# Copyright 2026 Individual Contributor: d fei
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collect compare-and-correct trajectories before one joint GSPO update."""

from math import lcm

import numpy as np
import torch
from omegaconf import OmegaConf

from recipe.ibpo.algorithms import (
    compare_correction,
    correction_messages,
    joint_weights,
    select_references,
    validate_ibpo_config,
)
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.utils import tensordict_utils as tu
from verl.utils.debug import marked_timer
from verl.utils.py_functional import rename_dict
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.utils.padding import left_right_2_no_padding


def _objects(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


class IBPORayPPOTrainer(RayPPOTrainer):
    rollout_postprocessing = True

    def __init__(self, config, tokenizer, *args, **kwargs):
        self._validate_config(config, kwargs.get("processor"))
        self.ibpo_config = config.ibpo
        super().__init__(config, tokenizer, *args, **kwargs)

    @staticmethod
    def _validate_config(config, processor=None):
        ibpo = config.ibpo
        validate_ibpo_config(
            ibpo.variant,
            ibpo.shaping_weight,
            ibpo.rewrite_threshold,
            ibpo.correction_n,
            ibpo.aux_weight,
            ibpo.mask_fallback,
            ibpo.k,
        )
        actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
        if config.algorithm.adv_estimator != "ibpo" or rollout.n < 2:
            raise ValueError("IBPO requires adv_estimator=ibpo and rollout.n >= 2")
        if ibpo.rho != 0.5:
            raise ValueError("the fixed IBPO instance requires rho=0.5")
        if actor.policy_loss.loss_mode != "gspo" or actor.loss_agg_mode != "seq-mean-token-mean":
            raise ValueError("IBPO requires GSPO with seq-mean-token-mean aggregation")
        if actor.clip_ratio_low != actor.clip_ratio_high:
            raise ValueError("IBPO requires equal lower and upper clipping thresholds")
        if actor.use_kl_loss or config.algorithm.use_kl_in_reward or actor.entropy_coeff != 0:
            raise ValueError("IBPO does not mix KL penalties or entropy regularization into its objective")
        if OmegaConf.select(config, "critic.enable") or OmegaConf.select(config, "distillation.enabled"):
            raise ValueError("IBPO does not use a critic or distillation")
        correction = config.algorithm.get("rollout_correction")
        if correction and any(correction.get(key) for key in ("bypass_mode", "rollout_rs", "rollout_is")):
            raise ValueError("IBPO requires recomputed old log-probs without rollout IS or rejection sampling")
        if (
            processor is not None
            or rollout.multi_turn.enable
            or rollout.agent.default_agent_loop != "single_turn_agent"
            or rollout.agent.get("agent_loop_config_path")
            or rollout.multi_turn.get("tool_config_path")
            or rollout.multi_turn.get("function_tool_path")
        ):
            raise ValueError("IBPO currently supports only text-only single_turn_agent rollouts")
        if actor.get("use_prefix_grouper", False) or rollout.get("skip", {}).get("enable", False):
            raise ValueError("IBPO requires fresh rollouts and row-level batch balancing")
        if actor.get("ppo_micro_batch_size") is not None or rollout.get("log_prob_micro_batch_size") is not None:
            raise ValueError("IBPO requires per-GPU micro batch sizes instead of deprecated global micro batch sizes")

    def _binary_rewards(self, batch, timing_raw):
        with marked_timer("ibpo_verify", timing_raw):
            if "rm_scores" not in batch.batch:
                manager = getattr(self, "reward_loop_manager", None)
                workers = (
                    len(manager.reward_loop_workers)
                    if manager is not None
                    else OmegaConf.select(self.config, "reward.num_workers", default=1)
                )
                verifier_input, pad_size = pad_dataproto_to_divisor(batch, workers)
                verified = unpad_dataproto(self._compute_reward_colocate(verifier_input), pad_size)
                batch.meta_info.pop("reward_extra_keys", None)
                batch = batch.union(verified)
            rewards = batch.batch["rm_scores"].sum(-1).detach().cpu()
            if not torch.isfinite(rewards).all() or not ((rewards == 0) | (rewards == 1)).all():
                raise ValueError("IBPO requires unshaped binary {0,1} terminal correctness rewards")
        return batch, rewards

    def _response_tokens(self, batch, row):
        positions = batch.batch["response_mask"][row].bool().nonzero(as_tuple=True)[0].tolist()
        return [int(batch.batch["responses"][row, p]) for p in positions], positions

    def _render_model_candidate(self, tokens):
        boundaries = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        return self.tokenizer.decode([token for token in tokens if token not in boundaries], skip_special_tokens=False)

    def _padding_divisor(self):
        actor, rollout = self.config.actor_rollout_ref.actor, self.config.actor_rollout_ref.rollout
        train_micro = 1 if actor.get("use_dynamic_bsz", False) else actor.get("ppo_micro_batch_size_per_gpu") or 1
        infer_micro = (
            1
            if rollout.get("log_prob_use_dynamic_bsz", False)
            else rollout.get("log_prob_micro_batch_size_per_gpu") or 1
        )
        return self._get_dp_size(self.actor_rollout_wg, "actor") * lcm(train_micro, infer_micro)

    @staticmethod
    def _terminal_scores(batch, rewards):
        mask = batch.batch["response_mask"].bool()
        if not mask.any(-1).all():
            raise ValueError("IBPO requires at least one valid response token per generated trajectory")
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        terminal = positions.masked_fill(~mask, -1).max(-1).values
        scores = torch.zeros_like(mask, dtype=torch.float32)
        scores[torch.arange(len(batch), device=mask.device), terminal] = rewards.to(mask.device)
        return scores

    def _prepare_training_batch(self, batch, metrics, timing_raw):
        if batch.meta_info.get("validate", False):
            return batch
        timing_raw.setdefault("ibpo_generate", 0.0)
        metrics.update(
            {
                "ibpo/correction_input_tokens": 0,
                "ibpo/correction_output_tokens": 0,
                "ibpo/correction_generated_input_tokens": 0,
                "ibpo/correction_generated_output_tokens": 0,
                "ibpo/correction_generation_padding_samples": 0,
                "ibpo/correction_generated_samples": 0,
                "ibpo/correction_reward_mean": 0.0,
            }
        )
        required = {"uid", "raw_prompt", "reward_model", "extra_info", "data_source"}
        if not required.issubset(batch.non_tensor_batch):
            raise ValueError(f"IBPO needs original task metadata: {sorted(required - batch.non_tensor_batch.keys())}")
        for messages in batch.non_tensor_batch["raw_prompt"]:
            if any(not isinstance(message.get("content"), str) for message in messages):
                raise ValueError("IBPO supports text-only raw_prompt messages")
        for key in ("multi_modal_inputs", "multi_modal_data"):
            if key in batch.non_tensor_batch and any(bool(value) for value in batch.non_tensor_batch[key]):
                raise ValueError("IBPO does not support multimodal trajectories")
        if "agent_name" in batch.non_tensor_batch and any(
            name != "single_turn_agent" for name in batch.non_tensor_batch["agent_name"]
        ):
            raise ValueError("IBPO supports only single_turn_agent trajectories")
        batch.non_tensor_batch["multi_modal_inputs"] = _objects([{} for _ in range(len(batch))])
        batch, original_rewards = self._binary_rewards(batch, timing_raw)
        main_uids = batch.non_tensor_batch["uid"].copy()
        # Derive the RNG from the iteration so checkpoint resume preserves reference selection.
        rng = np.random.default_rng(self.ibpo_config.seed + self.global_steps)
        references = select_references(
            original_rewards.numpy(), main_uids, rng, group_size=self.config.actor_rollout_ref.rollout.n
        )
        error_rows = list(references)
        correction_n = self.ibpo_config.correction_n if self.ibpo_config.aux_weight > 0 else 1
        corr_batch = None
        corr_rewards = torch.empty(0)
        corr_uids, parent_uids = [], []
        if error_rows:
            raw_prompts, prompt_lengths = [], []
            for row in error_rows:
                target, _ = self._response_tokens(batch, row)
                reference, _ = self._response_tokens(batch, references[row])
                messages = correction_messages(
                    list(batch.non_tensor_batch["raw_prompt"][row]),
                    self._render_model_candidate(target),
                    self._render_model_candidate(reference),
                )
                template_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
                custom_template = self.config.actor_rollout_ref.model.get("custom_chat_template")
                if custom_template is not None:
                    template_kwargs["chat_template"] = custom_template
                prompt_ids = normalize_token_ids(
                    self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        **template_kwargs,
                    )
                )
                if len(prompt_ids) > self.config.actor_rollout_ref.rollout.prompt_length:
                    raise ValueError(
                        f"IBPO correction prompt has {len(prompt_ids)} tokens, exceeding "
                        f"rollout.prompt_length={self.config.actor_rollout_ref.rollout.prompt_length}; "
                        "increase the budget to preserve both complete candidates"
                    )
                raw_prompts.append(messages)
                prompt_lengths.append(len(prompt_ids))
            corr_inputs = batch.select_idxs(error_rows)
            # AgentLoop consumes raw_prompt; no tensors from an old generated trajectory belong in a new request.
            corr_inputs.batch = None
            input_keys = required | {"agent_name", "index", "multi_modal_inputs"}
            corr_inputs.non_tensor_batch = {
                key: value for key, value in corr_inputs.non_tensor_batch.items() if key in input_keys
            }
            corr_inputs.non_tensor_batch["raw_prompt"] = _objects(raw_prompts)
            corr_inputs.non_tensor_batch["ibpo_parent_uid"] = main_uids[error_rows].copy()
            corr_inputs.non_tensor_batch["uid"] = _objects(
                [f"ibpo-corr:{self.global_steps}:{row}:{main_uids[row]}" for row in error_rows]
            )
            corr_inputs.non_tensor_batch["agent_name"] = _objects(["single_turn_agent"] * len(error_rows))
            corr_inputs.meta_info = {
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
                "global_steps": self.global_steps,
                "validate": False,
            }
            corr_inputs = corr_inputs.repeat(correction_n, interleave=True)
            generation_inputs, generation_pad = pad_dataproto_to_divisor(
                corr_inputs, self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            with marked_timer("ibpo_generate", timing_raw):
                corr_batch = self.async_rollout_manager.generate_sequences(generation_inputs)
            if "response_mask" not in corr_batch.batch:
                corr_batch.batch["response_mask"] = compute_response_mask(corr_batch)
            response_width = corr_batch.batch["responses"].shape[1]
            metrics["ibpo/correction_generation_padding_samples"] = generation_pad
            metrics["ibpo/correction_generated_samples"] = len(generation_inputs)
            metrics["ibpo/correction_generated_input_tokens"] = (
                corr_batch.batch["attention_mask"][:, :-response_width].sum().item()
            )
            metrics["ibpo/correction_generated_output_tokens"] = corr_batch.batch["response_mask"].sum().item()
            corr_batch = unpad_dataproto(corr_batch, generation_pad)
            if len(corr_batch) != len(error_rows) * correction_n:
                raise ValueError("IBPO correction rollout returned a different number of trajectories")
            # Asynchronous reward generation may omit input metadata; restore original task identities explicitly.
            corr_batch.non_tensor_batch.update(corr_inputs.non_tensor_batch)
            corr_batch, corr_rewards = self._binary_rewards(corr_batch, timing_raw)
            corr_batch.batch["rm_scores"] = self._terminal_scores(corr_batch, corr_rewards)
            corr_batch.batch["ibpo_token_mask"] = torch.ones_like(corr_batch.batch["response_mask"])
            corr_batch.batch["ibpo_raw_reward"] = corr_rewards.to(corr_batch.batch["rm_scores"].device)
            corr_uids = corr_batch.non_tensor_batch["uid"].tolist()
            parent_uids = corr_batch.non_tensor_batch["ibpo_parent_uid"].tolist()
            metrics["ibpo/correction_input_tokens"] = sum(prompt_lengths) * correction_n
            metrics["ibpo/correction_output_tokens"] = corr_batch.batch["response_mask"].sum().item()
            metrics["ibpo/correction_reward_mean"] = corr_rewards.float().mean().item()

        shaped_rewards = original_rewards.float().clone()
        token_mask = torch.ones_like(batch.batch["response_mask"])
        successful, rewritten, empty = 0, 0, 0
        corr_offsets = {row: i * correction_n for i, row in enumerate(error_rows)}
        with marked_timer("ibpo_compare", timing_raw):
            for row in range(len(batch)):
                original, positions = self._response_tokens(batch, row)
                empty += not original
                corrected, reference, reward = [], [], 0
                if row in corr_offsets:
                    offset = corr_offsets[row]
                    corrected, _ = self._response_tokens(corr_batch, offset)
                    reference, _ = self._response_tokens(batch, references[row])
                    reward = corr_rewards[offset].item()
                comparison = compare_correction(
                    original,
                    corrected,
                    reference,
                    original_rewards[row].item(),
                    reward,
                    variant=self.ibpo_config.variant,
                    rewrite_threshold=self.ibpo_config.rewrite_threshold,
                    mask_fallback=self.ibpo_config.mask_fallback,
                )
                shaped_rewards[row] += self.ibpo_config.shaping_weight * comparison.delta
                successful += comparison.valid
                rewritten += comparison.rewritten
                if self.ibpo_config.variant == "mask":
                    token_mask[row].zero_()
                    if not comparison.valid and self.ibpo_config.mask_fallback == "unmasked":
                        token_mask[row].fill_(1)
                    elif positions:
                        token_mask[row, positions] = comparison.token_mask.to(token_mask.device, token_mask.dtype)
        batch.batch["rm_scores"] = self._terminal_scores(batch, shaped_rewards)
        batch.batch["ibpo_token_mask"] = token_mask
        batch.batch["ibpo_raw_reward"] = original_rewards.to(batch.batch["rm_scores"].device)
        batch.non_tensor_batch["ibpo_parent_uid"] = main_uids.copy()
        batch.non_tensor_batch["ibpo_row_type"] = _objects(["main"] * len(batch))
        main_meta = {key: value for key, value in batch.meta_info.items() if key not in ("metrics", "timing")}
        if corr_batch is not None and self.ibpo_config.aux_weight > 0:
            corr_batch.non_tensor_batch["ibpo_row_type"] = _objects(["correction"] * len(corr_batch))
            tensor_keys = sorted(set(batch.batch.keys()) & set(corr_batch.batch.keys()))
            non_tensor_keys = sorted(set(batch.non_tensor_batch) & set(corr_batch.non_tensor_batch))
            batch = batch.select(batch_keys=tensor_keys, non_tensor_batch_keys=non_tensor_keys)
            corr_batch = corr_batch.select(batch_keys=tensor_keys, non_tensor_batch_keys=non_tensor_keys)
            main_meta["reward_extra_keys"] = [
                key for key in main_meta.get("reward_extra_keys", []) if key in non_tensor_keys
            ]
            batch.meta_info = corr_batch.meta_info = main_meta
            batch = DataProto.concat([batch, corr_batch])
        else:
            corr_uids, parent_uids = [], []
            batch.meta_info = main_meta

        real_size = len(batch)
        batch, pad_size = pad_dataproto_to_divisor(batch, self._padding_divisor())
        if pad_size:
            for key in ("rm_scores", "response_mask", "ibpo_token_mask", "ibpo_raw_reward"):
                batch.batch[key][real_size:].zero_()
            for i in range(real_size, len(batch)):
                batch.non_tensor_batch["uid"][i] = f"ibpo-pad:{self.global_steps}:{i}"
                batch.non_tensor_batch["ibpo_parent_uid"][i] = batch.non_tensor_batch["uid"][i]
                batch.non_tensor_batch["ibpo_row_type"][i] = "padding"
        batch.batch["ibpo_row_weight"] = joint_weights(
            main_uids, parent_uids, corr_uids, self.ibpo_config.aux_weight, padded_size=len(batch)
        ).to(batch.batch["rm_scores"].device)
        metrics.update(
            {
                "ibpo/main_reward_mean": original_rewards.float().mean().item(),
                "ibpo/main_shaped_reward_mean": shaped_rewards.mean().item(),
                "ibpo/main_input_tokens": batch.batch["attention_mask"][: len(main_uids), : -token_mask.shape[1]]
                .sum()
                .item(),
                "ibpo/main_output_tokens": batch.batch["response_mask"][: len(main_uids)].sum().item(),
                "ibpo/errors": len(error_rows),
                "ibpo/q_passed": successful,
                "ibpo/rewrites": rewritten,
                "ibpo/empty_comparison_responses": empty,
                "ibpo/correction_samples": len(error_rows) * correction_n,
                "ibpo/train_rows": real_size,
                "ibpo/padded_train_rows": len(batch),
                "ibpo/padding_rows": pad_size,
            }
        )
        return batch

    def _update_actor(self, batch):
        batch.meta_info["multi_turn"] = False
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        batch_td = batch.to_tensordict()
        # Group standardization has already happened in fit; apply reduction weights afterwards.
        batch_td["advantages"] = batch_td["advantages"] * batch_td["ibpo_row_weight"].unsqueeze(-1)
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=False,
            distillation_use_topk=False,
            global_batch_size=len(batch),
            mini_batch_size=len(batch),
            epochs=1,
            seed=self.config.actor_rollout_ref.actor.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.actor_rollout_ref.actor.shuffle},
            compute_loss=True,
        )
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_metrics = rename_dict(tu.get(actor_output, "metrics"), "actor/")
        actor_metrics["perf/mfu/actor"] = actor_metrics.pop("actor/mfu")
        return DataProto.from_single_dict(data={}, meta_info={"metrics": actor_metrics})
