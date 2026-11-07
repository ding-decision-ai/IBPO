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
"""Compare-and-correct IBPO math; defaults describe this implementation, not recovered runs."""

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

from verl.trainer.ppo.core_algos import register_adv_est

CORRECTION_INSTRUCTION = (
    "You are given two candidate solutions to the same problem. "
    "Compare them step by step, identify any inconsistencies or errors, "
    "and then produce a corrected solution and final answer."
)


def _real(name, value, minimum=0.0, maximum=None, upper_inclusive=True):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if value < minimum or (maximum is not None and (value > maximum if upper_inclusive else value >= maximum)):
        bracket = "]" if upper_inclusive else ")"
        raise ValueError(f"{name} must be in [{minimum}, {maximum if maximum is not None else 'infinity'}{bracket}")


def _integer(name, value, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _variant(variant, mask_fallback):
    if variant not in ("fixed", "ratio", "mask"):
        raise ValueError("variant must be fixed, ratio, or mask")
    if mask_fallback not in (None, "unmasked", "skip") or (variant == "mask" and mask_fallback is None):
        raise ValueError("mask requires an explicit mask_fallback: unmasked or skip")


def validate_ibpo_config(variant, shaping_weight, rewrite_threshold, correction_n, aux_weight, mask_fallback, k=2):
    """Validate the paper's strict shaping design domain and explicit implementation choices."""
    _variant(variant, mask_fallback)
    _real("shaping_weight", shaping_weight, maximum=0.5 if variant == "ratio" else 1.0, upper_inclusive=False)
    _real("rewrite_threshold", rewrite_threshold, maximum=1.0)
    _integer("correction_n", correction_n)
    _real("aux_weight", aux_weight)
    _integer("k", k, minimum=2)
    if k != 2:
        raise ValueError("only K=2 is specified; multi-reference aggregation is not implemented")


def _groups(index):
    groups = defaultdict(list)
    for row, uid in enumerate(index):
        groups[str(uid)].append(row)
    return groups


def _binary_rewards(rewards):
    if isinstance(rewards, torch.Tensor):
        rewards = rewards.detach().cpu().numpy()
    rewards = np.asarray(rewards)
    if rewards.ndim != 1 or not np.isin(rewards, (0, 1)).all():
        raise ValueError("terminal correctness rewards must be a one-dimensional binary {0,1} array")
    return rewards


def select_references(rewards, index, rng, group_size=None) -> dict[int, int]:
    """Uniformly choose one same-prompt reference for each failed response, never itself."""
    rewards = _binary_rewards(rewards)
    if len(index) != len(rewards):
        raise ValueError("index and rewards must have the same length")
    if group_size is not None:
        _integer("group_size", group_size, minimum=2)
    references = {}
    for rows in _groups(index).values():
        if len(rows) < 2 or (group_size is not None and len(rows) != group_size):
            raise ValueError("every original prompt must have a complete group of at least two responses")
        correct = [row for row in rows if rewards[row] == 1]
        for row in rows:
            if rewards[row] == 0:
                candidates = correct or [other for other in rows if other != row]
                references[row] = int(rng.choice(candidates))
    return references


def correction_messages(messages, target, reference) -> list[dict]:
    """Keep the original text-only conversation and extend its final user question."""
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("correction input must end in the original user question")
    if not isinstance(target, str) or not isinstance(reference, str):
        raise ValueError("candidate solutions must be text")
    if any(not isinstance(message.get("content"), str) for message in messages):
        raise ValueError("IBPO correction currently supports text-only messages")
    result = deepcopy(messages)
    result[-1]["content"] += (
        f"\n\n{CORRECTION_INSTRUCTION}\n\n"
        f"Candidate solution 1 (to correct):\n{target}\n\n"
        f"Candidate solution 2 (reference):\n{reference}"
    )
    return result


@dataclass(frozen=True)
class CorrectionComparison:
    delta: float
    token_mask: torch.Tensor
    valid: bool
    rewritten: bool


def _tokens(sequence):
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.detach().cpu().tolist()
    sequence = list(sequence)
    if any(isinstance(token, bool) or not isinstance(token, Integral) for token in sequence):
        raise ValueError("edit distance requires a one-dimensional integer token sequence")
    return sequence


def compare_correction(
    original,
    corrected,
    reference,
    original_reward,
    correction_reward,
    variant="fixed",
    rewrite_threshold=0.5,
    mask_fallback=None,
) -> CorrectionComparison:
    """Compare generated token IDs. Special-token removal is the caller's explicit convention.

    The rewrite gate applies to shaping/masks only; auxiliary correctness rewards are unchanged.
    Unit-cost Levenshtein is normalized by the longer sequence. Editops determine one alignment;
    insertions have no original-response gradient coordinate.
    """
    _variant(variant, mask_fallback)
    _real("rewrite_threshold", rewrite_threshold, maximum=1.0)
    original_reward, correction_reward = _binary_rewards([original_reward, correction_reward])
    original, corrected, reference = (_tokens(sequence) for sequence in (original, corrected, reference))
    eligible = original_reward == 0 and correction_reward == 1
    rewritten = False
    if eligible:
        distance = Levenshtein.normalized_distance(original, corrected)
        rewritten = distance > rewrite_threshold and distance > Levenshtein.normalized_distance(corrected, reference)
    valid = bool(eligible and not rewritten)
    changed = torch.zeros(len(original), dtype=torch.bool)
    if valid and variant in ("ratio", "mask"):
        for operation in Levenshtein.editops(original, corrected):
            if operation.tag != "insert":
                changed[operation.src_pos] = True
    if not valid:
        delta = 0.0
    elif variant == "ratio":
        delta = float((~changed).sum()) / len(original) if original else 0.0
    else:
        delta = 0.5
    token_mask = torch.ones(len(original), dtype=torch.bool)
    if variant == "mask":
        token_mask = changed if valid else torch.full((len(original),), mask_fallback == "unmasked", dtype=torch.bool)
    return CorrectionComparison(delta, token_mask, valid, bool(rewritten))


def joint_weights(main_uids, correction_parent_uids, correction_uids, aux_weight, padded_size=None) -> torch.Tensor:
    """Turn a combined row mean into a prompt mean plus eta * within-prompt correction mean."""
    _real("aux_weight", aux_weight)
    main_uids = [str(uid) for uid in main_uids]
    parents = [str(uid) for uid in correction_parent_uids]
    corrections = [str(uid) for uid in correction_uids]
    if not main_uids or len(parents) != len(corrections):
        raise ValueError("main rows must be nonempty and correction parent/group arrays must align")
    row_count = len(main_uids) + len(corrections)
    padded_size = row_count if padded_size is None else padded_size
    _integer("padded_size", padded_size)
    if padded_size < row_count:
        raise ValueError("padded_size cannot discard real training rows")
    main_counts, correction_counts = Counter(main_uids), Counter(corrections)
    correction_parent = {}
    groups_per_parent = defaultdict(set)
    for parent, uid in zip(parents, corrections, strict=True):
        if parent not in main_counts or (uid in correction_parent and correction_parent[uid] != parent):
            raise ValueError("each correction group must belong to exactly one existing original prompt")
        if uid in main_counts:
            raise ValueError("main and correction advantage group IDs must be distinct")
        correction_parent[uid] = parent
        groups_per_parent[parent].add(uid)
    prompt_count = len(main_counts)
    weights = torch.zeros(padded_size, dtype=torch.float32)
    for row, uid in enumerate(main_uids):
        weights[row] = padded_size / (prompt_count * main_counts[uid])
    for row, (parent, uid) in enumerate(zip(parents, corrections, strict=True), start=len(main_uids)):
        weights[row] = (
            padded_size * aux_weight / (prompt_count * len(groups_per_parent[parent]) * correction_counts[uid])
        )
    return weights


@register_adv_est("ibpo")
@torch.no_grad()
def compute_ibpo_outcome_advantage(token_level_rewards, response_mask, index, config=None, **kwargs):
    """Population-std outcome advantages, normalized separately for each original/correction input."""
    if token_level_rewards.ndim != 2 or token_level_rewards.shape != response_mask.shape:
        raise ValueError("rewards and response_mask must be matching [batch, response_length] tensors")
    if len(index) != token_level_rewards.shape[0]:
        raise ValueError("index must contain one advantage-group UID per response")
    if not torch.isfinite(token_level_rewards).all() or not ((response_mask == 0) | (response_mask == 1)).all():
        raise ValueError("outcome rewards must be finite and response_mask must be binary")
    rewards = token_level_rewards.detach()
    if not rewards.is_floating_point():
        rewards = rewards.float()
    mask = response_mask.to(device=rewards.device, dtype=torch.bool)
    scores = (rewards * mask).sum(-1)
    sequence_advantages = torch.zeros_like(scores)
    for rows in _groups(index).values():
        group = scores[rows]
        sequence_advantages[rows] = (group - group.mean()) / (group.std(correction=0) + 1e-8)
    advantages = sequence_advantages.unsqueeze(-1) * mask
    return advantages, advantages
