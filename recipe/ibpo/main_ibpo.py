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
"""Run the IBPO recipe using verl's standard worker and dataset setup."""

from pathlib import Path

import hydra
import ray
from omegaconf import open_dict

from recipe.ibpo.ibpo_ray_trainer import IBPORayPPOTrainer
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.utils.device import auto_set_device


class IBPOTaskRunner(TaskRunner):
    trainer_class = IBPORayPPOTrainer


@hydra.main(config_path="config", config_name="ibpo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    repo_root = Path(__file__).resolve().parents[2]
    with open_dict(config):
        runtime_env = config.ray_kwargs.ray_init.get("runtime_env") or {}
        env_vars = dict(runtime_env.get("env_vars") or {})
        env_vars["PYTHONPATH"] = str(repo_root)
        runtime_env["env_vars"] = env_vars
        config.ray_kwargs.ray_init.runtime_env = runtime_env
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(IBPOTaskRunner))


if __name__ == "__main__":
    main()
