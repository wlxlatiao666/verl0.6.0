# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)

    # Forward wandb credentials to Ray workers (workers do not inherit the caller's env).
    for _wandb_var in ("WANDB_API_KEY", "WANDB_KEY", "WANDB_MODE", "WANDB_DIR", "WANDB_RUN_ID", "WANDB_RESUME"):
        _val = os.environ.get(_wandb_var)
        if _val:
            runtime_env["env_vars"][_wandb_var] = _val

    # TensorBoard path (verl.utils.tracking._TensorboardAdapter reads TENSORBOARD_DIR).
    _tb = os.environ.get("TENSORBOARD_DIR")
    if _tb:
        runtime_env["env_vars"]["TENSORBOARD_DIR"] = _tb

    # verl rollout uses logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN")); forward so [TreeRollout] INFO appears in worker logs.
    _verl_log = os.environ.get("VERL_LOGGING_LEVEL")
    if _verl_log:
        runtime_env["env_vars"]["VERL_LOGGING_LEVEL"] = _verl_log

    # Forward VLLM_USE_V1 to Ray workers so we can force V0 engine for tree decoding.
    _vllm_use_v1 = os.environ.get("VLLM_USE_V1")
    if _vllm_use_v1 is not None:
        runtime_env["env_vars"]["VLLM_USE_V1"] = _vllm_use_v1

    return runtime_env
