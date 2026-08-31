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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import getpass
import inspect
import logging
import math
import os
import pickle
import socket
import time
from contextlib import contextmanager
from dataclasses import asdict
from types import MethodType
from typing import Any, Generator

import numpy as np
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import ListConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel, LoRAConfig
from vllm.lora.request import LoRARequest
from vllm.sampling_params import TreeSearchParams

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    # https://github.com/vllm-project/vllm/commit/6a113d9aed8221a9c234535958e70e34ab6cac5b
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils.device import is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.tree_training import is_tree_process_reward_enabled
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        if config.layered_summon:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        model_path = model_config.local_path
        tokenizer = model_config.tokenizer
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code
        self.lora_kwargs = (
            {"enable_lora": True, "max_loras": 1, "max_lora_rank": model_config.lora_rank}
            if model_config.lora_rank > 0
            else {}
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        # copy it to avoid secretly modifying the engine config
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}

        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_config["compilation_config"] = CompilationConfig(
                    level=CompilationLevel.PIECEWISE, cudagraph_capture_sizes=cudagraph_capture_sizes
                )
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=config.enable_prefix_caching,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **compilation_config,
            **self.lora_kwargs,
            **engine_kwargs,
        )

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            repetition_penalty=config.get("repetition_penalty", 1.0),
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        # Tree search params — read from rollout config's tree_search sub-config
        _tree_cfg = config.get("tree_search", None)
        if _tree_cfg is not None and _tree_cfg.get("enable", False):
            self.sampling_params.tree_search_params = TreeSearchParams(
                enable_tree_search=True,
                entropy_threshold=float(_tree_cfg.get("entropy_threshold", 1.0)),
                branching_factor=int(_tree_cfg.get("branching_factor", 2)),
                max_tree_depth=int(_tree_cfg.get("max_tree_depth", 3)),
                tau_importance=float(_tree_cfg.get("tau_importance", 0.0)),
                min_seg_length=int(_tree_cfg.get("min_seg_length", 10)),
                branch_trigger_mode=_tree_cfg.get("branch_trigger_mode", None),
                branch_sampling=str(_tree_cfg.get("branch_sampling", "sample")),
                branch_temperature=float(_tree_cfg.get("branch_temperature", 1.0)),
            )
            logger.info(f"[TreeRollout] TreeSearchParams enabled: {self.sampling_params.tree_search_params}")

        self.pad_token_id = tokenizer.pad_token_id

    def update_entropy_threshold(self, threshold: float):
        """Dynamically update the entropy_threshold for tree search."""
        if self.sampling_params.tree_search_params is not None:
            self.sampling_params.tree_search_params.entropy_threshold = threshold
            logger.info(f"[TreeRollout] entropy_threshold updated to {threshold:.4f}")

    def update_tau_importance(self, tau: float):
        """Dynamically update the tau_importance for tree search."""
        if self.sampling_params.tree_search_params is not None:
            self.sampling_params.tree_search_params.tau_importance = tau
            logger.info(f"[TreeRollout] tau_importance updated to {tau:.4f}")

    @GPUMemoryLogger(role="vllm rollout spmd collect_threshold_stats", logger=logger)
    @torch.no_grad()
    def collect_threshold_stats(self, prompts: DataProto) -> DataProto:
        """Run a forward pass with collect_threshold_stats=True to compute
        per-token entropy and importance lists, then return their p80 values.

        Returns a DataProto with meta_info containing:
            - "entropy_p80": float
            - "importance_p80": float or None (if importance_list is empty)
        """
        idx = prompts.batch["input_ids"]
        batch_size = idx.size(0)

        non_tensor_batch = dict(prompts.non_tensor_batch)
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_ids), "multi_modal_data": mm}
                for raw_ids, mm in zip(non_tensor_batch["raw_prompt_ids"], non_tensor_batch["multi_modal_data"])
            ]
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_ids)} for raw_ids in non_tensor_batch["raw_prompt_ids"]]

        _tree_cfg = self.config.get("tree_search", {})
        stats_n = int(_tree_cfg.get("threshold_stats_n", 1))
        _stats_max_tokens = int(_tree_cfg.get("threshold_stats_max_tokens", 0))
        _max_tokens = _stats_max_tokens if _stats_max_tokens > 0 else int(self.sampling_params.max_tokens)

        stats_params = SamplingParams(
            n=stats_n,
            temperature=float(self.sampling_params.temperature),
            max_tokens=_max_tokens,
            collect_threshold_stats=True,
        )

        outputs = self.inference_engine.generate(
            prompts=vllm_inputs,
            sampling_params=stats_params,
            use_tqdm=False,
        )

        all_entropy: list[float] = []
        all_importance: list[float] = []
        for output in outputs:
            for completion in output.outputs:
                # Filter out nan values that can arise from numerical instability
                # (e.g. 0.0 * (-inf) in entropy computation) before computing percentile.
                all_entropy.extend(v for v in completion.entropy_list if not math.isnan(v))
                all_importance.extend(v for v in completion.importance_list if v is not None)

        entropy_p80: float = float(np.percentile(all_entropy, 80)) if all_entropy else 1.0
        importance_p80: float | None = float(np.percentile(all_importance, 80)) if all_importance else None

        logger.info(
            f"[TreeRollout] collect_threshold_stats: entropy_p80={entropy_p80:.4f}, "
            f"importance_p80={importance_p80}"
        )

        result = DataProto.from_single_dict({})
        result.meta_info["entropy_p80"] = entropy_p80
        result.meta_info["importance_p80"] = importance_p80
        return result

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        _tree_cfg = self.config.get("tree_search", None)
        _tree_search_active = bool(
            _tree_cfg is not None
            and _tree_cfg.get("enable", False)
            and do_sample
            and not is_validate
        )
        _tree_process_reward = is_tree_process_reward_enabled(
            _tree_cfg,
            is_validate=is_validate,
            do_sample=do_sample,
        )
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
                "tree_search_params": None,  # disable tree search during validation
            }
        elif is_validate:
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
                "tree_search_params": None,
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        _tree_metrics: dict = {}
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            # unique_segments: one entry per unique tree node (deduped by seq_id)
            # leaf_segment_indices: for each leaf, the ordered list of indices into unique_segments representing its root→leaf path
            unique_segments: list[list[int]] = []
            seq_id_to_segment_idx: dict[int, int] = {}
            leaf_segment_indices: list[list[int]] = []
            rollout_log_probs = []
            prompt_indices = []  # Track which prompt each response belongs to
            # Per-prompt leaf counts (tree leaves now; top-ups added later at line ~690).
            # Indexed by prompt position within `outputs`.
            _leaves_per_prompt = [0] * len(outputs)
            # ----- Inverse-sharing weight support -----
            # Intermediate storage for leaf records (collected in the first pass so we can
            # compute node_descendant_count before producing per-token weights).
            _leaf_records: list[tuple[list[tuple[int, list[int]]], list[int], int]] = []
            _global_seq_map: dict[int, object] = {}
            token_share_weights: list[list[float]] = []
            _has_tree_by_prompt: list[bool] = []

            for out_idx, output in enumerate(outputs):
                seq_map = {out.seq_id: out for out in output.outputs}
                _global_seq_map.update(seq_map)
                has_tree = any(getattr(s, 'is_leaf', False) for s in output.outputs)
                _has_tree_by_prompt.append(has_tree)

                samples_to_collect = output.outputs
                for sample in samples_to_collect:
                    if has_tree:
                        response_ids = []
                        if getattr(sample, 'is_leaf', False):
                            # Walk from leaf to root, collecting (seq_id, tree_ids) along the path
                            path_nodes: list[tuple[int, list[int]]] = []
                            current = sample
                            while current is not None:
                                print(f"segment length: {len(current.tree_ids)}, segment depth: {current.tree_depth}")
                                path_nodes.append((current.seq_id, current.tree_ids))
                                if current.parent_seq_id is not None and current.parent_seq_id in seq_map:
                                    current = seq_map[current.parent_seq_id]
                                else:
                                    current = None
                            # Reverse to get root→leaf order
                            path_nodes.reverse()
                            response_ids = sum([seg for _, seg in path_nodes], [])

                            # Register each node in unique_segments (dedup by seq_id)
                            if _tree_process_reward:
                                path_indices: list[int] = []
                                for seq_id, seg in path_nodes:
                                    if seq_id not in seq_id_to_segment_idx:
                                        seq_id_to_segment_idx[seq_id] = len(unique_segments)
                                        unique_segments.append(seg)
                                    path_indices.append(seq_id_to_segment_idx[seq_id])
                                leaf_segment_indices.append(path_indices)

                            # Save the leaf record for second-pass processing
                            _leaf_records.append((path_nodes, response_ids, out_idx))
                    elif not has_tree:
                        response_ids = sample.token_ids
                        if response_ids:
                            response.append(response_ids)
                            if self.config.calculate_log_probs:
                                curr_log_prob = []
                                for i, logprob in enumerate(sample.logprobs):
                                    if i < len(response_ids):
                                        tok = response_ids[i]
                                        curr_log_prob.append(
                                            logprob[tok].logprob if tok in logprob else 0.0
                                        )
                                    else:
                                        curr_log_prob.append(0.0)
                                rollout_log_probs.append(curr_log_prob)

            if _tree_search_active:
                missing_tree_prompts = [
                    prompt_idx
                    for prompt_idx, prompt_has_tree in enumerate(_has_tree_by_prompt)
                    if not prompt_has_tree
                ]
                if len(outputs) != len(vllm_inputs) or missing_tree_prompts:
                    raise RuntimeError(
                        "Active tree search returned a mixed or incomplete output batch. "
                        f"Expected tree-formatted output for {len(vllm_inputs)} prompts, got {len(outputs)}; "
                        f"non-tree prompt indices={missing_tree_prompts[:8]}."
                    )

            # ----- Second pass: process tree leaves (count descendants + emit records) -----
            if _leaf_records:
                _node_descendant_count: dict[int, int] = {}
                # Inverse-sharing weights are part of tree process reward, not
                # plain tree rollout. Avoid changing the treerollout baseline's
                # loss merely because its responses came from a tree.
                if _tree_process_reward:
                    for path_nodes, _, _ in _leaf_records:
                        for seq_id, _ in path_nodes:
                            _node_descendant_count[seq_id] = _node_descendant_count.get(seq_id, 0) + 1

                # Emit each leaf's response / prompt index / rollout_log_probs.
                # token_share_weights are emitted only for tree process reward.
                for path_nodes, response_ids, out_idx in _leaf_records:
                    response.append(response_ids)
                    prompt_indices.append(out_idx)
                    _leaves_per_prompt[out_idx] += 1

                    if self.config.calculate_log_probs:
                        curr_log_prob: list[float] = []
                        for seg_id, seg_tokens in path_nodes:
                            seg_sample = _global_seq_map.get(seg_id)
                            if seg_sample is not None and getattr(seg_sample, 'logprobs', None) is not None:
                                for j, logprob_entry in enumerate(seg_sample.logprobs):
                                    if j < len(seg_tokens):
                                        tok = seg_tokens[j]
                                        if tok in logprob_entry:
                                            curr_log_prob.append(logprob_entry[tok].logprob)
                                        elif len(logprob_entry) > 0:
                                            first_key = next(iter(logprob_entry.keys()))
                                            curr_log_prob.append(logprob_entry[first_key].logprob)
                                        else:
                                            curr_log_prob.append(0.0)
                                    else:
                                        curr_log_prob.append(0.0)
                            else:
                                curr_log_prob.extend([0.0] * len(seg_tokens))
                        rollout_log_probs.append(curr_log_prob)

                    if _tree_process_reward:
                        # Token-level inverse-sharing weight:
                        # 1 / descendant_count for each node's tokens.
                        leaf_weights: list[float] = []
                        for seg_id, seg_tokens in path_nodes:
                            cnt = float(_node_descendant_count.get(seg_id, 1))
                            leaf_weights.extend([1.0 / cnt] * len(seg_tokens))
                        if len(leaf_weights) != len(response_ids):
                            raise RuntimeError(
                                "[token_share_weights] length mismatch: "
                                f"leaf_weights={len(leaf_weights)} response_ids={len(response_ids)}"
                            )
                        token_share_weights.append(leaf_weights)

                # Optional debug stats (print first 3 leaves)
                if _tree_process_reward:
                    for _i, (_p, r, _o) in enumerate(_leaf_records[:3]):
                        w = token_share_weights[_i]
                        counts = [_node_descendant_count.get(s, 0) for s, _ in _p]
                        segs = [(s, len(t), c) for (s, t), c in zip(_p, counts)]
                        print(
                            f"[DEBUG][token_share_weights] leaf {_i}: "
                            f"resp_len={len(r)} n_segments={len(_p)} "
                            f"segments(seg_id, seg_len, desc_count)={segs} "
                            f"w_min={min(w):.4f} w_max={max(w):.4f} w_sum={sum(w):.4f}"
                        )

            # ── Step 2 (NEW): TOP-UP leaves with conventional sampling ──────
            # Target = bf ** max_depth. For every prompt whose tree produced
            # fewer leaves, we run conventional (non-branching) sampling with
            # the SAME temperature / top_p / top_k / min_p / ... as the active
            # rollout params, but with tree_search_params = None. These
            # conventional samples are appended after tree leaves and share
            # the same prompt uid so GRPO groups them together.
            _topup_total = 0
            _topup_per_prompt: dict[int, int] = {}
            _tree_cfg = self.config.get("tree_search", None)
            if _tree_search_active and _tree_cfg is not None and _leaves_per_prompt:
                _bf    = int(_tree_cfg.get("branching_factor", 2))
                _md    = int(_tree_cfg.get("max_tree_depth",   3))
                _tgt   = _bf ** _md
                _topup_enable = bool(_tree_cfg.get("topup_leaves_to_target", True))
                if _topup_enable and min(_leaves_per_prompt) < _tgt:
                    # Build per-prompt deficit
                    _tu_inputs: list[dict] = []
                    _tu_n:      list[int]  = []
                    _tu_map:    list[int]  = []
                    for _p_out in range(len(outputs)):
                        deficit = max(0, _tgt - _leaves_per_prompt[_p_out])
                        if deficit > 0:
                            _tu_inputs.append(vllm_inputs[_p_out])
                            _tu_n.append(deficit)
                            _tu_map.append(_p_out)
                    if _tu_inputs:
                        # Clone sampling params (no tree branching) + set n=need
                        _base = self.sampling_params
                        _lp_req = 1 if self.config.calculate_log_probs else 0
                        _tu_sp_list = []
                        for _def in _tu_n:
                            _tu_sp_list.append(
                                SamplingParams(
                                    n=_def,
                                    temperature=float(_base.temperature),
                                    top_p=float(_base.top_p),
                                    top_k=int(_base.top_k),
                                    min_p=(float(_base.min_p)
                                           if _base.min_p is not None else 0.0),
                                    repetition_penalty=float(_base.repetition_penalty),
                                    presence_penalty=float(_base.presence_penalty),
                                    frequency_penalty=float(_base.frequency_penalty),
                                    max_tokens=int(_base.max_tokens),
                                    logprobs=_lp_req,
                                    tree_search_params=None,
                                )
                            )
                        logger.info(
                            f"[TreeRollout][TopUp] bf={_bf}, depth={_md}, target={_tgt}/prompt. "
                            f"Top-up needed for {len(_tu_inputs)}/{len(outputs)} prompts: "
                            f"total {sum(_tu_n)} conventional samples."
                        )
                        with torch.no_grad():
                            _tu_outputs = self.inference_engine.generate(
                                prompts=_tu_inputs,
                                sampling_params=_tu_sp_list,
                                lora_request=None,
                                use_tqdm=False,
                            )

                        _pre_leaf_count = len(response)

                        for _rel, (_tu_out, p_out) in enumerate(
                            zip(_tu_outputs, _tu_map)
                        ):
                            got = 0
                            for _tu_sample in _tu_out.outputs:
                                resp_ids = list(_tu_sample.token_ids)
                                if not resp_ids:
                                    continue

                                # 1) responses + prompt routing (same prompt index)
                                response.append(resp_ids)
                                prompt_indices.append(p_out)
                                got += 1

                                # 2) rollout_log_probs (if required)
                                if self.config.calculate_log_probs:
                                    curr_lp: list[float] = []
                                    for _j, _lpe in enumerate(
                                        getattr(_tu_sample, "logprobs", []) or []
                                    ):
                                        if _j < len(resp_ids):
                                            tok = resp_ids[_j]
                                            if _lpe and tok in _lpe:
                                                curr_lp.append(_lpe[tok].logprob)
                                            elif _lpe:
                                                k0 = next(iter(_lpe.keys()))
                                                curr_lp.append(_lpe[k0].logprob)
                                            else:
                                                curr_lp.append(0.0)
                                        else:
                                            curr_lp.append(0.0)
                                    pad_n = max(0, len(resp_ids) - len(curr_lp))
                                    if pad_n:
                                        curr_lp.extend([0.0] * pad_n)
                                    rollout_log_probs.append(curr_lp)

                                # 3) A conventional top-up has no shared
                                # ancestors, so its process-reward weight is 1.
                                if _tree_process_reward:
                                    token_share_weights.append([1.0] * len(resp_ids))

                                # 4) tree_process_reward single synthetic segment
                                if _tree_process_reward:
                                    seg_idx = len(unique_segments)
                                    unique_segments.append(list(resp_ids))
                                    leaf_segment_indices.append([seg_idx])

                            _topup_per_prompt[p_out] = got
                            _leaves_per_prompt[p_out] += got
                            _topup_total += got

                        logger.info(
                            f"[TreeRollout][TopUp] Done. Collected "
                            f"{len(response) - _pre_leaf_count} samples "
                            f"(requested {sum(_tu_n)}). "
                            f"Total responses now: {len(response)}."
                        )

            # ── Summary print for top-up ──────────────────────────────────────
            if _tree_search_active and _tree_cfg is not None:
                _bf = int(_tree_cfg.get("branching_factor", 2))
                _md = int(_tree_cfg.get("max_tree_depth",   3))
                _tgt = _bf ** _md
                _topup_enable = bool(_tree_cfg.get("topup_leaves_to_target", True))
                _hits = sum(1 for n in _leaves_per_prompt if n >= _tgt) if _leaves_per_prompt else 0
                print(
                    f"[TreeRollout][TopUp-Summary] target_leaves/prompt={_tgt} (bf={_bf}, depth={_md}). "
                    f"Prompts at target: {_hits}/{len(_leaves_per_prompt) if _leaves_per_prompt else 0}. "
                    f"Top-up samples total: {_topup_total}. "
                    f"Per-prompt leaves: min={min(_leaves_per_prompt) if _leaves_per_prompt else 0}, "
                    f"max={max(_leaves_per_prompt) if _leaves_per_prompt else 0}, "
                    f"mean={(sum(_leaves_per_prompt)/len(_leaves_per_prompt)) if _leaves_per_prompt else 0:.2f}."
                )
                if _topup_enable:
                    incomplete_prompts = [
                        prompt_idx
                        for prompt_idx, num_leaves in enumerate(_leaves_per_prompt)
                        if num_leaves != _tgt
                    ]
                    if incomplete_prompts:
                        raise RuntimeError(
                            "Tree top-up must produce the fixed response count used for PPO batch sizing. "
                            f"Expected {_tgt} leaves per prompt, but prompts {incomplete_prompts[:8]} "
                            f"have counts {[_leaves_per_prompt[i] for i in incomplete_prompts[:8]]}."
                        )

            if _tree_search_active and len(prompt_indices) != len(response):
                raise RuntimeError(
                    "Tree response routing must contain one prompt index per response, "
                    f"got prompt_indices={len(prompt_indices)} responses={len(response)}."
                )

            # When tree search produces more responses than prompts,
            # expand prompt tensors and non_tensor_batch to match
            if len(response) != batch_size:
                prompt_indices_t = torch.tensor(prompt_indices, device=idx.device)
                idx = idx[prompt_indices_t]
                attention_mask = attention_mask[prompt_indices_t]
                position_ids = position_ids[prompt_indices_t]
                # Expand non_tensor_batch so every key matches the new batch size
                expanded_ntb = {}
                for k, v in non_tensor_batch.items():
                    expanded_ntb[k] = v[np.array(prompt_indices)]
                non_tensor_batch = expanded_ntb
                batch_size = len(response)
                logger.info(f"[TreeRollout] Expanded batch: {len(outputs)} prompts -> {batch_size} leaf responses")

            # Always write tree routing metadata when tree search is active so that
            # DataProto.concat across workers sees consistent keys and lengths.
            print("prompt_indices:", prompt_indices)
            if prompt_indices:
                print("prompt_indices True")
                non_tensor_batch["tree_prompt_indices"] = np.array(prompt_indices)
                non_tensor_batch["tree_num_leaves"] = np.array([len(response)] * len(response))
                non_tensor_batch["tree_num_prompts"] = np.array([len(outputs)] * len(response))

            # Store segment-level data for tree responses.
            # unique_segments: token list per unique tree node (deduped by seq_id), shape (n_unique_nodes,)
            # leaf_segment_indices: for each leaf, ordered indices into unique_segments for its root→leaf path
            #                       shape (n_leaves,) of variable-len index arrays
            if _tree_process_reward:
                # Always write these keys (even if empty) so all workers have the same keys
                # NOTE: unique_segments is stored in non_tensor_batch
                # rather than meta_info['metrics'] so they are properly partitioned per-worker
                if unique_segments:
                    non_tensor_batch["unique_segments"] = np.array(unique_segments, dtype=object)
                else:
                    # Write empty arrays with the right dtype
                    non_tensor_batch["unique_segments"] = np.array([], dtype=object)
                # leaf_segment_indices aligns with batch dimension (one entry per leaf response)
                if leaf_segment_indices:
                    leaf_seg_arr = np.empty(len(leaf_segment_indices), dtype=object)
                    for i, v in enumerate(leaf_segment_indices):
                        leaf_seg_arr[i] = v
                    non_tensor_batch["leaf_segment_indices"] = leaf_seg_arr

            # ── Compute tree search metrics ──
            _tree_total, _tree_leaves = 0, 0
            _depth_sum, _depth_max = 0, 0
            # NOTE: _leaves_per_prompt already contains tree leaves + top-ups; we
            # just need tree-only stats (_tree_leaves) and quality signals.
            _tree_only_leaves: list[int] = []
            _leaf_resp_lens = []

            # Debug: Print per-prompt tree statistics
            print(f"[TreeRollout] Number of prompts (outputs): {len(outputs)}")
            print(f"[TreeRollout] Total unique_segments collected so far: {len(unique_segments)}")
            print(f"[TreeRollout] Total leaf_segment_indices collected so far: {len(leaf_segment_indices)}")

            for _out_idx, _out in enumerate(outputs):
                _seqs = _out.outputs
                _tree_total += len(_seqs)
                prompt_leaf_count = 0
                prompt_max_depth = 0
                prompt_total_nodes = 0
                prompt_segments_before = len(unique_segments)
                for _s in _seqs:
                    depth = getattr(_s, 'tree_depth', 0)
                    prompt_total_nodes += 1
                    if isinstance(depth, (int, float)):
                        prompt_max_depth = max(prompt_max_depth, depth)
                    if getattr(_s, 'is_leaf', True):
                        prompt_leaf_count += 1
                        _leaf_resp_lens.append(len(_s.token_ids))
                _tree_leaves += prompt_leaf_count
                _tree_only_leaves.append(prompt_leaf_count)
                _depth_sum += prompt_max_depth
                _depth_max = max(_depth_max, prompt_max_depth)

                # Debug per-prompt tree info
                prompt_new_segments = len(unique_segments) - prompt_segments_before
                _tu = _topup_per_prompt.get(_out_idx, 0) if '_topup_per_prompt' in locals() else 0
                print(f"[TreeRollout] Prompt {_out_idx}: {prompt_total_nodes} total nodes, "
                      f"{prompt_leaf_count} tree-leaves + {_tu} topup = {prompt_leaf_count+_tu} total, "
                      f"max_depth={prompt_max_depth}, "
                      f"{prompt_new_segments} new segments (segments/leaf ratio: {prompt_total_nodes / max(prompt_leaf_count, 1):.2f})")

            n_prompts = max(len(outputs), 1)
            _tree_branch_pts = _tree_total - _tree_leaves
            _all_n_leaves = _leaves_per_prompt if _leaves_per_prompt else [0]
            avg_leaves = sum(_all_n_leaves) / len(_all_n_leaves)
            _min_l  = min(_all_n_leaves) if _all_n_leaves else 0
            _max_l  = max(_all_n_leaves) if _all_n_leaves else 0
            _avg_tree = (sum(_tree_only_leaves) / len(_tree_only_leaves)) if _tree_only_leaves else 0

            # Metrics: merge top-up info (produced earlier if any, defaults 0)
            _tree_metrics["tree/topup_samples"] = _topup_total if '_topup_total' in locals() else 0
            _tree_metrics["tree/prompts_need_topup"] = (
                sum(1 for _p in range(len(outputs))
                    if (_topup_per_prompt.get(_p, 0) if '_topup_per_prompt' in locals() else 0) > 0)
            )

            _tree_metrics.update({
                # -- Structure: how large/deep is the tree? --
                "tree/total_nodes": _tree_total,
                "tree/leaf_nodes": _tree_leaves,
                "tree/branch_points": _tree_branch_pts,
                "tree/avg_max_depth": round(_depth_sum / n_prompts, 4),
                "tree/global_max_depth": _depth_max,
                # -- Efficiency: branching budget utilisation --
                "tree/branching_rate": round(_tree_branch_pts / max(_tree_total, 1), 4),
                "tree/avg_tree_leaves_per_prompt": round(_avg_tree, 2),
                "tree/avg_leaves_per_prompt": round(avg_leaves, 2),
                "tree/min_leaves_per_prompt": _min_l,
                "tree/max_leaves_per_prompt": _max_l,
                # -- Response quality signals (tree leaves only) --
                "tree/avg_leaf_resp_len": round(sum(_leaf_resp_lens) / max(len(_leaf_resp_lens), 1), 1) if _leaf_resp_lens else 0,
                "tree/min_leaf_resp_len": min(_leaf_resp_lens) if _leaf_resp_lens else 0,
                "tree/max_leaf_resp_len": max(_leaf_resp_lens) if _leaf_resp_lens else 0,
                # -- Expansion ratio (tree leaves + top-ups) --
                "tree/expansion_ratio": round(avg_leaves, 2),
            })

            if _tree_process_reward:
                if len(token_share_weights) != len(response):
                    raise RuntimeError(
                        "Tree process reward must emit exactly one token_share_weights row "
                        f"per response, got weights={len(token_share_weights)} responses={len(response)}."
                    )
                if len(leaf_segment_indices) != len(response):
                    raise RuntimeError(
                        "Tree process reward must emit exactly one segment path per response, "
                        f"got paths={len(leaf_segment_indices)} responses={len(response)}."
                    )
                if not all(
                    len(weights) == len(tokens)
                    for weights, tokens in zip(token_share_weights, response)
                ):
                    raise RuntimeError(
                        "Tree process reward token_share_weights must align with every unpadded response."
                    )
                invalid_segment_paths = [
                    path_idx
                    for path_idx, path in enumerate(leaf_segment_indices)
                    if any(segment_idx < 0 or segment_idx >= len(unique_segments) for segment_idx in path)
                ]
                if invalid_segment_paths:
                    raise RuntimeError(
                        "Tree process reward emitted out-of-range segment indices for response paths "
                        f"{invalid_segment_paths[:8]}."
                    )

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)
            # ----- token_share_weights (inverse-sharing) -----
            if _tree_process_reward and token_share_weights:
                # Pad with 0.0 so that padded positions contribute 0 in the loss weighting.
                token_share_weights_t = pad_2d_list_to_length(
                    token_share_weights, 0.0, max_length=self.config.response_length
                ).to(idx.device)
                token_share_weights_t = token_share_weights_t.to(torch.float32)
                print(
                    f"[DEBUG][token_share_weights] Emitted tensor: "
                    f"shape={tuple(token_share_weights_t.shape)} "
                    f"nonzero_rows={int((token_share_weights_t.sum(dim=-1) > 0).sum().item())} "
                    f"w_sum_total={float(token_share_weights_t.sum().item()):.2f} "
                    f"w_min={float(token_share_weights_t.min().item()):.4f} "
                    f"w_max={float(token_share_weights_t.max().item()):.4f}"
                )
            else:
                token_share_weights_t = None

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope (batch size, 4, seq len)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        # Write inverse-sharing weights only for tree process reward training.
        if 'token_share_weights_t' in locals() and token_share_weights_t is not None:
            batch["token_share_weights"] = token_share_weights_t

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"metrics": _tree_metrics})

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if not self.config.free_cache_engine:
            return

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=tags)
        else:
            self.inference_engine.wake_up()

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        self.inference_engine.reset_prefix_cache()

        if not self.config.free_cache_engine:
            return

        self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.llm_engine.add_lora(lora_reqest)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(weights)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout(BaseRollout):
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase, which is engine in single worker process."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = model_config.tokenizer
        self.inference_engine: WorkerWrapperBase = None
        self.address = self._init_zeromq()
        self.lora_config = (
            {"max_loras": 1, "max_lora_rank": model_config.lora_rank} if model_config.lora_rank > 0 else {}
        )

        # https://github.com/vllm-project/vllm/issues/25171
        if config.layered_summon or config.expert_parallel_size > 1:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        loop = asyncio.get_running_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    def _get_free_port(self):
        ip = ray.util.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    async def _loop_forever(self):
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")
                os._exit(-1)

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        device_name = "NPU" if is_npu_available else "GPU"
        all_kwargs[0]["local_rank"] = (
            0
            if not ray_noset_visible_devices()
            else int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        )
        self.vllm_config = all_kwargs[0]["vllm_config"]
        if self.lora_config:
            lora_dtype = getattr(torch, self.config.dtype)
            self.vllm_config.lora_config = LoRAConfig(lora_dtype=lora_dtype, **self.lora_config)
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        elif method == "sleep" or method == "wake_up":
            raise ValueError("wake_up and sleep should not be called through ZeroMQ")
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            self.inference_engine.wake_up(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=dict(weights),
            )
            self.inference_engine.worker.add_lora(lora_reqest)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(weights)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode."""
        raise NotImplementedError

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address
