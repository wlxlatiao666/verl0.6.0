# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import os
import random

import numpy as np
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import torch.distributed as dist

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, build_segment_tensors, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

        # Initialize gradient accumulation attribute for safety
        self.gradient_accumulation = 1

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        import time
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        print(f"[DEBUG] Rank {rank}: _forward_micro_batch starting at {time.time()}")

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            print(f"[DEBUG] Rank {rank}: batch_size={batch_size}, seqlen={seqlen}")
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                print(f"[DEBUG] Rank {rank}: starting actor_module forward pass at {time.time()}")
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                print(f"[DEBUG] Rank {rank}: finished actor_module forward pass at {time.time()}")

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                print(f"[DEBUG] Rank {rank}: use_remove_padding=False, starting forward pass at {time.time()}")
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating
                print(f"[DEBUG] Rank {rank}: use_remove_padding=False, forward pass done at {time.time()}")

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            print(f"[DEBUG] Rank {rank}: _forward_micro_batch returning at {time.time()}")
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        import time
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        print(f"[DEBUG] Rank {rank}/{world_size}: compute_log_prob started at {time.time()}")

        # Barrier 1: 确保所有rank同时开始
        if dist.is_initialized():
            print(f"[DEBUG] Rank {rank}/{world_size}: compute_log_prob waiting at start barrier")
            dist.barrier()
            print(f"[DEBUG] Rank {rank}/{world_size}: compute_log_prob passed start barrier")

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        print(f"[DEBUG] Rank {rank}: before data.select(), len(data)={len(data)}")
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        print(f"[DEBUG] Rank {rank}: after data.select()")

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            print(f"[DEBUG] Rank {rank}: use_dynamic_bsz=True, max_token_len={max_token_len}")
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            print(f"[DEBUG] Rank {rank}: use_dynamic_bsz=False, micro_batch_size={micro_batch_size}")
            micro_batches = data.split(micro_batch_size)

        print(f"[DEBUG] Rank {rank}: Number of micro_batches: {len(micro_batches)}")

        log_probs_lst = []
        entropy_lst = []
        for i, micro_batch in enumerate(micro_batches):
            print(f"[DEBUG] Rank {rank}: Processing micro_batch {i}/{len(micro_batches)} at {time.time()}")
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            print(f"[DEBUG] Rank {rank}: micro_batch {i} moved to device, starting forward pass")
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            print(f"[DEBUG] Rank {rank}: micro_batch {i} forward pass done at {time.time()}")
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        print(f"[DEBUG] Rank {rank}: All micro_batches done, starting concat at {time.time()}")

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        print(f"[DEBUG] Rank {rank}/{world_size}: compute_log_prob finished at {time.time()}")
        # Barrier 2: 确保所有rank同时结束
        if dist.is_initialized():
            print(f"[DEBUG] Rank {rank}/{world_size}: compute_log_prob waiting at end barrier")
            dist.barrier()
            print(f"[DEBUG] Rank {rank}/{world_size}: compute_log_prob passed end barrier")


        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        import time
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        print(f"[DEBUG] Rank {rank}/{world_size}: update_policy started at {time.time()}")
        # Barrier 1: 确保所有rank同时开始
        if dist.is_initialized():
            print(f"[DEBUG] Rank {rank}/{world_size}: update_policy waiting at start barrier")
            dist.barrier()
            print(f"[DEBUG] Rank {rank}/{world_size}: update_policy passed start barrier")
        # make sure we are in training mode
        self.actor_module.train()


        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if "leaf_segment_indices" in data.non_tensor_batch:
            non_tensor_select_keys.append("leaf_segment_indices")
        # Also select tree segment data if present
        if "unique_segments" in data.non_tensor_batch:
            non_tensor_select_keys.append("unique_segments")
        if "unique_segment_seq_ids" in data.non_tensor_batch:
            non_tensor_select_keys.append("unique_segment_seq_ids")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys if non_tensor_select_keys else None)

        # Pre-compute LOCAL segment-level targets FIRST (before splitting into mini_batches) for tree_segment loss
        # Note: data is already sliced per-worker (each worker has its own leaves and segments)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        local_tree_seg_targets = None
        local_batch_size = len(data)
        # We need to keep track of global seg indices for ownership check
        global_to_local_seg_idx_map = None
        local_to_global_seg_idx_map = None
        if loss_mode == "tree_segment":
            # Try to get unique_segments from non_tensor_batch first (new location),
            # then fall back to meta_info['metrics'] (old location for compatibility)
            global_unique_segments = data.non_tensor_batch.get("unique_segments")
            if global_unique_segments is None:
                global_unique_segments = data.meta_info.get("metrics", {}).get("unique_segments")
                print("get unique_segments from meta_info['metrics'] for tree_segment loss. This is the old location and may be None if not set properly in trainer.")
            global_leaf_segment_indices = data.non_tensor_batch.get("leaf_segment_indices")
            worker_segments_offsets = data.non_tensor_batch.get("worker_segments_offsets")

            rank = dist.get_rank() if dist.is_initialized() else 0
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if worker_segments_offsets is not None:
                print(f"[DEBUG] [tree_segment] Worker {rank}: found worker_segments_offsets={worker_segments_offsets}")

            if global_unique_segments is not None and global_leaf_segment_indices is not None:
                print(f"[DEBUG] [tree_segment] Worker {rank}: start building local segments")
                print(f"[DEBUG] [tree_segment] Worker {rank}: global_unique_segments={len(global_unique_segments)}, local_leaves={len(global_leaf_segment_indices)}")

                # ==== REBUILD LOCAL SEGMENTS FROM GLOBAL ====
                # Collect all segment indices referenced by this worker's leaves
                used_global_seg_indices = set()
                for path in global_leaf_segment_indices:
                    for seg_idx in path:
                        used_global_seg_indices.add(seg_idx)
                used_global_seg_indices = sorted(used_global_seg_indices)
                print(f"[DEBUG] [tree_segment] Worker {rank}: uses {len(used_global_seg_indices)} segments: {used_global_seg_indices[:10]}{'...' if len(used_global_seg_indices) > 10 else ''}")

                # Build mapping from global seg idx to local seg idx (and reverse)
                global_to_local_seg_idx_map = {
                    global_seg_idx: local_seg_idx
                    for local_seg_idx, global_seg_idx in enumerate(used_global_seg_indices)
                }
                local_to_global_seg_idx_map = {
                    local_seg_idx: global_seg_idx
                    for local_seg_idx, global_seg_idx in enumerate(used_global_seg_indices)
                }

                # Build local unique_segments
                local_unique_segments = [
                    global_unique_segments[global_seg_idx]
                    for global_seg_idx in used_global_seg_indices
                ]

                # Build local unique_segment_seq_ids if available
                local_unique_segment_seq_ids = None
                if "unique_segment_seq_ids" in data.non_tensor_batch:
                    global_unique_segment_seq_ids = data.non_tensor_batch["unique_segment_seq_ids"]
                    local_unique_segment_seq_ids = [
                        global_unique_segment_seq_ids[global_seg_idx]
                        for global_seg_idx in used_global_seg_indices
                    ]

                # Adjust leaf_segment_indices to use local indices
                local_leaf_segment_indices = []
                for path in global_leaf_segment_indices:
                    adjusted_path = [global_to_local_seg_idx_map[seg_idx] for seg_idx in path]
                    local_leaf_segment_indices.append(adjusted_path)
                print(f"[DEBUG] [tree_segment] Worker {rank}: leaf_segment_indices adjusted to local")

                # Print detailed info on each rank (not just rank 0)
                print(f"[DEBUG] [tree_segment] Worker {rank}: Building segment tensors:")
                print(f"[DEBUG] [tree_segment] Worker {rank}: - Batch size (leaves): {len(data)}")
                print(f"[DEBUG] [tree_segment] Worker {rank}: - Global unique segments: {len(global_unique_segments)}")
                print(f"[DEBUG] [tree_segment] Worker {rank}: - Local unique segments: {len(local_unique_segments)}")

                # Unpack TensorDict to plain dict so .get() works safely
                data_inputs = {**data.batch, **data.non_tensor_batch}

                _, seg_old_log_prob, seg_advantages, seg_response_mask, seg_canonical, seg_lens, seg_rollout_is = build_segment_tensors(
                    log_prob=None,
                    old_log_prob=data_inputs["old_log_probs"],
                    advantages=data_inputs["advantages"],
                    response_mask=data_inputs["response_mask"],
                    unique_segments=local_unique_segments,
                    leaf_segment_indices=local_leaf_segment_indices,
                    rollout_is_weights=data_inputs.get("rollout_is_weights"),
                )
                local_tree_seg_targets = {
                    "old_log_prob": seg_old_log_prob,
                    "advantages": seg_advantages,
                    "response_mask": seg_response_mask,
                    "seg_canonical": seg_canonical,
                    "seg_lens": seg_lens,
                    "rollout_is_weights": seg_rollout_is,
                    "local_to_global_seg_idx_map": local_to_global_seg_idx_map,
                }

                print(f"[DEBUG] [tree_segment] Worker {rank}: Local segment tensors built: seg_old_log_prob.shape={seg_old_log_prob.shape}")
                print(f"[DEBUG] [tree_segment] Worker {rank}: seg_advantages.shape={seg_advantages.shape}, seg_response_mask.shape={seg_response_mask.shape}")

        # === New code: Check batch strategy for tree_segment loss ===
        tree_segment_batch_strategy = getattr(self.config, "tree_segment_batch_strategy", "leaf")
        use_segment_batching = (
            loss_mode == "tree_segment" and
            tree_segment_batch_strategy == "segment" and
            local_tree_seg_targets is not None and
            not self.config.use_dynamic_bsz  # Dynamic bsz not supported with segment batching yet
        )

        on_policy = False
        metrics = {}

        if use_segment_batching:
            # === SEGMENT-BASED BATCHING STRATEGY - LOCAL WORKER ONLY ===
            ppo_micro_batch_segments = getattr(self.config, "ppo_micro_batch_segments", None)
            total_segments = len(local_tree_seg_targets["seg_lens"])

            rank = dist.get_rank() if dist.is_initialized() else 0
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            print(f"[DEBUG] [tree_segment] Worker {rank}/{world_size}: start update_policy with segment batching")
            print(f"[DEBUG] [tree_segment] Worker {rank}: local_batch_size={local_batch_size} leaves, total_segments={total_segments} segments")

            # === NO FILTER: PROCESS ALL LOCAL SEGMENTS ON THIS WORKER ===
            # Each worker processes ALL segments from its own rollout
            assigned_local_seg_indices = list(range(total_segments))
            num_assigned_segments = len(assigned_local_seg_indices)

            print(f"[DEBUG] [tree_segment] Worker {rank}: processing ALL {num_assigned_segments} local segments (no global reallocation)")

            if ppo_micro_batch_segments is None:
                # Default estimation: use similar ratio as leaf-based batching
                if self.config.ppo_micro_batch_size_per_gpu is not None:
                    avg_segments_per_leaf = num_assigned_segments / max(local_batch_size, 1)
                    ppo_micro_batch_segments = max(8, int(self.config.ppo_micro_batch_size_per_gpu * avg_segments_per_leaf))
                else:
                    ppo_micro_batch_segments = max(8, num_assigned_segments // 10)

            # Determine gradient accumulation: how many micro-batches per optimizer step
            # This should match the leaf-based strategy logic
            if self.config.ppo_micro_batch_size_per_gpu is not None and self.config.ppo_mini_batch_size is not None:
                self.gradient_accumulation = max(1, self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu)
            else:
                self.gradient_accumulation = 1

            print(f"[DEBUG] [tree_segment] Worker {rank}: ppo_micro_batch_segments={ppo_micro_batch_segments}, gradient_accumulation={self.gradient_accumulation}")

            on_policy = num_assigned_segments <= ppo_micro_batch_segments and self.config.ppo_epochs == 1

            for epoch in range(self.config.ppo_epochs):
                print(f"[DEBUG] [tree_segment] Worker {rank}: epoch {epoch+1}/{self.config.ppo_epochs}")
                # Shuffle all local segments
                segment_indices = assigned_local_seg_indices.copy()
                random.shuffle(segment_indices)
                print(f"[DEBUG] [tree_segment] Worker {rank}: shuffled {len(segment_indices)} segments")

                # Split into micro-batches (we use gradient accumulation for multiple micro-batches per step)
                segment_micro_batches = [
                    segment_indices[i:i+ppo_micro_batch_segments]
                    for i in range(0, len(segment_indices), ppo_micro_batch_segments)
                ]
                print(f"[DEBUG] [tree_segment] Worker {rank}: split into {len(segment_micro_batches)} micro-batches")

                # Zero grad at the start of each mini-batch cycle
                self.actor_optimizer.zero_grad()

                for m, seg_indices in enumerate(segment_micro_batches):
                    if not seg_indices:
                        continue

                    # Collect all required leaves for these segments
                    seg_canonical = local_tree_seg_targets["seg_canonical"]
                    required_leaves = list({seg_canonical[i][0] for i in seg_indices})
                    if m == 0:
                        print(f"[DEBUG] [tree_segment] Worker {rank}: micro-batch {m} has {len(seg_indices)} segments, requires {len(required_leaves)} leaves")
                        print(f"[DEBUG] [tree_segment] Worker {rank}: seg_indices={seg_indices[:5]}{'...' if len(seg_indices) > 5 else ''}")
                        print(f"[DEBUG] [tree_segment] Worker {rank}: required_leaves={required_leaves[:5]}{'...' if len(required_leaves) > 5 else ''}")

                    # Get the data for required leaves only
                    # Note: seg_canonical uses LOCAL leaf indices within this worker's data
                    mini_batch = data[required_leaves]
                    mini_batch = mini_batch.to(get_device_id())

                    # Forward pass on required leaves
                    model_inputs = {**mini_batch.batch, **mini_batch.non_tensor_batch}
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    loss_scale_factor = 1 / self.gradient_accumulation

                    calculate_entropy = entropy_coeff != 0
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    if m == 0:
                        print(f"[DEBUG] [tree_segment] Worker {rank}: forward done, log_prob.shape={log_prob.shape}")

                    # Build leaf inverse map: from original local leaf index to local index in mini_batch
                    leaf_inverse_map = {global_idx: local_idx for local_idx, global_idx in enumerate(required_leaves)}

                    # Build segment log_probs
                    max_seg_len = local_tree_seg_targets["old_log_prob"].shape[1]
                    seg_log_prob_local = torch.zeros(len(seg_indices), max_seg_len, device=log_prob.device, dtype=log_prob.dtype)

                    seg_lens = local_tree_seg_targets["seg_lens"]
                    for local_i, seg_idx in enumerate(seg_indices):
                        leaf_j, tok_offset = seg_canonical[seg_idx]
                        local_leaf_j = leaf_inverse_map[leaf_j]
                        seg_len = seg_lens[seg_idx]
                        end_tok = min(tok_offset + seg_len, log_prob.shape[1])
                        actual_len = end_tok - tok_offset
                        seg_log_prob_local[local_i, :actual_len] = log_prob[local_leaf_j, tok_offset:end_tok]

                    # Get other segment tensors
                    if on_policy:
                        seg_old_log_prob_local = seg_log_prob_local.detach()
                    else:
                        seg_old_log_prob_local = local_tree_seg_targets["old_log_prob"][seg_indices].to(log_prob.device)

                    seg_advantages_local = local_tree_seg_targets["advantages"][seg_indices].to(log_prob.device)
                    seg_response_mask_local = local_tree_seg_targets["response_mask"][seg_indices].to(log_prob.device)
                    seg_rollout_is_weights_local = local_tree_seg_targets["rollout_is_weights"]
                    if seg_rollout_is_weights_local is not None:
                        seg_rollout_is_weights_local = seg_rollout_is_weights_local[seg_indices].to(log_prob.device)

                    if m == 0:
                        print(f"[DEBUG] [tree_segment] Worker {rank}: seg_advantages.shape={seg_advantages_local.shape}")
                        print(f"[DEBUG] [tree_segment] Worker {rank}: seg_advantages range: [{seg_advantages_local.min():.4f}, {seg_advantages_local.max():.4f}]")

                    # Compute loss
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    micro_batch_metrics = {}

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=seg_old_log_prob_local,
                        log_prob=seg_log_prob_local,
                        advantages=seg_advantages_local,
                        response_mask=seg_response_mask_local,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=seg_rollout_is_weights_local,
                    )

                    # Entropy loss
                    if entropy_coeff != 0:
                        response_mask = model_inputs["response_mask"]
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    # KL loss
                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        response_mask = model_inputs["response_mask"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if m == 0:
                        print(f"[DEBUG] [tree_segment] Worker {rank}: pg_loss={pg_loss.detach().item():.6f}, policy_loss={policy_loss.detach().item():.6f}")

                    # Backward pass with gradient accumulation
                    loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                # Optimizer step after gradient accumulation
                grad_norm = self._optimizer_step()
                print(f"[DEBUG] [tree_segment] Worker {rank}: optimizer step done, grad_norm={grad_norm.detach().item():.6f}")
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        else:
            # === ORIGINAL LEAF-BASED BATCHING STRATEGY ===
            # Split to make minibatch iterator for updating the actor
            # See PPO paper for details. https://arxiv.org/abs/1707.06347
            mini_batches = data.split(self.config.ppo_mini_batch_size)

            # Track local leaf index ranges for each mini_batch (to correctly map segments to micro_batches)
            mini_batch_ranges = []
            current_start = 0
            for mb in mini_batches:
                mb_len = len(mb)
                mini_batch_ranges.append((current_start, current_start + mb_len))
                current_start += mb_len

            on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

            for _ in range(self.config.ppo_epochs):
                for batch_idx, mini_batch in enumerate(mini_batches):
                    local_leaf_start, local_leaf_end = mini_batch_ranges[batch_idx]
                    if self.config.use_dynamic_bsz:
                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        micro_batches, batch_idx_list = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                        # Adjust batch_idx_list to be LOCAL leaf indices instead of mini_batch local indices
                        if batch_idx_list is not None:
                            batch_idx_list = [[local_leaf_start + local_idx for local_idx in sublist] for sublist in batch_idx_list]
                    else:
                        self.gradient_accumulation = (
                            self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        )
                        micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                        batch_idx_list = None

                    # Use pre-computed LOCAL segment targets, don't rebuild at mini_batch level
                    if loss_mode == "tree_segment" and local_tree_seg_targets is not None:
                        mini_batch.meta_info["tree_seg_targets"] = local_tree_seg_targets

                    self.actor_optimizer.zero_grad()

                    for m, micro_batch in enumerate(micro_batches):
                        micro_batch = micro_batch.to(get_device_id())
                        micro_batch_metrics = {}
                        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                        response_mask = model_inputs["response_mask"]
                        old_log_prob = model_inputs["old_log_probs"]
                        advantages = model_inputs["advantages"]

                        entropy_coeff = self.config.entropy_coeff
                        loss_agg_mode = self.config.loss_agg_mode

                        if self.config.use_dynamic_bsz:
                            loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                        else:
                            loss_scale_factor = 1 / self.gradient_accumulation

                        # all return: (bsz, response_length)
                        calculate_entropy = False
                        if entropy_coeff != 0:
                            calculate_entropy = True
                        entropy, log_prob = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                        )

                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                        # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                        # Extract pre-computed rollout importance sampling weights if present
                        # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                        rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                        # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                        # are computed centrally in ray_trainer.py for consistency and efficiency.
                        # This ensures metrics are computed uniformly across all batches at the trainer level
                        # and avoids redundant computation across workers and micro-batches.

                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        # Compute policy loss (all functions return 4 values)
                        if loss_mode == "tree_segment":
                            tree_seg_targets = micro_batch.meta_info.get("tree_seg_targets")
                            if tree_seg_targets is not None:
                                seg_canonical = tree_seg_targets["seg_canonical"]
                                seg_lens = tree_seg_targets["seg_lens"]
                                local_to_global_seg_idx_map = tree_seg_targets.get("local_to_global_seg_idx_map")

                                # Fall back to original behavior if no mapping exists
                                if local_to_global_seg_idx_map is None:
                                    # Determine which leaves are present in this micro-batch (using LOCAL leaf indices)
                                    if self.config.use_dynamic_bsz:
                                        present_leaves = set(batch_idx_list[m])
                                    else:
                                        # For non-dynamic bsz, compute the LOCAL leaf indices in this micro_batch
                                        local_micro_batch_start = local_leaf_start + m * self.config.ppo_micro_batch_size_per_gpu
                                        local_micro_batch_end = min(local_micro_batch_start + self.config.ppo_micro_batch_size_per_gpu, local_leaf_end)
                                        present_leaves = set(range(local_micro_batch_start, local_micro_batch_end))

                                    # Select segments whose canonical leaf is in this micro-batch
                                    seg_indices = [i for i, (leaf_j, _) in enumerate(seg_canonical) if leaf_j in present_leaves]
                                else:
                                    rank = dist.get_rank() if dist.is_initialized() else 0
                                    world_size = dist.get_world_size() if dist.is_initialized() else 1

                                    # Determine which leaves are present in this micro-batch (using LOCAL leaf indices)
                                    if self.config.use_dynamic_bsz:
                                        present_leaves = set(batch_idx_list[m])
                                    else:
                                        # For non-dynamic bsz, compute the LOCAL leaf indices in this micro_batch
                                        local_micro_batch_start = local_leaf_start + m * self.config.ppo_micro_batch_size_per_gpu
                                        local_micro_batch_end = min(local_micro_batch_start + self.config.ppo_micro_batch_size_per_gpu, local_leaf_end)
                                        present_leaves = set(range(local_micro_batch_start, local_micro_batch_end))

                                    # Select segments whose canonical leaf is in this micro-batch AND assigned to this worker
                                    # This guarantees each segment is updated exactly once and load balanced
                                    seg_indices = []
                                    for i, (leaf_j, _) in enumerate(seg_canonical):
                                        if leaf_j in present_leaves:
                                            global_seg_idx = local_to_global_seg_idx_map[i]
                                            if global_seg_idx % world_size == rank:
                                                seg_indices.append(i)

                                if len(seg_indices) == 0:
                                    pg_loss = torch.tensor(0.0, device=log_prob.device)
                                    pg_clipfrac = torch.tensor(0.0, device=log_prob.device)
                                    ppo_kl = torch.tensor(0.0, device=log_prob.device)
                                    pg_clipfrac_lower = torch.tensor(0.0, device=log_prob.device)
                                else:
                                    # Build inverse map from LOCAL leaf index to local micro-batch index
                                    if self.config.use_dynamic_bsz:
                                        # batch_idx_list[m] already contains LOCAL indices
                                        leaf_inverse_map = {global_idx: local_idx for local_idx, global_idx in enumerate(batch_idx_list[m])}
                                    else:
                                        # Map from local leaf index to local micro_batch index
                                        leaf_inverse_map = {}
                                        for local_idx, global_idx in enumerate(range(local_micro_batch_start, local_micro_batch_end)):
                                            leaf_inverse_map[global_idx] = local_idx

                                    # Use the pre-computed local max_seg_len so shapes align with
                                    # tree_seg_targets tensors (old_log_prob / advantages / mask).
                                    max_seg_len = tree_seg_targets["old_log_prob"].shape[1]
                                    seg_log_prob_local = torch.zeros(len(seg_indices), max_seg_len, device=log_prob.device, dtype=log_prob.dtype)

                                    for local_i, seg_idx in enumerate(seg_indices):
                                        leaf_j, tok_offset = seg_canonical[seg_idx]
                                        local_leaf_j = leaf_inverse_map[leaf_j]
                                        seg_len = seg_lens[seg_idx]
                                        end_tok = min(tok_offset + seg_len, log_prob.shape[1])
                                        actual_len = end_tok - tok_offset
                                        seg_log_prob_local[local_i, :actual_len] = log_prob[local_leaf_j, tok_offset:end_tok]

                                    if on_policy:
                                        seg_old_log_prob_local = seg_log_prob_local.detach()
                                    else:
                                        seg_old_log_prob_local = tree_seg_targets["old_log_prob"][seg_indices].to(log_prob.device)

                                    seg_advantages_local = tree_seg_targets["advantages"][seg_indices].to(log_prob.device)
                                    seg_response_mask_local = tree_seg_targets["response_mask"][seg_indices].to(log_prob.device)
                                    seg_rollout_is_weights_local = tree_seg_targets["rollout_is_weights"]
                                    if seg_rollout_is_weights_local is not None:
                                        seg_rollout_is_weights_local = seg_rollout_is_weights_local[seg_indices].to(log_prob.device)

                                    if batch_idx == 0 and m == 0:  # Print once per epoch
                                        print(f"[tree_segment] Updating {len(seg_indices)} segments in this micro_batch")

                                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                                        old_log_prob=seg_old_log_prob_local,
                                        log_prob=seg_log_prob_local,
                                        advantages=seg_advantages_local,
                                        response_mask=seg_response_mask_local,
                                        loss_agg_mode=loss_agg_mode,
                                        config=self.config,
                                        rollout_is_weights=seg_rollout_is_weights_local,
                                    )
                            else:
                                # Fallback if targets not precomputed
                                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                                    old_log_prob=old_log_prob,
                                    log_prob=log_prob,
                                    advantages=advantages,
                                    response_mask=response_mask,
                                    loss_agg_mode=loss_agg_mode,
                                    config=self.config,
                                    rollout_is_weights=rollout_is_weights,
                                )
                        else:
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )

                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            # compute policy loss
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                        else:
                            policy_loss = pg_loss

                        if self.config.use_kl_loss:
                            ref_log_prob = model_inputs["ref_log_prob"]
                            # compute kl loss
                            kld = kl_penalty(
                                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                            )
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                            micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                        if self.config.use_dynamic_bsz:
                            # relative to the dynamic bsz
                            loss = policy_loss * loss_scale_factor
                        else:
                            loss = policy_loss * loss_scale_factor
                        loss.backward()

                        micro_batch_metrics.update(
                            {
                                "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                                "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                                "actor/ppo_kl": ppo_kl.detach().item(),
                                "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                            }
                        )
                        append_to_dict(metrics, micro_batch_metrics)

                    grad_norm = self._optimizer_step()
                    mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                    append_to_dict(metrics, mini_batch_metrics)

        self.actor_optimizer.zero_grad()
        # Barrier 2: 确保所有rank同时结束
        if dist.is_initialized():
            print(f"[DEBUG] Rank {rank}/{world_size}: update_policy waiting at end barrier")
            dist.barrier()
            print(f"[DEBUG] Rank {rank}/{world_size}: update_policy passed end barrier")

        return metrics
