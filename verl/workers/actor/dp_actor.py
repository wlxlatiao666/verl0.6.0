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
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, build_segment_tensors, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.tree_weighting import (
    compute_segment_reach_masses,
    normalize_tree_loss_scales,
    select_segments_for_present_leaves,
    tree_loss_scale_normalization_totals,
)
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.tree_training import build_optimizer_micro_batches
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _loss_reduction_units(response_mask: torch.Tensor, loss_agg_mode: str) -> torch.Tensor:
    """Return the denominator units used by a mean-style loss reducer."""

    if loss_agg_mode == "token-mean":
        return response_mask.to(torch.float32).sum()
    if loss_agg_mode in {"seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"}:
        return (response_mask.sum(dim=-1) > 0).to(torch.float32).sum()
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


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

        # Counts only actor updates that carry the tree-process data contract.
        # This keeps the sparse diagnostic cadence independent of validation or
        # conventional GRPO updates.
        self._tree_process_update_count = 0

    @staticmethod
    def _sample_pg_loss(
        policy_loss_fn,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        config: ActorConfig,
        rollout_is_weights: torch.Tensor | None = None,
    ) -> float:
        """Return one item's token-mean PG loss without extending autograd state."""
        rng_devices = [] if log_prob.device.index is None else [log_prob.device.index]
        # Some optional policy losses sample internally (for example clip_cov).
        # Isolate that RNG use so this rank-only diagnostic cannot change the
        # subsequent distributed training trajectory.
        with torch.no_grad(), torch.random.fork_rng(
            devices=rng_devices, device_type=log_prob.device.type
        ):
            pg_loss, _, _, _ = policy_loss_fn(
                old_log_prob=old_log_prob.detach(),
                log_prob=log_prob.detach(),
                advantages=advantages.detach(),
                response_mask=response_mask.detach(),
                loss_agg_mode="seq-mean-token-mean",
                config=config,
                rollout_is_weights=(
                    rollout_is_weights.detach() if rollout_is_weights is not None else None
                ),
            )
        return float(pg_loss.item())

    @staticmethod
    def _format_advantage_runs(
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        max_runs: int = 16,
    ) -> str:
        """Format valid-token advantages as compact inclusive-position RLE."""
        valid_positions = torch.nonzero(response_mask > 0, as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            return "[]"

        positions = valid_positions.detach().cpu().tolist()
        values = advantages[valid_positions].detach().float().cpu().tolist()
        runs = []
        start = end = positions[0]
        run_value = values[0]
        for position, value in zip(positions[1:], values[1:]):
            if position == end + 1 and math.isclose(value, run_value, rel_tol=1e-5, abs_tol=1e-6):
                end = position
                continue
            runs.append((start, end, run_value))
            start = end = position
            run_value = value
        runs.append((start, end, run_value))

        def _format_run(run) -> str:
            run_start, run_end, value = run
            if math.isclose(value, 0.0, abs_tol=5e-7):
                value = 0.0
            return f"{run_start}:{run_end}={value:.6g}"

        if len(runs) <= max_runs:
            parts = [_format_run(run) for run in runs]
        else:
            half = max_runs // 2
            omitted = len(runs) - 2 * half
            parts = (
                [_format_run(run) for run in runs[:half]]
                + [f"...(+{omitted} runs)"]
                + [_format_run(run) for run in runs[-half:]]
            )
        return "[" + ",".join(parts) + "]"

    def _print_tree_process_loss_sample(
        self,
        *,
        policy_loss_fn,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        process_advantages: torch.Tensor,
        effective_advantages: torch.Tensor,
        response_mask: torch.Tensor,
        rollout_is_weights: torch.Tensor | None,
        update_index: int,
        unit: str,
        identifiers: list[str],
        token_share_weights: torch.Tensor | None = None,
    ) -> None:
        """Print one informative tree-process PG-loss diagnostic on rank 0.

        A sequence carrying shared-prefix weights is preferred for TreePR.  For
        TreeSR, the segment with the largest mean absolute process advantage is
        selected.  The printed loss is a standalone token-mean diagnostic.
        """
        with torch.no_grad():
            valid = response_mask > 0
            valid_counts = valid.sum(dim=-1)
            eligible = valid_counts > 0
            if not bool(eligible.any().item()):
                print(
                    f"[TREE_PROCESS_LOSS] update={update_index} unit={unit} "
                    "skipped=no_valid_tokens",
                    flush=True,
                )
                return

            if token_share_weights is not None:
                weights = token_share_weights.to(
                    device=process_advantages.device,
                    dtype=process_advantages.dtype,
                )
                shared = valid & (weights > 0) & (weights < 1 - 1e-6)
                scores = shared.sum(dim=-1).to(torch.float32)
                if not bool((scores > 0).any().item()):
                    scores = (
                        process_advantages.detach().abs() * valid.to(process_advantages.dtype)
                    ).sum(dim=-1) / valid_counts.clamp_min(1)
            else:
                scores = (
                    process_advantages.detach().abs() * valid.to(process_advantages.dtype)
                ).sum(dim=-1) / valid_counts.clamp_min(1)

            scores = scores.masked_fill(~eligible, float("-inf"))
            sample_idx = int(scores.argmax().item())
            sample_slice = slice(sample_idx, sample_idx + 1)
            sample_is_weights = (
                rollout_is_weights[sample_slice] if rollout_is_weights is not None else None
            )
            process_pg_loss = self._sample_pg_loss(
                policy_loss_fn,
                old_log_prob[sample_slice],
                log_prob[sample_slice],
                process_advantages[sample_slice],
                response_mask[sample_slice],
                self.config,
                sample_is_weights,
            )
            effective_pg_loss = process_pg_loss
            if token_share_weights is not None:
                effective_pg_loss = self._sample_pg_loss(
                    policy_loss_fn,
                    old_log_prob[sample_slice],
                    log_prob[sample_slice],
                    effective_advantages[sample_slice],
                    response_mask[sample_slice],
                    self.config,
                    sample_is_weights,
                )

            advantage_runs = self._format_advantage_runs(
                process_advantages[sample_idx], response_mask[sample_idx]
            )
            identifier = identifiers[sample_idx] if sample_idx < len(identifiers) else f"item={sample_idx}"
            loss_fields = f"process_pg_loss={process_pg_loss:.8f}"
            if token_share_weights is not None:
                loss_fields += f" effective_pg_loss={effective_pg_loss:.8f}"
            print(
                f"[TREE_PROCESS_LOSS] update={update_index} unit={unit} {identifier} "
                f"{loss_fields} advantage_runs={advantage_runs}",
                flush=True,
            )

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

        num_micro_batches = len(micro_batches)
        print(f"[DEBUG] Rank {rank}: Number of micro_batches: {num_micro_batches}")

        # === ALIGN MICRO-BATCH COUNT ACROSS ALL WORKERS ===
        # Gather micro-batch counts from all workers and find the maximum
        max_micro_batches = num_micro_batches
        if dist.is_initialized() and world_size > 1:
            # Create tensor to hold local count
            count_tensor = torch.tensor([num_micro_batches], dtype=torch.int64, device=get_device_id())
            # Gather counts from all workers
            gathered_counts = [torch.tensor([0], dtype=torch.int64, device=get_device_id()) for _ in range(world_size)]
            dist.all_gather(gathered_counts, count_tensor)
            # Find maximum count
            max_micro_batches = max([cnt.item() for cnt in gathered_counts])
            print(f"[DEBUG] Rank {rank}: local={num_micro_batches}, max={max_micro_batches} micro-batches across {world_size} workers")

        log_probs_lst = []
        entropy_lst = []
        # Process both real and dummy micro-batches up to max_micro_batches
        for i in range(max_micro_batches):
            is_dummy = i >= num_micro_batches

            if is_dummy:
                # === DUMMY MICRO-BATCH: ONLY FOR SYNCHRONIZATION ===
                # Run full forward pass with dummy data to keep FSDP in sync
                print(f"[DEBUG] Rank {rank}: Processing micro_batch {i}/{max_micro_batches} (dummy) at {time.time()}")
                # Use first batch as dummy data if available
                if num_micro_batches > 0:
                    dummy_batch = micro_batches[0]
                    dummy_batch = dummy_batch.to(get_device_id())
                    model_inputs = {**dummy_batch.batch, **dummy_batch.non_tensor_batch}
                    with torch.no_grad():
                        # Run full forward pass (this will trigger FSDP communication)
                        _, _ = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=False
                        )
                continue

            # === REAL MICRO-BATCH ===
            micro_batch = micro_batches[i]
            print(f"[DEBUG] Rank {rank}: Processing micro_batch {i}/{max_micro_batches} at {time.time()}")
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

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        has_tree_process_data = "token_share_weights" in data.batch.keys()
        if has_tree_process_data:
            self._tree_process_update_count += 1
        tree_process_log_interval_value = self.config.get("tree_process_loss_log_interval", 10)
        tree_process_log_interval = (
            int(tree_process_log_interval_value) if tree_process_log_interval_value is not None else 0
        )
        should_log_tree_process_loss = (
            has_tree_process_data
            and rank == 0
            and tree_process_log_interval > 0
            and (
                self._tree_process_update_count == 1
                or self._tree_process_update_count % tree_process_log_interval == 0
            )
        )
        tree_process_loss_logged = False

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

        # Conditional actor probability mass for non-iid tree leaves.  Keep it
        # separate from rollout IS: the former corrects tree enumeration while
        # the latter corrects rollout/training implementation mismatch.
        if "tree_leaf_masses" in data.batch.keys():
            select_keys.append("tree_leaf_masses")
        if "tree_loss_scales" in data.batch.keys():
            select_keys.append("tree_loss_scales")

        # Include tree-process-reward inverse-sharing weights if present.
        if "token_share_weights" in data.batch.keys():
            select_keys.append("token_share_weights")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")

        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if loss_mode == "tree_segment" and self.use_ulysses_sp:
            raise NotImplementedError(
                "TreeSR/tree_segment currently requires ulysses_sequence_parallel_size=1. "
                "Independent segment shuffles across sequence-parallel ranks can otherwise "
                "feed different samples/shapes into the same collective."
            )
        if (
            loss_mode == "tree_segment"
            and self.config.loss_agg_mode == "seq-mean-token-sum-norm"
            and world_size > 1
        ):
            raise NotImplementedError(
                "Distributed TreeSR does not support seq-mean-token-sum-norm because each rank's "
                "local max segment width would define a different reducer denominator. Use "
                "token-mean or seq-mean-token-mean."
            )
        if loss_mode == "tree_segment":
            if "leaf_segment_indices" in data.non_tensor_batch:
                non_tensor_select_keys.append("leaf_segment_indices")
            if "unique_segments" in data.non_tensor_batch:
                non_tensor_select_keys.append("unique_segments")
            if "worker_segments_offsets" in data.non_tensor_batch:
                non_tensor_select_keys.append("worker_segments_offsets")
            if "worker_leaves_offsets" in data.non_tensor_batch:
                non_tensor_select_keys.append("worker_leaves_offsets")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys if non_tensor_select_keys else None)

        # Pre-compute LOCAL segment-level targets FIRST (before splitting into mini_batches) for tree_segment loss
        # Note: data is already sliced per-worker (each worker has its own leaves and segments)
        local_tree_seg_targets = None
        local_batch_size = len(data)
        # Global indices are remapped to compact worker-local indices once.
        global_to_local_seg_idx_map = None
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

                # Build mapping from global seg idx to local seg idx.
                global_to_local_seg_idx_map = {
                    global_seg_idx: local_seg_idx
                    for local_seg_idx, global_seg_idx in enumerate(used_global_seg_indices)
                }

                # Build local unique_segments
                local_unique_segments = [
                    global_unique_segments[global_seg_idx]
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
                    "loss_weights": None,
                }

                if "tree_leaf_masses" in data_inputs:
                    segment_reach_masses = compute_segment_reach_masses(
                        data_inputs["tree_leaf_masses"],
                        local_leaf_segment_indices,
                        len(local_unique_segments),
                    )
                    normalization_totals = tree_loss_scale_normalization_totals(
                        segment_reach_masses,
                        seg_response_mask,
                        self.config.loss_agg_mode,
                    )
                    global_normalization_totals = torch.stack(normalization_totals).to(get_device_id())
                    if dist.is_initialized() and world_size > 1:
                        dist.all_reduce(global_normalization_totals, op=dist.ReduceOp.SUM)
                    local_tree_seg_targets["loss_weights"] = normalize_tree_loss_scales(
                        segment_reach_masses,
                        seg_response_mask,
                        self.config.loss_agg_mode,
                        normalization_totals=(
                            float(global_normalization_totals[0].item()),
                            float(global_normalization_totals[1].item()),
                        ),
                    )

                print(f"[DEBUG] [tree_segment] Worker {rank}: Local segment tensors built: seg_old_log_prob.shape={seg_old_log_prob.shape}")
                print(f"[DEBUG] [tree_segment] Worker {rank}: seg_advantages.shape={seg_advantages.shape}, seg_response_mask.shape={seg_response_mask.shape}")

        if loss_mode == "tree_segment" and local_tree_seg_targets is None:
            raise RuntimeError(
                "tree_segment loss requires local unique_segments and leaf_segment_indices; "
                "refusing to fall back to leaf-shaped tensors."
            )

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
        optimizer_step_count = 0
        expected_optimizer_step_count = None

        if use_segment_batching:
            # === SEGMENT-BASED BATCHING STRATEGY - LOCAL WORKER ONLY ===
            ppo_micro_batch_segments = getattr(self.config, "ppo_micro_batch_segments", None)
            total_segments = len(local_tree_seg_targets["seg_lens"])

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

            # Match the number of logical optimizer updates used by the normal
            # leaf path. After FSDP normalization, ppo_mini_batch_size is the
            # per-rank number of responses in one base-GRPO optimizer update.
            local_target_optimizer_steps = max(
                1, math.ceil(local_batch_size / self.config.ppo_mini_batch_size)
            )
            target_optimizer_steps = local_target_optimizer_steps
            segment_counts = [num_assigned_segments]
            if dist.is_initialized() and world_size > 1:
                batching_state = torch.tensor(
                    [local_target_optimizer_steps, num_assigned_segments],
                    dtype=torch.int64,
                    device=get_device_id(),
                )
                gathered_states = [torch.zeros_like(batching_state) for _ in range(world_size)]
                dist.all_gather(gathered_states, batching_state)
                target_optimizer_steps = max(int(state[0].item()) for state in gathered_states)
                segment_counts = [int(state[1].item()) for state in gathered_states]

            if min(segment_counts) < target_optimizer_steps:
                raise ValueError(
                    "tree segment batching needs at least one segment per rank and optimizer step, "
                    f"got segment_counts={segment_counts}, steps={target_optimizer_steps}"
                )
            expected_optimizer_step_count = target_optimizer_steps * self.config.ppo_epochs

            est_micro_batches = max(1, math.ceil(num_assigned_segments / ppo_micro_batch_segments))
            print(
                f"[DEBUG] [tree_segment] Worker {rank}: "
                f"ppo_micro_batch_segments={ppo_micro_batch_segments}, "
                f"target_optimizer_steps={target_optimizer_steps}, "
                f"est_micro_batches={est_micro_batches}"
            )

            on_policy = (
                target_optimizer_steps == 1
                and num_assigned_segments <= ppo_micro_batch_segments
                and self.config.ppo_epochs == 1
            )

            for epoch in range(self.config.ppo_epochs):
                print(f"[DEBUG] [tree_segment] Worker {rank}: epoch {epoch+1}/{self.config.ppo_epochs}")
                # Shuffle all local segments
                segment_indices = assigned_local_seg_indices.copy()
                random.shuffle(segment_indices)
                print(f"[DEBUG] [tree_segment] Worker {rank}: shuffled {len(segment_indices)} segments")

                # First form the same number of optimizer groups as base GRPO,
                # then split every group into segment micro-batches. Counts are
                # padded independently per optimizer group so all FSDP ranks
                # cross the optimizer-step boundary at exactly the same time.
                local_optimizer_micro_batches = build_optimizer_micro_batches(
                    segment_indices,
                    num_optimizer_steps=target_optimizer_steps,
                    micro_batch_size=ppo_micro_batch_segments,
                )
                local_micro_counts = [len(group) for group in local_optimizer_micro_batches]
                local_group_units = []
                for optimizer_group in local_optimizer_micro_batches:
                    optimizer_group_indices = [idx for micro in optimizer_group for idx in micro]
                    local_group_units.append(
                        _loss_reduction_units(
                            local_tree_seg_targets["response_mask"][optimizer_group_indices],
                            self.config.loss_agg_mode,
                        )
                    )
                global_group_units = torch.stack(local_group_units).to(get_device_id())
                if dist.is_initialized() and world_size > 1:
                    dist.all_reduce(global_group_units, op=dist.ReduceOp.SUM)
                max_micro_counts = local_micro_counts.copy()
                if dist.is_initialized() and world_size > 1:
                    count_tensor = torch.tensor(local_micro_counts, dtype=torch.int64, device=get_device_id())
                    gathered_counts = [torch.zeros_like(count_tensor) for _ in range(world_size)]
                    dist.all_gather(gathered_counts, count_tensor)
                    max_micro_counts = [
                        max(int(rank_counts[group_idx].item()) for rank_counts in gathered_counts)
                        for group_idx in range(target_optimizer_steps)
                    ]

                segment_micro_batches: list[list[int] | None] = []
                optimizer_step_ends = []
                for group_idx, local_micro_batches in enumerate(local_optimizer_micro_batches):
                    segment_micro_batches.extend(local_micro_batches)
                    segment_micro_batches.extend(
                        [None] * (max_micro_counts[group_idx] - len(local_micro_batches))
                    )
                    optimizer_step_ends.append(len(segment_micro_batches))

                num_micro_batches = sum(local_micro_counts)
                max_micro_batches = len(segment_micro_batches)
                fallback_seg_indices = local_optimizer_micro_batches[0][0]
                print(
                    f"[DEBUG] [tree_segment] Worker {rank}: "
                    f"local_micro_counts={local_micro_counts}, "
                    f"max_micro_counts={max_micro_counts}, "
                    f"scheduled_micro_batches={max_micro_batches}"
                )

                optimizer_group_idx = 0
                optimizer_step_starts = [0, *optimizer_step_ends[:-1]]

                def _finish_segment_optimizer_group(micro_batch_idx: int):
                    nonlocal optimizer_group_idx, optimizer_step_count
                    if micro_batch_idx + 1 != optimizer_step_ends[optimizer_group_idx]:
                        return
                    grad_norm = self._optimizer_step()
                    optimizer_step_count += 1
                    print(
                        f"[DEBUG] [tree_segment] Worker {rank}: optimizer step "
                        f"{optimizer_group_idx + 1}/{target_optimizer_steps} done, "
                        f"grad_norm={grad_norm.detach().item():.6f}"
                    )
                    append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
                    optimizer_group_idx += 1

                # Process real and dummy micro-batches. Each optimizer group
                # has its own zero_grad/gradient accumulation/optimizer step.
                for m, scheduled_seg_indices in enumerate(segment_micro_batches):
                    if m == optimizer_step_starts[optimizer_group_idx]:
                        self.actor_optimizer.zero_grad()
                        # Preserve the previous per-rank mean-loss semantics;
                        # padded dummy micro-batches synchronize FSDP only and
                        # must not dilute this rank's real segment gradients.
                        self.gradient_accumulation = local_micro_counts[optimizer_group_idx]
                    is_dummy = scheduled_seg_indices is None

                    if is_dummy:
                        # === DUMMY MICRO-BATCH: NO GRADIENT UPDATE ===
                        # Run full forward/backward with dummy data and 0 loss scale
                        # This ensures FSDP communication patterns match across all workers
                        print(f"[DEBUG] [tree_segment] Worker {rank}: micro-batch {m} (dummy, no gradient update)")
                        if num_micro_batches > 0 and local_batch_size > 0:
                            # Reuse the first batch's data for dummy computation
                            first_seg_indices = fallback_seg_indices
                            first_seg_canonical = local_tree_seg_targets["seg_canonical"]
                            first_required_leaves = list({first_seg_canonical[i][0] for i in first_seg_indices})
                            mini_batch = data[first_required_leaves]
                            mini_batch = mini_batch.to(get_device_id())
                            model_inputs = {**mini_batch.batch, **mini_batch.non_tensor_batch}
                            entropy_coeff = self.config.entropy_coeff
                            # Use 0 loss scale to prevent actual gradient update
                            loss_scale_factor = 0.0

                            # Run normal forward pass
                            calculate_entropy = entropy_coeff != 0
                            entropy, log_prob = self._forward_micro_batch(
                                model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                            )

                            # Build dummy tensors for loss computation (just to keep same code path)
                            # NOTE: Use torch.stack + F.pad to preserve gradient flow.
                            max_seg_len = local_tree_seg_targets["old_log_prob"].shape[1]
                            dummy_seg_indices = first_seg_indices
                            # Build leaf inverse map
                            leaf_inverse_map = {global_idx: local_idx for local_idx, global_idx in enumerate(first_required_leaves)}
                            seg_lens = local_tree_seg_targets["seg_lens"]
                            log_prob_pieces = []
                            for seg_idx in dummy_seg_indices:
                                leaf_j, tok_offset = first_seg_canonical[seg_idx]
                                local_leaf_j = leaf_inverse_map[leaf_j]
                                seg_len = seg_lens[seg_idx]
                                end_tok = min(tok_offset + seg_len, log_prob.shape[1])
                                piece = log_prob[local_leaf_j, tok_offset:end_tok]
                                if piece.shape[0] < max_seg_len:
                                    piece = torch.nn.functional.pad(piece, (0, max_seg_len - piece.shape[0]))
                                log_prob_pieces.append(piece)
                            seg_log_prob_local = torch.stack(log_prob_pieces) if log_prob_pieces else torch.zeros(
                                len(dummy_seg_indices), max_seg_len, device=log_prob.device, dtype=log_prob.dtype
                            )

                            # Compute dummy loss - use real response_mask from first batch to avoid nan
                            seg_old_log_prob_local = seg_log_prob_local.detach()
                            seg_advantages_local = torch.zeros_like(seg_old_log_prob_local)
                            # Use the actual response_mask from model_inputs instead of all zeros
                            # This avoids division by zero in agg_loss while still being a dummy
                            seg_response_mask_local = torch.zeros_like(seg_old_log_prob_local)
                            # Set at least one token mask to 1 for each sequence to avoid division by zero
                            if seg_response_mask_local.size(0) > 0 and seg_response_mask_local.size(1) > 0:
                                seg_response_mask_local[:, 0] = 1.0

                            policy_loss_fn = get_policy_loss_fn(loss_mode)
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                                old_log_prob=seg_old_log_prob_local,
                                log_prob=seg_log_prob_local,
                                advantages=seg_advantages_local,
                                response_mask=seg_response_mask_local,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=None,
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

                            # Backward pass with 0 loss scale - no actual gradient update
                            loss = policy_loss * loss_scale_factor
                            loss.backward()
                        # Add dummy metrics
                        micro_batch_metrics = {
                            "actor/pg_loss": 0.0,
                            "actor/pg_clipfrac": 0.0,
                            "actor/ppo_kl": 0.0,
                            "actor/pg_clipfrac_lower": 0.0,
                        }
                        if self.config.use_kl_loss:
                            micro_batch_metrics["actor/kl_loss"] = 0.0
                            micro_batch_metrics["actor/kl_coef"] = 0.0
                        append_to_dict(metrics, micro_batch_metrics)
                        _finish_segment_optimizer_group(m)
                        continue

                    # === REAL MICRO-BATCH ===
                    seg_indices = scheduled_seg_indices
                    if not seg_indices:
                        raise RuntimeError("real segment micro-batches must be non-empty")

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
                    if loss_agg_mode == "seq-mean-token-sum-norm":
                        loss_scale_factor = float(world_size)
                    else:
                        micro_units = _loss_reduction_units(
                            local_tree_seg_targets["response_mask"][seg_indices], loss_agg_mode
                        ).to(get_device_id())
                        loss_scale_factor = float(
                            (
                                micro_units
                                * world_size
                                / global_group_units[optimizer_group_idx].clamp_min(1)
                            ).item()
                        )
                    if m == 0:
                        print(f"[DEBUG] [tree_segment] Worker {rank}: micro-batch {m}, loss_scale_factor={loss_scale_factor:.6f} "
                              f"(gradient_accumulation={self.gradient_accumulation})")

                    calculate_entropy = entropy_coeff != 0
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    if m == 0:
                        print(f"[DEBUG] [tree_segment] Worker {rank}: forward done, log_prob.shape={log_prob.shape}")

                    # Build leaf inverse map: from original local leaf index to local index in mini_batch
                    leaf_inverse_map = {global_idx: local_idx for local_idx, global_idx in enumerate(required_leaves)}

                    # Build segment log_probs
                    # NOTE: Use torch.stack + F.pad instead of torch.zeros + in-place indexing
                    # to preserve gradient flow from log_prob back to model parameters.
                    max_seg_len = local_tree_seg_targets["old_log_prob"].shape[1]
                    seg_lens = local_tree_seg_targets["seg_lens"]
                    log_prob_pieces = []
                    entropy_pieces = []
                    for seg_idx in seg_indices:
                        leaf_j, tok_offset = seg_canonical[seg_idx]
                        local_leaf_j = leaf_inverse_map[leaf_j]
                        seg_len = seg_lens[seg_idx]
                        end_tok = min(tok_offset + seg_len, log_prob.shape[1])
                        piece = log_prob[local_leaf_j, tok_offset:end_tok]
                        if piece.shape[0] < max_seg_len:
                            piece = torch.nn.functional.pad(piece, (0, max_seg_len - piece.shape[0]))
                        log_prob_pieces.append(piece)
                        if entropy is not None:
                            entropy_piece = entropy[local_leaf_j, tok_offset:end_tok]
                            if entropy_piece.shape[0] < max_seg_len:
                                entropy_piece = torch.nn.functional.pad(
                                    entropy_piece, (0, max_seg_len - entropy_piece.shape[0])
                                )
                            entropy_pieces.append(entropy_piece)
                    seg_log_prob_local = torch.stack(log_prob_pieces) if log_prob_pieces else torch.zeros(
                        len(seg_indices), max_seg_len, device=log_prob.device, dtype=log_prob.dtype
                    )
                    seg_entropy_local = torch.stack(entropy_pieces) if entropy_pieces else None

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
                    seg_loss_weights_local = local_tree_seg_targets["loss_weights"]
                    if seg_loss_weights_local is not None:
                        seg_loss_weights_local = seg_loss_weights_local[seg_indices].to(log_prob.device)

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
                        loss_weights=seg_loss_weights_local,
                    )

                    if should_log_tree_process_loss and not tree_process_loss_logged:
                        segment_identifiers = [
                            (
                                f"local_segment={seg_idx} "
                                f"canonical_sequence={seg_canonical[seg_idx][0]} "
                                f"token_offset={seg_canonical[seg_idx][1]}"
                            )
                            for seg_idx in seg_indices
                        ]
                        self._print_tree_process_loss_sample(
                            policy_loss_fn=policy_loss_fn,
                            old_log_prob=seg_old_log_prob_local,
                            log_prob=seg_log_prob_local,
                            process_advantages=seg_advantages_local,
                            effective_advantages=seg_advantages_local,
                            response_mask=seg_response_mask_local,
                            rollout_is_weights=seg_rollout_is_weights_local,
                            update_index=self._tree_process_update_count,
                            unit="segment",
                            identifiers=segment_identifiers,
                        )
                        tree_process_loss_logged = True

                    # Entropy loss
                    if entropy_coeff != 0:
                        assert seg_entropy_local is not None
                        entropy_loss = agg_loss(
                            loss_mat=seg_entropy_local,
                            loss_mask=seg_response_mask_local,
                            loss_agg_mode=loss_agg_mode,
                            loss_weights=seg_loss_weights_local,
                        )
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
                        kld_pieces = []
                        for seg_idx in seg_indices:
                            leaf_j, tok_offset = seg_canonical[seg_idx]
                            local_leaf_j = leaf_inverse_map[leaf_j]
                            seg_len = seg_lens[seg_idx]
                            end_tok = min(tok_offset + seg_len, kld.shape[1])
                            piece = kld[local_leaf_j, tok_offset:end_tok]
                            if piece.shape[0] < max_seg_len:
                                piece = torch.nn.functional.pad(piece, (0, max_seg_len - piece.shape[0]))
                            kld_pieces.append(piece)
                        seg_kld_local = torch.stack(kld_pieces)
                        kl_loss = agg_loss(
                            loss_mat=seg_kld_local,
                            loss_mask=seg_response_mask_local,
                            loss_agg_mode=loss_agg_mode,
                            loss_weights=seg_loss_weights_local,
                        )
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
                    _finish_segment_optimizer_group(m)

                if optimizer_group_idx != target_optimizer_steps:
                    raise RuntimeError(
                        f"expected {target_optimizer_steps} tree-segment optimizer steps, "
                        f"completed {optimizer_group_idx}"
                    )

        else:
            # === ORIGINAL LEAF-BASED BATCHING STRATEGY ===
            # Split to make minibatch iterator for updating the actor
            # See PPO paper for details. https://arxiv.org/abs/1707.06347
            mini_batches = data.split(self.config.ppo_mini_batch_size)
            expected_optimizer_step_count = len(mini_batches) * self.config.ppo_epochs

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
                        print(f"[DEBUG] [leaf] Worker {rank}: mini_batch_size={self.config.ppo_mini_batch_size}, "
                              f"micro_batch_size_per_gpu={self.config.ppo_micro_batch_size_per_gpu}, "
                              f"gradient_accumulation={self.gradient_accumulation}, "
                              f"actual_micro_batches={len(micro_batches)}, "
                              f"MATCH={self.gradient_accumulation == len(micro_batches)}")

                    # === ALIGN MICRO-BATCH COUNT ACROSS ALL WORKERS ===
                    # Critical fix: when tree_process_reward=True, different workers may have
                    # different numbers of leaves (due to worker_leaves_offsets-based chunking).
                    # This leads to different numbers of micro-batches, causing FSDP communication
                    # mismatch and NCCL timeout. We must align micro-batch counts across all workers.
                    local_num_micro_batches = len(micro_batches)
                    max_micro_batches = local_num_micro_batches
                    if dist.is_initialized() and world_size > 1:
                        count_tensor = torch.tensor([local_num_micro_batches], dtype=torch.int64, device=get_device_id())
                        gathered_counts = [torch.tensor([0], dtype=torch.int64, device=get_device_id()) for _ in range(world_size)]
                        dist.all_gather(gathered_counts, count_tensor)
                        max_micro_batches = max([cnt.item() for cnt in gathered_counts])
                        if rank == 0:
                            print(f"[DEBUG] [leaf] Worker {rank}: local={local_num_micro_batches}, max={max_micro_batches} micro-batches across {world_size} workers")

                    # Adjust gradient_accumulation to match actual max micro-batch count
                    self.gradient_accumulation = max_micro_batches

                    # Fixed tree scales are normalized on the complete rollout
                    # batch. Combine microbatch numerators using the reducer's
                    # original full optimizer-group denominator. With DDP/FSDP
                    # gradient averaging, the world-size factor recovers the
                    # global (all-rank) denominator exactly.
                    tree_group_global_units = None
                    if "tree_loss_scales" in mini_batch.batch.keys():
                        if loss_mode == "tree_segment" and local_tree_seg_targets is not None:
                            group_present_leaves = set(range(local_leaf_start, local_leaf_end))
                            group_segment_indices = select_segments_for_present_leaves(
                                local_tree_seg_targets["seg_canonical"], group_present_leaves
                            )
                            local_group_mask = local_tree_seg_targets["response_mask"][
                                group_segment_indices
                            ]
                        else:
                            local_group_mask = mini_batch.batch["response_mask"]
                        local_group_units = _loss_reduction_units(
                            local_group_mask, self.config.loss_agg_mode
                        ).to(get_device_id())
                        tree_group_global_units = local_group_units.clone()
                        if dist.is_initialized() and world_size > 1:
                            dist.all_reduce(tree_group_global_units, op=dist.ReduceOp.SUM)

                    # Use pre-computed LOCAL segment targets, don't rebuild at mini_batch level
                    if loss_mode == "tree_segment" and local_tree_seg_targets is not None:
                        mini_batch.meta_info["tree_seg_targets"] = local_tree_seg_targets

                    self.actor_optimizer.zero_grad()

                    for m in range(max_micro_batches):
                        is_dummy = m >= local_num_micro_batches

                        if is_dummy:
                            # === DUMMY MICRO-BATCH: NO GRADIENT UPDATE ===
                            # This ensures FSDP communication patterns match across all workers
                            if rank == 0:
                                print(f"[DEBUG] [leaf] Worker {rank}: micro-batch {m} (dummy, no gradient update)")
                            if local_num_micro_batches > 0:
                                # Reuse the first micro-batch's data for dummy computation
                                dummy_micro_batch = micro_batches[0]
                                dummy_micro_batch = dummy_micro_batch.to(get_device_id())
                                model_inputs = {**dummy_micro_batch.batch, **dummy_micro_batch.non_tensor_batch}
                                entropy_coeff = self.config.entropy_coeff
                                loss_scale_factor = 0.0

                                calculate_entropy = entropy_coeff != 0
                                entropy, log_prob = self._forward_micro_batch(
                                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                                )

                                # Compute dummy loss with 0 scale to prevent actual gradient update
                                response_mask = model_inputs["response_mask"]
                                old_log_prob = model_inputs["old_log_probs"]
                                advantages = torch.zeros_like(old_log_prob)
                                rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                                policy_loss_fn = get_policy_loss_fn(loss_mode)
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
                                    policy_loss = pg_loss - entropy_loss * entropy_coeff
                                else:
                                    policy_loss = pg_loss

                                if self.config.use_kl_loss:
                                    ref_log_prob = model_inputs["ref_log_prob"]
                                    kld = kl_penalty(
                                        logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                                    )
                                    kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

                                loss = policy_loss * loss_scale_factor
                                loss.backward()

                            # Add dummy metrics
                            micro_batch_metrics = {
                                "actor/pg_loss": 0.0,
                                "actor/pg_clipfrac": 0.0,
                                "actor/ppo_kl": 0.0,
                                "actor/pg_clipfrac_lower": 0.0,
                            }
                            if self.config.use_kl_loss:
                                micro_batch_metrics["actor/kl_loss"] = 0.0
                            append_to_dict(metrics, micro_batch_metrics)
                            continue

                        # === REAL MICRO-BATCH ===
                        micro_batch = micro_batches[m]
                        micro_batch = micro_batch.to(get_device_id())
                        micro_batch_metrics = {}
                        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                        response_mask = model_inputs["response_mask"]
                        old_log_prob = model_inputs["old_log_probs"]
                        advantages = model_inputs["advantages"]
                        process_advantages = advantages

                        entropy_coeff = self.config.entropy_coeff
                        loss_agg_mode = self.config.loss_agg_mode

                        # Retained only for legacy diagnostics. Actor-occupancy
                        # leaf weights already make a shared segment's summed
                        # contribution equal its reach mass; multiplying by the
                        # old inverse-descendant weight would underweight it.
                        token_share_weights = model_inputs.get("token_share_weights", None)
                        loss_weights = model_inputs.get("tree_loss_scales", None)

                        if tree_group_global_units is not None:
                            if loss_agg_mode == "seq-mean-token-sum-norm":
                                loss_scale_factor = float(world_size)
                            else:
                                if loss_mode == "tree_segment" and local_tree_seg_targets is not None:
                                    if self.config.use_dynamic_bsz:
                                        scale_present_leaves = set(batch_idx_list[m])
                                    else:
                                        scale_micro_start = (
                                            local_leaf_start + m * self.config.ppo_micro_batch_size_per_gpu
                                        )
                                        scale_micro_end = min(
                                            scale_micro_start + self.config.ppo_micro_batch_size_per_gpu,
                                            local_leaf_end,
                                        )
                                        scale_present_leaves = set(range(scale_micro_start, scale_micro_end))
                                    scale_segment_indices = select_segments_for_present_leaves(
                                        local_tree_seg_targets["seg_canonical"], scale_present_leaves
                                    )
                                    micro_unit_mask = local_tree_seg_targets["response_mask"][
                                        scale_segment_indices
                                    ]
                                else:
                                    micro_unit_mask = response_mask
                                micro_units = _loss_reduction_units(micro_unit_mask, loss_agg_mode)
                                loss_scale_factor = float(
                                    (micro_units * world_size / tree_group_global_units.clamp_min(1)).item()
                                )
                        elif self.config.use_dynamic_bsz:
                            loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                        else:
                            loss_scale_factor = 1 / self.gradient_accumulation

                        if m == 0:
                            print(f"[DEBUG] [leaf] Worker {rank}: micro-batch {m}, loss_scale_factor={loss_scale_factor:.6f} "
                                  f"(dynamic_bsz={self.config.use_dynamic_bsz})")

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
                        segment_regularizer_ctx = None

                        # Compute policy loss (all functions return 4 values)
                        if loss_mode == "tree_segment":
                            tree_seg_targets = micro_batch.meta_info.get("tree_seg_targets")
                            if tree_seg_targets is not None:
                                seg_canonical = tree_seg_targets["seg_canonical"]
                                seg_lens = tree_seg_targets["seg_lens"]

                                # Determine which LOCAL leaves are present in
                                # this microbatch. DataProto.chunk already gave
                                # every rank a disjoint local segment set and
                                # reindexed its paths, so a second
                                # global_segment_id % world_size filter would
                                # silently discard roughly 1 - 1/world_size of
                                # valid segments.
                                if self.config.use_dynamic_bsz:
                                    present_leaves = set(batch_idx_list[m])
                                else:
                                    local_micro_batch_start = (
                                        local_leaf_start + m * self.config.ppo_micro_batch_size_per_gpu
                                    )
                                    local_micro_batch_end = min(
                                        local_micro_batch_start + self.config.ppo_micro_batch_size_per_gpu,
                                        local_leaf_end,
                                    )
                                    present_leaves = set(range(local_micro_batch_start, local_micro_batch_end))

                                seg_indices = select_segments_for_present_leaves(seg_canonical, present_leaves)

                                if len(seg_indices) == 0:
                                    pg_loss = log_prob.sum() * 0.0
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
                                    # NOTE: Use torch.stack + F.pad to preserve gradient flow.
                                    max_seg_len = tree_seg_targets["old_log_prob"].shape[1]
                                    log_prob_pieces = []
                                    entropy_pieces = []
                                    for seg_idx in seg_indices:
                                        leaf_j, tok_offset = seg_canonical[seg_idx]
                                        local_leaf_j = leaf_inverse_map[leaf_j]
                                        seg_len = seg_lens[seg_idx]
                                        end_tok = min(tok_offset + seg_len, log_prob.shape[1])
                                        piece = log_prob[local_leaf_j, tok_offset:end_tok]
                                        if piece.shape[0] < max_seg_len:
                                            piece = torch.nn.functional.pad(piece, (0, max_seg_len - piece.shape[0]))
                                        log_prob_pieces.append(piece)
                                        if entropy is not None:
                                            entropy_piece = entropy[local_leaf_j, tok_offset:end_tok]
                                            if entropy_piece.shape[0] < max_seg_len:
                                                entropy_piece = torch.nn.functional.pad(
                                                    entropy_piece, (0, max_seg_len - entropy_piece.shape[0])
                                                )
                                            entropy_pieces.append(entropy_piece)
                                    seg_log_prob_local = torch.stack(log_prob_pieces) if log_prob_pieces else torch.zeros(
                                        len(seg_indices), max_seg_len, device=log_prob.device, dtype=log_prob.dtype
                                    )
                                    seg_entropy_local = torch.stack(entropy_pieces) if entropy_pieces else None

                                    if on_policy:
                                        seg_old_log_prob_local = seg_log_prob_local.detach()
                                    else:
                                        seg_old_log_prob_local = tree_seg_targets["old_log_prob"][seg_indices].to(log_prob.device)

                                    seg_advantages_local = tree_seg_targets["advantages"][seg_indices].to(log_prob.device)
                                    seg_response_mask_local = tree_seg_targets["response_mask"][seg_indices].to(log_prob.device)
                                    seg_rollout_is_weights_local = tree_seg_targets["rollout_is_weights"]
                                    if seg_rollout_is_weights_local is not None:
                                        seg_rollout_is_weights_local = seg_rollout_is_weights_local[seg_indices].to(log_prob.device)
                                    seg_loss_weights_local = tree_seg_targets["loss_weights"]
                                    if seg_loss_weights_local is not None:
                                        seg_loss_weights_local = seg_loss_weights_local[seg_indices].to(log_prob.device)
                                    segment_regularizer_ctx = (
                                        seg_indices,
                                        seg_entropy_local,
                                        seg_response_mask_local,
                                        seg_loss_weights_local,
                                        seg_canonical,
                                        seg_lens,
                                        leaf_inverse_map,
                                        max_seg_len,
                                    )

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
                                        loss_weights=seg_loss_weights_local,
                                    )

                                    if should_log_tree_process_loss and not tree_process_loss_logged:
                                        segment_identifiers = [
                                            (
                                                f"local_segment={seg_idx} "
                                                f"canonical_sequence={seg_canonical[seg_idx][0]} "
                                                f"token_offset={seg_canonical[seg_idx][1]}"
                                            )
                                            for seg_idx in seg_indices
                                        ]
                                        self._print_tree_process_loss_sample(
                                            policy_loss_fn=policy_loss_fn,
                                            old_log_prob=seg_old_log_prob_local,
                                            log_prob=seg_log_prob_local,
                                            process_advantages=seg_advantages_local,
                                            effective_advantages=seg_advantages_local,
                                            response_mask=seg_response_mask_local,
                                            rollout_is_weights=seg_rollout_is_weights_local,
                                            update_index=self._tree_process_update_count,
                                            unit="segment",
                                            identifiers=segment_identifiers,
                                        )
                                        tree_process_loss_logged = True
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
                                    **({"loss_weights": loss_weights} if loss_weights is not None else {}),
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
                                **({"loss_weights": loss_weights} if loss_weights is not None else {}),
                            )

                            if should_log_tree_process_loss and not tree_process_loss_logged:
                                sequence_identifiers = [
                                    f"micro_sequence={i}" for i in range(response_mask.shape[0])
                                ]
                                self._print_tree_process_loss_sample(
                                    policy_loss_fn=policy_loss_fn,
                                    old_log_prob=old_log_prob,
                                    log_prob=log_prob,
                                    process_advantages=process_advantages,
                                    effective_advantages=advantages,
                                    response_mask=response_mask,
                                    rollout_is_weights=rollout_is_weights,
                                    update_index=self._tree_process_update_count,
                                    unit="sequence",
                                    identifiers=sequence_identifiers,
                                    token_share_weights=token_share_weights,
                                )
                                tree_process_loss_logged = True

                        if entropy_coeff != 0:
                            if loss_mode == "tree_segment" and tree_seg_targets is not None:
                                if segment_regularizer_ctx is None:
                                    entropy_loss = entropy.sum() * 0.0
                                else:
                                    _, seg_entropy_local, seg_mask, seg_scales, *_ = segment_regularizer_ctx
                                    assert seg_entropy_local is not None
                                    entropy_loss = agg_loss(
                                        loss_mat=seg_entropy_local,
                                        loss_mask=seg_mask,
                                        loss_agg_mode=loss_agg_mode,
                                        loss_weights=seg_scales,
                                    )
                            else:
                                entropy_loss = agg_loss(
                                    loss_mat=entropy,
                                    loss_mask=response_mask,
                                    loss_agg_mode=loss_agg_mode,
                                    loss_weights=loss_weights,
                                )

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
                            if loss_mode == "tree_segment" and tree_seg_targets is not None:
                                if segment_regularizer_ctx is None:
                                    kl_loss = kld.sum() * 0.0
                                else:
                                    (
                                        reg_seg_indices,
                                        _,
                                        seg_mask,
                                        seg_scales,
                                        reg_seg_canonical,
                                        reg_seg_lens,
                                        reg_leaf_inverse_map,
                                        reg_max_seg_len,
                                    ) = segment_regularizer_ctx
                                    kld_pieces = []
                                    for seg_idx in reg_seg_indices:
                                        leaf_j, tok_offset = reg_seg_canonical[seg_idx]
                                        local_leaf_j = reg_leaf_inverse_map[leaf_j]
                                        end_tok = min(tok_offset + reg_seg_lens[seg_idx], kld.shape[1])
                                        piece = kld[local_leaf_j, tok_offset:end_tok]
                                        if piece.shape[0] < reg_max_seg_len:
                                            piece = torch.nn.functional.pad(
                                                piece, (0, reg_max_seg_len - piece.shape[0])
                                            )
                                        kld_pieces.append(piece)
                                    seg_kld_local = torch.stack(kld_pieces)
                                    kl_loss = agg_loss(
                                        loss_mat=seg_kld_local,
                                        loss_mask=seg_mask,
                                        loss_agg_mode=loss_agg_mode,
                                        loss_weights=seg_scales,
                                    )
                            else:
                                kl_loss = agg_loss(
                                    loss_mat=kld,
                                    loss_mask=response_mask,
                                    loss_agg_mode=loss_agg_mode,
                                    loss_weights=loss_weights,
                                )

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
                    optimizer_step_count += 1
                    mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                    append_to_dict(metrics, mini_batch_metrics)

        if should_log_tree_process_loss and not tree_process_loss_logged:
            print(
                f"[TREE_PROCESS_LOSS] update={self._tree_process_update_count} "
                "skipped=no_real_process_loss_was_consumed",
                flush=True,
            )

        if optimizer_step_count != expected_optimizer_step_count:
            raise RuntimeError(
                f"optimizer step mismatch: expected {expected_optimizer_step_count}, got {optimizer_step_count}"
            )
        append_to_dict(
            metrics,
            {
                "actor/optimizer_steps": optimizer_step_count,
                "actor/local_ppo_mini_batch_size": self.config.ppo_mini_batch_size,
            },
        )
        self.actor_optimizer.zero_grad()
        return metrics
