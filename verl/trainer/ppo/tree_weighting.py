# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Actor-occupancy weights for deterministic top-k tree rollouts.

Tree decoding enumerates top-k children instead of sampling one child from the
actor.  Consequently, treating every emitted leaf as an iid actor sample
overweights low-probability branches.  This module reconstructs the actor's
conditional distribution on the represented tree support from recomputed
``old_log_probs`` and the root-to-leaf segment paths.

At a branch, a child segment starts with the forced top-k token.  Softmax over
the old-actor log probabilities of those sibling tokens therefore gives the
conditional actor probability within the selected top-k set.  The result is a
conditional-on-tree-support objective: deterministic top-k has no support for
the omitted tail, so no finite weight can recover the full actor objective.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class TreeActorWeights:
    """Probability masses for leaves and unique segments.

    ``leaf_masses`` sums to one independently for every prompt.  A prompt may
    contain one expanded tree root and conventional top-up samples represented
    as one-node synthetic roots.  Those roots are treated as equal estimator
    strata; mass within an expanded root is distributed by actor probabilities.

    ``segment_masses[s]`` is the probability of reaching unique segment ``s``
    and equals the sum of masses of its descendant leaves.
    """

    leaf_masses: torch.Tensor
    segment_masses: torch.Tensor
    branch_coverages: torch.Tensor


def _as_paths(leaf_segment_indices: Sequence[Sequence[int]]) -> list[list[int]]:
    return [[int(segment_idx) for segment_idx in path] for path in leaf_segment_indices]


def compute_conditional_topk_tree_weights(
    *,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    unique_segments: Sequence[Sequence[int]],
    leaf_segment_indices: Sequence[Sequence[int]],
    prompt_ids: Sequence[Hashable] | np.ndarray,
) -> TreeActorWeights:
    """Compute actor leaf mass and segment reach mass for a rollout forest.

    Args:
        old_log_probs: Recomputed old-actor token log probabilities, shaped
            ``(num_leaves, response_length)``.
        response_mask: Valid response-token mask with the same shape.
        unique_segments: Token ids for each unique tree segment.
        leaf_segment_indices: Root-to-leaf segment-index path for each row.
        prompt_ids: Prompt/group id for each leaf row.

    Returns:
        :class:`TreeActorWeights` on ``old_log_probs.device``.  All returned
        tensors are detached because proposal/occupancy weights must not carry
        gradients into the actor update.
    """

    if old_log_probs.ndim != 2 or response_mask.shape != old_log_probs.shape:
        raise ValueError(
            "old_log_probs and response_mask must be aligned rank-2 tensors, "
            f"got {tuple(old_log_probs.shape)} and {tuple(response_mask.shape)}"
        )

    paths = _as_paths(leaf_segment_indices)
    num_leaves, response_length = old_log_probs.shape
    num_segments = len(unique_segments)
    if len(paths) != num_leaves or len(prompt_ids) != num_leaves:
        raise ValueError(
            "tree metadata must have one path and prompt id per leaf, "
            f"got leaves={num_leaves}, paths={len(paths)}, prompt_ids={len(prompt_ids)}"
        )

    for leaf_idx, path in enumerate(paths):
        if not path:
            raise ValueError(f"leaf {leaf_idx} has an empty segment path")
        if len(path) != len(set(path)):
            raise ValueError(f"leaf {leaf_idx} repeats a segment and would create a topology cycle")
        invalid = [idx for idx in path if idx < 0 or idx >= num_segments]
        if invalid:
            raise ValueError(f"leaf {leaf_idx} has out-of-range segment indices {invalid[:8]}")
        empty = [idx for idx in path if len(unique_segments[idx]) == 0]
        if empty:
            raise ValueError(f"leaf {leaf_idx} contains empty segments {empty[:8]}")
        path_token_count = sum(len(unique_segments[idx]) for idx in path)
        valid_token_count = int(response_mask[leaf_idx].sum().item())
        if path_token_count < valid_token_count:
            raise ValueError(
                f"leaf {leaf_idx} path contains {path_token_count} tokens but response_mask has "
                f"{valid_token_count} valid tokens"
            )

    # Tree topology and a canonical response offset for every segment.  A
    # shared segment has the same token prefix/logprob in all descendant rows,
    # so the first occurrence is sufficient and avoids averaging roundoff.
    children_of: dict[int, list[int]] = defaultdict(list)
    parent_of: dict[int, int] = {}
    canonical: dict[int, tuple[int, int]] = {}
    prompt_of_segment: dict[int, Hashable] = {}
    leaves_by_prompt: dict[Hashable, list[int]] = defaultdict(list)
    roots_by_prompt: dict[Hashable, list[int]] = defaultdict(list)

    for leaf_idx, path in enumerate(paths):
        prompt_id = prompt_ids[leaf_idx]
        leaves_by_prompt[prompt_id].append(leaf_idx)
        if path[0] not in roots_by_prompt[prompt_id]:
            roots_by_prompt[prompt_id].append(path[0])

        offset = 0
        for depth, segment_idx in enumerate(path):
            prior_prompt = prompt_of_segment.setdefault(segment_idx, prompt_id)
            if prior_prompt != prompt_id:
                raise ValueError(f"segment {segment_idx} is shared across prompts")
            canonical.setdefault(segment_idx, (leaf_idx, offset))
            if depth:
                parent_idx = path[depth - 1]
                prior_parent = parent_of.setdefault(segment_idx, parent_idx)
                if prior_parent != parent_idx:
                    raise ValueError(f"segment {segment_idx} has multiple parents")
                if segment_idx not in children_of[parent_idx]:
                    children_of[parent_idx].append(segment_idx)
            offset += len(unique_segments[segment_idx])

    cyclic_roots = [
        root
        for roots in roots_by_prompt.values()
        for root in roots
        if root in parent_of
    ]
    if cyclic_roots:
        raise ValueError(f"tree roots also appear as children; cyclic topology at {cyclic_roots[:8]}")

    dtype = torch.float32
    device = old_log_probs.device
    segment_masses = torch.zeros(num_segments, dtype=dtype, device=device)
    leaf_masses = torch.zeros(num_leaves, dtype=dtype, device=device)
    branch_coverages: list[torch.Tensor] = []

    with torch.no_grad():
        for prompt_id, leaf_indices in leaves_by_prompt.items():
            roots = roots_by_prompt[prompt_id]
            if not roots:
                raise ValueError(f"prompt {prompt_id!r} has no tree root")

            # The expanded tree and every conventional top-up sequence are
            # separate Monte-Carlo strata.  Equal root mass prevents the many
            # leaves of an expanded stratum from overwhelming iid top-ups.
            root_mass = 1.0 / len(roots)
            pending = list(roots)
            for root in roots:
                segment_masses[root] = root_mass

            while pending:
                parent_idx = pending.pop()
                children = children_of.get(parent_idx, [])
                if not children:
                    continue

                child_log_probs = []
                for child_idx in children:
                    leaf_idx, offset = canonical[child_idx]
                    if offset >= response_length or response_mask[leaf_idx, offset] <= 0:
                        raise ValueError(
                            f"branch token for segment {child_idx} is outside the valid response "
                            f"(leaf={leaf_idx}, offset={offset})"
                        )
                    child_log_probs.append(old_log_probs[leaf_idx, offset].detach().float())

                child_log_probs_t = torch.stack(child_log_probs)
                if not torch.all(torch.isfinite(child_log_probs_t)):
                    raise ValueError(
                        f"branch below segment {parent_idx} has non-finite old actor log probabilities"
                    )
                child_probs = torch.softmax(child_log_probs_t, dim=0)
                # Sum of old-actor top-k probabilities before conditioning.
                # It is useful for diagnosing how much omitted-tail bias the
                # conditional objective may have.
                branch_coverages.append(torch.exp(child_log_probs_t).sum().clamp(max=1.0))
                for child_idx, child_prob in zip(children, child_probs, strict=True):
                    segment_masses[child_idx] = segment_masses[parent_idx] * child_prob
                    pending.append(child_idx)

            for leaf_idx in leaf_indices:
                leaf_masses[leaf_idx] = segment_masses[paths[leaf_idx][-1]]

            # Only remove accumulated floating-point error. A materially
            # incomplete prompt/tree must be rejected by the rollout contract,
            # not silently reinterpreted as a newly conditioned support.
            prompt_mass = leaf_masses[leaf_indices].sum()
            if not torch.isfinite(prompt_mass) or prompt_mass <= 0:
                raise ValueError(f"prompt {prompt_id!r} has invalid leaf mass {prompt_mass.item()}")
            if not torch.isclose(prompt_mass, prompt_mass.new_tensor(1.0), atol=1e-5, rtol=1e-5):
                raise ValueError(
                    f"prompt {prompt_id!r} has incomplete tree mass {prompt_mass.item():.8f}; "
                    "refusing to silently renormalize a pruned tree"
                )
            leaf_masses[leaf_indices] /= prompt_mass

        # Derive reach mass from terminal leaves so shared-prefix occupancy is
        # exactly the sum of its represented descendants.
        segment_masses.zero_()
        for leaf_idx, path in enumerate(paths):
            for segment_idx in path:
                segment_masses[segment_idx] += leaf_masses[leaf_idx]

    coverages = (
        torch.stack(branch_coverages).to(device=device, dtype=dtype)
        if branch_coverages
        else torch.empty(0, device=device, dtype=dtype)
    )
    return TreeActorWeights(
        leaf_masses=leaf_masses.detach(),
        segment_masses=segment_masses.detach(),
        branch_coverages=coverages.detach(),
    )


def compute_segment_reach_masses(
    leaf_masses: torch.Tensor,
    leaf_segment_indices: Sequence[Sequence[int]],
    num_segments: int,
) -> torch.Tensor:
    """Aggregate per-leaf probability masses into unique-segment reach mass."""

    if leaf_masses.ndim != 1 or len(leaf_segment_indices) != leaf_masses.shape[0]:
        raise ValueError("leaf_masses must be rank-1 and aligned with leaf_segment_indices")
    segment_masses = torch.zeros(num_segments, dtype=leaf_masses.dtype, device=leaf_masses.device)
    for leaf_idx, path in enumerate(leaf_segment_indices):
        for segment_idx in path:
            segment_idx = int(segment_idx)
            if segment_idx < 0 or segment_idx >= num_segments:
                raise ValueError(
                    f"leaf {leaf_idx} has out-of-range segment index {segment_idx} for {num_segments} segments"
                )
            segment_masses[segment_idx] += leaf_masses[leaf_idx]
    return segment_masses.detach()


def normalize_tree_loss_scales(
    masses: torch.Tensor,
    unit_mask: torch.Tensor,
    loss_agg_mode: str,
    normalization_totals: tuple[float | torch.Tensor, float | torch.Tensor] | None = None,
) -> torch.Tensor:
    """Convert probability/reach mass into fixed mean-one reducer scales.

    The normalization must be performed before splitting an optimizer group
    into microbatches.  The returned scale is then multiplied into each unit's
    loss while the normal unweighted reducer denominator is retained.  This
    prevents a size-one microbatch from renormalizing every nonzero mass to one.
    """

    if masses.ndim != 1 or unit_mask.ndim != 2 or masses.shape[0] != unit_mask.shape[0]:
        raise ValueError(
            "masses must be rank-1 and aligned with the first dimension of rank-2 unit_mask"
        )
    masses = masses.detach().to(device=unit_mask.device, dtype=torch.float32)
    if not torch.all(torch.isfinite(masses)) or torch.any(masses < 0):
        raise ValueError("masses must be finite and non-negative")

    valid_lengths = unit_mask.to(torch.float32).sum(dim=-1)
    valid_units = valid_lengths > 0
    if not torch.any(valid_units):
        return torch.zeros_like(masses)

    if normalization_totals is None:
        unweighted_total, mass_weighted_total = tree_loss_scale_normalization_totals(
            masses, unit_mask, loss_agg_mode
        )
    else:
        unweighted_total = torch.as_tensor(
            normalization_totals[0], device=masses.device, dtype=masses.dtype
        )
        mass_weighted_total = torch.as_tensor(
            normalization_totals[1], device=masses.device, dtype=masses.dtype
        )

    if loss_agg_mode == "token-mean":
        scale = masses * unweighted_total / mass_weighted_total.clamp_min(1e-8)
    elif loss_agg_mode in {
        "seq-mean-token-sum",
        "seq-mean-token-mean",
        "seq-mean-token-sum-norm",
    }:
        scale = masses * unweighted_total / mass_weighted_total.clamp_min(1e-8)
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return torch.where(valid_units, scale, torch.zeros_like(scale)).detach()


def tree_loss_scale_normalization_totals(
    masses: torch.Tensor,
    unit_mask: torch.Tensor,
    loss_agg_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return additive sufficient statistics for global scale normalization.

    The two scalars can be summed across data-parallel ranks before passing
    them back to :func:`normalize_tree_loss_scales`.
    """

    if masses.ndim != 1 or unit_mask.ndim != 2 or masses.shape[0] != unit_mask.shape[0]:
        raise ValueError(
            "masses must be rank-1 and aligned with the first dimension of rank-2 unit_mask"
        )
    masses = masses.detach().to(device=unit_mask.device, dtype=torch.float32)
    valid_lengths = unit_mask.to(torch.float32).sum(dim=-1)
    valid_units = valid_lengths > 0
    if loss_agg_mode == "token-mean":
        return valid_lengths.sum(), torch.sum(masses * valid_lengths)
    if loss_agg_mode in {
        "seq-mean-token-sum",
        "seq-mean-token-mean",
        "seq-mean-token-sum-norm",
    }:
        return valid_units.to(torch.float32).sum(), torch.sum(masses * valid_units.to(masses.dtype))
    raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


def select_segments_for_present_leaves(
    segment_canonical: Sequence[tuple[int, int]],
    present_leaves: set[int],
) -> list[int]:
    """Select every local segment whose canonical leaf is in a microbatch.

    Segment ids are already local and disjoint after ``DataProto.chunk``; no
    rank-modulo ownership filter belongs here.
    """

    return [
        segment_idx
        for segment_idx, (leaf_idx, _) in enumerate(segment_canonical)
        if leaf_idx in present_leaves
    ]
