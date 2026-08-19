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

import numpy as np
import pytest
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_grpo_outcome_advantage
from verl.trainer.ppo.tree_weighting import (
    compute_conditional_topk_tree_weights,
    compute_segment_reach_masses,
    normalize_tree_loss_scales,
    select_segments_for_present_leaves,
    tree_loss_scale_normalization_totals,
)


def test_conditional_topk_weights_follow_actor_probability_and_reach_mass():
    # Tree: 0 -> {1 -> {3, 4}, 2}.  Segment 0 is the shared sampled prefix;
    # the first token of every child segment is its forced branch token.
    unique_segments = [[10, 11], [20], [30, 31], [40], [50]]
    paths = [[0, 1, 3], [0, 1, 4], [0, 2]]
    old_log_probs = torch.zeros(3, 5)
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )

    # P(child 1 | {1,2}) = .8, P(child 2 | {1,2}) = .2.
    old_log_probs[0, 2] = torch.log(torch.tensor(0.4))
    old_log_probs[1, 2] = torch.log(torch.tensor(0.4))
    old_log_probs[2, 2] = torch.log(torch.tensor(0.1))
    # P(child 3 | {3,4}) = .75, P(child 4 | {3,4}) = .25.
    old_log_probs[0, 3] = torch.log(torch.tensor(0.3))
    old_log_probs[1, 3] = torch.log(torch.tensor(0.1))

    result = compute_conditional_topk_tree_weights(
        old_log_probs=old_log_probs,
        response_mask=response_mask,
        unique_segments=unique_segments,
        leaf_segment_indices=paths,
        prompt_ids=np.array(["p", "p", "p"], dtype=object),
    )

    torch.testing.assert_close(result.leaf_masses, torch.tensor([0.6, 0.2, 0.2]))
    torch.testing.assert_close(result.segment_masses, torch.tensor([1.0, 0.8, 0.2, 0.6, 0.2]))
    torch.testing.assert_close(result.branch_coverages, torch.tensor([0.5, 0.4]))


def test_topup_mass_equals_mean_expanded_tree_leaf_mass():
    # Expanded root 0 has children 1/2.  Segments 3 and 4 are conventional
    # top-ups represented as synthetic one-node roots.
    unique_segments = [[10], [20], [30], [40], [50]]
    paths = [[0, 1], [0, 2], [3], [4]]
    old_log_probs = torch.zeros(4, 2)
    response_mask = torch.tensor([[1, 1], [1, 1], [1, 0], [1, 0]], dtype=torch.float32)
    old_log_probs[0, 1] = torch.log(torch.tensor(0.9))
    old_log_probs[1, 1] = torch.log(torch.tensor(0.1))

    result = compute_conditional_topk_tree_weights(
        old_log_probs=old_log_probs,
        response_mask=response_mask,
        unique_segments=unique_segments,
        leaf_segment_indices=paths,
        prompt_ids=[0, 0, 0, 0],
    )

    expected_leaf = torch.tensor([0.45, 0.05, 0.25, 0.25])
    torch.testing.assert_close(result.leaf_masses, expected_leaf)
    torch.testing.assert_close(result.segment_masses, torch.tensor([0.5, 0.45, 0.05, 0.25, 0.25]))
    tree_leaf_mean = result.leaf_masses[:2].mean()
    torch.testing.assert_close(result.leaf_masses[2:], tree_leaf_mean.expand(2))
    torch.testing.assert_close(result.leaf_masses.sum(), torch.tensor(1.0))


def test_multilevel_tree_and_topups_use_equal_mean_leaf_slots():
    # Expanded tree 0 -> {1 -> {3,4}, 2} has conditional leaf mass
    # [.6, .2, .2]. Three one-node top-ups make six emitted leaf slots.
    unique_segments = [[token] for token in range(8)]
    paths = [[0, 1, 3], [0, 1, 4], [0, 2], [5], [6], [7]]
    old_log_probs = torch.zeros(6, 3)
    response_mask = torch.tensor(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=torch.float32,
    )
    old_log_probs[0, 1] = torch.log(torch.tensor(0.4))
    old_log_probs[1, 1] = torch.log(torch.tensor(0.4))
    old_log_probs[2, 1] = torch.log(torch.tensor(0.1))
    old_log_probs[0, 2] = torch.log(torch.tensor(0.3))
    old_log_probs[1, 2] = torch.log(torch.tensor(0.1))

    result = compute_conditional_topk_tree_weights(
        old_log_probs=old_log_probs,
        response_mask=response_mask,
        unique_segments=unique_segments,
        leaf_segment_indices=paths,
        prompt_ids=["p"] * 6,
    )

    expected_leaf = torch.tensor([0.3, 0.1, 0.1, 1 / 6, 1 / 6, 1 / 6])
    expected_reach = torch.tensor([0.5, 0.4, 0.1, 0.3, 0.1, 1 / 6, 1 / 6, 1 / 6])
    torch.testing.assert_close(result.leaf_masses, expected_leaf)
    torch.testing.assert_close(result.segment_masses, expected_reach)
    torch.testing.assert_close(result.leaf_masses[3:], result.leaf_masses[:3].mean().expand(3))

    scales = normalize_tree_loss_scales(result.leaf_masses, response_mask, "token-mean")
    torch.testing.assert_close(scales[3:], scales[:3].mean().expand(3))


def test_topup_equal_mean_normalization_is_independent_for_interleaved_prompts():
    # Prompt A has two tree leaves plus one top-up. Prompt B has three tree
    # leaves plus one top-up. Rows are deliberately interleaved.
    unique_segments = [[token] for token in range(9)]
    paths = [[0, 1], [4, 5], [3], [4, 6], [0, 2], [8], [4, 7]]
    prompt_ids = np.array(["a", "b", "a", "b", "a", "b", "b"], dtype=object)
    old_log_probs = torch.zeros(7, 2)
    response_mask = torch.tensor(
        [
            [1, 1],
            [1, 1],
            [1, 0],
            [1, 1],
            [1, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=torch.float32,
    )
    old_log_probs[0, 1] = torch.log(torch.tensor(0.75))
    old_log_probs[4, 1] = torch.log(torch.tensor(0.25))
    old_log_probs[1, 1] = torch.log(torch.tensor(0.5))
    old_log_probs[3, 1] = torch.log(torch.tensor(0.3))
    old_log_probs[6, 1] = torch.log(torch.tensor(0.2))

    result = compute_conditional_topk_tree_weights(
        old_log_probs=old_log_probs,
        response_mask=response_mask,
        unique_segments=unique_segments,
        leaf_segment_indices=paths,
        prompt_ids=prompt_ids,
    )

    a_rows = torch.tensor([0, 2, 4])
    a_tree_rows = torch.tensor([0, 4])
    b_rows = torch.tensor([1, 3, 5, 6])
    b_tree_rows = torch.tensor([1, 3, 6])
    torch.testing.assert_close(result.leaf_masses[a_rows].sum(), torch.tensor(1.0))
    torch.testing.assert_close(result.leaf_masses[b_rows].sum(), torch.tensor(1.0))
    torch.testing.assert_close(result.leaf_masses[2], result.leaf_masses[a_tree_rows].mean())
    torch.testing.assert_close(result.leaf_masses[5], result.leaf_masses[b_tree_rows].mean())


def test_weights_normalize_each_prompt_independently():
    unique_segments = [[10], [20], [30], [40], [50], [60]]
    paths = [[0, 1], [0, 2], [3, 4], [3, 5]]
    old_log_probs = torch.zeros(4, 2)
    response_mask = torch.ones_like(old_log_probs)
    old_log_probs[:, 1] = torch.log(torch.tensor([0.7, 0.3, 0.2, 0.8]))

    result = compute_conditional_topk_tree_weights(
        old_log_probs=old_log_probs,
        response_mask=response_mask,
        unique_segments=unique_segments,
        leaf_segment_indices=paths,
        prompt_ids=["a", "a", "b", "b"],
    )

    torch.testing.assert_close(result.leaf_masses, torch.tensor([0.7, 0.3, 0.2, 0.8]))
    assert result.leaf_masses[:2].sum() == pytest.approx(1.0)
    assert result.leaf_masses[2:].sum() == pytest.approx(1.0)


def test_segment_reach_mass_does_not_use_a_canonical_leaf_weight():
    leaf_masses = torch.tensor([0.6, 0.2, 0.2])
    paths = [[0, 1, 3], [0, 1, 4], [0, 2]]
    segment_masses = compute_segment_reach_masses(leaf_masses, paths, num_segments=5)
    torch.testing.assert_close(segment_masses, torch.tensor([1.0, 0.8, 0.2, 0.6, 0.2]))


def test_concat_then_chunk_preserves_tree_tensors_and_local_topology():
    def worker(tensor_values, segments, paths):
        segment_array = np.empty(len(segments), dtype=object)
        for idx, segment in enumerate(segments):
            segment_array[idx] = segment
        path_array = np.empty(len(paths), dtype=object)
        for idx, path in enumerate(paths):
            path_array[idx] = path
        return DataProto.from_dict(
            tensors={
                "tree_leaf_masses": torch.tensor(tensor_values, dtype=torch.float32),
                "tree_loss_scales": torch.tensor(tensor_values, dtype=torch.float32) * 2,
            },
            non_tensors={
                "uid": np.array(["p"] * len(paths), dtype=object),
                "unique_segments": segment_array,
                "leaf_segment_indices": path_array,
            },
        )

    first = worker([0.7, 0.3], [[10], [20], [30]], [[0, 1], [0, 2]])
    second = worker([1.0], [[40]], [[0]])
    chunks = DataProto.concat([first, second], keep_local_workers=True).chunk(2)

    torch.testing.assert_close(chunks[0].batch["tree_leaf_masses"], torch.tensor([0.7, 0.3]))
    torch.testing.assert_close(chunks[0].batch["tree_loss_scales"], torch.tensor([1.4, 0.6]))
    assert chunks[0].non_tensor_batch["leaf_segment_indices"].tolist() == [[0, 1], [0, 2]]
    assert len(chunks[0].non_tensor_batch["unique_segments"]) == 3

    torch.testing.assert_close(chunks[1].batch["tree_leaf_masses"], torch.tensor([1.0]))
    torch.testing.assert_close(chunks[1].batch["tree_loss_scales"], torch.tensor([2.0]))
    assert chunks[1].non_tensor_batch["leaf_segment_indices"].tolist() == [[0]]
    assert len(chunks[1].non_tensor_batch["unique_segments"]) == 1


def test_leaf_row_reordering_does_not_change_tree_mass():
    unique_segments = [[10], [20], [30]]
    paths = [[0, 1], [0, 2]]
    old_log_probs = torch.tensor([[0.0, torch.log(torch.tensor(0.8))], [0.0, torch.log(torch.tensor(0.2))]])
    response_mask = torch.ones_like(old_log_probs)
    original = compute_conditional_topk_tree_weights(
        old_log_probs=old_log_probs,
        response_mask=response_mask,
        unique_segments=unique_segments,
        leaf_segment_indices=paths,
        prompt_ids=[0, 0],
    )
    order = torch.tensor([1, 0])
    reordered = compute_conditional_topk_tree_weights(
        old_log_probs=old_log_probs[order],
        response_mask=response_mask[order],
        unique_segments=unique_segments,
        leaf_segment_indices=[paths[i] for i in order.tolist()],
        prompt_ids=[0, 0],
    )
    torch.testing.assert_close(reordered.leaf_masses[order], original.leaf_masses)
    torch.testing.assert_close(reordered.segment_masses, original.segment_masses)


def test_invalid_branch_offset_is_rejected():
    with pytest.raises(ValueError, match="outside the valid response"):
        compute_conditional_topk_tree_weights(
            old_log_probs=torch.zeros(2, 2),
            response_mask=torch.tensor([[1, 0], [1, 0]], dtype=torch.float32),
            unique_segments=[[10], [20], [30]],
            leaf_segment_indices=[[0, 1], [0, 2]],
            prompt_ids=[0, 0],
        )


@pytest.mark.parametrize("bad_log_prob", [float("nan"), float("-inf")])
def test_nonfinite_branch_logprob_is_rejected(bad_log_prob):
    with pytest.raises(ValueError, match="non-finite"):
        compute_conditional_topk_tree_weights(
            old_log_probs=torch.tensor([[0.0, bad_log_prob], [0.0, -1.0]]),
            response_mask=torch.ones(2, 2),
            unique_segments=[[10], [20], [30]],
            leaf_segment_indices=[[0, 1], [0, 2]],
            prompt_ids=[0, 0],
        )


def test_empty_segment_is_rejected():
    with pytest.raises(ValueError, match="empty segments"):
        compute_conditional_topk_tree_weights(
            old_log_probs=torch.zeros(1, 1),
            response_mask=torch.ones(1, 1),
            unique_segments=[[]],
            leaf_segment_indices=[[0]],
            prompt_ids=[0],
        )


def test_repeated_segment_path_is_rejected_before_topology_walk():
    with pytest.raises(ValueError, match="topology cycle"):
        compute_conditional_topk_tree_weights(
            old_log_probs=torch.zeros(1, 3),
            response_mask=torch.ones(1, 3),
            unique_segments=[[10], [20]],
            leaf_segment_indices=[[0, 1, 0]],
            prompt_ids=[0],
        )


def test_cross_path_cycle_is_rejected_before_topology_walk():
    with pytest.raises(ValueError, match="cyclic topology"):
        compute_conditional_topk_tree_weights(
            old_log_probs=torch.zeros(2, 2),
            response_mask=torch.ones(2, 2),
            unique_segments=[[10], [20]],
            leaf_segment_indices=[[0, 1], [1, 0]],
            prompt_ids=[0, 0],
        )


def test_segment_reach_rejects_invalid_indices():
    with pytest.raises(ValueError, match="out-of-range"):
        compute_segment_reach_masses(torch.tensor([1.0]), [[-1]], num_segments=1)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("token-mean", 2.6),
        ("seq-mean-token-sum", 13.0 / 3.0),
        ("seq-mean-token-mean", 3.0),
    ],
)
def test_weighted_loss_reducer(mode, expected):
    loss_mat = torch.tensor([[1.0, 3.0], [5.0, 0.0]])
    loss_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    masses = torch.tensor([2.0 / 3.0, 1.0 / 3.0])
    scales = normalize_tree_loss_scales(masses, loss_mask, mode)
    actual = agg_loss(loss_mat, loss_mask, mode, loss_weights=scales)
    torch.testing.assert_close(actual, torch.tensor(expected))


def test_weighted_loss_reducer_recovers_uniform_behavior():
    loss_mat = torch.tensor([[1.0, 3.0], [5.0, 0.0]])
    loss_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    for mode in ("token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"):
        unweighted = agg_loss(loss_mat, loss_mask, mode)
        weighted = agg_loss(loss_mat, loss_mask, mode, loss_weights=torch.ones(2))
        torch.testing.assert_close(weighted, unweighted)


def test_weighted_loss_reducer_zero_weight_preserves_autograd():
    loss_mat = torch.tensor([[1.0, 2.0]], requires_grad=True)
    loss = agg_loss(
        loss_mat,
        torch.ones_like(loss_mat),
        "token-mean",
        loss_weights=torch.zeros(1),
    )
    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(loss_mat.grad, torch.zeros_like(loss_mat))


def test_weighted_loss_reducer_rejects_nonbroadcastable_shape():
    with pytest.raises(ValueError, match="cannot broadcast"):
        agg_loss(
            torch.ones(2, 3),
            torch.ones(2, 3),
            "token-mean",
            loss_weights=torch.ones(2, 2),
        )


def test_fixed_scales_survive_size_one_microbatches():
    # If each microbatch divided by its own weight sum, these would become
    # equal-weight samples. Fixed full-group scales preserve the intended 9:1.
    loss_mat = torch.tensor([[2.0], [10.0]])
    loss_mask = torch.ones_like(loss_mat)
    masses = torch.tensor([0.9, 0.1])
    scales = normalize_tree_loss_scales(masses, loss_mask, "seq-mean-token-mean")

    full_loss = agg_loss(
        loss_mat,
        loss_mask,
        "seq-mean-token-mean",
        loss_weights=scales,
    )
    micro_losses = [
        agg_loss(
            loss_mat[i : i + 1],
            loss_mask[i : i + 1],
            "seq-mean-token-mean",
            loss_weights=scales[i : i + 1],
        )
        for i in range(2)
    ]
    accumulated_loss = torch.stack(micro_losses).mean()

    torch.testing.assert_close(full_loss, torch.tensor(2.8))
    torch.testing.assert_close(accumulated_loss, full_loss)


def test_token_mean_full_loss_matches_unequal_length_micro_accumulation():
    loss_mat = torch.tensor([[1.0, 3.0], [5.0, 0.0]])
    loss_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    masses = torch.tensor([2.0 / 3.0, 1.0 / 3.0])
    scales = normalize_tree_loss_scales(masses, loss_mask, "token-mean")
    full_loss = agg_loss(loss_mat, loss_mask, "token-mean", loss_weights=scales)

    total_tokens = loss_mask.sum()
    accumulated_loss = loss_mat.new_tensor(0.0)
    for row in range(2):
        micro_loss = agg_loss(
            loss_mat[row : row + 1],
            loss_mask[row : row + 1],
            "token-mean",
            loss_weights=scales[row : row + 1],
        )
        accumulated_loss = accumulated_loss + micro_loss * (loss_mask[row].sum() / total_tokens)

    torch.testing.assert_close(full_loss, torch.tensor(2.6))
    torch.testing.assert_close(accumulated_loss, full_loss)


def test_local_segment_selection_does_not_apply_rank_modulo_filter():
    canonical = [(0, 0), (0, 2), (1, 0), (2, 0)]
    assert select_segments_for_present_leaves(canonical, {0, 2}) == [0, 1, 3]


@pytest.mark.parametrize("mode", ["token-mean", "seq-mean-token-mean"])
def test_global_segment_scale_normalization_matches_two_shards(mode):
    losses = torch.tensor([[1.0, 3.0, 0.0], [5.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    masses = torch.tensor([0.5, 0.3, 0.2])
    full_scales = normalize_tree_loss_scales(masses, mask, mode)
    full_loss = agg_loss(losses, mask, mode, loss_weights=full_scales)

    shard_slices = (slice(0, 1), slice(1, 3))
    shard_totals = [
        tree_loss_scale_normalization_totals(masses[sl], mask[sl], mode) for sl in shard_slices
    ]
    global_totals = tuple(sum(stats[i] for stats in shard_totals) for i in range(2))
    shard_scales = [
        normalize_tree_loss_scales(masses[sl], mask[sl], mode, normalization_totals=global_totals)
        for sl in shard_slices
    ]
    torch.testing.assert_close(torch.cat(shard_scales), full_scales)

    if mode == "token-mean":
        shard_units = [mask[sl].sum() for sl in shard_slices]
    else:
        shard_units = [torch.tensor(mask[sl].shape[0], dtype=torch.float32) for sl in shard_slices]
    global_units = sum(shard_units)
    combined = sum(
        agg_loss(losses[sl], mask[sl], mode, loss_weights=scales) * units / global_units
        for sl, scales, units in zip(shard_slices, shard_scales, shard_units, strict=True)
    )
    torch.testing.assert_close(combined, full_loss)


def test_leaf_batched_tree_segment_uses_segment_units_for_accumulation():
    # Three unique segments are owned by two canonical leaves. The first
    # leaf's microbatch owns two segments, so using leaf count (1/2) instead of
    # segment count (2/3) would not reproduce the full segment objective.
    canonical = [(0, 0), (0, 2), (1, 0)]
    segment_losses = torch.tensor([[1.0], [3.0], [9.0]])
    segment_mask = torch.ones_like(segment_losses)
    reach_masses = torch.tensor([1.0, 0.6, 0.4])
    scales = normalize_tree_loss_scales(
        reach_masses, segment_mask, "seq-mean-token-mean"
    )
    full_loss = agg_loss(
        segment_losses,
        segment_mask,
        "seq-mean-token-mean",
        loss_weights=scales,
    )

    accumulated = segment_losses.new_tensor(0.0)
    for present_leaves in ({0}, {1}):
        segment_indices = select_segments_for_present_leaves(canonical, present_leaves)
        micro_loss = agg_loss(
            segment_losses[segment_indices],
            segment_mask[segment_indices],
            "seq-mean-token-mean",
            loss_weights=scales[segment_indices],
        )
        accumulated = accumulated + micro_loss * (len(segment_indices) / len(canonical))
    torch.testing.assert_close(accumulated, full_loss)


def test_grpo_reward_statistics_use_tree_leaf_mass():
    rewards = torch.tensor([[0.0], [1.0], [10.0]])
    response_mask = torch.ones_like(rewards)
    masses = torch.tensor([0.6, 0.3, 0.1])
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.array(["p", "p", "p"], dtype=object),
        norm_adv_by_std_in_grpo=False,
        sample_weights=masses,
    )
    # Weighted mean = 0*.6 + 1*.3 + 10*.1 = 1.3.
    torch.testing.assert_close(advantages[:, 0], torch.tensor([-1.3, -0.3, 8.7]))


def test_grpo_weighted_population_standardization():
    rewards = torch.tensor([[10.0], [2.0], [0.0]])
    response_mask = torch.ones_like(rewards)
    masses = torch.tensor([0.6, 0.2, 0.2])
    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.array(["p", "p", "p"], dtype=object),
        norm_adv_by_std_in_grpo=True,
        sample_weights=masses,
    )
    mean = torch.sum(masses * advantages[:, 0])
    variance = torch.sum(masses * (advantages[:, 0] - mean).square())
    torch.testing.assert_close(mean, torch.tensor(0.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(variance, torch.tensor(1.0), atol=1e-5, rtol=0)
