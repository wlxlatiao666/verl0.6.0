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

from types import SimpleNamespace

import pytest

from verl.utils.tree_training import (
    build_optimizer_micro_batches,
    get_ppo_rollout_batch_multiplier,
    is_tree_process_reward_enabled,
    split_balanced,
)


def test_ppo_rollout_batch_multiplier_uses_conventional_n():
    config = SimpleNamespace(n=64, tree_search={"enable": False})
    assert get_ppo_rollout_batch_multiplier(config) == 64


def test_ppo_rollout_batch_multiplier_uses_fixed_tree_target():
    config = SimpleNamespace(
        n=1,
        tree_search={
            "enable": True,
            "branching_factor": 2,
            "max_tree_depth": 6,
            "topup_leaves_to_target": True,
        },
    )
    assert get_ppo_rollout_batch_multiplier(config) == 64


def test_ppo_rollout_batch_multiplier_does_not_infer_variable_tree_size():
    config = SimpleNamespace(
        n=1,
        tree_search={
            "enable": True,
            "branching_factor": 2,
            "max_tree_depth": 6,
            "topup_leaves_to_target": False,
        },
    )
    assert get_ppo_rollout_batch_multiplier(config) == 1


def test_fixed_tree_rejects_multiple_input_copies():
    config = SimpleNamespace(
        n=2,
        tree_search={
            "enable": True,
            "branching_factor": 2,
            "max_tree_depth": 6,
            "topup_leaves_to_target": True,
        },
    )
    with pytest.raises(ValueError, match="rollout.n=1"):
        get_ppo_rollout_batch_multiplier(config)


@pytest.mark.parametrize(
    ("tree_config", "is_validate", "do_sample", "expected"),
    [
        (None, False, True, False),
        ({"enable": False, "tree_process_reward": True}, False, True, False),
        ({"enable": True, "tree_process_reward": False}, False, True, False),
        ({"enable": True, "tree_process_reward": True}, False, True, True),
        ({"enable": True, "tree_process_reward": True}, True, True, False),
        ({"enable": True, "tree_process_reward": True}, False, False, False),
    ],
)
def test_tree_process_reward_controls_process_only_features(tree_config, is_validate, do_sample, expected):
    assert (
        is_tree_process_reward_enabled(tree_config, is_validate=is_validate, do_sample=do_sample) is expected
    )


def test_split_balanced_preserves_coverage_and_order():
    groups = split_balanced(list(range(10)), 3)
    assert [len(group) for group in groups] == [4, 3, 3]
    assert [item for group in groups for item in group] == list(range(10))


def test_math7b_tree_optimizer_groups_match_base_grpo():
    prompts = 96
    prompt_mini_batch_size = 16
    responses_per_prompt = 2**6
    dp_size = 8
    response_micro_batch_size = 2

    local_responses = prompts * responses_per_prompt // dp_size
    local_mini_batch_size = prompt_mini_batch_size * responses_per_prompt // dp_size
    optimizer_steps = local_responses // local_mini_batch_size

    assert local_responses == 768
    assert local_mini_batch_size == 128
    assert local_mini_batch_size // response_micro_batch_size == 64
    assert optimizer_steps == prompts // prompt_mini_batch_size == 6

    segment_groups = split_balanced(list(range(local_responses)), optimizer_steps)
    assert [len(group) for group in segment_groups] == [128] * optimizer_steps

    optimizer_micro_batches = build_optimizer_micro_batches(
        list(range(local_responses)),
        num_optimizer_steps=optimizer_steps,
        micro_batch_size=response_micro_batch_size,
    )
    assert len(optimizer_micro_batches) == optimizer_steps
    assert [len(group) for group in optimizer_micro_batches] == [64] * optimizer_steps
    assert [item for group in optimizer_micro_batches for micro in group for item in micro] == list(
        range(local_responses)
    )


@pytest.mark.parametrize("num_groups", [0, -1])
def test_split_balanced_rejects_non_positive_group_count(num_groups):
    with pytest.raises(ValueError, match="num_groups must be positive"):
        split_balanced([0], num_groups)


def test_split_balanced_rejects_empty_groups():
    with pytest.raises(ValueError, match="cannot split"):
        split_balanced([0, 1], 3)
