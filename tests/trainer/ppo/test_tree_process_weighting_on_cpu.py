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

import math

import numpy as np
import torch

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import compute_tree_process_advantage


def _tree_process_batch() -> DataProto:
    # Tree: 0 -> {1 -> {3, 4}, 2}; leaf rewards [10, 2, 0] and actor
    # masses [.6, .2, .2]. Thus score(1)=8 and score(0)=6.4.
    unique_segments = np.empty(5, dtype=object)
    for idx, segment in enumerate(([10], [20], [30], [40], [50])):
        unique_segments[idx] = segment
    leaf_paths = np.empty(3, dtype=object)
    for idx, path in enumerate(([0, 1, 3], [0, 1, 4], [0, 2])):
        leaf_paths[idx] = path
    return DataProto.from_dict(
        tensors={
            "response_mask": torch.tensor(
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]
            ),
            "token_level_rewards": torch.tensor(
                [[0.0, 0.0, 10.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]
            ),
            "tree_leaf_masses": torch.tensor([0.6, 0.2, 0.2]),
        },
        non_tensors={
            "uid": np.array(["p", "p", "p"], dtype=object),
            "unique_segments": unique_segments,
            "leaf_segment_indices": leaf_paths,
        },
    )


def test_tree_process_local_advantage_uses_actor_conditional_backup():
    data = compute_tree_process_advantage(
        _tree_process_batch(),
        proc_agg_mode="raw",
        local_adv_weight=1.0,
        global_adv_weight=0.0,
    )
    advantages = data.batch["advantages"]
    expected = torch.tensor(
        [
            [0.0, 0.5, 2.0 / math.sqrt(12.0)],
            [0.0, 0.5, -6.0 / math.sqrt(12.0)],
            [0.0, -2.0, 0.0],
        ]
    )
    torch.testing.assert_close(advantages, expected, atol=2e-6, rtol=0)


def test_tree_process_global_advantage_uses_weighted_leaf_statistics():
    data = compute_tree_process_advantage(
        _tree_process_batch(),
        proc_agg_mode="raw",
        local_adv_weight=0.0,
        global_adv_weight=1.0,
    )
    advantages = data.batch["advantages"]
    std = math.sqrt(19.84)
    expected_terminal = torch.tensor([(10.0 - 6.4) / std, (2.0 - 6.4) / std, (0.0 - 6.4) / std])
    torch.testing.assert_close(
        torch.stack([advantages[0, 2], advantages[1, 2], advantages[2, 1]]),
        expected_terminal,
        atol=2e-6,
        rtol=0,
    )


def test_length_balancing_uses_unique_not_leaf_duplicated_segment_lengths():
    raw = compute_tree_process_advantage(
        _tree_process_batch(),
        proc_agg_mode="raw",
        local_adv_weight=1.0,
        global_adv_weight=0.0,
    ).batch["advantages"]
    balanced = compute_tree_process_advantage(
        _tree_process_batch(),
        proc_agg_mode="length_balanced",
        local_adv_weight=1.0,
        global_adv_weight=0.0,
    ).batch["advantages"]

    # Every unique segment has one valid token. Repeated copies of the shared
    # root must not inflate mean_seg_len from 1 to total_leaf_tokens / 5.
    torch.testing.assert_close(balanced, raw)
