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

"""Small, dependency-free helpers for tree-rollout actor batching."""


def _config_get(config, key, default=None):
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def get_ppo_rollout_batch_multiplier(rollout_config) -> int:
    """Return the response multiplicity used to normalize PPO mini-batches.

    Conventional rollout expands by ``rollout.n``. Tree rollout starts with
    ``n=1`` and expands inside vLLM, so a fixed-size, topped-up tree instead
    uses ``branching_factor ** max_tree_depth``.
    """
    rollout_n = int(_config_get(rollout_config, "n", 1))
    tree_config = _config_get(rollout_config, "tree_search", None)
    if tree_config is None or not bool(_config_get(tree_config, "enable", False)):
        return rollout_n
    if not bool(_config_get(tree_config, "topup_leaves_to_target", True)):
        return rollout_n
    if rollout_n != 1:
        raise ValueError(
            "fixed-size tree rollout requires rollout.n=1; tree expansion supplies the response multiplicity"
        )

    branching_factor = int(_config_get(tree_config, "branching_factor", 1))
    max_tree_depth = int(_config_get(tree_config, "max_tree_depth", 0))
    if branching_factor < 1:
        raise ValueError(f"tree_search.branching_factor must be positive, got {branching_factor}")
    if max_tree_depth < 0:
        raise ValueError(f"tree_search.max_tree_depth must be non-negative, got {max_tree_depth}")
    # Leaf-budget control: when max_num_leaves is set it is the binding size
    # target (top-up fills to it and vLLM caps splits at it), so the response
    # multiplicity is exactly that budget.
    max_num_leaves = _config_get(tree_config, "max_num_leaves", None)
    if max_num_leaves:
        max_num_leaves = int(max_num_leaves)
        if max_num_leaves < 1:
            raise ValueError(f"tree_search.max_num_leaves must be positive, got {max_num_leaves}")
        return max_num_leaves
    return branching_factor**max_tree_depth


def is_tree_process_reward_enabled(
    tree_config,
    *,
    is_validate: bool = False,
    do_sample: bool = True,
) -> bool:
    """Return whether process-reward-only tree training features may be emitted."""
    return bool(
        tree_config is not None
        and _config_get(tree_config, "enable", False)
        and _config_get(tree_config, "tree_process_reward", False)
        and do_sample
        and not is_validate
    )


def split_balanced(items: list[int], num_groups: int) -> list[list[int]]:
    """Split items into non-empty, order-preserving groups of near-equal size."""
    if num_groups < 1:
        raise ValueError(f"num_groups must be positive, got {num_groups}")
    if len(items) < num_groups:
        raise ValueError(f"cannot split {len(items)} items into {num_groups} non-empty groups")

    base_size, remainder = divmod(len(items), num_groups)
    groups = []
    start = 0
    for group_idx in range(num_groups):
        group_size = base_size + (1 if group_idx < remainder else 0)
        groups.append(items[start : start + group_size])
        start += group_size
    return groups


def build_optimizer_micro_batches(
    items: list[int], num_optimizer_steps: int, micro_batch_size: int
) -> list[list[list[int]]]:
    """Partition items into optimizer groups and then into micro-batches."""
    if micro_batch_size < 1:
        raise ValueError(f"micro_batch_size must be positive, got {micro_batch_size}")
    optimizer_groups = split_balanced(items, num_optimizer_steps)
    return [
        [group[start : start + micro_batch_size] for start in range(0, len(group), micro_batch_size)]
        for group in optimizer_groups
    ]
