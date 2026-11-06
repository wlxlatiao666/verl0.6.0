"""CPU checks for TreePR weighting, without importing distributed workers."""
import ast
import gc
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def compute(data):
    source = Path(__file__).resolve().parents[3] / 'verl/trainer/ppo/ray_trainer.py'
    function = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'compute_tree_process_advantage')
    namespace = dict(DataProto=object, torch=torch, np=np, gc=gc,
                     logging=logging, logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[function.name](data, local_adv_weight=0., global_adv_weight=1.)


def batch(logprobs=None):
    # Two trees with the same prompt UID, plus a conventional single-leaf top-up.
    # Tree A has unequal depth: root -> internal -> leaves 2,3; root -> leaf 4.
    paths = [[0, 1, 2], [0, 1, 3], [0, 4], [5, 6], [5, 7], [8]]
    return SimpleNamespace(
        batch={'response_mask': torch.tensor([[1,1,1], [1,1,1], [1,1,0],
                                               [1,1,0], [1,1,0], [1,0,0]]),
               'token_level_rewards': torch.tensor([[0.,0.,1.], [0.,0.,0.], [0.,0.,1.],
                                                     [0.,0.,0.], [0.,0.,1.], [0.,0.,0.]])},
        non_tensor_batch={'unique_segments': [[1]] * 9,
                          'leaf_segment_indices': paths,
                          'uid': np.array(['same'] * 6, dtype=object),
                          'tree_root_ids': np.array(['a','a','a','b','b','topup']),
                          'tree_leaf_logprobs': (np.log([.3,.12,.3,.8,.2,1.])
                                                if logprobs is None else logprobs)},
        meta_info={})


def test_probability_weights_apply_to_existing_advantages_and_mask():
    data = compute(batch())
    weights = torch.tensor([5/12, 1/6, 5/12, .8, .2, 1.])
    torch.testing.assert_close(data.batch['tree_leaf_weights'], weights)
    # Leaf reward baseline is still the original unweighted prompt mean/std.
    rewards = torch.tensor([1.,0.,1.,0.,1.,0.])
    expected = (rewards - rewards.mean()) / (rewards.std() + 1e-6) * weights
    positions = torch.tensor([2,2,1,1,1,0])
    torch.testing.assert_close(data.batch['advantages'][torch.arange(6), positions], expected)
    assert torch.count_nonzero(data.batch['advantages'][data.batch['response_mask'] == 0]) == 0
    torch.testing.assert_close(data.batch['returns'], data.batch['advantages'])


def test_deep_paths_do_not_underflow_and_groups_survive_row_reordering():
    data = batch(np.array([-10000.,-10001.,-10000.,-20000.,-20001.,0.]))
    order = [5, 3, 0, 4, 2, 1]
    for key in data.batch:
        data.batch[key] = data.batch[key][order]
    for key in ('uid', 'tree_root_ids', 'tree_leaf_logprobs'):
        data.non_tensor_batch[key] = data.non_tensor_batch[key][order]
    data.non_tensor_batch['leaf_segment_indices'] = [data.non_tensor_batch['leaf_segment_indices'][i] for i in order]
    data = compute(data)
    for root in ('a','b','topup'):
        selected = data.non_tensor_batch['tree_root_ids'] == root
        assert data.batch['tree_leaf_weights'][selected].sum().item() == pytest.approx(1.)
    assert torch.isfinite(data.batch['advantages']).all()


def test_missing_or_invalid_probability_metadata_fails():
    data = batch()
    del data.non_tensor_batch['tree_leaf_logprobs']
    with pytest.raises(KeyError):
        compute(data)
    data = batch([float('nan')] * 6)
    with pytest.raises(ValueError, match='finite'):
        compute(data)
