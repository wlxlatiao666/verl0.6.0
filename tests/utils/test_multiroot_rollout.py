"""Exercise actual rollout assembly with a small deterministic fake engine."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.utils.tree_training import get_ppo_rollout_batch_multiplier
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
from vllm import SamplingParams
from vllm.sampling_params import TreeSearchParams

class FakeEngine:
    def __init__(self): self.calls = []
    def generate(self, prompts, sampling_params, **kwargs):
        self.calls.append((prompts, sampling_params))
        params = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(prompts)
        result = []
        for i, sp in enumerate(params):
            if sp.tree_search_params is None:
                seqs = [NS(token_ids=[40, 41, 2]) for _ in range(sp.n)]
            elif i % 2:
                seqs = [NS(seq_id=i*10, tree_ids=[20, 2], token_ids=[20, 2], is_leaf=True, parent_seq_id=None, tree_depth=0)]
            else:
                # One actual binary split, plus top-ups to per-root budget.
                seqs = [NS(seq_id=i*10, tree_ids=[10], token_ids=[10], is_leaf=False, parent_seq_id=None, tree_depth=0)]
                seqs += [NS(seq_id=i*10+j, tree_ids=[20+j, 2], token_ids=[20+j, 2], is_leaf=True, parent_seq_id=i*10, tree_depth=1) for j in (1,2)]
            result.append(NS(outputs=seqs))
        return result

def make_rollout(process, roots=8):
    obj = object.__new__(vLLMRollout)
    obj.config = OmegaConf.create(dict(n=1, response_length=8, calculate_log_probs=False,
        tree_search=dict(enable=True, num_roots=roots, branching_factor=2, max_tree_depth=3,
                         topup_leaves_to_target=True, tree_process_reward=process)))
    obj.sampling_params = SamplingParams(n=1, seed=42, max_tokens=8,
        tree_search_params=TreeSearchParams(enable_tree_search=True, branch_sampling='sample_with_replacement'))
    obj.inference_engine = FakeEngine()
    obj.pad_token_id = 0
    obj.lora_kwargs = {}
    return obj

def prompts():
    return DataProto(batch=TensorDict(dict(input_ids=torch.tensor([[3,4],[5,6]]),
        attention_mask=torch.ones(2,2,dtype=torch.long),position_ids=torch.tensor([[0,1],[0,1]])),batch_size=2),
        non_tensor_batch={'uid':np.array(['a','b'],dtype=object)},meta_info={'eos_token_id':2})

@pytest.mark.parametrize('process', [False,True])
@pytest.mark.parametrize('roots', [1,8])
def test_multiroot_assembly(process,roots):
    obj=make_rollout(process,roots)
    out=obj.generate_sequences(prompts())
    assert len(out.batch['responses']) == 2*roots*8
    assert np.bincount(out.non_tensor_batch['tree_prompt_indices']).tolist() == [roots*8]*2
    assert set(out.non_tensor_batch['tree_num_prompts']) == {2}
    assert np.unique(out.non_tensor_batch['uid'], return_counts=True)[1].tolist() == [roots*8]*2
    assert out.meta_info['metrics']['tree/avg_leaves_per_prompt'] == roots*8
    assert get_ppo_rollout_batch_multiplier(obj.config) == roots*8
    assert ('token_share_weights' in out.batch) == process
    assert ('leaf_segment_indices' in out.non_tensor_batch) == process
    ps=obj.inference_engine.calls[0][1]
    if roots>1:
        assert len({p.seed for p in ps}) == 2*roots
        assert len({id(p.tree_search_params) for p in ps}) == 2*roots
    if process:
        paths=out.non_tensor_batch['leaf_segment_indices']
        # No shared node can belong to two original questions.
        owners={}
        for uid,path in zip(out.non_tensor_batch['uid'],paths):
            for node in path:
                assert owners.setdefault(node,uid) == uid
        # Actual treePR computation: two roots share global question statistics,
        # but local siblings are connected only by their explicit parent path.
        from verl.trainer.ppo.ray_trainer import compute_tree_process_advantage
        out.batch['response_mask']=(out.batch['responses']!=0).long()
        rewards=torch.zeros_like(out.batch['responses'],dtype=torch.float)
        rewards[::2,0]=1
        out.batch['token_level_rewards']=rewards
        result=compute_tree_process_advantage(out)
        assert torch.isfinite(result.batch['advantages']).all()


def test_root_seed_and_state_isolation():
    p=SamplingParams(seed=19,tree_search_params=TreeSearchParams(enable_tree_search=True))
    a,b=p.clone_for_tree_root(0),p.clone_for_tree_root(1)
    assert a.seed != b.seed and a.seed == p.clone_for_tree_root(0).seed
    a.tree_search_params.entropy_threshold=99
    assert b.tree_search_params.entropy_threshold == p.tree_search_params.entropy_threshold
    assert SamplingParams().clone_for_tree_root(1).seed is None
    with pytest.raises(ValueError): p.clone_for_tree_root(-1)

def test_treepr_roots_share_only_global_statistics():
    from verl.trainer.ppo.ray_trainer import compute_tree_process_advantage
    paths=np.empty(4,dtype=object)
    paths[:]=[[0,1],[0,2],[3,4],[3,5]]
    segments=np.empty(6,dtype=object)
    segments[:]=[[10],[11],[12],[13],[14],[15]]
    data=DataProto(batch=TensorDict(dict(
        response_mask=torch.ones(4,2,dtype=torch.long),
        token_level_rewards=torch.tensor([[0.,0.],[0.,0.],[1.,0.],[1.,0.]])),batch_size=4),
        non_tensor_batch={'uid':np.array(['same']*4,dtype=object),
          'unique_segments':segments,'leaf_segment_indices':paths})
    out=compute_tree_process_advantage(data)
    expected=.25/(torch.tensor([0.,0.,1.,1.]).std()+1e-6)
    assert torch.allclose(out.batch['advantages'][:2], torch.full((2,2),-expected))
    assert torch.allclose(out.batch['advantages'][2:], torch.full((2,2),expected))

def test_entropy_quantile_prepass():
    obj=make_rollout(False)
    obj.config.tree_search.threshold_stats_quantile=.9
    obj.inference_engine=NS(generate=lambda **kw:[NS(outputs=[NS(
        entropy_list=list(range(10)),importance_list=[])])])
    stats=obj.collect_threshold_stats(prompts())
    assert stats.meta_info['entropy_threshold_stat']==pytest.approx(8.1)
    assert stats.meta_info['entropy_p80']==pytest.approx(7.2)
