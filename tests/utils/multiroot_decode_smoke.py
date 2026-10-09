"""Small real-model assembly smoke test; does not start RL."""
import argparse, json, os
os.environ['VLLM_USE_V1']='0'
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import TreeSearchParams
from verl import DataProto
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout

parser=argparse.ArgumentParser()
parser.add_argument('--model',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
tok=AutoTokenizer.from_pretrained(args.model)
engine=LLM(model=args.model,tensor_parallel_size=1,gpu_memory_utilization=.7,
    max_model_len=512,max_num_seqs=128,enforce_eager=True,disable_async_output_proc=True)
texts=['Solve step by step: If x + 7 = 19, what is x?',
       'Solve step by step: Find all real roots of x^2 - 5x + 6 = 0.']
ids=[tok.apply_chat_template([{'role':'user','content':t}],tokenize=True,add_generation_prompt=True) for t in texts]
width=max(map(len,ids)); pad=tok.pad_token_id or tok.eos_token_id
input_ids=torch.tensor([[pad]*(width-len(x))+x for x in ids])
mask=torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in ids])
positions=(mask.cumsum(-1)-1).clamp_min(0)
class CachedEngine:
    def __init__(self): self.cache=[]; self.index=0; self.replay=False
    def generate(self,**kw):
        if self.replay:
            result=self.cache[self.index];self.index+=1;return result
        result=engine.generate(**kw);self.cache.append(result);return result
cached=CachedEngine();results={}
for process in [False,True]:
    obj=object.__new__(vLLMRollout)
    obj.config=OmegaConf.create(dict(n=1,response_length=256,calculate_log_probs=False,
      tree_search=dict(enable=True,num_roots=8,branching_factor=2,max_tree_depth=3,
      topup_leaves_to_target=True,tree_process_reward=process)))
    obj.pad_token_id=pad;obj.lora_kwargs={};obj.inference_engine=cached
    obj.sampling_params=SamplingParams(n=1,seed=42,max_tokens=256,temperature=1.,logprobs=0,
      tree_search_params=TreeSearchParams(enable_tree_search=True,branch_trigger_mode='entropy',
      entropy_threshold=.8,tau_importance=None,branching_factor=2,max_tree_depth=3,
      min_seg_length=64,max_num_leaves=8,branch_sampling='sample_with_replacement'))
    data=DataProto(batch=TensorDict(dict(input_ids=input_ids,attention_mask=mask,position_ids=positions),batch_size=2),
      non_tensor_batch={'uid':np.array(['q0','q1'],dtype=object)},meta_info={'eos_token_id':tok.eos_token_id})
    out=obj.generate_sequences(data)
    assert len(out.batch['responses'])==128
    assert np.bincount(out.non_tensor_batch['tree_prompt_indices']).tolist()==[64,64]
    assert ('token_share_weights' in out.batch)==process
    if process:
        from verl.trainer.ppo.ray_trainer import compute_tree_process_advantage
        out.batch['response_mask']=(out.batch['responses']!=pad).long()
        out.batch['token_level_rewards']=torch.zeros_like(out.batch['responses'],dtype=torch.float32)
        out.batch['token_level_rewards'][::3,0]=1  # Synthetic rewards test assembly, not accuracy.
        compute_tree_process_advantage(out)
        assert torch.isfinite(out.batch['advantages']).all()
    results[str(process)]=dict(responses=len(out.batch['responses']),counts=[64,64],metrics=out.meta_info['metrics'])
    cached.replay=True
with open(args.output,'w') as f: json.dump(results,f,indent=2)
print('REAL_DECODE_SMOKE_OK',args.output)
