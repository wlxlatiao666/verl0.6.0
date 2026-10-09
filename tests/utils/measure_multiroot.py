"""Measure leaves/top-ups with production response budget and periodic P80."""
import argparse, json, os, time
from pathlib import Path
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
p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--questions',required=True)
p.add_argument('--output-dir',required=True)
p.add_argument('--num-questions',type=int,default=20)
args=p.parse_args()
outdir=Path(args.output_dir);outdir.mkdir(parents=True,exist_ok=True)
qs=[q for q in json.loads(Path(args.questions).read_text()) if q['split']=='test'][:args.num_questions]
assert len(qs)==args.num_questions
(outdir/'questions.json').write_text(json.dumps(qs,indent=2))
tokenizer=AutoTokenizer.from_pretrained(args.model)
pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
obj=object.__new__(vLLMRollout)
obj.config=OmegaConf.create(dict(n=1,response_length=2048,calculate_log_probs=False,
    tree_search=dict(enable=True,num_roots=8,branching_factor=2,max_tree_depth=3,
       threshold_stats_n=1,threshold_stats_max_tokens=64,threshold_stats_quantile=.8,
       threshold_stats_interval=10,topup_leaves_to_target=True,tree_process_reward=False)))
obj.pad_token_id=pad;obj.lora_kwargs={}
obj.inference_engine=LLM(model=args.model,tensor_parallel_size=1,gpu_memory_utilization=.7,
    max_model_len=4096,max_num_seqs=128,enforce_eager=True,disable_async_output_proc=True)
obj.sampling_params=SamplingParams(n=1,seed=42,max_tokens=2048,temperature=1.,logprobs=0,
    tree_search_params=TreeSearchParams(enable_tree_search=True,branch_trigger_mode='entropy',
       entropy_threshold=.8,tau_importance=None,branching_factor=2,max_tree_depth=3,
       min_seg_length=64,max_num_leaves=8,branch_sampling='sample_with_replacement'))
records=[]
for offset in range(0,len(qs),2):
    step=offset//2+1; batch=qs[offset:offset+2];ids=[q['prompt_ids'] for q in batch]
    width=max(map(len,ids))
    tokens=torch.tensor([[pad]*(width-len(x))+x for x in ids])
    mask=torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in ids])
    data=DataProto(batch=TensorDict(dict(input_ids=tokens,attention_mask=mask,
        position_ids=(mask.cumsum(-1)-1).clamp_min(0)),batch_size=len(batch)),
        non_tensor_batch={'uid':np.array([q['id'] for q in batch],dtype=object)},
        meta_info={'eos_token_id':tokenizer.eos_token_id})
    updated=step==1 or step%10==0
    if updated:
        stat=obj.collect_threshold_stats(data)
        obj.update_entropy_threshold(stat.meta_info['entropy_threshold_stat'])
    obj.sampling_params.seed=42+step
    start=time.time()
    result=obj.generate_sequences(data)
    counts=np.bincount(result.non_tensor_batch['tree_prompt_indices']).tolist()
    assert counts==[64]*len(batch),counts
    metrics=result.meta_info['metrics']
    leaves=int(metrics['tree/leaf_nodes']);topups=int(metrics['tree/topup_samples'])
    assert leaves+topups==64*len(batch)
    row=dict(step=step,questions=len(batch),threshold_updated=updated,
        entropy_threshold=obj.sampling_params.tree_search_params.entropy_threshold,
        tree_leaves=leaves,topups=topups,total=leaves+topups,seconds=time.time()-start,
        metrics=metrics)
    records.append(row)
    summary=dict(num_questions=sum(r['questions'] for r in records),target_questions=len(qs),
       tree_leaves=sum(r['tree_leaves'] for r in records),topups=sum(r['topups'] for r in records),
       total=sum(r['total'] for r in records),response_limit=2048,interval=10,
       quantile=.8,calibration_tokens=64,calibration_questions_per_update=2,
       model_updated=False,records=records)
    summary['tree_fraction']=summary['tree_leaves']/summary['total']
    summary['topup_fraction']=summary['topups']/summary['total']
    (outdir/'summary.json.tmp').write_text(json.dumps(summary,indent=2))
    (outdir/'summary.json.tmp').replace(outdir/'summary.json')
    print('MEASURE_PROGRESS',json.dumps({k:v for k,v in summary.items() if k!='records'}),flush=True)
print('MEASURE_COMPLETE',flush=True)
