"""Fresh question-disjoint data, full-trajectory entropy scan, conditional sampling."""
import argparse
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

from common import (atomic_json, check_artifact, digest, ensure_manifest,
                    file_digest, question_id, read_json, seed_for, utility_label)
from run import (load_run, load_verifier, model_identity, sample_branch_candidates,
                 sampling, tokenizer_for)


def select_positions(entropies, minimum, threshold, count, gap, seed):
    """Temporal strata with randomized order; never pad with low-entropy points."""
    eligible = [t for t in range(minimum, len(entropies))
                if math.isfinite(entropies[t]) and entropies[t] >= threshold]
    rng = random.Random(seed)
    bins = [[] for _ in range(count)]
    for t in eligible:
        bins[min(count-1, (t-minimum)*count//max(1, len(entropies)-minimum))].append(t)
    for bucket in bins:
        rng.shuffle(bucket)
    order = list(range(count))
    rng.shuffle(order)
    selected = []
    while len(selected) < count:
        added = False
        for index in order:
            bucket = bins[index]
            while bucket:
                t = bucket.pop()
                if all(abs(t-s) >= gap for s in selected):
                    selected.append(t)
                    added = True
                    break
            if len(selected) == count:
                break
        if not added:
            break
    return sorted(selected)


def prepare(args):
    import pyarrow.parquet as pq
    root, source = Path(args.work_dir), Path(args.source_dir)
    old = read_json(source/'config.json')
    if file_digest(old['data']) != old['data_sha256']:
        raise ValueError('Source dataset changed')
    excluded = set()
    exclusion_files = [source/'questions.json', *map(Path, args.exclude_questions)]
    for path in exclusion_files:
        excluded.update(q['id'] for q in read_json(path))
    cfg = dict(old, num_questions=700, val_fraction=1/7, test_fraction=1/7,
               trajectories=2, positions=8, repeats=4, eval_repeats=8,
               branch_sampling='sample', entropy_quantile=.8, position_gap=16,
               position_policy='full_trajectory_high_entropy_temporal_strata_v1',
               excluded_question_ids=sorted(excluded))
    if model_identity(cfg['model']) != cfg['model_identity']:
        raise ValueError('Model changed')
    unique, conflicts, offset = {}, set(), 0
    for batch in pq.ParquetFile(cfg['data']).iter_batches(batch_size=8192,
            columns=[cfg['prompt_key'], 'reward_model', 'data_source']):
        for i, row in enumerate(batch.to_pylist()):
            messages = row[cfg['prompt_key']]
            qid = question_id(messages)
            gt = row['reward_model']['ground_truth']
            if not isinstance(gt, str):
                raise ValueError('Expected string ground truth')
            if qid in unique:
                if gt != unique[qid]['ground_truth']:
                    conflicts.add(qid)
            else:
                unique[qid] = dict(id=qid, messages=messages, ground_truth=gt,
                                  data_source=row['data_source'], row_index=offset+i)
        offset += batch.num_rows
    pool = sorted(set(unique)-conflicts-excluded)
    random.Random(cfg['seed']).shuffle(pool)
    tokenizer = tokenizer_for(cfg['model'])
    questions, overlong = [], 0
    for qid in pool:
        q = dict(unique[qid])
        raw = tokenizer.apply_chat_template(q.pop('messages'), tokenize=False, add_generation_prompt=True)
        ids = tokenizer(raw, add_special_tokens=False)['input_ids']
        if len(ids) > cfg['max_prompt']:
            overlong += 1
            continue
        s = q['data_source']
        if not (s in {'math_dapo','math','math_dapo_reasoning','math500','amc','olympiad_bench'} or s.startswith(('aime','math_dapo_'))):
            raise ValueError(f'Unsupported verifier source: {s}')
        split = 'train' if len(questions) < 500 else 'val' if len(questions) < 600 else 'test'
        questions.append(dict(q, prompt_ids=ids, split=split))
        if len(questions) == 700:
            break
    if len(questions) != 700:
        raise ValueError('Insufficient unique questions')
    ensure_manifest(root/'config.json', cfg)
    ensure_manifest(root/'questions.json', questions)
    ensure_manifest(root/'prepare_summary.json', dict(splits=dict(Counter(q['split'] for q in questions)),
        source_rows=offset, conflict_groups=len(conflicts), excluded_prior_questions=len(excluded),
        overlong_skipped=overlong, questions_sha256=file_digest(root/'questions.json'),
        max_positions=11200, max_continuations=230400))
    print('prepared', root, flush=True)


def make_engine(cfg):
    from vllm import LLM
    os.environ['VLLM_USE_V1'] = '0'
    return LLM(model=cfg['model'], tensor_parallel_size=1, gpu_memory_utilization=.7,
               enforce_eager=True, max_model_len=cfg['max_prompt']+cfg['max_response'],
               dtype='bfloat16', enable_prefix_caching=True, seed=cfg['seed'],
               max_num_seqs=64, disable_async_output_proc=True)


def rollout(args):
    root, cfg, questions = load_run(args)
    todo = []
    for q in questions:
        provenance = digest([cfg,q,'conditional_rollout_v1'])
        path = root/'trajectories'/f"{q['id']}.json"
        if not check_artifact(path, provenance):
            todo.append((q,path,provenance))
    if not todo:
        return
    llm = make_engine(cfg)
    for start in range(0,len(todo),16):
        batch = todo[start:start+16]
        requests, params = [], []
        for q,_,_ in batch:
            for j in range(cfg['trajectories']):
                requests.append({'prompt_token_ids':q['prompt_ids']})
                params.append(sampling(cfg,seed_for(cfg['seed'],q['id'],j),cfg['max_response']))
        outputs = llm.generate(requests, params, use_tqdm=False)
        cursor = 0
        for q,path,provenance in batch:
            records = []
            for _ in range(cfg['trajectories']):
                s = outputs[cursor].outputs[0]
                records.append(dict(tokens=list(s.token_ids),finish_reason=s.finish_reason))
                cursor += 1
            atomic_json(path,dict(provenance=provenance,records=records))
        print('rollout',min(start+16,len(todo)),'/',len(todo),flush=True)


def hf_model(cfg):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(cfg['model'],torch_dtype=torch.bfloat16,
                                               attn_implementation='sdpa').to('cuda').eval()
    if model.config.model_type != 'qwen2':
        raise ValueError('Expected Qwen2 architecture')
    model.requires_grad_(False)
    return model


def scan(args):
    """Scan all eligible next-token states, chunking the vocabulary projection."""
    import torch
    root,cfg,questions = load_run(args)
    todo = []
    for q in questions:
        source=root/'trajectories'/f"{q['id']}.json"
        provenance=digest([cfg,q,file_digest(source),'full_entropy_v1'])
        path=root/'entropy'/f"{q['id']}.json"
        if not check_artifact(path,provenance):todo.append((q,source,path,provenance))
    if not todo:return
    model=hf_model(cfg)
    vocab=len(tokenizer_for(cfg['model']))
    with torch.inference_mode():
        for qi,(q,source,path,provenance) in enumerate(todo):
            records=[]
            for tr in read_json(source)['records']:
                response=tr['tokens']
                ids=torch.tensor([q['prompt_ids']+response],device='cuda')
                h=model.model(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state[0]
                base=len(q['prompt_ids'])-1
                values=[]
                for start in range(0,len(response),32):
                    lp=model.lm_head(h[base+start:base+min(start+32,len(response))]).float()[:,:vocab].log_softmax(-1)
                    values.extend((-(lp.exp()*lp).sum(-1)).tolist())
                if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite entropy')
                records.append(dict(entropy=values,finish_reason=tr['finish_reason']))
                del h,ids
            atomic_json(path,dict(provenance=provenance,records=records))
            print('scan',qi+1,'/',len(todo),q['split'],flush=True)


def calibrate(args):
    import numpy as np
    root,cfg,questions=load_run(args)
    values,sources=[],{}
    for q in questions:
        if q['split']!='train':continue
        path=root/'entropy'/f"{q['id']}.json"
        sources[q['id']]=file_digest(path)
        for r in read_json(path)['records']:
            values.extend(r['entropy'][cfg['min_prefix']:])
    if not values:raise ValueError('No eligible training entropy')
    cutoff=float(np.quantile(values,cfg['entropy_quantile']))
    ensure_manifest(root/'entropy_calibration.json',dict(threshold=cutoff,quantile=cfg['entropy_quantile'],
        eligible_training_tokens=len(values),training_sources_sha256=digest(sources),
        policy='pooled_training_tokens_only',gate='entropy >= threshold'))
    print('entropy threshold',cutoff,'training tokens',len(values),flush=True)


def features(args):
    import torch
    root,cfg,questions=load_run(args)
    calibration=read_json(root/'entropy_calibration.json')
    cutoff=calibration['threshold']
    todo=[]
    for q in questions:
        tr=root/'trajectories'/f"{q['id']}.json"
        en=root/'entropy'/f"{q['id']}.json"
        path=root/'features'/f"{q['id']}.json"
        provenance=digest([cfg,q,file_digest(tr),file_digest(en),calibration,'conditional_features_v1'])
        if check_artifact(path,provenance):
            if file_digest(path.with_suffix('.pt'))!=read_json(path)['tensor_sha256']:raise ValueError('Corrupt hidden')
        else:todo.append((q,tr,en,path,provenance))
    if not todo:return
    model=hf_model(cfg)
    vocab=len(tokenizer_for(cfg['model']))
    with torch.inference_mode():
        for qi,(q,tr,en,path,provenance) in enumerate(todo):
            records,hidden=[],[]
            trajectories=read_json(tr)['records']
            entropies=read_json(en)['records']
            for j,(trajectory,ent) in enumerate(zip(trajectories,entropies,strict=True)):
                response=trajectory['tokens']
                positions=select_positions(ent['entropy'],cfg['min_prefix'],cutoff,cfg['positions'],
                    cfg['position_gap'],seed_for(cfg['seed'],q['id'],j,'conditional_positions'))
                if not positions:continue
                ids=torch.tensor([q['prompt_ids']+response],device='cuda')
                states=model.model(input_ids=ids,use_cache=False,return_dict=True).last_hidden_state[0]
                h=states[[len(q['prompt_ids'])+t-1 for t in positions]]
                lp=model.lm_head(h).float()[:,:vocab].log_softmax(-1)
                for i,t in enumerate(positions):
                    g=torch.Generator(device='cuda').manual_seed(seed_for(cfg['seed'],q['id'],j,t,'candidates'))
                    candidates=sample_branch_candidates(lp[i],cfg['branching_factor'],cfg['branch_temperature'],g)
                    top=lp[i].topk(2).values
                    cp=lp[i,candidates].tolist()
                    records.append(dict(id=f"{q['id']}:{j}:{t}",trajectory=j,position=t,
                        entropy=ent['entropy'][t],candidate_ids=candidates.tolist(),candidate_logprobs=cp,
                        auxiliary=[ent['entropy'][t],float(top[0]),float(top[0]-top[1]),
                            *sorted(cp,reverse=True),float(lp[i,candidates].exp().sum()),
                            t/cfg['max_response'],(cfg['max_response']-t-1)/cfg['max_response']]))
                hidden.append(h.cpu().to(torch.float16))
                del ids,states,h,lp
            path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix('.pt.tmp')
            torch.save(torch.cat(hidden) if hidden else torch.empty((0,model.config.hidden_size)),tmp)
            os.replace(tmp,path.with_suffix('.pt'))
            atomic_json(path,dict(provenance=provenance,records=records,hidden_size=model.config.hidden_size,
                tensor_sha256=file_digest(path.with_suffix('.pt')),feature='qwen2.model.last_hidden_state_after_final_norm'))
            print('features',qi+1,'/',len(todo),'positions',len(records),flush=True)


def label(args):
    from transformers import GenerationConfig
    root,cfg,questions=load_run(args)
    # Fixed pilot chosen by question order before labels; no success-based selection.
    pilot=[]
    for split,n in [('train',20),('val',10),('test',10)]:
        pilot.extend([q['id'] for q in questions if q['split']==split][:n])
    ensure_manifest(root/'pilot_questions.json',pilot)
    questions=sorted(questions,key=lambda q:q['id'] not in pilot)
    if args.pilot_only:questions=[q for q in questions if q['id'] in pilot]
    verifier,verifier_hash=load_verifier()
    todo=[]
    for q in questions:
        source=root/'features'/f"{q['id']}.json"
        repeats=cfg['repeats'] if q['split']=='train' else cfg['eval_repeats']
        provenance=digest([cfg,q,file_digest(source),'conditional_labels_v1',repeats,verifier_hash])
        path=root/'labels/main'/f"{q['id']}.json"
        if not check_artifact(path,provenance):todo.append((q,source,repeats,path,provenance))
    if not todo:return
    llm=make_engine(cfg)
    tokenizer=tokenizer_for(cfg['model'])
    eos=GenerationConfig.from_pretrained(cfg['model']).eos_token_id
    eos_ids={tokenizer.eos_token_id,*(eos if isinstance(eos,list) else [eos])}
    for qi,(q,source,repeats,path,provenance) in enumerate(todo):
        started=time.monotonic()
        fs=read_json(source)['records']
        trajectories=read_json(root/'trajectories'/f"{q['id']}.json")['records']
        requests,params,slots,values=[],[],[],{}
        for fi,f in enumerate(fs):
            prefix=trajectories[f['trajectory']]['tokens'][:f['position']]
            for ci,token in enumerate(f['candidate_ids']):
                response_prefix=prefix+[token]
                remaining=cfg['max_response']-len(response_prefix)
                for ri in range(repeats):
                    seed=seed_for(cfg['seed'],f['id'],token,ri,'conditional_main')
                    slot=fi,ci,ri
                    if token in eos_ids or remaining<=0:
                        result=verifier(tokenizer.decode(response_prefix,skip_special_tokens=True),q['ground_truth'])
                        values[slot]=(int(result['acc']),dict(seed=seed,score=result['score'],pred=result['pred'],
                            finish_reason='stop' if token in eos_ids else 'length',generated_tokens=0,response_length=len(response_prefix)))
                    else:
                        requests.append({'prompt_token_ids':q['prompt_ids']+response_prefix})
                        params.append(sampling(cfg,seed,remaining))
                        slots.append((slot,response_prefix,seed))
        for start in range(0,len(requests),512):
            outputs=llm.generate(requests[start:start+512],params[start:start+512],use_tqdm=False)
            for (slot,prefix,seed),output in zip(slots[start:start+512],outputs,strict=True):
                s=output.outputs[0]
                result=verifier(tokenizer.decode(prefix+list(s.token_ids),skip_special_tokens=True),q['ground_truth'])
                values[slot]=(int(result['acc']),dict(seed=seed,score=result['score'],pred=result['pred'],
                    finish_reason=s.finish_reason,generated_tokens=len(s.token_ids),response_length=len(prefix)+len(s.token_ids)))
        records=[]
        for fi,f in enumerate(fs):
            outcomes=[[values[fi,ci,ri][0] for ri in range(repeats)] for ci in range(len(f['candidate_ids']))]
            details=[[values[fi,ci,ri][1] for ri in range(repeats)] for ci in range(len(f['candidate_ids']))]
            records.append(dict(id=f['id'],outcomes=outcomes,details=details,**utility_label(outcomes)))
        atomic_json(path,dict(provenance=provenance,feature_sha256=file_digest(source),verifier_sha256=verifier_hash,
            repeats=repeats,records=records,elapsed_seconds=time.monotonic()-started))
        print('label',qi+1,'/',len(todo),q['split'],'positions',len(fs),'seconds',round(time.monotonic()-started,1),flush=True)


def summarize(args):
    from abcd import category
    root,cfg,questions=load_run(args)
    report={}
    for split in ('train','val','test'):
        qs=[q for q in questions if q['split']==split]
        counts=Counter();total=0;done=0;truncated=0;continuations=0;seconds=0.;tokens=0
        for q in qs:
            p=root/'labels/main'/f"{q['id']}.json"
            if not p.exists():continue
            obj=read_json(p);done+=1;seconds+=obj['elapsed_seconds']
            for r in obj['records']:
                total+=1;counts[category(r)]+=1
                flat=[v for a in r['outcomes'] for v in a]
                counts['all_failed']+=int(not any(flat))
                for ds in r['details']:
                    for d in ds:
                        continuations+=1;truncated+=d['finish_reason']=='length';tokens+=d['generated_tokens']
        report[split]=dict(questions=len(qs),labeled_questions=done,positions=total,counts=dict(counts),
            fractions={k:v/total for k,v in counts.items()} if total else {},continuations=continuations,
            truncated_fraction=truncated/continuations if continuations else None,
            elapsed_seconds=seconds,generated_tokens=tokens)
    atomic_json(root/args.summary_name,report)
    print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','rollout','scan','calibrate','features','label','summarize'])
    p.add_argument('--work-dir',required=True)
    p.add_argument('--source-dir')
    p.add_argument('--exclude-questions',action='append',default=[])
    p.add_argument('--pilot-only',action='store_true')
    p.add_argument('--summary-name',default='label_summary.json')
    args=p.parse_args()
    if args.command=='prepare' and not args.source_dir:p.error('prepare requires source-dir')
    globals()[args.command](args)
