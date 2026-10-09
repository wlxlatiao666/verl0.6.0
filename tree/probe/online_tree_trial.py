"""Single-seed online trial of frozen A / entropy / random tree triggers.

Same leaf count (independent top-ups when needed), same response length cap.
Actual decoded-token work is reported separately: it is NOT equal FLOPs.
"""
import argparse
import os
import time
from pathlib import Path

os.environ['VLLM_USE_V1'] = '0'

import numpy as np
import torch

from common import atomic_json, ensure_manifest, file_digest, read_json, seed_for
from run import load_verifier, model_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--questions-file', help='Fresh held-out questions; calibration still uses work-dir')
    p.add_argument('--mode', choices=['probe', 'probe_entropy', 'entropy', 'random'], required=True)
    p.add_argument('--branch-sampling', choices=['sample', 'sample_with_replacement', 'topk'], default='sample')
    p.add_argument('--probe-checkpoint', help='Conditional high-entropy checkpoint or original A')
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--max-response', type=int, default=2048)
    p.add_argument('--leaves', type=int, default=16)
    args = p.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import TreeSearchParams
    from vllm.model_executor.layers.branch_probe import BranchProbe
    root, out = Path(args.work_dir), Path(args.output_dir)
    cfg = read_json(root / 'config.json')
    if model_identity(cfg['model']) != cfg['model_identity']:
        raise ValueError('Frozen model identity changed')
    ckpt_path = Path(args.probe_checkpoint) if args.probe_checkpoint else root / 'abcd/A.pt'
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    all_questions = read_json(root / 'questions.json')
    questions = [q for q in all_questions if q['split'] == args.split]
    if args.questions_file:
        questions = read_json(args.questions_file)
        old_ids = {q['id'] for q in all_questions}
        if any(q['id'] in old_ids for q in questions):
            raise ValueError('External evaluation questions overlap probe development data')
        if len({q['id'] for q in questions}) != len(questions):
            raise ValueError('Duplicate evaluation questions')
    if args.limit:
        questions = questions[:args.limit]
    val_entropy = [r['entropy'] for q in all_questions if q['split'] == 'val'
                   for r in read_json(root / 'features' / f"{q['id']}.json")['records']]
    threshold = float(checkpoint.get('entropy_threshold', np.quantile(val_entropy, .8)))
    if args.mode == 'probe_entropy' and not checkpoint.get('conditional_high_entropy'):
        raise ValueError('Combined trial requires a high-entropy-trained checkpoint')
    verifier, verifier_hash = load_verifier()
    manifest = dict(version=1, seed=42, mode=args.mode, split=args.split,
        questions=[q['id'] for q in questions], checkpoint_sha256=file_digest(ckpt_path),
        model_identity=cfg['model_identity'], verifier_sha256=verifier_hash,
        probe_threshold=checkpoint['threshold'], entropy_threshold=threshold,
        random_probability=.2, min_seg_length=10, depth=3, k=4,
        branch_sampling=args.branch_sampling, temperature=1., max_response=args.max_response,
        leaves=args.leaves, top_up='independent samples to fixed leaf count')
    if args.questions_file:
        manifest['questions_file_sha256'] = file_digest(args.questions_file)
        manifest['fresh_holdout'] = True
    ensure_manifest(out / args.mode / 'manifest.json', manifest)
    # Validate the folded online head against the original PCA pipeline on all
    # stored validation states before starting expensive generation.
    scorer = BranchProbe(str(ckpt_path), 'cpu')
    transform, head = checkpoint['transform'], checkpoint['head']
    max_error = 0.
    for q in all_questions:
        if q['split'] != 'val':
            continue
        h = torch.load(root / 'features' / f"{q['id']}.pt", weights_only=True).float()
        hn = h / h.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        ref = ((hn-transform['mean']) @ transform['components'] / transform['scale']) @ head['weight'] + head['bias']
        max_error = max(max_error, float((scorer(h)-ref).abs().max()))
        torch.testing.assert_close(scorer(h), ref, atol=2e-5, rtol=2e-5)
    print(f'Folded probe parity max_abs_error={max_error:.8g}', flush=True)
    llm = LLM(model=cfg['model'], tensor_parallel_size=1, dtype='bfloat16',
        enforce_eager=True, enable_prefix_caching=True, gpu_memory_utilization=.7,
        max_model_len=cfg['max_prompt']+cfg['max_response'], max_num_seqs=32,
        seed=42, disable_async_output_proc=True)
    tokenizer = llm.get_tokenizer()
    for qi, q in enumerate(questions):
        path = out / args.mode / f"{q['id']}.json"
        if path.exists():
            continue
        seed = seed_for(42, q['id'], 'online-tree')
        # Global branch Gumbel draws and ordinary sampler seed both reset per
        # question. Different tree shapes still consume different random draws.
        torch.manual_seed(seed)
        tree = TreeSearchParams(enable_tree_search=True, branch_trigger_mode=args.mode,
            branch_probe_path=str(ckpt_path) if args.mode in ('probe', 'probe_entropy') else None,
            branch_probe_threshold=checkpoint['threshold'] if args.mode in ('probe', 'probe_entropy') else None,
            entropy_threshold=threshold, random_branch_probability=.2,
            branching_factor=4, max_tree_depth=3, min_seg_length=10,
            max_num_leaves=args.leaves, branch_sampling=args.branch_sampling, branch_temperature=1.)
        params = SamplingParams(temperature=1., top_p=1., top_k=-1, seed=seed,
            max_tokens=args.max_response, tree_search_params=tree)
        started = time.monotonic()
        result = llm.generate([dict(prompt_token_ids=q['prompt_ids'])], params, use_tqdm=False)[0]
        nodes = {s.seq_id: s for s in result.outputs}
        samples, node_records = [], []
        for s in result.outputs:
            node_records.append(dict(seq_id=s.seq_id, parent_seq_id=s.parent_seq_id,
                depth=s.tree_depth, is_leaf=s.is_leaf, tree_ids=list(s.tree_ids),
                generated_tokens=len(s.token_ids), finish_reason=s.finish_reason))
            if not s.is_leaf:
                continue
            segments, current, visited = [], s, set()
            while current is not None:
                if current.seq_id in visited:
                    raise RuntimeError('Cycle in tree output')
                visited.add(current.seq_id)
                segments.append(list(current.tree_ids))
                if current.parent_seq_id is not None and current.parent_seq_id not in nodes:
                    raise RuntimeError('Missing parent in tree output')
                current = nodes.get(current.parent_seq_id)
            tokens = [t for segment in reversed(segments) for t in segment]
            if len(tokens) > args.max_response:
                raise RuntimeError('Tree response exceeded token cap')
            samples.append(dict(tokens=tokens, source='tree', finish_reason=s.finish_reason))
        if not samples:
            raise RuntimeError('Tree output has no leaves')
        tree_leaves = len(samples)
        if tree_leaves > args.leaves:
            raise RuntimeError('Leaf cap exceeded')
        top_up_tokens = 0
        if tree_leaves < args.leaves:
            deficits = args.leaves-tree_leaves
            prompts = [dict(prompt_token_ids=q['prompt_ids']) for _ in range(deficits)]
            params = [SamplingParams(temperature=1., top_p=1., top_k=-1,
                max_tokens=args.max_response, seed=seed_for(seed, 'topup', i)) for i in range(deficits)]
            for r in llm.generate(prompts, params, use_tqdm=False):
                s = r.outputs[0]
                top_up_tokens += len(s.token_ids)
                samples.append(dict(tokens=list(s.token_ids), source='topup', finish_reason=s.finish_reason))
        elapsed = time.monotonic()-started
        for sample in samples:
            sample['text'] = tokenizer.decode(sample['tokens'], skip_special_tokens=True)
            sample['reward'] = verifier(sample['text'], q['ground_truth'])
        correct = sum(bool(s['reward']['acc']) for s in samples)
        record = dict(question=q['id'], mode=args.mode, correct=correct, leaves=len(samples),
            tree_leaves=tree_leaves, topups=len(samples)-tree_leaves,
            accuracy=correct/len(samples), any_correct=bool(correct),
            mixed_rewards=0 < correct < len(samples), elapsed_seconds=elapsed,
            generated_tokens=sum(n['generated_tokens'] for n in node_records)+top_up_tokens,
            truncated_fraction=sum(s['finish_reason']=='length' for s in samples)/len(samples),
            nodes=node_records, samples=samples)
        atomic_json(path, record)
        print(f"{args.mode} {qi+1}/{len(questions)} correct={correct}/{len(samples)} "
              f"tree_leaves={tree_leaves} tokens={record['generated_tokens']} seconds={elapsed:.1f}", flush=True)
    records = [read_json(out / args.mode / f"{q['id']}.json") for q in questions]
    summary = dict(mode=args.mode, questions=len(records),
        leaf_accuracy=float(np.mean([r['accuracy'] for r in records])),
        any_correct_questions=sum(r['any_correct'] for r in records),
        mixed_reward_questions=sum(r['mixed_rewards'] for r in records),
        mean_tree_leaves=float(np.mean([r['tree_leaves'] for r in records])),
        topups=sum(r['topups'] for r in records),
        generated_tokens=sum(r['generated_tokens'] for r in records),
        elapsed_seconds=sum(r['elapsed_seconds'] for r in records),
        truncated_fraction=float(np.mean([r['truncated_fraction'] for r in records])),
        folded_probe_max_error=max_error)
    atomic_json(out / args.mode / 'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
