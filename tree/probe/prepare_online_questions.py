"""Draw fresh unique questions, excluding all existing probe development IDs."""
import argparse
import random
from pathlib import Path

from common import ensure_manifest, file_digest, question_id, read_json
from run import tokenizer_for


def main():
    import pyarrow.parquet as pq

    p = argparse.ArgumentParser()
    p.add_argument('--work-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--num-questions', type=int, default=100)
    args = p.parse_args()
    root, out = Path(args.work_dir), Path(args.output_dir)
    cfg = read_json(root / 'config.json')
    excluded = {q['id'] for q in read_json(root / 'questions.json')}
    if file_digest(cfg['data']) != cfg['data_sha256']:
        raise ValueError('Source dataset changed')
    unique, conflicts, offset = {}, set(), 0
    for batch in pq.ParquetFile(cfg['data']).iter_batches(
            batch_size=8192, columns=[cfg['prompt_key'], 'reward_model', 'data_source']):
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
    random.Random(42).shuffle(pool)
    tokenizer = tokenizer_for(cfg['model'])
    selected, overlong = [], 0
    for qid in pool:
        q = dict(unique[qid])
        raw = tokenizer.apply_chat_template(q.pop('messages'), tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(raw, add_special_tokens=False)['input_ids']
        if len(tokens) > cfg['max_prompt']:
            overlong += 1
            continue
        source = q['data_source']
        if not (source in {'math_dapo', 'math', 'math_dapo_reasoning', 'math500', 'amc', 'olympiad_bench'}
                or source.startswith(('aime', 'math_dapo_'))):
            raise ValueError(f'Unsupported verifier source: {source}')
        selected.append(dict(q, prompt_ids=tokens, split='test'))
        if len(selected) == args.num_questions:
            break
    if len(selected) != args.num_questions:
        raise ValueError('Not enough eligible fresh questions')
    ensure_manifest(out / 'questions.json', selected)
    manifest = dict(seed=42, count=len(selected), source_sha256=cfg['data_sha256'],
        excluded_development_count=len(excluded), excluded_ids=sorted(excluded),
        conflict_groups=len(conflicts), eligible_unique_questions=len(pool),
        overlong_skipped=overlong, questions_sha256=file_digest(out / 'questions.json'))
    ensure_manifest(out / 'selection.json', manifest)
    print({k:v for k,v in manifest.items() if k != 'excluded_ids'}, flush=True)


if __name__ == '__main__':
    main()
