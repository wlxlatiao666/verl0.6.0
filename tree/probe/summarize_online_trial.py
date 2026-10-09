"""Paired question-level comparisons for the completed online tree trial."""
import argparse
from pathlib import Path

import numpy as np

from common import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--probe-mode', choices=['probe', 'probe_entropy'], default='probe')
    args = parser.parse_args()
    probe_mode = args.probe_mode
    root = Path(args.output_dir)
    summaries, rows, protocols = {}, {}, {}
    for mode in (probe_mode, 'entropy', 'random'):
        protocol = read_json(root / mode / 'manifest.json')
        protocols[mode] = {k: v for k, v in protocol.items() if k != 'mode'}
        summaries[mode] = read_json(root / mode / 'summary.json')
        rows[mode] = [read_json(root / mode / f'{qid}.json') for qid in protocol['questions']]
        branch_positions = []
        for question in rows[mode]:
            nodes = {n['seq_id']: n for n in question['nodes']}
            for node in nodes.values():
                if node['is_leaf']:
                    continue
                position, cur = 0, node
                while cur is not None:
                    position += len(cur['tree_ids'])
                    cur = nodes.get(cur['parent_seq_id'])
                branch_positions.append(position)
        summaries[mode]['branch_count'] = len(branch_positions)
        summaries[mode]['mean_branch_response_position'] = (
            float(np.mean(branch_positions)) if branch_positions else None)
    if not (protocols[probe_mode] == protocols['entropy'] == protocols['random']):
        raise ValueError('Trial protocols/questions differ')
    rng = np.random.default_rng(42)
    count = len(rows[probe_mode])
    indices = rng.integers(count, size=(10000, count))
    comparisons = {}
    for baseline in ('entropy', 'random'):
        comparisons[baseline] = {}
        for metric in ('accuracy', 'any_correct', 'mixed_rewards', 'generated_tokens'):
            delta = np.array([float(a[metric])-float(b[metric])
                              for a, b in zip(rows[probe_mode], rows[baseline])])
            comparisons[baseline][metric] = dict(
                probe_minus_baseline=float(delta.mean()),
                question_bootstrap_95_ci=np.quantile(delta[indices].mean(1), [.025, .975]).tolist())
    report = dict(summaries=summaries, paired_comparisons=comparisons,
                  note=f"{count}-question {'fresh' if protocols[probe_mode].get('fresh_holdout') else 'reused'} holdout. Bootstrap resamples questions, not leaves. Single training/evaluation seed.")
    atomic_json(root / 'comparison.json', report)
    print(report)


if __name__ == '__main__':
    main()
