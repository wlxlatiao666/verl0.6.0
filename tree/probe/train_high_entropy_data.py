"""Fit setting-A classification on the fresh, preselected high-entropy dataset.
Uses the frozen full-training-trajectory entropy threshold without refiltering P80.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from abcd import baseline, fit_head, category
from common import atomic_json, ensure_manifest, file_digest, read_json, digest
from screened_classifier import comparison, metrics, predict, transform, write_predictions


def load_rows(root, splits):
    states, rows = [], []
    threshold = read_json(root/'entropy_calibration.json')['threshold']
    for q in read_json(root/'questions.json'):
        if q['split'] not in splits:
            continue
        path = root/'features'/f"{q['id']}.json"
        feature = read_json(path)
        labels = read_json(root/'labels/main'/f"{q['id']}.json")
        if labels['feature_sha256'] != file_digest(path):
            raise ValueError('Feature/label fingerprint mismatch')
        if feature['tensor_sha256'] != file_digest(path.with_suffix('.pt')):
            raise ValueError('Hidden state fingerprint mismatch')
        h = torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=True).float()
        if len(h) != len(feature['records']):
            raise ValueError('Hidden row mismatch')
        states.append(h)
        for f,l in zip(feature['records'],labels['records'],strict=True):
            if f['id'] != l['id'] or f['entropy'] < threshold:
                raise ValueError('Row alignment or high-entropy condition failed')
            c = category(l)
            rows.append(dict(id=f['id'],question=q['id'],trajectory=f['trajectory'],
                split=q['split'],entropy=f['entropy'],target=l['utility_raw'],
                positive=int(c=='positive'),category=c))
    return torch.cat(states),rows


def subset(h, rows, threshold):
    indices = [i for i, r in enumerate(rows) if r['entropy'] >= threshold]
    return h[indices], [rows[i] for i in indices]


def report(rows, scores, cutoff):
    if not rows:
        return {'positions': 0, 'note': 'No eligible positions; metrics undefined'}
    mask = scores >= cutoff
    return dict(metrics=metrics(rows, scores), baselines=baseline(rows),
                vs_entropy=comparison(rows, scores, np.array([r['entropy'] for r in rows])),
                fixed_threshold=dict(count=int(mask.sum()), fraction=float(mask.mean()),
                    mean_utility_raw=float(np.mean([r['target'] for r, yes in zip(rows, mask) if yes]))
                    if mask.any() else None))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--probe-quantile', type=float, default=.8)
    p.add_argument('--pca-dim', type=int, default=32)
    args = p.parse_args()
    if not 0 < args.probe_quantile < 1:
        p.error('quantiles must lie strictly between 0 and 1')
    if args.pca_dim < 1:
        p.error('pca-dim must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    root, out = Path(args.work_dir), Path(args.output_dir)
    calibration = read_json(root/'entropy_calibration.json')
    label_hashes = {q['id']: file_digest(root/'labels/main'/f"{q['id']}.json") for q in read_json(root/'questions.json')}
    ensure_manifest(out/'protocol.json', dict(version=1, source=str(root.resolve()), seed=42,
        dataset_fingerprint=digest(label_hashes), calibration=calibration, probe_quantile=args.probe_quantile,
        pca_dim=args.pca_dim, candidate_sampling='sample_without_replacement',
        entropy_calibration='frozen full training trajectories', probe_calibration='high-entropy validation positions',
        labels='setting A screened balanced classification', l2=1.0))
    h, rows = load_rows(root, ('train', 'val'))
    threshold = calibration['threshold']
    h, selected = subset(h, rows, threshold)
    ti = [i for i, r in enumerate(selected) if r['split'] == 'train']
    vi = [i for i, r in enumerate(selected) if r['split'] == 'val']
    counts = {split: dict(total=sum(r['split'] == split for r in rows),
                         eligible=sum(r['split'] == split for r in selected),
                         categories=dict(Counter(r['category'] for r in selected if r['split'] == split)),
                         questions=len({r['question'] for r in selected if r['split'] == split}))
              for split in ('train', 'val')}
    atomic_json(out/'screening.json', dict(entropy_threshold=threshold, splits=counts))
    print('High-entropy screening:', counts, 'threshold=', threshold, flush=True)
    if len(ti) < 3 or len(vi) < 2:
        raise ValueError('Insufficient eligible training/validation data; collect additional high-entropy positions')
    checkpoint_path = out/'A.pt'
    frozen_path = out/'frozen.json'
    if checkpoint_path.exists() or frozen_path.exists():
        raise FileExistsError('Use a new output directory; do not overwrite a frozen experiment')
    normalized = h / h.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
    mean = normalized[ti].mean(0)
    _, _, components = torch.pca_lowrank(normalized[ti]-mean,
        q=min(args.pca_dim, len(ti)-1, h.shape[1]), center=False, niter=4)
    scale = ((normalized[ti]-mean)@components).std(0, unbiased=False).clamp_min(1e-4)
    state = dict(mean=mean, components=components, scale=scale)
    x = transform(h, state)
    head, fit_counts = fit_head(x[ti], [selected[i] for i in ti], 'A')
    scores = predict(x[vi], head)
    cutoff = float(np.quantile(scores, args.probe_quantile))
    checkpoint = dict(group='A', transform=state, head=head, threshold=cutoff,
                      entropy_threshold=threshold, conditional_high_entropy=True)
    torch.save(checkpoint, checkpoint_path)
    ensure_manifest(frozen_path, dict(checkpoint_sha256=file_digest(checkpoint_path),
                                    entropy_threshold=threshold, probe_threshold=cutoff))
    vr = [selected[i] for i in vi]
    atomic_json(out/'validation.json', dict(fit_counts=fit_counts, **report(vr, scores, cutoff)))
    write_predictions(out/'validation.csv', vr, scores)
    # No test labels or hidden states were used in fitting either threshold.
    th, test = load_rows(root, ('test',))
    total = len(test)
    th, test = subset(th, test, threshold)
    ts = predict(transform(th, state), head) if test else np.array([])
    atomic_json(out/'test.json', dict(total_positions=total, eligible_positions=len(test),
        checkpoint_sha256=file_digest(checkpoint_path), **report(test, ts, cutoff)))
    if test:
        write_predictions(out/'test.csv', test, ts)
    print('Complete:', out, flush=True)


if __name__ == '__main__':
    main()
