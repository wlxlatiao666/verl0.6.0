"""Screened/balanced branch classifier experiments using existing offline labels.

Fit reads train/val only. Test evaluation requires a separate, explicit command.
Run with --help; all computations run on CPU and no new rollouts are generated.
"""

import argparse
import csv
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import atomic_json, ensure_manifest, file_digest, read_json
from learning import correlation, load_data, matched_selection, ranks


def dataset(root, splits):
    h, rows, fingerprint = load_data(root, splits=splits)
    labels = {}
    for q in read_json(root / "questions.json"):
        if q["split"] in splits:
            for record in read_json(root / "labels/main" / f"{q['id']}.json")["records"]:
                labels[record["id"]] = record
    for row in rows:
        record = labels[row["id"]]
        q = [sum(o) / len(o) for o in record["outcomes"]]
        gap = max(q) - min(q)
        # Proxy labels, NOT a declaration that a small-sample all-wrong prefix
        # has genuinely zero utility. Freeze this rule before looking at val/test.
        positive = record["utility_raw"] > 1e-12 and gap >= 0.5 and max(map(sum, record["outcomes"])) >= 2
        background = record["utility_raw"] <= 1e-12 and gap <= 0.25
        row.update(positive=int(positive), gap=gap,
                   category="positive" if positive else "background" if background else "uncertain")
    return h, rows, fingerprint


def average_precision(y, scores):
    if not y.sum():
        return None
    order = np.argsort(-scores, kind="stable")
    yy, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
    tp = np.cumsum(yy)[ends]
    return float(np.sum(np.diff(np.r_[0, tp]) / y.sum() * tp / (ends + 1)))


def metrics(rows, scores, fraction=0.2):
    y = np.array([r["positive"] for r in rows])
    utility = np.array([r["target"] for r in rows])
    n = max(1, math.ceil(len(rows) * fraction))
    top = np.zeros(len(rows), dtype=bool)
    top[np.argsort(-scores, kind="stable")[:n]] = True
    matched, groups = matched_selection(scores, rows, fraction)

    def selection(mask):
        return dict(count=int(mask.sum()), fraction=float(mask.mean()),
                    proxy_precision=float(y[mask].mean()),
                    proxy_recall=float(y[mask].sum() / y.sum()) if y.sum() else None,
                    mean_utility_raw=float(utility[mask].mean()))

    within = [correlation(ranks(scores[idx]), ranks(utility[idx])) for idx in groups.values()]
    within = [v for v in within if v is not None]
    return dict(positions=len(rows), questions=len(groups), proxy_positive_count=int(y.sum()),
                proxy_prevalence=float(y.mean()), average_precision=average_precision(y, scores),
                global_top_fraction=selection(top), retrospective_matched_budget=selection(matched),
                within_question_utility_spearman=float(np.mean(within)) if within else None)


def comparison(rows, scores, baseline, fraction=0.2):
    target = np.array([r["target"] for r in rows])
    a, groups = matched_selection(scores, rows, fraction)
    b, _ = matched_selection(baseline, rows, fraction)
    differences = [target[np.array(idx)[a[idx]]].mean() - target[np.array(idx)[b[idx]]].mean()
                   for idx in groups.values()]
    rng = np.random.default_rng(42)
    draws = [np.mean(rng.choice(differences, len(differences), replace=True)) for _ in range(1000)]
    return dict(mean=float(np.mean(differences)), question_bootstrap_95_ci=np.quantile(draws, [.025, .975]).tolist())


def transform(h, state):
    # Per-token RMS uses no future information; PCA and scaling fitted only on train.
    h = h / h.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
    return ((h - state["mean"]) @ state["components"]) / state["scale"]


def logistic(x, y, selected, balanced, strength):
    xx, yy = x[selected].double(), y[selected].double()
    if len(torch.unique(yy)) < 2:
        raise ValueError("Screening left fewer than two training classes")
    head = nn.Linear(x.shape[1], 1, dtype=torch.float64)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    weights = torch.ones_like(yy)
    if balanced:
        for value in (0, 1):
            mask = yy == value
            weights[mask] = len(yy) / (2 * int(mask.sum()))
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=200, tolerance_grad=1e-8,
                                  tolerance_change=1e-10, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = (nn.functional.binary_cross_entropy_with_logits(head(xx).squeeze(1), yy, reduction="none") * weights).mean()
        loss = loss + strength / 2 * head.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        result = dict(weight=head.weight[0].float().clone(), bias=head.bias[0].float().clone())
    return result


def predict(x, state):
    return (x @ state["weight"] + state["bias"]).numpy()


def write_predictions(path, rows, scores):
    with open(path, "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*rows[0], "score"])
        writer.writeheader()
        writer.writerows(dict(row, score=float(score)) for row, score in zip(rows, scores, strict=True))


def fit(args):
    root, output = Path(args.work_dir), Path(args.work_dir) / "models" / args.name
    h, rows, fingerprint = dataset(root, ("train", "val"))
    config = dict(version=1, fingerprint=fingerprint, seed=42,
                  positive_rule="U_raw>1e-12 AND max(q)-min(q)>=0.5 AND max(success_count)>=2",
                  background_rule="U_raw<=1e-12 AND max(q)-min(q)<=0.25",
                  dimensions=[32, 128], l2=[0.1, 1.0],
                  selection="validation proxy average precision; ties: smaller dimension, stronger L2",
                  warning="balanced classifier output is a ranking logit, not calibrated branch probability")
    ensure_manifest(output / "experiment.json", config)
    if (output / "results.json").exists():
        print(f"Already complete: {output}")
        return
    train = torch.tensor([i for i, r in enumerate(rows) if r["split"] == "train"])
    val = torch.tensor([i for i, r in enumerate(rows) if r["split"] == "val"])
    train_rows, val_rows = [rows[i] for i in train.tolist()], [rows[i] for i in val.tolist()]
    counts = {split: dict(Counter(r["category"] for r in rows if r["split"] == split)) for split in ("train", "val")}
    atomic_json(output / "screening.json", counts)
    print("Screening:", counts, flush=True)
    if not any(r["positive"] for r in val_rows):
        raise ValueError("No validation proxy positives; cannot select by AP")
    torch.manual_seed(42)
    normalized = h / h.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6)
    mean = normalized[train].mean(0)
    rank = min(128, len(train) - 1, h.shape[1])
    _, _, components = torch.pca_lowrank(normalized[train] - mean, q=rank, center=False, niter=4)
    y = torch.tensor([r["positive"] for r in train_rows], dtype=torch.float32)
    entropy = np.array([r["entropy"] for r in val_rows])
    baselines = {"entropy": metrics(val_rows, entropy)}
    # Multiple random rankings expose single-seed baseline variability.
    random_metrics = [metrics(val_rows, np.random.default_rng(seed).random(len(val_rows))) for seed in range(200)]
    baselines["random_200_seeds"] = dict(
        mean_average_precision=float(np.mean([m["average_precision"] for m in random_metrics])),
        mean_matched_utility=float(np.mean([m["retrospective_matched_budget"]["mean_utility_raw"] for m in random_metrics])))
    experiments, best = [], None
    for dim in sorted({min(32, rank), rank}):
        state = dict(mean=mean, components=components[:, :dim])
        projection = (normalized[train] - mean) @ state["components"]
        state["scale"] = projection.std(0, unbiased=False).clamp_min(1e-4)
        x = transform(h, state)
        for mode in ("screened_balanced", "all_balanced", "all_natural"):
            selected = torch.tensor([r["category"] != "uncertain" or mode != "screened_balanced" for r in train_rows])
            for strength in (0.1, 1.0):
                candidate = f"{mode}_pca{dim}_l2{strength}"
                head = logistic(x[train], y, selected, mode != "all_natural", strength)
                scores = predict(x[val], head)
                result = dict(name=candidate, mode=mode, dimension=dim, l2=strength,
                              selected_training_positions=int(selected.sum()),
                              train=metrics(train_rows, predict(x[train], head)),
                              validation=metrics(val_rows, scores))
                experiments.append(result)
                atomic_json(output / f"{candidate}.json", result)
                print(candidate, result["validation"], flush=True)
                # Only screened experiments compete for the primary checkpoint.
                if mode == "screened_balanced":
                    key = (result["validation"]["average_precision"], -dim, strength)
                    if best is None or key > best[0]:
                        best = (key, dict(name=candidate, transform=state, head=head, fingerprint=fingerprint,
                                          experiment=config, threshold=float(np.quantile(scores, 0.8))), scores, result)
    _, checkpoint, scores, winner = best
    tmp = output / "probe.pt.tmp"
    torch.save(checkpoint, tmp)
    os.replace(tmp, output / "probe.pt")
    write_predictions(output / "validation_predictions.csv", val_rows, scores)
    atomic_json(output / "results.json", dict(screening=counts, baselines=baselines, experiments=experiments,
                selected=winner, selected_vs_entropy=comparison(val_rows, scores, entropy),
                note="Validation used for model selection; no test labels read. Positive labels remain noisy proxies."))
    print(f"Selected {winner['name']}; saved {output}", flush=True)


def evaluate(args):
    root, output = Path(args.work_dir), Path(args.work_dir) / "models" / args.name
    checkpoint = torch.load(output / "probe.pt", map_location="cpu", weights_only=True)
    _, _, fingerprint = dataset(root, ("train", "val"))
    if fingerprint != checkpoint["fingerprint"]:
        raise ValueError("Train/validation artifacts changed after fitting")
    h, rows, fingerprint = dataset(root, ("test",))
    scores = predict(transform(h, checkpoint["transform"]), checkpoint["head"])
    entropy = np.array([r["entropy"] for r in rows])
    selected = scores >= checkpoint["threshold"]
    target = np.array([r["target"] for r in rows])
    report = dict(checkpoint_sha256=file_digest(output / "probe.pt"), test_fingerprint=fingerprint,
                  probe=metrics(rows, scores), entropy=metrics(rows, entropy),
                  probe_vs_entropy=comparison(rows, scores, entropy),
                  fixed_val_threshold=dict(count=int(selected.sum()), fraction=float(selected.mean()),
                      mean_utility_raw=float(target[selected].mean()) if selected.any() else None))
    atomic_json(output / "test.json", report)
    write_predictions(output / "test_predictions.csv", rows, scores)
    print(report, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["fit", "evaluate"])
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--name", default="screened_classifier_v1")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    (fit if args.command == "fit" else evaluate)(args)
