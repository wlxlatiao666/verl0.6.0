"""CPU/GPU probe fitting and question-held-out evaluation; no base model needed."""

import csv
import math
import os
from pathlib import Path

import numpy as np
import torch
from common import atomic_json, digest, ensure_manifest, file_digest, read_json
from torch import nn


class BranchProbe(nn.Module):
    """Output estimates 4 * Var_i P(correct | prefix, action_i); larger means branch."""

    def __init__(self, dimension, architecture):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("scale", torch.ones(dimension))
        self.network = (
            nn.Linear(dimension, 1)
            if architecture == "linear"
            else nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Linear(128, 1))
        )

    def forward(self, features):
        return self.network((features.float() - self.mean) / self.scale).squeeze(-1)


def feature_contract(root):
    questions = read_json(root / "questions.json")
    return digest([questions, [file_digest(root / "features" / f"{q['id']}.json") for q in questions]])


def load_data(root, replica="main", splits=("train", "val", "test")):
    root = Path(root)
    cfg = read_json(root / "config.json")
    questions = read_json(root / "questions.json")
    all_rows, hidden, provenance = [], [], []
    for q in questions:
        if q["split"] not in splits:
            continue
        feature_path = root / "features" / f"{q['id']}.json"
        label_path = root / "labels" / replica / f"{q['id']}.json"
        if not feature_path.exists() or not label_path.exists():
            raise ValueError(f"Missing features/labels for {q['id']}; complete all shards first")
        feature = read_json(feature_path)
        labels = read_json(label_path)
        if labels["feature_sha256"] != file_digest(feature_path):
            raise ValueError(f"Labels do not match features: {label_path}")
        tensor_path = feature_path.with_suffix(".pt")
        if file_digest(tensor_path) != feature["tensor_sha256"]:
            raise ValueError(f"Corrupt tensor {tensor_path}")
        tensor = torch.load(tensor_path, map_location="cpu", weights_only=True).float()
        if len(tensor) != len(feature["records"]) or len(tensor) != len(labels["records"]):
            raise ValueError(f"Feature/label row mismatch for {q['id']}")
        for f, label in zip(feature["records"], labels["records"], strict=True):
            if f["id"] != label["id"]:
                raise ValueError("Feature/label ID mismatch")
            if not math.isfinite(f["entropy"]) or not math.isfinite(label["utility_raw"]):
                raise ValueError("Non-finite feature/label")
            all_rows.append(
                dict(
                    id=f["id"],
                    question=q["id"],
                    split=q["split"],
                    entropy=f["entropy"],
                    target=label["utility_raw"],
                    mean_success=label["mean_success"],
                    truncated=sum(d["finish_reason"] == "length" for ds in label["details"] for d in ds)
                    / sum(len(ds) for ds in label["details"]),
                )
            )
        hidden.append(tensor)
        provenance.append([q["id"], file_digest(feature_path), file_digest(label_path)])
    if not all_rows:
        raise ValueError("No eligible positions; inspect trajectory lengths/min-prefix")
    h = torch.cat(hidden)
    if not torch.isfinite(h).all():
        raise ValueError("Non-finite hidden states")
    for split in splits:
        if not any(r["split"] == split for r in all_rows):
            raise ValueError(f"No usable positions in {split}")
    return h, all_rows, digest([cfg, questions, provenance])


def input_features(hidden, rows, inputs):
    entropy = torch.tensor([r["entropy"] for r in rows], dtype=torch.float32).unsqueeze(1)
    if inputs == "entropy":
        return entropy
    return torch.cat([hidden, entropy], dim=1) if inputs == "hidden_entropy" else hidden


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[inverse]


def correlation(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def selected_stats(target, mask):
    return dict(
        count=int(mask.sum()),
        fraction=float(mask.mean()),
        mean_utility_raw=float(target[mask].mean()) if mask.any() else None,
    )


def matched_selection(scores, rows, fraction):
    """Retrospective per-question ranking diagnostic, NOT an online branching policy."""
    chosen = np.zeros(len(rows), dtype=bool)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["question"], []).append(i)
    for indices in groups.values():
        indices = np.array(indices)
        n = max(1, math.ceil(len(indices) * fraction))
        chosen[indices[np.argsort(-scores[indices], kind="stable")[:n]]] = True
    return chosen, groups


def report_metrics(rows, predictions, threshold, entropy_threshold, fraction, bootstrap=500):
    target = np.array([r["target"] for r in rows])
    entropy = np.array([r["entropy"] for r in rows])
    rng = np.random.default_rng(1729)
    scores = {"probe": predictions, "entropy": entropy, "random": rng.random(len(rows))}
    result = dict(
        positions=len(rows),
        questions=len({r["question"] for r in rows}),
        mse=float(np.mean((predictions - target) ** 2)),
        zero_predictor_mse=float(np.mean(target**2)),
        spearman=correlation(ranks(predictions), ranks(target)),
        mean_utility_raw=float(target.mean()),
        negative_label_fraction=float((target < 0).mean()),
        mean_truncated_fraction=float(np.mean([r["truncated"] for r in rows])),
        calibrated_threshold={
            "probe": selected_stats(target, predictions >= threshold),
            "entropy": selected_stats(target, entropy >= entropy_threshold),
        },
        retrospective_matched_budget={},
    )
    selections = {}
    for name, score in scores.items():
        mask, groups = matched_selection(score, rows, fraction)
        selections[name] = mask
        result["retrospective_matched_budget"][name] = selected_stats(target, mask)
    # Independent unit is a question, not a token. CI compares question-mean selected utility.
    differences = []
    for indices in groups.values():
        indices = np.array(indices)
        a = target[indices[selections["probe"][indices]]].mean()
        b = target[indices[selections["entropy"][indices]]].mean()
        differences.append(a - b)
    differences = np.array(differences)
    if bootstrap > 0 and len(differences) >= 2:
        means = [float(rng.choice(differences, len(differences), replace=True).mean()) for _ in range(bootstrap)]
        result["probe_minus_entropy_question_mean"] = dict(
            value=float(differences.mean()), bootstrap_95_ci=np.quantile(means, [0.025, 0.975]).tolist()
        )
    within = []
    for indices in groups.values():
        indices = np.array(indices)
        value = correlation(ranks(predictions[indices]), ranks(target[indices]))
        if value is not None:
            within.append(value)
    result["within_question_spearman"] = float(np.mean(within)) if within else None
    return result


def train(args):
    if not (0 < args.select_fraction < 1) or min(args.epochs, args.batch_size, args.patience) < 1:
        raise ValueError("Invalid training counts or select-fraction")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("lr must be positive and weight-decay nonnegative")
    root = Path(args.work_dir)
    output = root / "models" / args.name
    h, rows, fingerprint = load_data(root)
    config = dict(
        architecture=args.architecture,
        inputs=args.inputs,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        select_fraction=args.select_fraction,
        dataset_fingerprint=fingerprint,
    )
    ensure_manifest(output / "train_config.json", config)
    if (output / "validation.json").exists():
        if not (output / "probe.pt").exists():
            raise ValueError(f"Completed training is missing its checkpoint: {output}")
        print(f"Training already complete: {output}. Use --name for another experiment.")
        return
    torch.manual_seed(args.seed)
    x = input_features(h, rows, args.inputs)
    y = torch.tensor([r["target"] * 4 for r in rows], dtype=torch.float32)
    train_idx = torch.tensor([i for i, r in enumerate(rows) if r["split"] == "train"])
    val_idx = torch.tensor([i for i, r in enumerate(rows) if r["split"] == "val"])
    model = BranchProbe(x.shape[1], args.architecture)
    model.mean.copy_(x[train_idx].mean(0))
    model.scale.copy_(x[train_idx].std(0, unbiased=False).clamp_min(1e-4))
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best, stale, history = float("inf"), 0, []
    for epoch in range(args.epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx))]
        losses = []
        for indices in order.split(args.batch_size):
            predicted = model(x[indices].to(args.device))
            # Raw corrected labels can be negative. Keeping them avoids clipping-induced positive bias.
            loss = nn.functional.mse_loss(predicted, y[indices].to(args.device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            val_predictions = torch.cat([model(x[idx].to(args.device)).cpu() for idx in val_idx.split(1024)])
        val_mse = float(nn.functional.mse_loss(val_predictions, y[val_idx]))
        if not math.isfinite(val_mse):
            raise ValueError("Training produced non-finite validation loss")
        history.append(dict(epoch=epoch + 1, train_mse_scaled=float(np.mean(losses)), val_mse_scaled=val_mse))
        print(history[-1], flush=True)
        if val_mse < best:
            best, stale = val_mse, 0
            checkpoint = dict(
                state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                dimension=x.shape[1],
                hidden_size=h.shape[1],
                architecture=args.architecture,
                inputs=args.inputs,
                target_scale=4.0,
                epoch=epoch + 1,
                train_config=config,
                run_config=read_json(root / "config.json"),
                feature_contract=feature_contract(root),
            )
            tmp = output / "probe.pt.tmp"
            torch.save(checkpoint, tmp)
            os.replace(tmp, output / "probe.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    atomic_json(output / "history.json", history)
    checkpoint, model = load_probe(output / "probe.pt")
    with torch.inference_mode():
        predictions = model(x[val_idx]).numpy() / checkpoint["target_scale"]
    val_rows = [rows[i] for i in val_idx.tolist()]
    threshold = float(np.quantile(predictions, 1 - args.select_fraction))
    entropy_threshold = float(np.quantile([r["entropy"] for r in val_rows], 1 - args.select_fraction))
    checkpoint.update(
        threshold_raw_utility=threshold,
        threshold_model_output=threshold * 4,
        entropy_threshold=entropy_threshold,
        select_fraction=args.select_fraction,
    )
    tmp = output / "probe.pt.tmp"
    torch.save(checkpoint, tmp)
    os.replace(tmp, output / "probe.pt")
    atomic_json(
        output / "validation.json",
        report_metrics(val_rows, predictions, threshold, entropy_threshold, args.select_fraction),
    )
    print(f"Saved {output / 'probe.pt'}; test remains untouched until evaluate", flush=True)


def load_probe(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = BranchProbe(checkpoint["dimension"], checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint, model.eval()


def evaluate(args):
    root = Path(args.work_dir)
    output = root / "models" / args.name
    checkpoint, model = load_probe(output / "probe.pt")
    h, rows, fingerprint = load_data(
        root, args.replica, ("train", "val", "test") if args.replica == "main" else ("test",)
    )
    if checkpoint["run_config"] != read_json(root / "config.json"):
        raise ValueError("Checkpoint experiment configuration does not match this dataset")
    if checkpoint["feature_contract"] != feature_contract(root):
        raise ValueError("Question splits or feature tensors changed after training")
    if args.replica == "main" and fingerprint != checkpoint["train_config"]["dataset_fingerprint"]:
        raise ValueError("Main labels/features changed after training")
    if "threshold_raw_utility" not in checkpoint:
        raise ValueError("Training/calibration did not finish; rerun train")
    x = input_features(h, rows, checkpoint["inputs"])
    idx = [i for i, r in enumerate(rows) if r["split"] == "test"]
    test_rows = [rows[i] for i in idx]
    with torch.inference_mode():
        predictions = torch.cat([model(batch) for batch in x[idx].split(1024)]).numpy() / checkpoint["target_scale"]
    metrics = report_metrics(
        test_rows,
        predictions,
        checkpoint["threshold_raw_utility"],
        checkpoint["entropy_threshold"],
        checkpoint["select_fraction"],
        args.bootstrap,
    )
    metrics.update(
        replica=args.replica, dataset_fingerprint=fingerprint, checkpoint_sha256=file_digest(output / "probe.pt")
    )
    atomic_json(output / f"test_{args.replica}.json", metrics)
    with open(output / f"predictions_{args.replica}.csv", "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*test_rows[0], "prediction", "selected"])
        writer.writeheader()
        writer.writerows(
            dict(row, prediction=float(prediction), selected=bool(prediction >= checkpoint["threshold_raw_utility"]))
            for row, prediction in zip(test_rows, predictions, strict=True)
        )
    print(metrics, flush=True)
