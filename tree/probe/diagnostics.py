"""Read-only label diagnostics, including exact zero classification from binary outcomes."""

from fractions import Fraction
from pathlib import Path

from common import atomic_json, read_json

CATEGORIES = (
    "zero_all_wrong", "zero_all_correct", "zero_mixed_cancellation", "positive", "negative",
)


def label_category(record):
    outcomes = record["outcomes"]
    if len(outcomes) < 2 or any(len(row) < 2 for row in outcomes):
        raise ValueError(f"Invalid outcomes for {record['id']}")
    if any(value not in (0, 1) for row in outcomes for value in row):
        raise ValueError(f"Nonbinary outcomes for {record['id']}")
    # Rational arithmetic distinguishes mathematical cancellation from display rounding
    # and floating-point residue. This does not change the stored training target.
    k = len(outcomes)
    q = [Fraction(sum(row), len(row)) for row in outcomes]
    mean = sum(q) / k
    utility = sum((value - mean) ** 2 for value in q) / k - Fraction(k - 1, k * k) * sum(
        value * (1 - value) / (len(row) - 1) for value, row in zip(q, outcomes, strict=True)
    )
    if utility:
        return "positive" if utility > 0 else "negative"
    if all(value == 0 for value in q):
        return "zero_all_wrong"
    if all(value == 1 for value in q):
        return "zero_all_correct"
    return "zero_mixed_cancellation"


def summarize_records(records):
    counts = {category: dict(count=0, outcomes=0, truncated_outcomes=0,
                            positions_with_truncation=0) for category in CATEGORIES}
    display_zero = rounded_nonzero = float_residue = 0
    for record in records:
        category = label_category(record)
        stats = counts[category]
        details = record["details"]
        if len(details) != len(record["outcomes"]) or any(
            len(d) != len(o) for d, o in zip(details, record["outcomes"], strict=True)
        ):
            raise ValueError(f"Outcome/detail mismatch for {record['id']}")
        truncated = sum(d["finish_reason"] == "length" for group in details for d in group)
        stats["count"] += 1
        stats["outcomes"] += sum(len(row) for row in record["outcomes"])
        stats["truncated_outcomes"] += truncated
        stats["positions_with_truncation"] += int(truncated > 0)
        raw = record["utility_raw"]
        displayed_zero = float(f"{raw:.5f}") == 0.0
        display_zero += int(displayed_zero)
        rounded_nonzero += int(displayed_zero and category in ("positive", "negative"))
        float_residue += int(category.startswith("zero_") and raw != 0.0)
    total = len(records)
    zeros = sum(stats["count"] for category, stats in counts.items() if category.startswith("zero_"))
    for category, stats in counts.items():
        stats["fraction_all_positions"] = stats["count"] / total if total else None
        stats["fraction_zero_positions"] = (
            stats["count"] / zeros if zeros and category.startswith("zero_") else None
        )
        stats["truncated_outcome_fraction"] = (
            stats["truncated_outcomes"] / stats["outcomes"] if stats["outcomes"] else None
        )
    return dict(positions=total, exact_zero_positions=zeros,
                exact_zero_fraction=zeros / total if total else None,
                displayed_zero_positions=display_zero,
                nonzero_but_displayed_zero_positions=rounded_nonzero,
                exact_zero_with_float_residue_positions=float_residue, categories=counts)


def summarize(args):
    root = Path(args.work_dir)
    questions = read_json(root / "questions.json")
    selected = [q for i, q in enumerate(questions)
                if i % args.num_shards == args.shard_index and (args.split == "all" or q["split"] == args.split)]
    records, by_split, missing = [], {name: [] for name in ("train", "val", "test")}, []
    for q in selected:
        path = root / "labels" / args.replica / f"{q['id']}.json"
        if not path.exists():
            missing.append(q["id"])
            continue
        rows = read_json(path)["records"]
        records.extend(rows)
        by_split[q["split"]].extend(rows)
    summary = dict(replica=args.replica, split=args.split, num_shards=args.num_shards,
                   shard_index=args.shard_index, expected_questions=len(selected),
                   completed_questions=len(selected) - len(missing), missing_question_ids=missing,
                   complete=not missing,
                   zero_definition="exact rational corrected utility computed from binary outcomes",
                   overall=summarize_records(records),
                   by_split={name: summarize_records(rows) for name, rows in by_split.items()})
    suffix = f"_shard{args.shard_index}-of-{args.num_shards}" if args.num_shards > 1 else ""
    output = root / f"label_summary_{args.replica}_{args.split}{suffix}.json"
    atomic_json(output, summary)
    print(f"[label-summary] questions={summary['completed_questions']}/{len(selected)} complete={not missing}", flush=True)
    for name, stats in [("overall", summary["overall"]), *summary["by_split"].items()]:
        if not stats["positions"]:
            continue
        print(f"[label-summary:{name}] positions={stats['positions']} "
              f"exact_zero={stats['exact_zero_positions']} ({stats['exact_zero_fraction']:.2%}) "
              f"displayed_zero={stats['displayed_zero_positions']} "
              f"nonzero_but_displayed_zero={stats['nonzero_but_displayed_zero_positions']}", flush=True)
        for category, values in stats["categories"].items():
            within_zero = values["fraction_zero_positions"]
            zero_text = f"{within_zero:.2%}" if within_zero is not None else "N/A"
            truncated = values["truncated_outcome_fraction"]
            truncated_text = f"{truncated:.2%}" if truncated is not None else "N/A"
            print(f"  {category}: count={values['count']} all={values['fraction_all_positions']:.2%} "
                  f"within_zero={zero_text} truncated_outcomes={truncated_text}", flush=True)
    print(f"[label-summary] saved {output}", flush=True)
