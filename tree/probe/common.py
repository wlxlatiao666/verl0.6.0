"""Pure-Python data contracts for the offline branch-utility probe."""

import hashlib
import json
import math
import os
import random
import re
from pathlib import Path

SCHEMA = 1


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def seed_for(seed, *parts):
    return int(digest([seed, *parts])[:8], 16) % (2**31)


def question_id(messages):
    # Conservative exact deduplication after whitespace normalization; NOT semantic deduplication.
    normalized = [{"role": m["role"], "content": re.sub(r"\s+", " ", m["content"]).strip()} for m in messages]
    return digest(normalized)


def split_questions(ids, seed, val_fraction, test_fraction):
    if len(set(ids)) != len(ids):
        raise ValueError("Deduplicate questions before splitting")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than 1")
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    nv, nt = max(1, int(len(ids) * val_fraction)), max(1, int(len(ids) * test_fraction))
    if len(ids) - nv - nt < 1:
        raise ValueError("Need at least one question in each split")
    return {qid: "val" if i < nv else "test" if i < nv + nt else "train" for i, qid in enumerate(ids)}


def choose_positions(response_length, count, min_prefix, max_response, seed):
    """t is number of response tokens ALREADY present; action y_t is not in the prefix."""
    eligible = list(range(min_prefix, min(response_length, max_response - 1)))
    if len(eligible) <= count:
        return eligible
    # One uniform draw per length stratum. No entropy gate or future reward selection.
    rng = random.Random(seed)
    return [rng.choice(eligible[i * len(eligible) // count : (i + 1) * len(eligible) // count]) for i in range(count)]


def utility_label(outcomes):
    """Noise-corrected finite-candidate variance of Bernoulli success probabilities."""
    k = len(outcomes)
    if k < 2 or any(len(row) < 2 for row in outcomes):
        raise ValueError("Need >=2 candidates and >=2 independent outcomes per candidate")
    if any(x not in (0, 1) for row in outcomes for x in row):
        raise ValueError("Labels must be binary accuracy, not signed DAPO rewards")
    q = [sum(row) / len(row) for row in outcomes]
    mean = sum(q) / k
    naive = sum((x - mean) ** 2 for x in q) / k
    noise = (k - 1) / k**2 * sum(x * (1 - x) / (len(row) - 1) for x, row in zip(q, outcomes, strict=True))
    return {
        "q": q,
        "mean_success": mean,
        "utility_raw": naive - noise,
        "utility_clipped": max(0.0, naive - noise),
        "utility_naive": naive,
        "noise_correction": noise,
    }


def accuracy_from_reward(result):
    if not isinstance(result, dict) or "acc" not in result:
        raise ValueError("Expected DAPO verifier dict containing binary 'acc'")
    acc = float(result["acc"])
    if not math.isfinite(acc) or acc not in (0, 1):
        raise ValueError(f"Expected binary accuracy; got {acc}")
    return int(acc)


def ensure_manifest(path, manifest):
    path = Path(path)
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError(f"Configuration/input changed: {path}. Use a new output directory.")
    else:
        atomic_json(path, manifest)


def check_artifact(path, provenance):
    path = Path(path)
    if not path.exists():
        return False
    if read_json(path).get("provenance") != provenance:
        raise ValueError(f"Stale artifact: {path}. Use a new output directory.")
    return True
