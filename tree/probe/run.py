"""Offline pipeline. Run `python tree/probe/run.py --help` from the verl checkout."""

import argparse
import importlib.util
import os
import random
from collections import Counter
from pathlib import Path

from common import (
    SCHEMA,
    accuracy_from_reward,
    atomic_json,
    check_artifact,
    choose_positions,
    digest,
    ensure_manifest,
    file_digest,
    question_id,
    read_json,
    seed_for,
    split_questions,
    utility_label,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-Math-7B"
DEFAULT_DATA = "/inspire/hdd/global_user/weilongxuan-253108120168/verl0.6.0/data/dapo-math-17k.parquet"


def tokenizer_for(model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    if not tokenizer.chat_template:
        raise ValueError("Model tokenizer has no chat template; use the same tokenizer/template as verl")
    return tokenizer


def model_identity(model):
    path = Path(model)
    if not path.is_dir():
        raise ValueError("--model must be an existing local checkpoint directory on the server")
    # Hash small configuration/tokenizer files; fingerprint large weights by size and mtime.
    configs = {p.name: file_digest(p) for p in sorted(path.glob("*.json"))}
    weights = {
        p.name: [p.stat().st_size, p.stat().st_mtime_ns]
        for pattern in ("*.safetensors", "*.bin")
        for p in sorted(path.glob(pattern))
    }
    if not weights:
        raise ValueError(f"No safetensors/bin model weights in {path}")
    return {"path": str(path.resolve()), "configs": configs, "weights": weights}


def prepare(args):
    import pyarrow.parquet as pq

    root = Path(args.work_dir)
    root.mkdir(parents=True, exist_ok=True)
    config = {
        k: getattr(args, k)
        for k in (
            "model",
            "data",
            "prompt_key",
            "num_questions",
            "val_fraction",
            "test_fraction",
            "seed",
            "trajectories",
            "positions",
            "min_prefix",
            "max_prompt",
            "max_response",
            "branching_factor",
            "branch_sampling",
            "branch_temperature",
            "temperature",
            "top_p",
            "repeats",
            "eval_repeats",
        )
    }
    config.update(schema=SCHEMA, model_identity=model_identity(args.model), data_sha256=file_digest(args.data))
    ensure_manifest(root / "config.json", config)
    tokenizer = tokenizer_for(args.model)
    table = pq.read_table(args.data)
    rows = table.to_pylist()
    unique, row_ids, group_rows, conflicts = {}, [], {}, set()
    for index, row in enumerate(rows):
        messages = row[args.prompt_key]
        if (
            not isinstance(messages, list)
            or not messages
            or any(not isinstance(m.get("content"), str) for m in messages)
        ):
            raise ValueError(f"Row {index}: expected text-only chat messages in {args.prompt_key!r}")
        qid = question_id(messages)
        row_ids.append(qid)
        group_rows.setdefault(qid, []).append(index)
        gt = row["reward_model"]["ground_truth"]
        if not isinstance(gt, str):
            raise ValueError(f"Row {index}: math_dapo ground_truth must be a string")
        if qid in unique:
            if unique[qid]["ground_truth"] != gt:
                conflicts.add(qid)
            continue
        raw = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(raw, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > args.max_prompt:
            # Match data.truncation=error and filter_overlong_prompts=False.
            raise ValueError(
                f"Row {index}: prompt length {len(prompt_ids)} exceeds {args.max_prompt}; no silent truncation"
            )
        source = row["data_source"]
        if not (
            source in {"math_dapo", "math", "math_dapo_reasoning", "math500", "amc", "olympiad_bench"}
            or source.startswith(("aime", "math_dapo_"))
        ):
            raise ValueError(f"This pipeline uses the local math_dapo verifier; unsupported source {source!r}")
        unique[qid] = dict(id=qid, row_index=index, prompt_ids=prompt_ids, ground_truth=gt, data_source=source)
    # Quarantine the WHOLE group: choosing the first/majority answer would silently
    # change supervision. Keep original rows for inspection, even if answers might
    # turn out to be equivalent spellings. Do not modify the source parquet.
    conflict_groups = [
        dict(
            id=qid,
            rows=[
                dict(row_index=i, prompt=rows[i][args.prompt_key],
                     ground_truth=rows[i]["reward_model"]["ground_truth"],
                     data_source=rows[i]["data_source"])
                for i in group_rows[qid]
            ],
        )
        for qid in sorted(conflicts)
    ]
    conflict_row_count = sum(len(group_rows[qid]) for qid in conflicts)
    atomic_json(root / "conflicting_ground_truth.json", dict(
        policy="exclude_entire_question_group",
        group_count=len(conflicts), row_count=conflict_row_count, groups=conflict_groups,
    ))
    for qid in conflicts:
        del unique[qid]
    if conflicts:
        print(f"Excluded {len(conflicts)} conflicting question groups ({conflict_row_count} rows); "
              f"see {root / 'conflicting_ground_truth.json'}", flush=True)
    selected = sorted(unique)
    random.Random(args.seed).shuffle(selected)
    selected = selected[: args.num_questions]
    split = split_questions(selected, args.seed, args.val_fraction, args.test_fraction)
    questions = [dict(unique[qid], split=split[qid]) for qid in sorted(selected)]
    # A failed old prepare left only config.json and can resume. Never overwrite
    # a different completed split that may already have downstream artifacts.
    ensure_manifest(root / "questions.json", questions)
    held_out = {q["id"] for q in questions if q["split"] != "train"}
    for name in ("train", "val", "test"):
        ids = [q["row_index"] for q in questions if q["split"] == name]
        pq.write_table(table.take(ids), root / f"probe_{name}.parquet")
    # Exclude ALL held-out duplicates and conflicting groups from the GRPO export.
    pq.write_table(
        table.take([i for i, qid in enumerate(row_ids) if qid not in held_out and qid not in conflicts]),
        root / "grpo_train_without_probe_holdout.parquet",
    )
    summary = dict(
        source_rows=len(rows),
        unique_questions=len(unique),
        conflicting_question_groups=len(conflicts),
        conflicting_rows=conflict_row_count,
        selected=len(questions),
        splits=dict(Counter(q["split"] for q in questions)),
        max_annotation_rollouts=sum(
            args.trajectories
            * args.positions
            * args.branching_factor
            * (args.repeats if q["split"] == "train" else args.eval_repeats)
            for q in questions
        ),
    )
    atomic_json(root / "prepare_summary.json", summary)
    print(summary, flush=True)


def load_run(args):
    root = Path(args.work_dir)
    cfg = read_json(root / "config.json")
    if cfg["schema"] != SCHEMA or cfg["model_identity"] != model_identity(cfg["model"]):
        raise ValueError("Schema or checkpoint changed; create a new work directory")
    questions = read_json(root / "questions.json")
    return root, cfg, questions


def engine(cfg, args):
    os.environ.setdefault("VLLM_USE_V1", "0")
    from vllm import LLM

    return LLM(
        model=cfg["model"],
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=cfg["max_prompt"] + cfg["max_response"],
        seed=cfg["seed"],
        dtype="bfloat16",
        enable_prefix_caching=True,
    )


def sampling(cfg, seed, max_tokens, n=1):
    from vllm import SamplingParams

    return SamplingParams(
        n=n, temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=-1, max_tokens=max_tokens, seed=seed
    )


def selected_questions(questions, args):
    return [
        q
        for i, q in enumerate(questions)
        if i % args.num_shards == args.shard_index
        and (getattr(args, "split", "all") == "all" or q["split"] == args.split)
    ]


def rollout(args):
    root, cfg, questions = load_run(args)
    todo = []
    for q in selected_questions(questions, args):
        provenance = digest([cfg, q, "rollout"])
        path = root / "trajectories" / f"{q['id']}.json"
        if not check_artifact(path, provenance):
            todo.append((q, provenance, path))
    if not todo:
        print("All assigned trajectories already complete")
        return
    llm = engine(cfg, args)
    for q, provenance, path in todo:
        params = [
            sampling(cfg, seed_for(cfg["seed"], q["id"], j), cfg["max_response"]) for j in range(cfg["trajectories"])
        ]
        results = llm.generate([{"prompt_token_ids": q["prompt_ids"]} for _ in params], params, use_tqdm=False)
        records = [
            {"tokens": list(r.outputs[0].token_ids), "finish_reason": r.outputs[0].finish_reason} for r in results
        ]
        atomic_json(path, {"provenance": provenance, "records": records})
        print(f"rollout {q['id'][:12]} {q['split']}", flush=True)


def prefix_features(model, ids, prompt_length, positions, vocab_size):
    """The selected causal state predicts the first token NOT yet in the prefix."""
    states = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
    h = states[0, [prompt_length + t - 1 for t in positions]]
    logp = model.lm_head(h).float()[:, :vocab_size].log_softmax(-1)
    entropy = -(logp.exp() * logp).sum(-1)
    return h, logp, entropy


def features(args):
    import torch
    from transformers import AutoModelForCausalLM

    root, cfg, questions = load_run(args)
    todo = []
    for q in selected_questions(questions, args):
        source = root / "trajectories" / f"{q['id']}.json"
        if not source.exists():
            raise ValueError(f"Missing trajectories: {source}. Finish rollout first.")
        provenance = digest([cfg, q, file_digest(source), "features"])
        path = root / "features" / f"{q['id']}.json"
        tensor_path = path.with_suffix(".pt")
        if check_artifact(path, provenance):
            if not tensor_path.exists() or file_digest(tensor_path) != read_json(path)["tensor_sha256"]:
                raise ValueError(f"Missing/corrupt tensor: {tensor_path}")
        else:
            todo.append((q, source, provenance, path))
    if not todo:
        print("All assigned features already complete")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Feature extraction requires a CUDA GPU with room for the 7B BF16 model")
    tokenizer = tokenizer_for(cfg["model"])
    model = (
        AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        .to("cuda")
        .eval()
    )
    if model.config.model_type != "qwen2":
        raise ValueError("Initial implementation targets Qwen2/Qwen2.5 text models only")
    model.requires_grad_(False)
    for q, source, provenance, path in todo:
        records, hidden = [], []
        for j, trajectory in enumerate(read_json(source)["records"]):
            response = trajectory["tokens"]
            positions = choose_positions(
                len(response),
                cfg["positions"],
                cfg["min_prefix"],
                cfg["max_response"],
                seed_for(cfg["seed"], q["id"], j, "positions"),
            )
            if not positions:
                continue
            # Causal teacher forcing: row P+t-1 predicts response[t], without seeing response[t].
            ids = torch.tensor([q["prompt_ids"] + response], device="cuda")
            with torch.inference_mode():
                h, logp, entropy = prefix_features(model, ids, len(q["prompt_ids"]), positions, len(tokenizer))
                for row, t in enumerate(positions):
                    if cfg["branch_sampling"] == "topk":
                        candidates = logp[row].topk(cfg["branching_factor"]).indices
                    else:
                        generator = torch.Generator(device="cuda").manual_seed(
                            seed_for(cfg["seed"], q["id"], j, t, "candidates")
                        )
                        u = torch.rand(logp.shape[-1], generator=generator, device="cuda").clamp_(1e-7, 1 - 1e-7)
                        candidates = (
                            (logp[row] / cfg["branch_temperature"] - (-u.log()).log())
                            .topk(cfg["branching_factor"])
                            .indices
                        )
                    records.append(
                        dict(
                            id=f"{q['id']}:{j}:{t}",
                            trajectory=j,
                            position=t,
                            entropy=entropy[row].item(),
                            candidate_ids=candidates.tolist(),
                            candidate_logprobs=logp[row, candidates].tolist(),
                        )
                    )
                hidden.append(h.cpu().to(torch.float16))
            del h, logp, ids
        path.parent.mkdir(parents=True, exist_ok=True)
        tensor_path = path.with_suffix(".pt")
        temp = tensor_path.with_suffix(".pt.tmp")
        torch.save(torch.cat(hidden) if hidden else torch.empty((0, model.config.hidden_size)), temp)
        os.replace(temp, tensor_path)
        atomic_json(
            path,
            dict(
                provenance=provenance,
                records=records,
                tensor_sha256=file_digest(tensor_path),
                feature="qwen2.model.last_hidden_state_after_final_norm",
                hidden_size=model.config.hidden_size,
            ),
        )
        print(f"features {q['id'][:12]}: {len(records)} positions", flush=True)


def load_verifier():
    # Load the very same local verifier without importing verl's Ray/FSDP initialization.
    path = REPO / "verl/utils/reward_score/math_dapo.py"
    spec = importlib.util.spec_from_file_location("probe_math_dapo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score, file_digest(path)


def label(args):
    root, cfg, questions = load_run(args)
    score, verifier_hash = load_verifier()
    todo = []
    for q in selected_questions(questions, args):
        source = root / "features" / f"{q['id']}.json"
        if not source.exists():
            raise ValueError(f"Missing features: {source}")
        repeats = args.repeats_override or (cfg["repeats"] if q["split"] == "train" else cfg["eval_repeats"])
        provenance = digest([cfg, q, file_digest(source), "labels", args.replica, repeats, verifier_hash])
        path = root / "labels" / args.replica / f"{q['id']}.json"
        if not check_artifact(path, provenance):
            todo.append((q, source, repeats, provenance, path))
    if not todo:
        print("All assigned labels already complete")
        return
    llm = engine(cfg, args)
    tokenizer = tokenizer_for(cfg["model"])
    from transformers import GenerationConfig

    eos_ids = {tokenizer.eos_token_id}
    generation_file = Path(cfg["model"]) / "generation_config.json"
    if generation_file.exists():
        eos = GenerationConfig.from_pretrained(cfg["model"]).eos_token_id
        eos_ids.update(eos if isinstance(eos, list) else [eos])
    for q, source, repeats, provenance, path in todo:
        trajectories = read_json(root / "trajectories" / f"{q['id']}.json")["records"]
        labeled = []
        for record in read_json(source)["records"]:
            prefix = trajectories[record["trajectory"]]["tokens"][: record["position"]]
            outcomes, details = [], []
            requests, params, slots, completed = [], [], [], {}
            for ci, candidate in enumerate(record["candidate_ids"]):
                response_prefix = prefix + [candidate]
                remaining = cfg["max_response"] - len(response_prefix)
                seeds = [seed_for(cfg["seed"], record["id"], ci, r, args.replica) for r in range(repeats)]
                for ri, seed in enumerate(seeds):
                    if candidate in eos_ids or remaining == 0:
                        completed[ci, ri] = ([], "stop" if candidate in eos_ids else "length")
                    else:
                        requests.append({"prompt_token_ids": q["prompt_ids"] + response_prefix})
                        params.append(sampling(cfg, seed, remaining))
                        slots.append((ci, ri))
            if requests:
                outputs = llm.generate(requests, params, use_tqdm=False)
                for slot, output in zip(slots, outputs, strict=True):
                    completed[slot] = (list(output.outputs[0].token_ids), output.outputs[0].finish_reason)
            for ci, candidate in enumerate(record["candidate_ids"]):
                response_prefix = prefix + [candidate]
                seeds = [seed_for(cfg["seed"], record["id"], ci, r, args.replica) for r in range(repeats)]
                continuations = [completed[ci, ri] for ri in range(repeats)]
                candidate_outcomes, candidate_details = [], []
                for seed, (tokens, finish) in zip(seeds, continuations, strict=True):
                    text = tokenizer.decode(response_prefix + tokens, skip_special_tokens=True)
                    result = score(text, q["ground_truth"])
                    candidate_outcomes.append(accuracy_from_reward(result))
                    detail = dict(
                        seed=seed,
                        score=result["score"],
                        pred=result["pred"],
                        finish_reason=finish,
                        response_length=len(response_prefix) + len(tokens),
                        generated_tokens=len(tokens),
                    )
                    if args.save_text:
                        detail["text"] = text
                    candidate_details.append(detail)
                outcomes.append(candidate_outcomes)
                details.append(candidate_details)
            labeled.append(dict(id=record["id"], outcomes=outcomes, details=details, **utility_label(outcomes)))
            print(f"label {record['id'][-18:]} U={labeled[-1]['utility_raw']:.5f}", flush=True)
        atomic_json(
            path,
            dict(
                provenance=provenance,
                feature_sha256=file_digest(source),
                verifier_sha256=verifier_hash,
                repeats=repeats,
                records=labeled,
            ),
        )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Deduplicate/split parquet and freeze the experiment configuration")
    prep.add_argument("--model", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    prep.add_argument("--data", default=os.environ.get("TRAIN_FILE", DEFAULT_DATA))
    prep.add_argument("--prompt-key", default="prompt")
    prep.add_argument("--num-questions", type=int, default=1000)
    prep.add_argument("--val-fraction", type=float, default=0.1)
    prep.add_argument("--test-fraction", type=float, default=0.1)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--trajectories", type=int, default=2)
    prep.add_argument("--positions", type=int, default=6)
    prep.add_argument("--min-prefix", type=int, default=10)
    prep.add_argument("--max-prompt", type=int, default=2048)
    prep.add_argument("--max-response", type=int, default=2048)
    prep.add_argument("--branching-factor", type=int, default=4)
    prep.add_argument("--branch-sampling", choices=["topk", "sample"], default="topk")
    prep.add_argument("--branch-temperature", type=float, default=1.0)
    prep.add_argument("--temperature", type=float, default=1.0)
    prep.add_argument("--top-p", type=float, default=1.0)
    prep.add_argument("--repeats", type=int, default=4)
    prep.add_argument("--eval-repeats", type=int, default=8)
    parsers = [prep]
    for command in ("rollout", "features", "label"):
        s = sub.add_parser(command)
        s.add_argument("--num-shards", type=int, default=1)
        s.add_argument("--shard-index", type=int, default=0)
        if command != "features":
            s.add_argument("--tensor-parallel-size", type=int, default=1)
            s.add_argument("--gpu-memory-utilization", type=float, default=0.7)
        if command == "label":
            s.add_argument("--replica", default="main")
            s.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
            s.add_argument("--repeats-override", type=int)
            s.add_argument("--save-text", action="store_true")
        parsers.append(s)
    train = sub.add_parser("train")
    train.add_argument("--name", default="mlp_hidden")
    train.add_argument("--architecture", choices=["linear", "mlp"], default="mlp")
    train.add_argument("--inputs", choices=["hidden", "entropy", "hidden_entropy"], default="hidden")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cpu")
    train.add_argument("--select-fraction", type=float, default=0.2)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--name", default="mlp_hidden")
    evaluation.add_argument("--replica", default="main")
    evaluation.add_argument("--bootstrap", type=int, default=500)
    parsers.extend([train, evaluation])
    for s in parsers:
        s.add_argument("--work-dir", required=True)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if hasattr(args, "num_shards") and not (args.num_shards > 0 and 0 <= args.shard_index < args.num_shards):
        p.error("Require num-shards > 0 and 0 <= shard-index < num-shards")
    if args.command == "prepare":
        if min(args.num_questions, args.trajectories, args.positions, args.max_prompt, args.min_prefix) < 1:
            p.error("Counts/lengths must be positive")
        if min(args.repeats, args.eval_repeats, args.branching_factor) < 2:
            p.error("Need >=2 repeats and >=2 candidate branches")
        if args.max_response <= args.min_prefix + 1 or args.temperature <= 0 or args.branch_temperature <= 0:
            p.error("Invalid length/temperature")
        if not 0 < args.top_p <= 1:
            p.error("top-p must be in (0,1]")
    if getattr(args, "repeats_override", None) is not None and args.repeats_override < 2:
        p.error("repeats-override must be >=2")
    if args.command in {"train", "evaluate"}:
        from learning import evaluate, train

        (train if args.command == "train" else evaluate)(args)
    else:
        globals()[args.command](args)


if __name__ == "__main__":
    main()
