"""Single-seed controlled A/B/C/D experiment; see ABCD.md for the protocol."""

import argparse
import gc
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import atomic_json, digest, ensure_manifest, file_digest, read_json, seed_for, utility_label
from learning import matched_selection
from run import load_verifier, model_identity, sample_branch_candidates, sampling, tokenizer_for
from screened_classifier import comparison, metrics, predict, transform, write_predictions


def category(record):
    q = record["q"]
    gap = max(q) - min(q)
    if record["utility_raw"] > 1e-12 and gap >= 0.5 and max(map(sum, record["outcomes"])) >= 2:
        return "positive"
    if record["utility_raw"] <= 1e-12 and gap <= 0.25:
        return "background"
    return "uncertain"


def initialize(args):
    """Reuse frozen prefix hidden states, but recompute the entire candidate distribution."""
    from safetensors import safe_open

    root, source = Path(args.work_dir), Path(args.source_dir)
    old = read_json(source / "config.json")
    if model_identity(old["model"]) != old["model_identity"]:
        raise ValueError("Source checkpoint fingerprint changed")
    cfg = dict(old, branch_sampling="sample")
    ensure_manifest(root / "config.json", cfg)
    questions = read_json(source / "questions.json")
    ensure_manifest(root / "questions.json", questions)
    protocol = dict(version=1, seed=42, source_dir=str(source.resolve()), sampling="sample",
                    sampling_temperature=cfg["branch_temperature"], k=cfg["branching_factor"],
                    initial_train_repeats=cfg["repeats"], eval_repeats=cfg["eval_repeats"],
                    independent_confirm_repeats=8, pca=128, l2=1.0, ranking_weight=1.0,
                    pair_min_utility_gap=0.01, max_pairs_per_question=32,
                    groups={"A": "initial screened balanced classification",
                            "B": "A with independently confirmed training labels",
                            "C": "B plus within-question ranking (prefer same trajectory)",
                            "D": "C plus candidate probability and remaining-budget features"})
    ensure_manifest(root / "protocol.json", protocol)
    tokenizer = tokenizer_for(cfg["model"])
    index = read_json(Path(cfg["model"]) / "model.safetensors.index.json")
    path = Path(cfg["model"]) / index["weight_map"]["lm_head.weight"]
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        lm_head = weights.get_tensor("lm_head.weight").to(device="cuda", dtype=torch.bfloat16)
    for q in questions:
        old_feature = source / "features" / f"{q['id']}.json"
        f = read_json(old_feature)
        if file_digest(old_feature.with_suffix(".pt")) != f["tensor_sha256"]:
            raise ValueError("Source hidden state checksum mismatch")
        output = root / "features" / f"{q['id']}.json"
        provenance = digest([protocol, cfg, q, file_digest(old_feature)])
        if output.exists():
            existing = read_json(output)
            if existing["provenance"] != provenance or file_digest(output.with_suffix(".pt")) != existing["tensor_sha256"]:
                raise ValueError("Existing resampled feature does not match this run")
            continue
        states = torch.load(old_feature.with_suffix(".pt"), map_location="cpu", weights_only=True)
        records = []
        if len(states):
            with torch.inference_mode():
                logits = nn.functional.linear(states.to("cuda", dtype=torch.bfloat16), lm_head).float()[:, :len(tokenizer)]
                logp = logits.log_softmax(-1)
                for i, original in enumerate(f["records"]):
                    generator = torch.Generator(device="cuda").manual_seed(seed_for(
                        cfg["seed"], q["id"], original["trajectory"], original["position"], "candidates"))
                    candidates = sample_branch_candidates(logp[i], cfg["branching_factor"], cfg["branch_temperature"], generator)
                    if len(candidates) < 2:
                        raise ValueError("Insufficient finite candidates")
                    entropy = float(-(logp[i].exp() * logp[i]).sum())
                    top = logp[i].topk(2).values
                    candidate_logp = logp[i, candidates].tolist()
                    auxiliary = [entropy, float(top[0]), float(top[0] - top[1]),
                                 *sorted(candidate_logp, reverse=True),
                                 float(logp[i, candidates].exp().sum()),
                                 original["position"] / cfg["max_response"],
                                 (cfg["max_response"] - original["position"] - 1) / cfg["max_response"]]
                    records.append(dict(original, entropy=entropy, candidate_ids=candidates.tolist(),
                                        candidate_logprobs=candidate_logp, auxiliary=auxiliary))
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(".pt.tmp")
        torch.save(states, tmp)
        os.replace(tmp, output.with_suffix(".pt"))
        atomic_json(output, dict(provenance=provenance, records=records, hidden_size=states.shape[1],
                                tensor_sha256=file_digest(output.with_suffix(".pt")),
                                feature=f["feature"], source_feature_sha256=file_digest(old_feature)))
        old_trajectory = read_json(source / "trajectories" / f"{q['id']}.json")
        atomic_json(root / "trajectories" / f"{q['id']}.json", old_trajectory)
        print(f"resampled {q['id'][:12]} positions={len(records)}", flush=True)
    del lm_head
    gc.collect()
    torch.cuda.empty_cache()


def annotate(args):
    """Batch independent continuations; atomically checkpoint each completed question."""
    from transformers import GenerationConfig
    from vllm import LLM

    root = Path(args.work_dir)
    cfg, protocol = read_json(root / "config.json"), read_json(root / "protocol.json")
    if model_identity(cfg["model"]) != cfg["model_identity"]:
        raise ValueError("Checkpoint changed")
    source = Path(protocol["source_dir"])
    questions = read_json(root / "questions.json")
    if args.phase == "confirm":
        questions = [q for q in questions if q["split"] == "train"]
    score, verifier_hash = load_verifier()
    replica = "main" if args.phase == "initial" else "confirm"
    todo = []
    for q in questions:
        path = root / "labels" / replica / f"{q['id']}.json"
        feature_path = root / "features" / f"{q['id']}.json"
        repeats = 8 if args.phase == "confirm" else cfg["repeats"] if q["split"] == "train" else cfg["eval_repeats"]
        provenance = digest([cfg, protocol, q, file_digest(feature_path), args.phase, repeats, verifier_hash])
        if path.exists():
            if read_json(path)["provenance"] != provenance:
                raise ValueError(f"Stale labels: {path}")
        else:
            todo.append((q, path, feature_path, repeats, provenance))
    if not todo:
        print(f"{args.phase}: already complete", flush=True)
        return
    tokenizer = tokenizer_for(cfg["model"])
    eos = GenerationConfig.from_pretrained(cfg["model"]).eos_token_id
    eos_ids = {tokenizer.eos_token_id, *(eos if isinstance(eos, list) else [eos])}
    os.environ["VLLM_USE_V1"] = "0"
    llm = LLM(model=cfg["model"], tensor_parallel_size=1, gpu_memory_utilization=0.7,
              enforce_eager=True, max_model_len=cfg["max_prompt"] + cfg["max_response"],
              dtype="bfloat16", enable_prefix_caching=True, seed=42, max_num_seqs=64)
    for qi, (q, path, feature_path, repeats, provenance) in enumerate(todo):
        fs = read_json(feature_path)["records"]
        trajectories = read_json(root / "trajectories" / f"{q['id']}.json")["records"]
        requests, params, slots, values = [], [], [], {}
        # Reuse only outcomes from exactly the same prefix and forced candidate;
        # old candidate selection never depended on rollout reward.
        cached = {}
        if args.phase == "initial":
            old_fs = read_json(source / "features" / f"{q['id']}.json")["records"]
            old_labels = read_json(source / "labels/main" / f"{q['id']}.json")["records"]
            for old_f, old_l in zip(old_fs, old_labels, strict=True):
                if old_f["id"] != old_l["id"]:
                    raise ValueError("Old label/feature mismatch")
                for ci, token in enumerate(old_f["candidate_ids"]):
                    cached[old_f["id"], token] = (old_l["outcomes"][ci], old_l["details"][ci])
        for fi, f in enumerate(fs):
            prefix = trajectories[f["trajectory"]]["tokens"][:f["position"]]
            for ci, token in enumerate(f["candidate_ids"]):
                response_prefix = prefix + [token]
                remaining = cfg["max_response"] - len(response_prefix)
                old = cached.get((f["id"], token))
                for ri in range(repeats):
                    slot = fi, ci, ri
                    if old is not None and ri < len(old[0]):
                        values[slot] = (old[0][ri], dict(old[1][ri], reused_from_topk=True))
                        continue
                    seed = seed_for(42, "abcd", args.phase, f["id"], token, ri)
                    if token in eos_ids or remaining <= 0:
                        answer = score(tokenizer.decode(response_prefix, skip_special_tokens=True), q["ground_truth"])
                        values[slot] = (int(answer["acc"]), dict(seed=seed, score=answer["score"], pred=answer["pred"],
                            finish_reason="stop" if token in eos_ids else "length", generated_tokens=0,
                            response_length=len(response_prefix), reused_from_topk=False))
                    else:
                        requests.append({"prompt_token_ids": q["prompt_ids"] + response_prefix})
                        params.append(sampling(cfg, seed, remaining))
                        slots.append((slot, response_prefix, seed))
        # Queue the question's continuations together. max_num_seqs=64 above
        # bounds active KV use while vLLM refills finished slots continuously.
        for start in range(0, len(requests), 512):
            outputs = llm.generate(requests[start:start+512], params[start:start+512], use_tqdm=False)
            for (slot, prefix, seed), output in zip(slots[start:start+512], outputs, strict=True):
                generated = output.outputs[0]
                answer = score(tokenizer.decode(prefix + list(generated.token_ids), skip_special_tokens=True), q["ground_truth"])
                values[slot] = (int(answer["acc"]), dict(seed=seed, score=answer["score"], pred=answer["pred"],
                    finish_reason=generated.finish_reason, generated_tokens=len(generated.token_ids),
                    response_length=len(prefix) + len(generated.token_ids), reused_from_topk=False))
        records = []
        for fi, f in enumerate(fs):
            outcomes = [[values[fi, ci, ri][0] for ri in range(repeats)] for ci in range(len(f["candidate_ids"]))]
            details = [[values[fi, ci, ri][1] for ri in range(repeats)] for ci in range(len(f["candidate_ids"]))]
            records.append(dict(id=f["id"], outcomes=outcomes, details=details, **utility_label(outcomes)))
        atomic_json(path, dict(provenance=provenance, feature_sha256=file_digest(feature_path),
                              verifier_sha256=verifier_hash, repeats=repeats, records=records))
        print(f"{args.phase} {qi+1}/{len(todo)} {q['split']} {q['id'][:12]} "
              f"new_requests={len(requests)} reused={sum(d.get('reused_from_topk',False) for _,d in values.values())}", flush=True)


def load_rows(root, splits):
    states, rows = [], []
    for q in read_json(root / "questions.json"):
        if q["split"] not in splits:
            continue
        path = root / "features" / f"{q['id']}.json"
        feature = read_json(path)
        initial = read_json(root / "labels/main" / f"{q['id']}.json")
        confirmed = read_json(root / "labels/confirm" / f"{q['id']}.json") if q["split"] == "train" else initial
        if any(labels["feature_sha256"] != file_digest(path) for labels in (initial, confirmed)):
            raise ValueError("Labels and features changed")
        if file_digest(path.with_suffix(".pt")) != feature["tensor_sha256"]:
            raise ValueError("Hidden checksum mismatch")
        h = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True).float()
        if len(h) != len(feature["records"]):
            raise ValueError("Hidden row mismatch")
        states.append(h)
        for f, a, b in zip(feature["records"], initial["records"], confirmed["records"], strict=True):
            if f["id"] != a["id"] or a["id"] != b["id"]:
                raise ValueError("Row ID mismatch")
            ca, cb = category(a), category(b)
            confirmed_category = ca if ca == cb else "uncertain"
            rows.append(dict(id=f["id"], question=q["id"], trajectory=f["trajectory"], split=q["split"],
                entropy=f["entropy"], target=a["utility_raw"], confirmed_target=b["utility_raw"],
                positive=int(ca == "positive"), category=ca, confirmed_category=confirmed_category,
                auxiliary=f["auxiliary"]))
    return torch.cat(states), rows


def fit_head(x, rows, group):
    key = "category" if group == "A" else "confirmed_category"
    indices = [i for i, r in enumerate(rows) if r[key] != "uncertain"]
    xx = x[indices].double()
    yy = torch.tensor([rows[i][key] == "positive" for i in indices], dtype=torch.float64)
    if len(yy.unique()) != 2:
        raise ValueError(f"{group}: insufficient confirmed positives/background")
    weights = torch.where(yy == 1, len(yy)/(2*yy.sum()), len(yy)/(2*(1-yy).sum()))
    head = nn.Linear(x.shape[1], 1, dtype=torch.float64)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    groups = defaultdict(list)
    for i in indices:
        groups[rows[i]["question"]].append(i)
    pairs, pair_weights = [], []
    if group in ("C", "D"):
        for qid, ids in groups.items():
            possible = [(i,j) for i in ids for j in ids
                        if rows[i][key] == "positive" and rows[j][key] == "background"
                        and rows[i]["confirmed_target"] - rows[j]["confirmed_target"] >= .01]
            rng = random.Random(seed_for(42, qid, "pairs"))
            rng.shuffle(possible)
            possible.sort(key=lambda pair: rows[pair[0]]["trajectory"] != rows[pair[1]]["trajectory"])
            possible = possible[:32]
            pairs.extend(possible)
            pair_weights.extend([1/len(possible)]*len(possible) if possible else [])
    dx = (x[[i for i,j in pairs]]-x[[j for i,j in pairs]]).double() if pairs else None
    pw = torch.tensor(pair_weights, dtype=torch.float64)
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=200, tolerance_grad=1e-8,
                                  tolerance_change=1e-10, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = (nn.functional.binary_cross_entropy_with_logits(head(xx).squeeze(1), yy, reduction="none")*weights).mean()
        loss = loss + .5 * head.weight.square().sum()
        if pairs:
            loss = loss + (nn.functional.softplus(-(dx @ head.weight[0]))*pw).sum()/pw.sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return dict(weight=head.weight[0].detach().float(), bias=head.bias[0].detach().float()), dict(
        training_positions=len(indices), positives=int(yy.sum()), backgrounds=int((1-yy).sum()),
        rank_pairs=len(pairs), rank_questions=len({rows[i]['question'] for i,j in pairs}))


def baseline(rows):
    entropy = np.array([r["entropy"] for r in rows])
    random_scores = np.random.default_rng(42).random(len(rows))
    mask, groups = matched_selection(entropy, rows, .2)
    expected = sum(math.ceil(.2*len(idx))*np.mean([rows[i]["target"] for i in idx]) for idx in groups.values())/mask.sum()
    return dict(entropy=metrics(rows, entropy), random_seed42=metrics(rows, random_scores),
                random_exact_expected_matched_utility=float(expected))


def train(args):
    root = Path(args.work_dir)
    output = root / "abcd"
    protocol = read_json(root / "protocol.json")
    ensure_manifest(output / "protocol.json", protocol)
    if (output / "validation.json").exists():
        print("ABCD already trained", flush=True)
        return
    h, rows = load_rows(root, ("train", "val"))
    ti = [i for i,r in enumerate(rows) if r["split"] == "train"]
    vi = [i for i,r in enumerate(rows) if r["split"] == "val"]
    tr, vr = [rows[i] for i in ti], [rows[i] for i in vi]
    torch.manual_seed(42)
    normalized = h/h.square().mean(1,keepdim=True).sqrt().clamp_min(1e-6)
    mean = normalized[ti].mean(0)
    _,_,components = torch.pca_lowrank(normalized[ti]-mean, q=min(128,len(ti)-1,h.shape[1]), center=False,niter=4)
    scale = ((normalized[ti]-mean)@components).std(0,unbiased=False).clamp_min(1e-4)
    state = dict(mean=mean,components=components,scale=scale)
    x = transform(h,state)
    auxiliary = torch.tensor([r["auxiliary"] for r in rows])
    amean,ascale = auxiliary[ti].mean(0),auxiliary[ti].std(0,unbiased=False).clamp_min(1e-4)
    augmented = torch.cat([x,(auxiliary-amean)/ascale],dim=1)
    results = dict(baselines=baseline(vr),groups={})
    for group in "ABCD":
        xx = augmented if group == "D" else x
        head, counts = fit_head(xx[ti],tr,group)
        scores = predict(xx[vi],head)
        checkpoint = dict(group=group,protocol=protocol,transform=state,head=head,
                          auxiliary_mean=amean,auxiliary_scale=ascale,threshold=float(np.quantile(scores,.8)))
        torch.save(checkpoint,output/f"{group}.pt")
        results["groups"][group] = dict(counts=counts,validation=metrics(vr,scores),
            vs_entropy=comparison(vr,scores,np.array([r['entropy'] for r in vr])))
        write_predictions(output/f"{group}_validation.csv",vr,scores)
        print(group,results["groups"][group],flush=True)
    ensure_manifest(output/"frozen_checkpoints.json",{g:file_digest(output/f"{g}.pt") for g in "ABCD"})
    atomic_json(output/"validation.json",results)


def evaluate(args):
    root = Path(args.work_dir)
    output = root/"abcd"
    if (output/"test.json").exists():
        print("ABCD test already complete",flush=True)
        return
    frozen = read_json(output/"frozen_checkpoints.json")
    h,rows = load_rows(root,("test",))
    result = dict(baselines=baseline(rows),groups={})
    for group in "ABCD":
        if file_digest(output/f"{group}.pt") != frozen[group]:
            raise ValueError("Checkpoint changed after freeze")
        checkpoint = torch.load(output/f"{group}.pt",map_location="cpu",weights_only=True)
        x = transform(h,checkpoint["transform"])
        if group == "D":
            auxiliary = torch.tensor([r['auxiliary'] for r in rows])
            x = torch.cat([x,(auxiliary-checkpoint['auxiliary_mean'])/checkpoint['auxiliary_scale']],dim=1)
        scores = predict(x,checkpoint['head'])
        result['groups'][group] = dict(metrics=metrics(rows,scores),
            vs_entropy=comparison(rows,scores,np.array([r['entropy'] for r in rows])),
            fixed_val_threshold_count=int((scores>=checkpoint['threshold']).sum()))
        write_predictions(output/f"{group}_test.csv",rows,scores)
        print(group,result['groups'][group],flush=True)
    atomic_json(output/'test.json',result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['initialize','annotate','train','evaluate'])
    parser.add_argument('--work-dir',required=True)
    parser.add_argument('--source-dir')
    parser.add_argument('--phase',choices=['initial','confirm'],default='initial')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.command=='initialize' and not args.source_dir:
        parser.error('--source-dir is required for initialize')
    globals()[args.command](args)
