import itertools
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common
import learning
import run


def test_noise_correction_is_unbiased():
    # Enumerate all outcomes instead of a flaky Monte Carlo assertion.
    expectation = 0.0
    for bits in itertools.product((0, 1), repeat=4):
        probability = 1.0
        for bit, p in zip(bits, (0.2, 0.2, 0.8, 0.8), strict=True):
            probability *= p if bit else 1 - p
        expectation += probability * common.utility_label([bits[:2], bits[2:]])["utility_raw"]
    assert expectation == pytest.approx(0.09)
    assert common.utility_label([[0, 0], [1, 1]])["utility_raw"] == 0.25
    assert common.utility_label([[0, 1], [0, 1]])["utility_raw"] < 0
    with pytest.raises(ValueError):
        common.utility_label([[-1, 1], [1, 1]])


def test_positions_and_question_split():
    ids = [str(i) for i in range(100)]
    split = common.split_questions(ids, 42, 0.1, 0.1)
    assert split == common.split_questions(list(reversed(ids)), 42, 0.1, 0.1)
    assert list(split.values()).count("train") == 80
    positions = common.choose_positions(2048, 6, 10, 2048, 42)
    assert len(positions) == len(set(positions)) == 6
    assert 10 <= min(positions) <= max(positions) < 2047
    assert common.choose_positions(5, 6, 10, 2048, 42) == []
    assert common.question_id([dict(role="user", content="a  b")]) == common.question_id(
        [dict(role="user", content="a b")]
    )


def test_signed_reward_and_provenance(tmp_path):
    assert common.accuracy_from_reward({"score": -1, "acc": False}) == 0
    assert common.accuracy_from_reward({"score": 1, "acc": True}) == 1
    with pytest.raises(ValueError):
        common.accuracy_from_reward({"score": -1})
    path = tmp_path / "manifest.json"
    common.ensure_manifest(path, {"model": "a"})
    with pytest.raises(ValueError):
        common.ensure_manifest(path, {"model": "b"})


def test_label_prefix_budget_eos_and_resume(tmp_path, monkeypatch):
    cfg = dict(model=str(tmp_path), seed=1, max_response=8, repeats=2, eval_repeats=4, temperature=1.0, top_p=1.0)
    question = dict(id="q", split="train", prompt_ids=[90, 91], ground_truth="ok")
    common.atomic_json(tmp_path / "trajectories/q.json", {"records": [{"tokens": [10, 11, 12, 13]}]})
    common.atomic_json(
        tmp_path / "features/q.json", {"records": [dict(id="q:0:2", trajectory=0, position=2, candidate_ids=[20, 99])]}
    )
    monkeypatch.setattr(run, "load_run", lambda args: (tmp_path, cfg, [question]))
    monkeypatch.setattr(
        run, "load_verifier", lambda: (lambda text, gt: {"acc": "20" in text, "score": 1, "pred": text}, "hash")
    )
    monkeypatch.setattr(run, "sampling", lambda c, seed, max_tokens: dict(seed=seed, max_tokens=max_tokens))
    monkeypatch.setattr(
        run, "tokenizer_for", lambda path: SimpleNamespace(eos_token_id=99, decode=lambda ids, **kw: str(ids))
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(GenerationConfig=object))
    calls = []

    class Engine:
        def generate(self, prompts, params, **kwargs):
            calls.append((prompts, params))
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[30], finish_reason="stop")]) for _ in prompts]

    monkeypatch.setattr(run, "engine", lambda *args: Engine())
    args = Namespace(
        work_dir=str(tmp_path),
        num_shards=1,
        shard_index=0,
        split="all",
        replica="main",
        repeats_override=None,
        save_text=True,
    )
    run.label(args)
    assert len(calls) == 1  # EOS candidate does not generate a suffix.
    assert calls[0][0][0]["prompt_token_ids"] == [90, 91, 10, 11, 20]
    assert calls[0][1][0]["max_tokens"] == 5
    assert calls[0][1][0]["seed"] != calls[0][1][1]["seed"]
    record = common.read_json(tmp_path / "labels/main/q.json")["records"][0]
    assert record["outcomes"] == [[1, 1], [0, 0]]
    assert record["details"][0][0]["response_length"] == 4
    run.label(args)
    assert len(calls) == 1
    args.repeats_override = 4
    with pytest.raises(ValueError, match="Stale"):
        run.label(args)


def make_dataset(root):
    cfg = {"schema": 1, "model": "synthetic-only"}
    common.atomic_json(root / "config.json", cfg)
    questions = []
    for qi in range(9):
        split = "train" if qi < 5 else "val" if qi < 7 else "test"
        qid = str(qi)
        questions.append(dict(id=qid, split=split))
        records, labels, h = [], [], []
        for i in range(6):
            outcomes = [[0, 0, 0, 0], [1, 1, 1, 1]] if i >= 3 else [[0, 1, 0, 1], [0, 1, 0, 1]]
            records.append(dict(id=f"{qid}:0:{i}", entropy=float(6 - i)))
            labels.append(
                dict(
                    id=f"{qid}:0:{i}",
                    **common.utility_label(outcomes),
                    details=[[dict(finish_reason="stop") for _ in range(4)] for _ in range(2)],
                )
            )
            h.append([float(i >= 3), i / 6, qi / 10, 1.0])
        path = root / "features" / f"{qid}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.tensor(h), path.with_suffix(".pt"))
        common.atomic_json(path, dict(records=records, tensor_sha256=common.file_digest(path.with_suffix(".pt"))))
        common.atomic_json(
            root / "labels/main" / f"{qid}.json", dict(records=labels, feature_sha256=common.file_digest(path))
        )
    common.atomic_json(root / "questions.json", questions)


def test_train_evaluate_roundtrip_and_leakage_guard(tmp_path):
    torch.set_num_threads(1)
    make_dataset(tmp_path)
    args = Namespace(
        work_dir=str(tmp_path),
        name="test",
        architecture="linear",
        inputs="hidden",
        epochs=40,
        patience=8,
        batch_size=32,
        lr=0.03,
        weight_decay=0.001,
        seed=42,
        device="cpu",
        select_fraction=0.5,
    )
    learning.train(args)
    checkpoint, model = learning.load_probe(tmp_path / "models/test/probe.pt")
    h, rows, _ = learning.load_data(tmp_path)
    training = [i for i, r in enumerate(rows) if r["split"] == "train"]
    assert torch.allclose(model.mean, h[training].mean(0))
    assert checkpoint["threshold_model_output"] == pytest.approx(4 * checkpoint["threshold_raw_utility"])
    args.replica, args.bootstrap = "main", 50
    learning.evaluate(args)
    metrics = common.read_json(tmp_path / "models/test/test_main.json")
    assert metrics["questions"] == 2
    assert metrics["mse"] < metrics["zero_predictor_mse"]
    assert metrics["probe_minus_entropy_question_mean"]["value"] > 0
    path = tmp_path / "labels/main/8.json"
    content = common.read_json(path)
    content["records"][0]["utility_raw"] = 0.1
    common.atomic_json(path, content)
    with pytest.raises(ValueError, match="changed after training"):
        learning.evaluate(args)


def test_matched_budget_and_constant_metrics():
    rows = [dict(question=str(i // 4), target=0.0, entropy=0.0, truncated=0.0) for i in range(8)]
    report = learning.report_metrics(rows, np.zeros(8), 0, 0, 0.25, bootstrap=20)
    counts = [m["count"] for m in report["retrospective_matched_budget"].values()]
    assert counts == [2, 2, 2]
    assert report["spearman"] is None


def test_prepare_real_parquet_holdout_and_duplicates(tmp_path, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    data = [
        dict(
            prompt=[dict(role="user", content=f"question {i}")],
            reward_model=dict(ground_truth="1"),
            data_source="math_dapo",
        )
        for i in range(10)
    ]
    data += data  # Every held-out question also has a duplicate row to exclude.
    source = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist(data), source)
    monkeypatch.setattr(run, "model_identity", lambda path: {"path": path})

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3]}

    monkeypatch.setattr(run, "tokenizer_for", lambda path: Tokenizer())
    root = tmp_path / "work"
    args = run.parser().parse_args(
        ["prepare", "--work-dir", str(root), "--data", str(source), "--model", "fake", "--num-questions", "10"]
    )
    run.prepare(args)
    questions = common.read_json(root / "questions.json")
    assert len(questions) == 10
    heldout = {q["id"] for q in questions if q["split"] != "train"}
    grpo = pq.read_table(root / "grpo_train_without_probe_holdout.parquet").to_pylist()
    assert len(grpo) == 16
    assert all(common.question_id(r["prompt"]) not in heldout for r in grpo)


def test_real_qwen_causal_feature_alignment():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    torch.manual_seed(13)
    model = transformers.Qwen2ForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.inference_mode():
        states, logp, _ = run.prefix_features(model, ids, 3, [1, 3], 64)
        for i, t in enumerate([1, 3]):
            prefix = ids[:, : 3 + t]
            reference = model.model(input_ids=prefix, use_cache=False).last_hidden_state[0, -1]
            assert torch.allclose(states[i], reference, atol=1e-5)
            reference_logp = model(prefix).logits[0, -1].float().log_softmax(-1)
            assert torch.allclose(logp[i], reference_logp, atol=1e-5)
        altered = ids.clone()
        altered[:, 6:] = 9
        changed, _, _ = run.prefix_features(model, altered, 3, [1, 3], 64)
        assert torch.allclose(states, changed, atol=1e-5)
