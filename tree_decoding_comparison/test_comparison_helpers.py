"""CPU-only tests for candidate budgets and order-invariant pass@k."""

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("tree_decoding_comparison.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "tree_decoding_comparison_main", MODULE_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
comparison = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = comparison
MODULE_SPEC.loader.exec_module(comparison)


class FakeParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeLLM:
    def __init__(self, tree_outputs, filler_outputs=None):
        self.tree_outputs = tree_outputs
        self.filler_outputs = filler_outputs or []
        self.calls = 0
        self.params = []

    def generate(self, prompts, params):
        self.calls += 1
        self.params.append(params)
        return self.tree_outputs if self.calls == 1 else self.filler_outputs


def node(seq_id, text, *, is_leaf, parent_seq_id=None):
    return SimpleNamespace(
        seq_id=seq_id,
        tree_text=text,
        text=text,
        is_leaf=is_leaf,
        parent_seq_id=parent_seq_id,
    )


def request_output(*nodes):
    return SimpleNamespace(outputs=list(nodes))


def test_pass_at_k_is_order_invariant():
    first = [[True, False, False, False]]
    last = [[False, False, False, True]]
    expected = [0.25, 0.5, 0.75, 1.0]
    assert [comparison.compute_pass_at_k(first, k) for k in range(1, 5)] \
        == pytest.approx(expected)
    assert [comparison.compute_pass_at_k(last, k) for k in range(1, 5)] \
        == pytest.approx(expected)


def test_pass_at_k_rejects_an_empty_candidate_pool():
    with pytest.raises(ValueError, match="no candidates"):
        comparison.compute_pass_at_k([[]], 1)
    with pytest.raises(ValueError, match="at least one example"):
        comparison.compute_pass_at_k([], 1)


def test_leaf_extraction_preserves_duplicate_and_empty_candidates():
    output = request_output(
        node(1, "", is_leaf=True),
        node(2, "same", is_leaf=True),
        node(3, "same", is_leaf=True),
    )
    candidates, shape = comparison.extract_tree_leaf_candidates(output)
    assert candidates == ["", "same", "same"]
    assert shape["tree_leaf_count"] == 3


def test_leaf_extraction_rejects_missing_parent_and_duplicate_ids():
    missing_parent = request_output(
        node(2, "suffix", is_leaf=True, parent_seq_id=1))
    with pytest.raises(RuntimeError, match="missing parent_seq_id=1"):
        comparison.extract_tree_leaf_candidates(missing_parent)

    duplicate_ids = request_output(
        node(1, "a", is_leaf=True),
        node(1, "b", is_leaf=True),
    )
    with pytest.raises(RuntimeError, match="Duplicate seq_id=1"):
        comparison.extract_tree_leaf_candidates(duplicate_ids)


def test_multilevel_leaf_path_reconstruction():
    output = request_output(
        node(1, "root-", is_leaf=False),
        node(2, "middle-", is_leaf=False, parent_seq_id=1),
        node(3, "leaf", is_leaf=True, parent_seq_id=2),
    )
    candidates, shape = comparison.extract_tree_leaf_candidates(output)
    assert candidates == ["root-middle-leaf"]
    assert shape["tree_internal_node_count"] == 2


@pytest.mark.parametrize(
    ("target_n", "tree_leaf_count"),
    [(1, 1), (3, 1), (8, 4), (10, 7)],
)
def test_tree_topup_produces_exact_budget(
    monkeypatch, target_n, tree_leaf_count,
):
    monkeypatch.setattr(comparison, "SamplingParams", FakeParams, raising=False)
    monkeypatch.setattr(comparison, "TreeSearchParams", FakeParams, raising=False)

    tree = request_output(*[
        node(index, "" if index == 0 else f"tree-{index}", is_leaf=True)
        for index in range(tree_leaf_count)
    ])
    needed = target_n - tree_leaf_count
    filler = request_output(*[
        SimpleNamespace(text=f"filler-{index}") for index in range(needed)
    ])
    llm = FakeLLM([tree], [filler] if needed else None)

    generations, metadata = comparison.generate_tree_decoding(
        llm=llm,
        prompts=["prompt"],
        branch_trigger_mode="random",
        target_n=target_n,
        random_branch_probability=0.2,
    )

    assert len(generations[0]) == target_n
    assert generations[0][0] == ""
    assert metadata[0]["tree_leaf_count"] == tree_leaf_count
    assert metadata[0]["filler_count"] == needed
    assert metadata[0]["total_count"] == target_n
    assert metadata[0]["candidate_sources"] == (
        ["tree"] * tree_leaf_count + ["grpo_filler"] * needed)
    assert llm.calls == (2 if needed else 1)


def test_tree_over_budget_fails_instead_of_slicing(monkeypatch):
    monkeypatch.setattr(comparison, "SamplingParams", FakeParams, raising=False)
    monkeypatch.setattr(comparison, "TreeSearchParams", FakeParams, raising=False)
    tree = request_output(*[
        node(index, str(index), is_leaf=True) for index in range(4)
    ])

    with pytest.raises(RuntimeError, match="exceeding max_num_leaves"):
        comparison.generate_tree_decoding(
            llm=FakeLLM([tree]),
            prompts=["prompt"],
            branch_trigger_mode="entropy",
            target_n=3,
        )


@pytest.mark.parametrize(
    ("mode", "input_tau", "effective_tau"),
    [
        ("random", 9.0, None),
        ("entropy", 9.0, None),
        ("entropy_waad", 0.5, 0.5),
    ],
)
def test_tree_modes_pass_explicit_vllm_semantics(
    monkeypatch, mode, input_tau, effective_tau,
):
    monkeypatch.setattr(comparison, "SamplingParams", FakeParams, raising=False)
    monkeypatch.setattr(comparison, "TreeSearchParams", FakeParams, raising=False)
    llm = FakeLLM([request_output(node(1, "answer", is_leaf=True))])

    comparison.generate_tree_decoding(
        llm=llm,
        prompts=["prompt"],
        branch_trigger_mode=mode,
        target_n=1,
        tau_importance=input_tau,
        branching_factor=4,
        max_tree_depth=5,
        entropy_threshold=6.0,
        random_branch_probability=0.3,
        min_seg_length=7,
    )

    sampling_params = llm.params[0][0]
    tree_params = sampling_params.tree_search_params
    assert tree_params.branch_trigger_mode == mode
    assert tree_params.tau_importance == effective_tau
    assert tree_params.max_num_leaves == 1
    assert tree_params.branching_factor == 4
    assert tree_params.max_tree_depth == 5
    assert tree_params.entropy_threshold == 6.0
    assert tree_params.random_branch_probability == 0.3
    assert tree_params.min_seg_length == 7


def test_base_grpo_rejects_missing_candidates(monkeypatch):
    monkeypatch.setattr(comparison, "SamplingParams", FakeParams, raising=False)
    llm = FakeLLM([
        request_output(SimpleNamespace(text="only-one")),
    ])
    with pytest.raises(RuntimeError, match="expected exactly 2"):
        comparison.generate_base_grpo(llm, ["prompt"], n=2)


def test_tree_rejects_wrong_filler_count(monkeypatch):
    monkeypatch.setattr(comparison, "SamplingParams", FakeParams, raising=False)
    monkeypatch.setattr(comparison, "TreeSearchParams", FakeParams, raising=False)
    llm = FakeLLM(
        [request_output(node(1, "tree", is_leaf=True))],
        [request_output(SimpleNamespace(text="only-one-filler"))],
    )
    with pytest.raises(RuntimeError, match="expected exactly 2"):
        comparison.generate_tree_decoding(
            llm=llm,
            prompts=["prompt"],
            branch_trigger_mode="entropy",
            target_n=3,
        )


def test_variable_filler_counts_map_back_to_each_prompt(monkeypatch):
    monkeypatch.setattr(comparison, "SamplingParams", FakeParams, raising=False)
    monkeypatch.setattr(comparison, "TreeSearchParams", FakeParams, raising=False)
    tree_outputs = [
        request_output(node(1, "tree-0", is_leaf=True)),
        request_output(
            node(2, "tree-1a", is_leaf=True),
            node(3, "tree-1b", is_leaf=True),
        ),
    ]
    filler_outputs = [
        request_output(
            SimpleNamespace(text="fill-0a"),
            SimpleNamespace(text="fill-0b"),
        ),
        request_output(SimpleNamespace(text="fill-1")),
    ]
    llm = FakeLLM(tree_outputs, filler_outputs)

    generations, metadata = comparison.generate_tree_decoding(
        llm=llm,
        prompts=["prompt-0", "prompt-1"],
        branch_trigger_mode="entropy",
        target_n=3,
        seed=7,
    )

    assert generations == [
        ["tree-0", "fill-0a", "fill-0b"],
        ["tree-1a", "tree-1b", "fill-1"],
    ]
    assert [item["filler_count"] for item in metadata] == [2, 1]
    assert [params.n for params in llm.params[1]] == [2, 1]
