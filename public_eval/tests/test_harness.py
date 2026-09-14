import json
from argparse import Namespace
from pathlib import Path

import pytest

from public_eval import harness


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return text.split()


def record(task, gold, labels=None):
    return {
        "record_id": f"{task}:0",
        "task": task,
        "gold": gold,
        "choice_labels": labels,
    }


@pytest.mark.parametrize(
    ("item", "response", "correct"),
    [
        (record("gsm8k", "18"), "work\nFinal answer: 18", True),
        (record("gsm8k", "18"), "Final answer: 18.0", True),
        (record("arc_challenge", "B", ["A", "B", "C"]), "Final answer: b", True),
        (record("boolq", "yes"), "Final answer: YES", True),
        (record("boolq", "no"), "Final answer: yes", False),
    ],
)
def test_strict_scorer_accepts_only_contract(item, response, correct):
    result = harness.parse_final_answer(item, response)
    assert result["answer_format_ok"] is True
    assert result["answer_correct"] is correct


@pytest.mark.parametrize(
    "response",
    [
        "The answer is yes.",
        "Final answer: yes\na trailing explanation",
        "Final answer: yes\nFinal answer: yes",
        "**Final answer: yes**",
        "Final answer: yes.",
    ],
)
def test_strict_scorer_flags_format_failures(response):
    result = harness.parse_final_answer(record("boolq", "yes"), response)
    assert result["answer_format_ok"] is False
    assert result["format_failure"] is True
    assert result["answer_correct"] is False


@pytest.mark.parametrize(
    ("item", "response"),
    [
        (record("gsm8k", "18"), "<answer>18.0</answer>"),
        (record("arc_challenge", "B", ["A", "B", "C"]), "<answer>b</answer>"),
        (record("boolq", "yes"), "<think>check</think>\n<answer> YES </answer>"),
    ],
)
def test_relaxed_scorer_accepts_one_atomic_answer_block(item, response):
    result = harness.parse_final_answer(item, response)
    assert result["answer_correct"] is False
    assert result["format_failure"] is True
    assert result["answer_wrapper_format_ok"] is True
    assert result["relaxed_answer_source"] == "answer_block"
    assert result["relaxed_answer_correct"] is True


@pytest.mark.parametrize(
    "response",
    [
        "<answer>yes</answer><answer>yes</answer>",
        "<answer>yes",
        "</answer><answer>yes</answer>",
        "<answer><answer>yes</answer></answer>",
        "<answer>The answer is yes.</answer>",
    ],
)
def test_relaxed_scorer_rejects_ambiguous_or_non_atomic_blocks(response):
    result = harness.parse_final_answer(record("boolq", "yes"), response)
    assert result["format_failure"] is True
    assert result["relaxed_answer_correct"] is False


def test_relaxed_scorer_does_not_override_valid_but_wrong_strict_answer():
    result = harness.parse_final_answer(
        record("boolq", "yes"), "<answer>yes</answer>\nFinal answer: no"
    )
    assert result["answer_format_ok"] is True
    assert result["answer_correct"] is False
    assert result["relaxed_answer_source"] == "strict_final"
    assert result["relaxed_answer_correct"] is False


def test_think_metrics_separate_reasoning_and_output():
    metrics = harness.analyze_thinking(
        "<think>one two three</think>\nFinal answer: yes", WordTokenizer()
    )
    assert metrics == {
        "think_incidence": True,
        "think_block_count": 1,
        "think_open_tag_count": 1,
        "think_close_tag_count": 1,
        "think_format_failure": False,
        "reasoning_tokens": 3,
        "output_tokens": 6,
        "non_reasoning_output_tokens": 3,
        "completion_tokens_local": 6,
    }


@pytest.mark.parametrize(
    "response",
    ["<think>unclosed", "</think>orphan", "<think>a<think>b</think></think>"],
)
def test_think_metrics_detect_malformed_tags(response):
    assert harness.analyze_thinking(response, WordTokenizer())["think_format_failure"] is True


def test_selection_is_deterministic_and_not_prefix():
    first = harness.select_indices("gsm8k", 100, 10, "salt")
    second = harness.select_indices("gsm8k", 100, 10, "salt")
    assert first == second
    assert first != list(range(10))
    assert first == sorted(first)


def test_subset_hash_verification(tmp_path):
    subset = tmp_path / "tiny.jsonl"
    item = {
        "record_id": "boolq:0",
        "task": "boolq",
        "messages": [{"role": "user", "content": "q"}],
        "gold": "yes",
        "answer_type": "boolean",
    }
    subset.write_text(harness.canonical_json(item) + "\n", encoding="utf-8")
    manifest_path = harness.default_manifest_path(subset)
    manifest_path.write_text(
        json.dumps(
            {
                "subset_sha256": harness.file_sha256(subset),
                "protocol_sha256": harness.PROTOCOL_SHA256,
                "record_count": 1,
            }
        ),
        encoding="utf-8",
    )
    rows, _ = harness.load_and_verify_subset(subset, manifest_path)
    assert rows == [item]
    subset.write_text(subset.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        harness.load_and_verify_subset(subset, manifest_path)


def test_resume_manifest_refuses_config_drift(tmp_path):
    path = tmp_path / "run.manifest.json"
    first = harness.validate_resume_manifest(path, {"model": "a"})
    assert first["run_config_sha256"] == harness.value_sha256({"model": "a"})
    harness.validate_resume_manifest(path, {"model": "a"})
    with pytest.raises(ValueError, match="does not match"):
        harness.validate_resume_manifest(path, {"model": "b"})


def test_repair_truncated_jsonl_preserves_complete_records(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_bytes(b'{"record_id":"a"}\n{"record_id":')
    assert harness.repair_truncated_jsonl(path) is True
    assert path.read_bytes() == b'{"record_id":"a"}\n'
    assert harness.latest_results(path)["a"]["record_id"] == "a"


def test_repair_jsonl_adds_newline_to_complete_final_record(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_bytes(b'{"record_id":"a"}')
    assert harness.repair_truncated_jsonl(path) is False
    assert path.read_bytes().endswith(b"\n")


def make_result(record_id, task, correct, think, reasoning, output, format_failure=False):
    return {
        "record_id": record_id,
        "task": task,
        "status": "ok",
        "answer_correct": correct,
        "relaxed_answer_correct": correct,
        "think_incidence": think,
        "reasoning_tokens": reasoning,
        "output_tokens": output,
        "format_failure": format_failure,
        "think_format_failure": False,
        "hit_max_tokens": False,
    }


def test_summary_metrics():
    rows = {
        "gsm8k:0": make_result("gsm8k:0", "gsm8k", True, True, 10, 2),
        "gsm8k:1": make_result("gsm8k:1", "gsm8k", False, False, 0, 4, True),
    }
    summary = harness.summarize_run(rows)
    gsm = summary["per_task"]["gsm8k"]
    assert gsm["answer_correct_rate"] == 0.5
    assert gsm["relaxed_answer_correct_rate"] == 0.5
    assert gsm["think_incidence_rate"] == 0.5
    assert gsm["reasoning_tokens_mean"] == 5
    assert gsm["output_tokens_mean"] == 3
    assert gsm["format_failure_rate"] == 0.5


def test_run_config_records_selected_limit(tmp_path):
    args = Namespace(
        role="method",
        task="gsm8k",
        model_label="m",
        served_model="served",
        tokenizer=tmp_path,
        max_tokens=16,
        seed=3407,
        api_base="http://127.0.0.1:1/v1",
    )
    config = harness.run_config(
        args,
        {"subset_sha256": "abc"},
        {"inventory_sha256": "def"},
        ["a", "b"],
    )
    assert config["selected_count"] == 2
    assert config["selected_ids_sha256"] == harness.value_sha256(["a", "b"])
    assert config["generation"]["temperature"] == 0.0


def test_run_is_single_task_and_resume_skips_successes(tmp_path, monkeypatch):
    subset = tmp_path / "frozen.jsonl"
    rows = [
        {
            "record_id": f"boolq:{index}",
            "task": "boolq",
            "source_index": index,
            "messages": [{"role": "user", "content": "question"}],
            "gold": "yes",
            "answer_type": "boolean",
            "choice_labels": None,
        }
        for index in range(2)
    ]
    subset.write_text("".join(harness.canonical_json(row) + "\n" for row in rows))
    subset_manifest = harness.default_manifest_path(subset)
    subset_manifest.write_text(
        json.dumps(
            {
                "subset_sha256": harness.file_sha256(subset),
                "protocol_sha256": harness.PROTOCOL_SHA256,
                "record_count": 2,
            }
        )
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    output = tmp_path / "run.jsonl"

    calls = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def verify_model(self, model):
            assert model == "served"

        def complete(self, payload):
            calls.append(payload)
            return {
                "choices": [
                    {
                        "message": {"content": "Final answer: yes"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 4},
            }

    monkeypatch.setattr(harness, "APIClient", FakeClient)
    monkeypatch.setattr(harness, "load_tokenizer", lambda path: WordTokenizer())
    monkeypatch.setattr(
        harness,
        "directory_fingerprint",
        lambda path: {"path": str(path), "inventory_sha256": "checkpoint-hash"},
    )
    args = Namespace(
        role="time_baseline",
        task="boolq",
        model_label="TIME",
        served_model="served",
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        subset=subset,
        subset_manifest=subset_manifest,
        output=output,
        api_base="http://127.0.0.1:1/v1",
        api_key="key",
        max_tokens=16,
        seed=3407,
        timeout=1,
        retries=0,
        limit=None,
        resume=False,
    )
    harness.run_evaluation(args)
    assert len(calls) == 2
    assert all(item["temperature"] == 0 for item in calls)
    assert all(row["answer_correct"] for row in harness.latest_results(output).values())

    args.resume = True
    harness.run_evaluation(args)
    assert len(calls) == 2


def test_compare_accepts_only_paired_single_task_runs(tmp_path):
    common_config = {
        "subset_sha256": "subset",
        "protocol_sha256": "protocol",
        "harness_sha256": "harness",
        "selected_ids_sha256": harness.value_sha256(["boolq:0"]),
        "selected_count": 1,
        "generation": {"temperature": 0.0},
        "task": "boolq",
    }
    paths = {}
    for role, correct, reasoning in (
        ("time_baseline", False, 4),
        ("method", True, 2),
    ):
        path = tmp_path / f"{role}.jsonl"
        config = {**common_config, "role": role, "checkpoint": role}
        config_hash = harness.value_sha256(config)
        result = make_result("boolq:0", "boolq", correct, True, reasoning, 3)
        result["run_config_sha256"] = config_hash
        path.write_text(harness.canonical_json(result) + "\n")
        Path(str(path) + ".manifest.json").write_text(
            json.dumps({"run_config": config, "run_config_sha256": config_hash})
        )
        paths[role] = path
    args = Namespace(
        time_run=paths["time_baseline"],
        method_run=paths["method"],
        output=tmp_path / "comparison.json",
        csv_output=tmp_path / "comparison.csv",
        allow_incomplete=False,
        bootstrap_reps=200,
        bootstrap_seed=3407,
    )
    comparison = harness.compare_runs(args)
    paired = comparison["paired"]["overall"]
    assert paired["delta_method_minus_time"]["answer_correct"] == 1.0
    assert paired["delta_method_minus_time"]["relaxed_answer_correct"] == 1.0
    assert paired["delta_method_minus_time_95_ci"]["answer_correct"] == [1.0, 1.0]
    assert paired["delta_method_minus_time"]["reasoning_tokens"] == -2.0
    assert paired["accuracy_pair_counts"]["method_only_correct"] == 1
    assert list(comparison["paired"]["per_task"]) == ["boolq"]
    assert args.csv_output.exists()


def test_paired_bootstrap_is_deterministic_and_paired():
    time_rows = {
        f"boolq:{index}": make_result(
            f"boolq:{index}", "boolq", correct, False, index, index + 1
        )
        for index, correct in enumerate((False, True, False, True))
    }
    method_rows = {
        key: {**value, "answer_correct": not value["answer_correct"],
              "relaxed_answer_correct": not value["relaxed_answer_correct"],
              "reasoning_tokens": value["reasoning_tokens"] + 2}
        for key, value in time_rows.items()
    }
    ids = sorted(time_rows)
    first = harness.paired_bootstrap_ci(
        time_rows, method_rows, ids, replications=250, seed=3407
    )
    second = harness.paired_bootstrap_ci(
        time_rows, method_rows, ids, replications=250, seed=3407
    )
    assert first == second
    assert first["reasoning_tokens"] == [2.0, 2.0]
    assert first["output_tokens"] == [0.0, 0.0]
