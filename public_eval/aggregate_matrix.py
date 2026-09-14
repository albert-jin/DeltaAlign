#!/usr/bin/env python3
"""Fail-closed paper-table aggregation for the frozen public evaluation.

This is intentionally a CPU-only post-processing command.  It accepts either
the nine single-model/single-task run JSONL files or the six TIME-versus-method
comparison JSON files.  Comparison inputs are resolved back to their recorded
run files because the v1 comparison schema predates the required
``non_reasoning_output_tokens`` metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "deltatime-public-eval-matrix-1.0"
RUN_SCHEMA_VERSION = "deltatime-public-eval-1.0"
TASKS = ("gsm8k", "arc_challenge", "boolq")
MODELS = ("time", "primary", "consistency")
EXPECTED_PER_TASK = 256

# (paper/JSON name, raw-result field, presentation kind)
METRICS = (
    ("strict_accuracy", "answer_correct", "rate"),
    ("relaxed_accuracy", "relaxed_answer_correct", "rate"),
    ("think_incidence", "think_incidence", "rate"),
    ("reasoning_tokens", "reasoning_tokens", "tokens"),
    ("output_tokens", "output_tokens", "tokens"),
    ("non_reasoning_output_tokens", "non_reasoning_output_tokens", "tokens"),
    ("answer_format_failure", "format_failure", "rate"),
    ("think_format_failure", "think_format_failure", "rate"),
    ("length_cap", "hit_max_tokens", "rate"),
)
METRIC_NAMES = tuple(item[0] for item in METRICS)
RAW_METRIC_FIELDS = tuple(item[1] for item in METRICS)
METRIC_KINDS = {item[0]: item[2] for item in METRICS}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from None
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"invalid JSON on {path}:{line_number}") from None
        if not isinstance(value, dict):
            raise ValueError(f"JSONL value on {path}:{line_number} is not an object")
        records.append(value)
    return records


def _require_keys(value: dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = sorted(set(keys) - value.keys())
    if missing:
        raise ValueError(f"{context} missing required fields: {missing}")


def load_frozen_subset(
    subset_path: Path, manifest_path: Path
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    manifest = load_json(manifest_path)
    _require_keys(
        manifest,
        ("schema_version", "protocol_sha256", "record_count", "task_counts", "subset_sha256"),
        f"subset manifest {manifest_path}",
    )
    if manifest["schema_version"] != RUN_SCHEMA_VERSION:
        raise ValueError(f"unexpected subset schema: {manifest['schema_version']!r}")
    actual_hash = file_sha256(subset_path)
    if actual_hash != manifest["subset_sha256"]:
        raise ValueError(
            f"subset hash mismatch: manifest={manifest['subset_sha256']} actual={actual_hash}"
        )
    records = load_jsonl(subset_path)
    if len(records) != EXPECTED_PER_TASK * len(TASKS):
        raise ValueError(
            f"frozen subset must contain exactly {EXPECTED_PER_TASK}x{len(TASKS)} records; "
            f"observed {len(records)}"
        )
    if len(records) != int(manifest["record_count"]):
        raise ValueError("frozen subset record count differs from its manifest")
    ids_by_task: dict[str, list[str]] = {task: [] for task in TASKS}
    seen: set[str] = set()
    for item in records:
        _require_keys(item, ("record_id", "task"), f"subset record in {subset_path}")
        task = str(item["task"])
        record_id = str(item["record_id"])
        if task not in ids_by_task:
            raise ValueError(f"unknown frozen task {task!r}")
        if record_id in seen:
            raise ValueError(f"duplicate frozen record_id: {record_id}")
        seen.add(record_id)
        ids_by_task[task].append(record_id)
    expected_counts = {task: EXPECTED_PER_TASK for task in TASKS}
    observed_counts = {task: len(ids) for task, ids in ids_by_task.items()}
    if observed_counts != expected_counts or manifest["task_counts"] != expected_counts:
        raise ValueError(
            f"frozen task counts must be {expected_counts}; subset={observed_counts}, "
            f"manifest={manifest['task_counts']}"
        )
    return ids_by_task, manifest


def _validated_run(
    path: Path,
    *,
    expected_model: str,
    expected_ids: Sequence[str],
    subset_manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = Path(str(path) + ".manifest.json")
    manifest = load_json(manifest_path)
    _require_keys(
        manifest,
        ("schema_version", "kind", "run_config", "run_config_sha256"),
        f"run manifest {manifest_path}",
    )
    if manifest["schema_version"] != RUN_SCHEMA_VERSION or manifest["kind"] != "public_eval_run":
        raise ValueError(f"unsupported run manifest in {manifest_path}")
    config = manifest["run_config"]
    if not isinstance(config, dict):
        raise ValueError(f"run_config in {manifest_path} is not an object")
    if value_sha256(config) != manifest["run_config_sha256"]:
        raise ValueError(f"run_config hash mismatch in {manifest_path}")
    _require_keys(
        config,
        (
            "role",
            "model_label",
            "checkpoint",
            "subset_sha256",
            "protocol_sha256",
            "harness_sha256",
            "selected_ids_sha256",
            "selected_count",
            "task",
            "generation",
        ),
        f"run_config in {manifest_path}",
    )
    expected_role = "time_baseline" if expected_model == "time" else "method"
    if config["role"] != expected_role:
        raise ValueError(
            f"{path} has role={config['role']!r}; {expected_model} requires {expected_role!r}"
        )
    task = str(config["task"])
    if task not in TASKS:
        raise ValueError(f"{path} has unsupported task {task!r}")
    if int(config["selected_count"]) != EXPECTED_PER_TASK:
        raise ValueError(
            f"{path} must select exactly {EXPECTED_PER_TASK} examples; "
            f"observed {config['selected_count']}"
        )
    if config["selected_ids_sha256"] != value_sha256(list(expected_ids)):
        raise ValueError(f"{path} selected IDs do not match the frozen {task} subset")
    if config["subset_sha256"] != subset_manifest["subset_sha256"]:
        raise ValueError(f"{path} subset hash differs from the frozen manifest")
    if config["protocol_sha256"] != subset_manifest["protocol_sha256"]:
        raise ValueError(f"{path} protocol hash differs from the frozen manifest")

    attempts = load_jsonl(path)
    latest: dict[str, dict[str, Any]] = {}
    expected_id_set = set(expected_ids)
    for item in attempts:
        record_id = item.get("record_id")
        if not record_id:
            raise ValueError(f"a record in {path} lacks record_id")
        if str(record_id) not in expected_id_set:
            raise ValueError(f"{path} contains foreign record_id {record_id}")
        for field, expected in (
            ("schema_version", RUN_SCHEMA_VERSION),
            ("task", task),
            ("role", expected_role),
            ("model_label", config["model_label"]),
            ("served_model", config.get("served_model")),
            ("run_config_sha256", manifest["run_config_sha256"]),
            ("subset_sha256", subset_manifest["subset_sha256"]),
        ):
            if item.get(field) != expected:
                raise ValueError(f"{path} has foreign {field} in attempt for {record_id}")
        latest[str(record_id)] = item
    if set(latest) != expected_id_set:
        missing = sorted(expected_id_set - set(latest))[:5]
        extra = sorted(set(latest) - expected_id_set)[:5]
        raise ValueError(
            f"{path} does not contain exactly the frozen {task} IDs; "
            f"missing={missing}, extra={extra}"
        )
    if len(latest) != EXPECTED_PER_TASK:
        raise ValueError(f"{path} has {len(latest)} unique results, expected {EXPECTED_PER_TASK}")
    for record_id in expected_ids:
        item = latest[record_id]
        _require_keys(
            item,
            (
                "record_id",
                "task",
                "status",
                "role",
                "run_config_sha256",
                "subset_sha256",
                *RAW_METRIC_FIELDS,
            ),
            f"result {record_id} in {path}",
        )
        if item["status"] != "ok":
            raise ValueError(f"{path} has non-successful final result for {record_id}")
        if item["task"] != task or item["role"] != expected_role:
            raise ValueError(f"{path} has foreign task/role in result {record_id}")
        if item["run_config_sha256"] != manifest["run_config_sha256"]:
            raise ValueError(f"{path} mixes run_config hashes at {record_id}")
        if item["subset_sha256"] != subset_manifest["subset_sha256"]:
            raise ValueError(f"{path} mixes subset hashes at {record_id}")
        for field in RAW_METRIC_FIELDS:
            if field in {"reasoning_tokens", "output_tokens", "non_reasoning_output_tokens"}:
                if isinstance(item[field], bool) or not isinstance(item[field], (int, float)):
                    raise ValueError(f"{path} has non-numeric {field} at {record_id}")
            elif not isinstance(item[field], bool):
                raise ValueError(f"{path} has non-boolean {field} at {record_id}")
            numeric = float(item[field])
            if not math.isfinite(numeric):
                raise ValueError(f"{path} has non-finite {field} at {record_id}")
            if field.endswith("tokens") and numeric < 0:
                raise ValueError(f"{path} has negative {field} at {record_id}")

    checkpoint = config["checkpoint"]
    if not isinstance(checkpoint, dict) or not checkpoint.get("inventory_sha256"):
        raise ValueError(f"{path} lacks a checkpoint inventory hash")
    return {
        "path": path.resolve(),
        "file_sha256": file_sha256(path),
        "manifest_path": manifest_path.resolve(),
        "manifest_sha256": file_sha256(manifest_path),
        "manifest": manifest,
        "config": config,
        "task": task,
        "rows": latest,
        "attempt_records": len(attempts),
    }


def _task_index(paths: Sequence[Path], context: str) -> dict[str, Path]:
    if len(paths) != len(TASKS):
        raise ValueError(f"{context} requires exactly {len(TASKS)} files")
    indexed: dict[str, Path] = {}
    for path in paths:
        manifest_path = Path(str(path) + ".manifest.json")
        manifest = load_json(manifest_path)
        config = manifest.get("run_config")
        task = config.get("task") if isinstance(config, dict) else None
        if task not in TASKS:
            raise ValueError(f"cannot infer a valid task from {manifest_path}")
        if task in indexed:
            raise ValueError(f"duplicate {context} task: {task}")
        indexed[str(task)] = path
    if set(indexed) != set(TASKS):
        raise ValueError(f"{context} does not cover exactly {list(TASKS)}")
    return indexed


def _runs_from_comparisons(
    primary_paths: Sequence[Path], consistency_paths: Sequence[Path]
) -> tuple[dict[str, list[Path]], list[dict[str, Any]]]:
    if len(primary_paths) != len(TASKS) or len(consistency_paths) != len(TASKS):
        raise ValueError("compare mode requires exactly three primary and three consistency files")
    sources: list[dict[str, Any]] = []
    indexed: dict[str, dict[str, Path]] = {model: {} for model in MODELS}
    baseline_identity: dict[str, tuple[str, str]] = {}
    for model, paths in (("primary", primary_paths), ("consistency", consistency_paths)):
        for path in paths:
            comparison = load_json(path)
            _require_keys(
                comparison,
                (
                    "schema_version",
                    "kind",
                    "subset_sha256",
                    "protocol_sha256",
                    "paired_bootstrap",
                    "time_baseline",
                    "method",
                    "paired",
                ),
                f"comparison {path}",
            )
            if (
                comparison["schema_version"] != RUN_SCHEMA_VERSION
                or comparison["kind"] != "time_vs_method_public_eval"
            ):
                raise ValueError(f"unsupported comparison artifact: {path}")
            per_task = comparison["paired"].get("per_task")
            if not isinstance(per_task, dict) or len(per_task) != 1:
                raise ValueError(f"{path} must contain exactly one per-task comparison")
            task = next(iter(per_task))
            if task not in TASKS or int(per_task[task].get("n", -1)) != EXPECTED_PER_TASK:
                raise ValueError(f"{path} is not a complete {EXPECTED_PER_TASK}-example task")
            if task in indexed[model]:
                raise ValueError(f"duplicate {model} comparison for {task}")
            time_run = Path(comparison["time_baseline"]["run"])
            method_run = Path(comparison["method"]["run"])
            for block_name, run_path in (("time_baseline", time_run), ("method", method_run)):
                run_manifest = load_json(Path(str(run_path) + ".manifest.json"))
                recorded_hash = comparison[block_name].get("run_config_sha256")
                if run_manifest.get("run_config_sha256") != recorded_hash:
                    raise ValueError(
                        f"{path} {block_name} run_config hash does not match {run_path}"
                    )
            indexed[model][task] = method_run
            time_key = (
                str(comparison["time_baseline"].get("run_config_sha256")),
                file_sha256(time_run),
            )
            if task in baseline_identity and baseline_identity[task] != time_key:
                raise ValueError(
                    f"primary and consistency comparisons use different TIME runs for {task}"
                )
            baseline_identity[task] = time_key
            indexed["time"][task] = time_run
            sources.append(
                {
                    "model": model,
                    "task": task,
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                    "document": comparison,
                }
            )
    for model in MODELS:
        if set(indexed[model]) != set(TASKS):
            raise ValueError(f"compare inputs do not cover all tasks for {model}")
    return (
        {model: [indexed[model][task] for task in TASKS] for model in MODELS},
        sources,
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty metric")
    return sum(values) / len(values)


def summarize_rows(rows: dict[str, dict[str, Any]], ids: Sequence[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(ids)}
    for name, field, kind in METRICS:
        values = [float(rows[record_id][field]) for record_id in ids]
        key = name if kind == "rate" else name + "_mean"
        summary[key] = _mean(values)
        if kind == "tokens":
            summary[name + "_median"] = statistics.median(values)
    return summary


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_comparison(
    time_rows: dict[str, dict[str, Any]],
    method_rows: dict[str, dict[str, Any]],
    ids: Sequence[str],
    *,
    replications: int,
    seed: int,
) -> dict[str, Any]:
    deltas: dict[str, list[float]] = {}
    for name, field, _ in METRICS:
        deltas[name] = [
            float(method_rows[record_id][field]) - float(time_rows[record_id][field])
            for record_id in ids
        ]
    point = {name: _mean(values) for name, values in deltas.items()}
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    n = len(ids)
    for _ in range(replications):
        selected = [rng.randrange(n) for _ in range(n)]
        for name in METRIC_NAMES:
            values = deltas[name]
            samples[name].append(sum(values[index] for index in selected) / n)
    intervals = {
        name: [percentile(values, 0.025), percentile(values, 0.975)]
        for name, values in samples.items()
    }
    strict_pairs = Counter(
        (
            bool(time_rows[record_id]["answer_correct"]),
            bool(method_rows[record_id]["answer_correct"]),
        )
        for record_id in ids
    )
    return {
        "n": n,
        "delta_method_minus_time": {
            name: {"estimate": point[name], "ci95": intervals[name]}
            for name in METRIC_NAMES
        },
        "strict_accuracy_pair_counts": {
            "both_correct": strict_pairs[(True, True)],
            "method_only_correct": strict_pairs[(False, True)],
            "time_only_correct": strict_pairs[(True, False)],
            "both_incorrect": strict_pairs[(False, False)],
        },
    }


def _all_equal(values: Sequence[Any], context: str) -> Any:
    if not values:
        raise ValueError(f"no values for {context}")
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(f"frozen invariant differs across runs: {context}")
    return first


def _validate_model_identity(model: str, runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    labels = [runs[task]["config"]["model_label"] for task in TASKS]
    checkpoints = [runs[task]["config"]["checkpoint"] for task in TASKS]
    label = _all_equal(labels, f"{model} model_label")
    inventory_sha256 = _all_equal(
        [item["inventory_sha256"] for item in checkpoints],
        f"{model} checkpoint inventory_sha256",
    )
    checkpoint_path = _all_equal(
        [item.get("path") for item in checkpoints], f"{model} checkpoint path"
    )
    return {
        "role": "time_baseline" if model == "time" else "method",
        "model_label": label,
        "checkpoint_path": checkpoint_path,
        "checkpoint_inventory_sha256": inventory_sha256,
    }


def _assert_close(actual: Any, expected: Any, context: str) -> None:
    if actual is None or expected is None:
        if actual != expected:
            raise ValueError(f"comparison mismatch for {context}: {actual!r} != {expected!r}")
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"comparison mismatch for {context}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_close(left, right, f"{context}[{index}]")
        return
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"comparison mismatch for {context}: {actual!r} != {expected!r}")


def _verify_comparison_sources(
    sources: Sequence[dict[str, Any]],
    matrix: dict[str, Any],
) -> None:
    legacy_metric_map = {
        "strict_accuracy": "answer_correct",
        "relaxed_accuracy": "relaxed_answer_correct",
        "think_incidence": "think_incidence",
        "reasoning_tokens": "reasoning_tokens",
        "output_tokens": "output_tokens",
        "answer_format_failure": "format_failure",
        "think_format_failure": "think_format_failure",
        "length_cap": "hit_max_tokens",
    }
    for source in sources:
        comparison = source["document"]
        model = source["model"]
        task = source["task"]
        if comparison["subset_sha256"] != matrix["validation"]["subset_sha256"]:
            raise ValueError(f"comparison {source['path']} has a different subset hash")
        if comparison["protocol_sha256"] != matrix["validation"]["protocol_sha256"]:
            raise ValueError(f"comparison {source['path']} has a different protocol hash")
        if comparison["paired_bootstrap"] != matrix["paired_bootstrap"]:
            raise ValueError(f"comparison {source['path']} uses different bootstrap settings")
        task_block = matrix["tasks"][task]
        expected = task_block["comparisons_vs_time"][model]["delta_method_minus_time"]
        legacy = comparison["paired"]["per_task"][task]
        for matrix_name, legacy_name in legacy_metric_map.items():
            _assert_close(
                legacy["delta_method_minus_time"][legacy_name],
                expected[matrix_name]["estimate"],
                f"{source['path']} {legacy_name} delta",
            )
            _assert_close(
                legacy["delta_method_minus_time_95_ci"][legacy_name],
                expected[matrix_name]["ci95"],
                f"{source['path']} {legacy_name} CI",
            )


def aggregate_matrix(
    run_paths: dict[str, Sequence[Path]],
    *,
    subset_path: Path,
    subset_manifest_path: Path,
    replications: int,
    seed: int,
    input_mode: str,
    comparison_sources: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    if replications < 1:
        raise ValueError("bootstrap replications must be positive")
    ids_by_task, subset_manifest = load_frozen_subset(subset_path, subset_manifest_path)
    indexed_paths = {
        model: _task_index(list(run_paths[model]), f"{model} runs") for model in MODELS
    }
    runs: dict[str, dict[str, dict[str, Any]]] = {model: {} for model in MODELS}
    for model in MODELS:
        for task in TASKS:
            runs[model][task] = _validated_run(
                indexed_paths[model][task],
                expected_model=model,
                expected_ids=ids_by_task[task],
                subset_manifest=subset_manifest,
            )

    all_runs = [runs[model][task] for model in MODELS for task in TASKS]
    subset_sha256 = _all_equal(
        [item["config"]["subset_sha256"] for item in all_runs], "subset_sha256"
    )
    protocol_sha256 = _all_equal(
        [item["config"]["protocol_sha256"] for item in all_runs], "protocol_sha256"
    )
    harness_sha256 = _all_equal(
        [item["config"]["harness_sha256"] for item in all_runs], "harness_sha256"
    )
    generation = _all_equal(
        [item["config"]["generation"] for item in all_runs], "generation settings"
    )
    model_metadata = {
        model: _validate_model_identity(model, runs[model]) for model in MODELS
    }
    inventories = [model_metadata[model]["checkpoint_inventory_sha256"] for model in MODELS]
    if len(set(inventories)) != len(inventories):
        raise ValueError("TIME, Primary, and Consistency must have distinct checkpoint inventories")

    bootstrap = {
        "unit": "paired example",
        "statistic": "mean(method - TIME)",
        "interval": "95% percentile",
        "replications": replications,
        "seed": seed,
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "time_primary_consistency_public_eval_matrix",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "validation": {
            "status": "pass",
            "expected_tasks": list(TASKS),
            "expected_examples_per_task": EXPECTED_PER_TASK,
            "expected_models_per_example": len(MODELS),
            "validated_task_model_cells": len(TASKS) * len(MODELS),
            "validated_paired_examples": EXPECTED_PER_TASK * len(TASKS),
            "validated_result_rows": EXPECTED_PER_TASK * len(TASKS) * len(MODELS),
            "subset_path": str(subset_path.resolve()),
            "subset_manifest_path": str(subset_manifest_path.resolve()),
            "subset_sha256": subset_sha256,
            "subset_manifest_sha256": file_sha256(subset_manifest_path),
            "protocol_sha256": protocol_sha256,
            "harness_sha256": harness_sha256,
            "generation": generation,
        },
        "paired_bootstrap": bootstrap,
        "models": model_metadata,
        "tasks": {},
        "provenance": {
            "input_mode": input_mode,
            "aggregator_path": str(Path(__file__).resolve()),
            "aggregator_sha256": file_sha256(Path(__file__)),
            "runs": {},
            "comparisons": [
                {key: source[key] for key in ("model", "task", "path", "sha256")}
                for source in comparison_sources
            ],
        },
    }
    for model in MODELS:
        result["provenance"]["runs"][model] = {
            task: {
                "path": str(runs[model][task]["path"]),
                "sha256": runs[model][task]["file_sha256"],
                "manifest_path": str(runs[model][task]["manifest_path"]),
                "manifest_sha256": runs[model][task]["manifest_sha256"],
                "run_config_sha256": runs[model][task]["manifest"]["run_config_sha256"],
                "attempt_records": runs[model][task]["attempt_records"],
                "successful_unique_records": EXPECTED_PER_TASK,
            }
            for task in TASKS
        }

    for task in TASKS:
        ids = ids_by_task[task]
        # The frozen subset preserves upstream source order, while the v1 pair
        # comparator deliberately bootstraps lexicographically sorted IDs.
        # Match that recorded ordering so a fixed bootstrap seed reproduces
        # the task comparison artifacts byte-for-number.
        paired_ids = sorted(ids)
        for model in MODELS:
            if set(runs[model][task]["rows"]) != set(ids):
                raise ValueError(f"three-way pairing failed for {model}/{task}")
        task_models = {
            model: summarize_rows(runs[model][task]["rows"], ids) for model in MODELS
        }
        result["tasks"][task] = {
            "n": EXPECTED_PER_TASK,
            "selected_ids_sha256": value_sha256(list(ids)),
            "models": task_models,
            "comparisons_vs_time": {
                model: paired_comparison(
                    runs["time"][task]["rows"],
                    runs[model][task]["rows"],
                    paired_ids,
                    replications=replications,
                    seed=seed,
                )
                for model in ("primary", "consistency")
            },
        }

    result["macro_task"] = {
        "note": "unweighted mean of the three task point estimates; no macro bootstrap CI",
        "models": {},
        "comparisons_vs_time": {},
    }
    for model in MODELS:
        keys = [key for key in result["tasks"][TASKS[0]]["models"][model] if key != "n"]
        result["macro_task"]["models"][model] = {
            "n_tasks": len(TASKS),
            **{
                key: _mean([result["tasks"][task]["models"][model][key] for task in TASKS])
                for key in keys
            },
        }
    for model in ("primary", "consistency"):
        result["macro_task"]["comparisons_vs_time"][model] = {
            "n_tasks": len(TASKS),
            "delta_method_minus_time": {
                name: _mean(
                    [
                        result["tasks"][task]["comparisons_vs_time"][model][
                            "delta_method_minus_time"
                        ][name]["estimate"]
                        for task in TASKS
                    ]
                )
                for name in METRIC_NAMES
            },
        }

    if comparison_sources:
        _verify_comparison_sources(comparison_sources, result)
    return result


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _delta_cell(metric: str, value: dict[str, Any]) -> str:
    estimate = float(value["estimate"])
    low, high = [float(item) for item in value["ci95"]]
    if METRIC_KINDS[metric] == "rate":
        return f"{100 * estimate:+.2f} pp [{100 * low:+.2f}, {100 * high:+.2f}]"
    return f"{estimate:+.2f} [{low:+.2f}, {high:+.2f}]"


def render_markdown(matrix: dict[str, Any]) -> str:
    validation = matrix["validation"]
    lines = [
        "# Frozen public-task evaluation matrix",
        "",
        (
            f"Validation: **PASS** — {validation['validated_paired_examples']} frozen examples "
            f"({validation['expected_examples_per_task']} per task) paired across TIME, Primary, "
            "and Consistency."
        ),
        "",
        f"Subset SHA-256: `{validation['subset_sha256']}`  ",
        f"Protocol SHA-256: `{validation['protocol_sha256']}`  ",
        f"Harness SHA-256: `{validation['harness_sha256']}`",
        "",
        "Rates are percentages. Token columns are per-example means. Deltas are method minus TIME; "
        "brackets are paired-example 95% percentile-bootstrap intervals.",
        "",
        "## Absolute results",
        "",
        "| Task | Model | N | Strict acc. | Relaxed acc. | Think incidence | Think tokens | Output tokens | Non-reasoning tokens | Answer format fail | Think format fail | Length cap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    display_models = {"time": "TIME", "primary": "Primary", "consistency": "Consistency"}
    for task in TASKS:
        for model in MODELS:
            stats = matrix["tasks"][task]["models"][model]
            lines.append(
                "| "
                + " | ".join(
                    (
                        task,
                        display_models[model],
                        str(stats["n"]),
                        _percent(stats["strict_accuracy"]),
                        _percent(stats["relaxed_accuracy"]),
                        _percent(stats["think_incidence"]),
                        f"{stats['reasoning_tokens_mean']:.2f}",
                        f"{stats['output_tokens_mean']:.2f}",
                        f"{stats['non_reasoning_output_tokens_mean']:.2f}",
                        _percent(stats["answer_format_failure"]),
                        _percent(stats["think_format_failure"]),
                        _percent(stats["length_cap"]),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Paired accuracy and validity deltas",
            "",
            "| Task | Contrast | Strict acc. | Relaxed acc. | Think incidence | Answer format fail | Think format fail | Length cap |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        for model in ("primary", "consistency"):
            values = matrix["tasks"][task]["comparisons_vs_time"][model][
                "delta_method_minus_time"
            ]
            lines.append(
                "| "
                + " | ".join(
                    (
                        task,
                        f"{display_models[model]} - TIME",
                        _delta_cell("strict_accuracy", values["strict_accuracy"]),
                        _delta_cell("relaxed_accuracy", values["relaxed_accuracy"]),
                        _delta_cell("think_incidence", values["think_incidence"]),
                        _delta_cell("answer_format_failure", values["answer_format_failure"]),
                        _delta_cell("think_format_failure", values["think_format_failure"]),
                        _delta_cell("length_cap", values["length_cap"]),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Paired token deltas",
            "",
            "| Task | Contrast | Think tokens | Output tokens | Non-reasoning tokens |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for task in TASKS:
        for model in ("primary", "consistency"):
            values = matrix["tasks"][task]["comparisons_vs_time"][model][
                "delta_method_minus_time"
            ]
            lines.append(
                "| "
                + " | ".join(
                    (
                        task,
                        f"{display_models[model]} - TIME",
                        _delta_cell("reasoning_tokens", values["reasoning_tokens"]),
                        _delta_cell("output_tokens", values["output_tokens"]),
                        _delta_cell(
                            "non_reasoning_output_tokens", values["non_reasoning_output_tokens"]
                        ),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Macro-task values in the JSON are unweighted means of the three task point estimates; "
            "they deliberately have no pseudo-replicated macro confidence interval.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-runs", type=Path, nargs=3, metavar=("GSM8K", "ARC", "BOOLQ"))
    parser.add_argument("--primary-runs", type=Path, nargs=3, metavar=("GSM8K", "ARC", "BOOLQ"))
    parser.add_argument(
        "--consistency-runs", type=Path, nargs=3, metavar=("GSM8K", "ARC", "BOOLQ")
    )
    parser.add_argument(
        "--primary-compares", type=Path, nargs=3, metavar=("GSM8K", "ARC", "BOOLQ")
    )
    parser.add_argument(
        "--consistency-compares", type=Path, nargs=3, metavar=("GSM8K", "ARC", "BOOLQ")
    )
    default_subset = Path(__file__).resolve().parent / "data" / "frozen_v1.jsonl"
    parser.add_argument("--subset", type=Path, default=default_subset)
    parser.add_argument("--subset-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="machine-readable JSON output")
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=3407)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_mode = any((args.time_runs, args.primary_runs, args.consistency_runs))
        compare_mode = any((args.primary_compares, args.consistency_compares))
        if run_mode == compare_mode:
            raise ValueError("choose exactly one input mode: all run files or all compare files")
        comparison_sources: Sequence[dict[str, Any]] = ()
        if run_mode:
            if not all((args.time_runs, args.primary_runs, args.consistency_runs)):
                raise ValueError("run mode requires --time-runs, --primary-runs, and --consistency-runs")
            run_paths = {
                "time": args.time_runs,
                "primary": args.primary_runs,
                "consistency": args.consistency_runs,
            }
            input_mode = "runs"
        else:
            if not all((args.primary_compares, args.consistency_compares)):
                raise ValueError(
                    "compare mode requires --primary-compares and --consistency-compares"
                )
            run_paths, comparison_sources = _runs_from_comparisons(
                args.primary_compares, args.consistency_compares
            )
            input_mode = "comparisons_with_recorded_runs"
        subset_manifest = args.subset_manifest or args.subset.with_suffix(".manifest.json")
        matrix = aggregate_matrix(
            run_paths,
            subset_path=args.subset,
            subset_manifest_path=subset_manifest,
            replications=args.bootstrap_reps,
            seed=args.bootstrap_seed,
            input_mode=input_mode,
            comparison_sources=comparison_sources,
        )
        atomic_write_json(args.output, matrix)
        atomic_write_text(args.markdown_output, render_markdown(matrix))
        print(
            f"PASS: wrote {args.output} and {args.markdown_output}; "
            f"validated {matrix['validation']['validated_result_rows']} model-example rows"
        )
        return 0
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
