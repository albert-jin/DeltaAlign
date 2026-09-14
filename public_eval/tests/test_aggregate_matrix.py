import json
from pathlib import Path

import pytest

from public_eval import aggregate_matrix as matrix


def _write_fixture(tmp_path: Path):
    subset = tmp_path / "frozen.jsonl"
    subset_rows = []
    ids_by_task = {}
    for task in matrix.TASKS:
        ids = [f"{task}:{index}" for index in range(matrix.EXPECTED_PER_TASK)]
        ids_by_task[task] = ids
        subset_rows.extend({"record_id": record_id, "task": task} for record_id in ids)
    subset.write_text(
        "".join(matrix.canonical_json(item) + "\n" for item in subset_rows), encoding="utf-8"
    )
    subset_manifest = subset.with_suffix(".manifest.json")
    frozen = {
        "schema_version": matrix.RUN_SCHEMA_VERSION,
        "protocol_sha256": "protocol-frozen",
        "record_count": len(subset_rows),
        "task_counts": {task: matrix.EXPECTED_PER_TASK for task in matrix.TASKS},
        "subset_sha256": matrix.file_sha256(subset),
    }
    subset_manifest.write_text(json.dumps(frozen), encoding="utf-8")

    run_paths = {model: [] for model in matrix.MODELS}
    for model_index, model in enumerate(matrix.MODELS):
        role = "time_baseline" if model == "time" else "method"
        for task in matrix.TASKS:
            path = tmp_path / f"{model}_{task}.jsonl"
            config = {
                "schema_version": matrix.RUN_SCHEMA_VERSION,
                "role": role,
                "model_label": model.upper(),
                "served_model": model,
                "checkpoint": {
                    "path": f"/models/{model}",
                    "inventory_sha256": f"checkpoint-{model}",
                },
                "tokenizer_path": f"/models/{model}",
                "subset_sha256": frozen["subset_sha256"],
                "protocol_sha256": frozen["protocol_sha256"],
                "harness_sha256": "harness-frozen",
                "selected_ids_sha256": matrix.value_sha256(ids_by_task[task]),
                "selected_count": matrix.EXPECTED_PER_TASK,
                "task": task,
                "generation": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_tokens": 1024,
                    "master_seed": 3407,
                },
                "endpoint": "http://127.0.0.1:1/v1",
            }
            config_hash = matrix.value_sha256(config)
            output_rows = []
            for index, record_id in enumerate(ids_by_task[task]):
                strict = (index % 2 == 0) if model == "time" else (index % 4 != 0)
                relaxed = strict or (model != "time" and index % 8 == 0)
                output_rows.append(
                    {
                        "record_id": record_id,
                        "task": task,
                        "status": "ok",
                        "role": role,
                        "schema_version": matrix.RUN_SCHEMA_VERSION,
                        "model_label": model.upper(),
                        "served_model": model,
                        "run_config_sha256": config_hash,
                        "subset_sha256": frozen["subset_sha256"],
                        "answer_correct": strict,
                        "relaxed_answer_correct": relaxed,
                        "think_incidence": index % (model_index + 2) == 0,
                        "reasoning_tokens": 10 - model_index,
                        "output_tokens": 20 - 2 * model_index,
                        "non_reasoning_output_tokens": 15 - model_index,
                        "format_failure": not strict,
                        "think_format_failure": False,
                        "hit_max_tokens": model == "time" and index % 16 == 0,
                    }
                )
            path.write_text(
                "".join(matrix.canonical_json(item) + "\n" for item in output_rows),
                encoding="utf-8",
            )
            Path(str(path) + ".manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": matrix.RUN_SCHEMA_VERSION,
                        "kind": "public_eval_run",
                        "run_config": config,
                        "run_config_sha256": config_hash,
                    }
                ),
                encoding="utf-8",
            )
            run_paths[model].append(path)
    return subset, subset_manifest, run_paths


def _aggregate(tmp_path: Path, *, replications: int = 20):
    subset, manifest, run_paths = _write_fixture(tmp_path)
    result = matrix.aggregate_matrix(
        run_paths,
        subset_path=subset,
        subset_manifest_path=manifest,
        replications=replications,
        seed=3407,
        input_mode="runs",
    )
    return subset, manifest, run_paths, result


def test_aggregate_matrix_validates_three_way_pairing_and_all_paper_metrics(tmp_path):
    _, _, _, result = _aggregate(tmp_path)
    assert result["status"] == "pass"
    assert result["validation"]["validated_task_model_cells"] == 9
    assert result["validation"]["validated_paired_examples"] == 768
    assert result["validation"]["validated_result_rows"] == 2304
    gsm = result["tasks"]["gsm8k"]
    assert gsm["models"]["time"]["strict_accuracy"] == 0.5
    assert gsm["models"]["primary"]["strict_accuracy"] == 0.75
    assert gsm["models"]["primary"]["non_reasoning_output_tokens_mean"] == 14
    delta = gsm["comparisons_vs_time"]["primary"]["delta_method_minus_time"]
    assert delta["strict_accuracy"]["estimate"] == 0.25
    assert delta["reasoning_tokens"]["estimate"] == -1
    assert delta["output_tokens"]["estimate"] == -2
    assert delta["non_reasoning_output_tokens"]["estimate"] == -1
    assert len(delta["length_cap"]["ci95"]) == 2
    rendered = matrix.render_markdown(result)
    assert "Validation: **PASS**" in rendered
    assert "Strict acc." in rendered
    assert "Non-reasoning tokens" in rendered
    assert "Length cap" in rendered


def test_aggregate_matrix_rejects_incomplete_256_cell(tmp_path):
    subset, manifest, run_paths = _write_fixture(tmp_path)
    broken = run_paths["consistency"][0]
    lines = broken.read_text(encoding="utf-8").splitlines()
    broken.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain exactly"):
        matrix.aggregate_matrix(
            run_paths,
            subset_path=subset,
            subset_manifest_path=manifest,
            replications=5,
            seed=3407,
            input_mode="runs",
        )


def test_aggregate_matrix_rejects_frozen_hash_drift(tmp_path):
    subset, manifest, run_paths = _write_fixture(tmp_path)
    run = run_paths["primary"][1]
    run_manifest_path = Path(str(run) + ".manifest.json")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest["run_config"]["subset_sha256"] = "different-subset"
    run_manifest["run_config_sha256"] = matrix.value_sha256(run_manifest["run_config"])
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="subset hash differs"):
        matrix.aggregate_matrix(
            run_paths,
            subset_path=subset,
            subset_manifest_path=manifest,
            replications=5,
            seed=3407,
            input_mode="runs",
        )


def _legacy_comparisons(tmp_path: Path, result, run_paths):
    reverse = {
        "strict_accuracy": "answer_correct",
        "relaxed_accuracy": "relaxed_answer_correct",
        "think_incidence": "think_incidence",
        "reasoning_tokens": "reasoning_tokens",
        "output_tokens": "output_tokens",
        "answer_format_failure": "format_failure",
        "think_format_failure": "think_format_failure",
        "length_cap": "hit_max_tokens",
    }
    comparison_paths = {"primary": [], "consistency": []}
    for model in ("primary", "consistency"):
        for task_index, task in enumerate(matrix.TASKS):
            block = result["tasks"][task]["comparisons_vs_time"][model]
            delta = {
                legacy: block["delta_method_minus_time"][current]["estimate"]
                for current, legacy in reverse.items()
            }
            ci = {
                legacy: block["delta_method_minus_time"][current]["ci95"]
                for current, legacy in reverse.items()
            }
            time_manifest = json.loads(
                Path(str(run_paths["time"][task_index]) + ".manifest.json").read_text()
            )
            method_manifest = json.loads(
                Path(str(run_paths[model][task_index]) + ".manifest.json").read_text()
            )
            document = {
                "schema_version": matrix.RUN_SCHEMA_VERSION,
                "kind": "time_vs_method_public_eval",
                "subset_sha256": result["validation"]["subset_sha256"],
                "protocol_sha256": result["validation"]["protocol_sha256"],
                "paired_bootstrap": result["paired_bootstrap"],
                "time_baseline": {
                    "run": str(run_paths["time"][task_index]),
                    "run_config_sha256": time_manifest["run_config_sha256"],
                },
                "method": {
                    "run": str(run_paths[model][task_index]),
                    "run_config_sha256": method_manifest["run_config_sha256"],
                },
                "paired": {
                    "overall": {"n": matrix.EXPECTED_PER_TASK},
                    "per_task": {
                        task: {
                            "n": matrix.EXPECTED_PER_TASK,
                            "delta_method_minus_time": delta,
                            "delta_method_minus_time_95_ci": ci,
                        }
                    },
                },
            }
            path = tmp_path / f"compare_{model}_{task}.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            comparison_paths[model].append(path)
    return comparison_paths


def test_compare_mode_resolves_and_verifies_recorded_runs(tmp_path):
    subset, manifest, run_paths, expected = _aggregate(tmp_path, replications=10)
    comparisons = _legacy_comparisons(tmp_path, expected, run_paths)
    resolved, sources = matrix._runs_from_comparisons(
        comparisons["primary"], comparisons["consistency"]
    )
    observed = matrix.aggregate_matrix(
        resolved,
        subset_path=subset,
        subset_manifest_path=manifest,
        replications=10,
        seed=3407,
        input_mode="comparisons_with_recorded_runs",
        comparison_sources=sources,
    )
    assert observed["status"] == "pass"
    assert observed["provenance"]["input_mode"] == "comparisons_with_recorded_runs"
    assert len(observed["provenance"]["comparisons"]) == 6


def test_compare_mode_rejects_tampered_bootstrap_ci(tmp_path):
    subset, manifest, run_paths, expected = _aggregate(tmp_path, replications=5)
    comparisons = _legacy_comparisons(tmp_path, expected, run_paths)
    path = comparisons["primary"][0]
    document = json.loads(path.read_text(encoding="utf-8"))
    document["paired"]["per_task"]["gsm8k"]["delta_method_minus_time_95_ci"][
        "answer_correct"
    ][0] += 0.1
    path.write_text(json.dumps(document), encoding="utf-8")
    resolved, sources = matrix._runs_from_comparisons(
        comparisons["primary"], comparisons["consistency"]
    )
    with pytest.raises(ValueError, match="comparison mismatch"):
        matrix.aggregate_matrix(
            resolved,
            subset_path=subset,
            subset_manifest_path=manifest,
            replications=5,
            seed=3407,
            input_mode="comparisons_with_recorded_runs",
            comparison_sources=sources,
        )
