#!/usr/bin/env python3
"""Paired, group-aware comparison of two DeltaTIMEBench output files.

The counterfactual siblings sharing a ``group_id`` are the dependence unit.
This script therefore resamples whole paired groups for confidence intervals
and swaps model labels for whole groups in the randomization test.  The
reported estimand is always ``model_b - model_a``; an additional normalized
``improvement`` field flips lower-is-better metrics for easier reading.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import score_delta_outputs
from score_delta_outputs import score


LOWER_IS_BETTER = {
    "false_retrigger_rate",
    "missed_state_change_rate",
    "mean_think_characters",
    "format_error_rate",
    "protocol_format_error_rate",
}


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_score(path: Path) -> list[dict]:
    records = json.loads(path.read_text())
    # score() performs the authoritative schema/group validation and annotates
    # each record with parsed trigger, answer, format, and joint fields.
    records = copy.deepcopy(records)
    score(records, reps=10, seed=0, allow_incomplete=False)
    return records


def align_groups(a: list[dict], b: list[dict]) -> list[dict[str, dict[str, dict]]]:
    by_id_a = {str(item["scenario_id"]): item for item in a}
    by_id_b = {str(item["scenario_id"]): item for item in b}
    if set(by_id_a) != set(by_id_b):
        missing_a = sorted(set(by_id_b) - set(by_id_a))[:5]
        missing_b = sorted(set(by_id_a) - set(by_id_b))[:5]
        raise ValueError(
            "scenario sets differ: "
            f"missing_from_a={missing_a}, missing_from_b={missing_b}"
        )

    metadata = ("group_id", "domain", "pair_role", "oracle_trigger", "expected_answer")
    grouped: dict[str, dict[str, dict[str, dict]]] = defaultdict(
        lambda: {"a": {}, "b": {}}
    )
    for scenario_id in sorted(by_id_a):
        left, right = by_id_a[scenario_id], by_id_b[scenario_id]
        mismatch = {key: (left.get(key), right.get(key)) for key in metadata if left.get(key) != right.get(key)}
        if mismatch:
            raise ValueError(f"metadata mismatch for {scenario_id}: {mismatch}")
        group_id = str(left["group_id"])
        role = str(left["pair_role"])
        grouped[group_id]["a"][role] = left
        grouped[group_id]["b"][role] = right

    result = []
    for group_id in sorted(grouped):
        pair = grouped[group_id]
        if set(pair["a"]) != set(pair["b"]):
            raise ValueError(f"role mismatch in group {group_id}")
        result.append(pair)
    return result


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def summarize(groups: list[dict[str, dict[str, dict]]], side: str) -> dict[str, float | None]:
    tp = fp = tn = fn = 0
    n = answers = protocol_answers = compatible_answers = 0
    format_errors = protocol_format_errors = 0
    think_total = 0.0
    inert_time: list[float] = []
    break_time: list[float] = []
    semantic: list[float] = []
    irrelevant: list[float] = []
    temporal: list[float] = []
    group_joint: list[float] = []
    protocol_group_joint: list[float] = []

    for pair in groups:
        roles = pair[side]
        for item in roles.values():
            oracle = int(item["oracle_trigger"])
            predicted = int(item["predicted_trigger"])
            if oracle and predicted:
                tp += 1
            elif oracle:
                fn += 1
            elif predicted:
                fp += 1
            else:
                tn += 1
            n += 1
            answers += int(item["answer_correct"])
            compatible_answers += int(item["time_compatible_answer_correct"])
            protocol_answers += int(item["protocol_answer_correct"])
            format_errors += int(not item["time_compatible_format_ok"])
            protocol_format_errors += int(not item["protocol_format_ok"])
            think_total += float(item["think_characters"])

        if "inert_short" in roles:
            inert_time.append(
                float(
                    roles["inert_short"]["predicted_trigger"] == 0
                    and roles["inert_long"]["predicted_trigger"] == 0
                )
            )
            break_time.append(
                float(
                    roles["assumption_break_short"]["predicted_trigger"] == 1
                    and roles["assumption_break_long"]["predicted_trigger"] == 1
                )
            )
            semantic.append(
                float(
                    roles["relevant_stable"]["predicted_trigger"] == 0
                    and roles["assumption_break_short"]["predicted_trigger"] == 1
                )
            )
            irrelevant.append(
                float(
                    roles["irrelevant_change"]["predicted_trigger"] == 0
                    and roles["assumption_break_short"]["predicted_trigger"] == 1
                )
            )
        elif "temporal_before_long" in roles:
            temporal.append(
                float(
                    roles["temporal_before_long"]["predicted_trigger"] == 0
                    and roles["temporal_crossing"]["predicted_trigger"] == 1
                )
            )
        group_joint.append(float(all(x["joint_success"] for x in roles.values())))
        protocol_group_joint.append(
            float(all(x["protocol_joint_success"] for x in roles.values()))
        )

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "trigger_accuracy": safe_div(tp + tn, tp + fp + tn + fn),
        "trigger_precision": precision,
        "trigger_recall": recall,
        "trigger_f1": safe_div(2 * precision * recall, precision + recall),
        "false_retrigger_rate": safe_div(fp, fp + tn),
        "missed_state_change_rate": safe_div(fn, fn + tp),
        "answer_accuracy": safe_div(answers, n),
        "time_compatible_answer_accuracy": safe_div(compatible_answers, n),
        "protocol_answer_accuracy": safe_div(protocol_answers, n),
        "mean_think_characters": safe_div(think_total, n),
        "format_error_rate": safe_div(format_errors, n),
        "protocol_format_error_rate": safe_div(protocol_format_errors, n),
        "counterfactual_time_consistency": mean(inert_time),
        "assumption_break_time_consistency": mean(break_time),
        "semantic_delta_separation_accuracy": mean(semantic),
        "irrelevant_delta_separation_accuracy": mean(irrelevant),
        "temporal_boundary_sensitivity": mean(temporal),
        "joint_group_success": mean(group_joint),
        "protocol_joint_group_success": mean(protocol_group_joint),
    }


def differences(groups: list[dict[str, dict[str, dict]]]) -> tuple[dict, dict, dict]:
    a = summarize(groups, "a")
    b = summarize(groups, "b")
    diff = {
        key: (None if a[key] is None or b[key] is None else float(b[key] - a[key]))
        for key in a
    }
    return a, b, diff


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        value = min(1.0, (count - rank) * p_values[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def compare(
    groups: list[dict[str, dict[str, dict]]],
    *,
    bootstrap_reps: int,
    permutation_reps: int,
    seed: int,
) -> dict:
    a, b, observed = differences(groups)
    metric_names = [key for key, value in observed.items() if value is not None]
    rng = np.random.default_rng(seed)
    bootstrap = {key: np.empty(bootstrap_reps) for key in metric_names}
    permutation = {key: np.empty(permutation_reps) for key in metric_names}
    n_groups = len(groups)

    for rep in range(bootstrap_reps):
        indices = rng.integers(0, n_groups, size=n_groups)
        sample = [groups[int(index)] for index in indices]
        _, _, diff = differences(sample)
        for key in metric_names:
            bootstrap[key][rep] = diff[key]

    for rep in range(permutation_reps):
        swaps = rng.integers(0, 2, size=n_groups)
        permuted = []
        for index, pair in enumerate(groups):
            if swaps[index]:
                permuted.append({"a": pair["b"], "b": pair["a"]})
            else:
                permuted.append(pair)
        _, _, diff = differences(permuted)
        for key in metric_names:
            permutation[key][rep] = diff[key]

    raw_p = {
        key: float(
            (1 + np.count_nonzero(np.abs(permutation[key]) >= abs(observed[key]) - 1e-15))
            / (permutation_reps + 1)
        )
        for key in metric_names
    }
    adjusted = holm_adjust(raw_p)
    metrics = {}
    for key in metric_names:
        lower_better = key in LOWER_IS_BETTER
        improvement = -observed[key] if lower_better else observed[key]
        boot_improvement = -bootstrap[key] if lower_better else bootstrap[key]
        metrics[key] = {
            "model_a": a[key],
            "model_b": b[key],
            "effect_b_minus_a": observed[key],
            "effect_ci95_group_bootstrap": [
                float(np.quantile(bootstrap[key], 0.025)),
                float(np.quantile(bootstrap[key], 0.975)),
            ],
            "preferred_direction": "lower" if lower_better else "higher",
            "improvement": improvement,
            "probability_b_better_group_bootstrap": float(np.mean(boot_improvement > 0)),
            "randomization_p_two_sided": raw_p[key],
            "holm_adjusted_p": adjusted[key],
        }
    return {"metrics": metrics}


def markdown_table(result: dict) -> str:
    lines = [
        "| Metric | A | B | B-A | 95% paired group bootstrap CI | Preferred | Holm p |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for key, value in result["metrics"].items():
        ci = value["effect_ci95_group_bootstrap"]
        lines.append(
            f"| {key} | {value['model_a']:.4f} | {value['model_b']:.4f} | "
            f"{value['effect_b_minus_a']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{value['preferred_direction']} | {value['holm_adjusted_p']:.4g} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-a", required=True, type=Path)
    parser.add_argument("--file-b", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--label-a", default="model_a")
    parser.add_argument("--label-b", default="model_b")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--permutation-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.bootstrap_reps < 1 or args.permutation_reps < 1:
        parser.error("replication counts must be positive")

    records_a = load_and_score(args.file_a)
    records_b = load_and_score(args.file_b)
    groups = align_groups(records_a, records_b)
    result = compare(
        groups,
        bootstrap_reps=args.bootstrap_reps,
        permutation_reps=args.permutation_reps,
        seed=args.seed,
    )
    result = {
        "analysis": {
            "estimand": "model_b_minus_model_a",
            "dependence_unit": "group_id",
            "confidence_interval": "paired cluster percentile bootstrap",
            "hypothesis_test": "two-sided Monte Carlo group-label-swap randomization",
            "multiple_comparison": "Holm family-wise correction across reported metrics",
            "bootstrap_reps": args.bootstrap_reps,
            "permutation_reps": args.permutation_reps,
            "seed": args.seed,
            "n_records": len(records_a),
            "n_groups": len(groups),
        },
        "software": {
            "comparison_script": str(Path(__file__).resolve()),
            "comparison_script_sha256": file_sha256(Path(__file__).resolve()),
            "scorer_script": str(Path(score_delta_outputs.__file__).resolve()),
            "scorer_script_sha256": file_sha256(
                Path(score_delta_outputs.__file__).resolve()
            ),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
        },
        "model_a": {
            "label": args.label_a,
            "path": str(args.file_a.resolve()),
            "sha256": file_sha256(args.file_a),
        },
        "model_b": {
            "label": args.label_b,
            "path": str(args.file_b.resolve()),
            "sha256": file_sha256(args.file_b),
        },
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    markdown = args.markdown or args.output.with_suffix(".md")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(markdown_table(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
