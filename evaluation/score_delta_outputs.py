#!/usr/bin/env python3
"""Score DeltaTIMEBench outputs with group-aware paired metrics.

The v0.2 benchmark is a clustered design: several counterfactual siblings share
one latent state and one ``group_id``.  Point estimates that operate on records
remain record-level, while uncertainty estimates resample whole groups.  Paired
metrics are only computed for groups containing every role required by their
schema; the command-line interface rejects incomplete groups by default.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

V02_THRESHOLD_ROLES = frozenset(
    {
        "inert_short",
        "inert_long",
        "irrelevant_change",
        "relevant_stable",
        "assumption_break_short",
        "assumption_break_long",
    }
)
V02_TEMPORAL_ROLES = frozenset(
    {"temporal_before_short", "temporal_before_long", "temporal_crossing"}
)
V0_ROLES = frozenset(
    {"preserve_short", "preserve_long", "irrelevant_change", "relevant_change"}
)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def bootstrap_mean(
    values: list[float], rng: np.random.Generator, reps: int
) -> list[float] | None:
    """Percentile bootstrap for values that already have one value per group."""
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    samples = rng.choice(arr, size=(reps, len(arr)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def bootstrap_clusters(
    groups: list[list[dict]],
    metric: Callable[[list[dict]], float],
    rng: np.random.Generator,
    reps: int,
) -> list[float] | None:
    """Bootstrap a record-level statistic while resampling whole groups."""
    if not groups:
        return None
    estimates = np.empty(reps, dtype=float)
    for rep in range(reps):
        indices = rng.integers(0, len(groups), size=len(groups))
        sampled = [item for index in indices for item in groups[int(index)]]
        estimates[rep] = metric(sampled)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def normalize_answer(value: object) -> str:
    return str(value).strip().upper()


def parse_response(response: object) -> dict:
    """Extract trigger/answer fields and validate both output format notions.

    An opening ``<think>`` counts as a trigger even if generation was truncated;
    malformed structure is separately penalized.  TIME-compatible validity
    permits visible prose and mid-response thinking; Phase-5 protocol validity
    requires only optional think blocks followed by a final answer.  This avoids
    incorrectly rewarding a truncated reasoning response as a no-think decision.
    """
    text = response if isinstance(response, str) else ""
    lower = text.lower()
    predicted = int("<think>" in lower)
    answer_matches = list(ANSWER_RE.finditer(text))
    # Keep relaxed answer correctness comparable with the original scorer: if
    # several answers occur, use the last one semantically but fail strict
    # format validation below.
    answer = answer_matches[-1].group(1).strip() if answer_matches else ""

    depth = 0
    think_structure_ok = True
    for match in THINK_TAG_RE.finditer(text):
        is_close = match.group(0).lower().startswith("</")
        if is_close:
            if depth != 1:
                think_structure_ok = False
            else:
                depth = 0
        else:
            if depth != 0:
                think_structure_ok = False
            depth += 1
    if depth:
        think_structure_ok = False

    answer_well_formed = (
        len(answer_matches) == 1
        and bool(answer)
        and lower.count("<answer>") == 1
        and lower.count("</answer>") == 1
    )
    answer_is_final = bool(answer_matches) and not text[answer_matches[-1].end() :].strip()
    think_before_answer = not answer_matches or all(
        match.start() < answer_matches[0].start() for match in THINK_TAG_RE.finditer(text)
    )
    prefix = text[: answer_matches[0].start()] if answer_matches else text
    no_visible_prefix = not THINK_RE.sub("", prefix).strip()
    time_compatible_format_ok = bool(think_structure_ok and answer_well_formed)
    protocol_format_ok = bool(
        think_structure_ok
        and answer_well_formed
        and answer_is_final
        and think_before_answer
        and no_visible_prefix
    )

    think_blocks = THINK_RE.findall(text)
    if think_blocks:
        think_characters = sum(len(value) for value in think_blocks)
    elif predicted:
        think_characters = len(lower.split("<think>", 1)[1])
    else:
        think_characters = 0
    return {
        "predicted_trigger": predicted,
        "answer_extracted": answer,
        "answer_normalized": normalize_answer(answer),
        "think_tags_balanced": bool(think_structure_ok),
        "time_compatible_format_ok": time_compatible_format_ok,
        "protocol_format_ok": protocol_format_ok,
        # Backward-compatible alias.  TIME-compatible validity is the main
        # format notion; Phase-5 protocol adherence is reported separately.
        "format_ok": time_compatible_format_ok,
        "think_characters": think_characters,
    }


def expected_roles(item: dict) -> frozenset[str] | None:
    if item.get("schema_version") == "cf-deltatime-0.2":
        if item.get("domain") == "temporal_deadline":
            return V02_TEMPORAL_ROLES
        return V02_THRESHOLD_ROLES
    role = item.get("pair_role")
    if role in V0_ROLES:
        return V0_ROLES
    return None


def validate_and_group(
    records: list[dict], *, allow_incomplete: bool
) -> tuple[dict[str, dict[str, dict]], dict[str, frozenset[str] | None]]:
    if not isinstance(records, list) or not records:
        raise ValueError("input must be a non-empty JSON list")

    by_group: dict[str, dict[str, dict]] = defaultdict(dict)
    expected_by_group: dict[str, frozenset[str] | None] = {}
    signature_by_group: dict[str, tuple[object, object, frozenset[str] | None]] = {}
    scenario_ids: set[str] = set()
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"record {index} is not an object")
        required = {"scenario_id", "domain", "pair_role", "oracle_trigger", "expected_answer"}
        missing = sorted(required - item.keys())
        if missing:
            raise ValueError(f"record {index} missing required fields: {missing}")
        scenario_id = str(item["scenario_id"])
        if scenario_id in scenario_ids:
            raise ValueError(f"duplicate scenario_id: {scenario_id}")
        scenario_ids.add(scenario_id)
        if item["oracle_trigger"] not in (0, 1, False, True):
            raise ValueError(f"invalid oracle_trigger for {scenario_id}")

        if item.get("schema_version") == "cf-deltatime-0.2":
            group_id = item.get("group_id")
            if not group_id:
                raise ValueError(f"v0.2 record lacks group_id: {scenario_id}")
        else:
            group_id = item.get("group_id") or item.get("pair_id") or item.get("domain")
        group_id = str(group_id)
        role = str(item["pair_role"])
        if role in by_group[group_id]:
            raise ValueError(f"duplicate role {role!r} in group {group_id!r}")
        by_group[group_id][role] = item

        required_roles = expected_roles(item)
        signature = (item.get("schema_version"), item.get("domain"), required_roles)
        if group_id in signature_by_group and signature_by_group[group_id] != signature:
            raise ValueError(f"mixed schemas or domains in group {group_id!r}")
        signature_by_group[group_id] = signature
        expected_by_group[group_id] = required_roles

    errors = []
    for group_id, roles in by_group.items():
        required_roles = expected_by_group[group_id]
        if required_roles is None:
            continue
        present = frozenset(roles)
        missing = sorted(required_roles - present)
        unexpected = sorted(present - required_roles)
        if missing or unexpected:
            errors.append(
                f"{group_id}: missing={missing or '[]'}, unexpected={unexpected or '[]'}"
            )
    if errors and not allow_incomplete:
        preview = "; ".join(errors[:5])
        suffix = f"; ... ({len(errors)} invalid groups total)" if len(errors) > 5 else ""
        raise ValueError(f"incomplete/invalid counterfactual groups: {preview}{suffix}")
    return dict(by_group), expected_by_group


def trigger_f1(items: list[dict]) -> float:
    tp = sum(x["oracle_trigger"] == 1 and x["predicted_trigger"] == 1 for x in items)
    fp = sum(x["oracle_trigger"] == 0 and x["predicted_trigger"] == 1 for x in items)
    fn = sum(x["oracle_trigger"] == 1 and x["predicted_trigger"] == 0 for x in items)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return safe_div(2 * precision * recall, precision + recall)


def confusion_rate(items: list[dict], rate: str) -> float:
    tp = sum(x["oracle_trigger"] == 1 and x["predicted_trigger"] == 1 for x in items)
    fp = sum(x["oracle_trigger"] == 0 and x["predicted_trigger"] == 1 for x in items)
    tn = sum(x["oracle_trigger"] == 0 and x["predicted_trigger"] == 0 for x in items)
    fn = sum(x["oracle_trigger"] == 1 and x["predicted_trigger"] == 0 for x in items)
    if rate == "precision":
        return safe_div(tp, tp + fp)
    if rate == "recall":
        return safe_div(tp, tp + fn)
    if rate == "false_retrigger":
        return safe_div(fp, fp + tn)
    if rate == "missed_state_change":
        return safe_div(fn, fn + tp)
    raise ValueError(rate)


def score(
    records: list[dict], reps: int, seed: int, *, allow_incomplete: bool = False
) -> dict:
    by_group, expected_by_group = validate_and_group(
        records, allow_incomplete=allow_incomplete
    )
    rng = np.random.default_rng(seed)
    tp = fp = tn = fn = 0
    by_role: dict[str, list[dict]] = defaultdict(list)
    by_domain: dict[str, list[dict]] = defaultdict(list)

    for item in records:
        parsed = parse_response(item.get("response", ""))
        item.update(parsed)
        oracle = int(item["oracle_trigger"])
        answer_ok = int(
            item["answer_normalized"] == normalize_answer(item["expected_answer"])
        )
        trigger_ok = int(item["predicted_trigger"] == oracle)
        item["answer_correct"] = answer_ok
        item["time_compatible_answer_correct"] = int(
            answer_ok and item["time_compatible_format_ok"]
        )
        item["protocol_answer_correct"] = int(
            answer_ok and item["protocol_format_ok"]
        )
        # Backward-compatible alias for the stricter Phase-5 protocol score.
        item["strict_answer_correct"] = item["protocol_answer_correct"]
        item["trigger_correct"] = trigger_ok
        item["joint_success"] = int(
            trigger_ok and answer_ok and item["time_compatible_format_ok"]
        )
        item["protocol_joint_success"] = int(
            trigger_ok and answer_ok and item["protocol_format_ok"]
        )
        by_role[item["pair_role"]].append(item)
        by_domain[item["domain"]].append(item)
        if oracle and item["predicted_trigger"]:
            tp += 1
        elif oracle:
            fn += 1
        elif item["predicted_trigger"]:
            fp += 1
        else:
            tn += 1

    complete_groups: list[dict[str, dict]] = []
    incomplete_groups: list[str] = []
    for group_id, roles in by_group.items():
        required_roles = expected_by_group[group_id]
        if required_roles is None or frozenset(roles) == required_roles:
            complete_groups.append(roles)
        else:
            incomplete_groups.append(group_id)

    time_consistency: list[float] = []
    time_joint: list[float] = []
    assumption_break_time_consistency: list[float] = []
    semantic_separation: list[float] = []
    semantic_joint: list[float] = []
    irrelevant_separation: list[float] = []
    temporal_boundary: list[float] = []
    temporal_boundary_joint: list[float] = []
    joint_group_success: list[float] = []
    protocol_joint_group_success: list[float] = []
    joint_group_success_by_type: dict[str, list[float]] = defaultdict(list)
    joint_group_success_by_domain: dict[str, list[float]] = defaultdict(list)

    for roles in complete_groups:
        if "inert_short" in roles:
            short, long = roles["inert_short"], roles["inert_long"]
            time_consistency.append(
                float(short["predicted_trigger"] == 0 and long["predicted_trigger"] == 0)
            )
            time_joint.append(float(short["joint_success"] and long["joint_success"]))

            changed_short = roles["assumption_break_short"]
            changed_long = roles["assumption_break_long"]
            assumption_break_time_consistency.append(
                float(
                    changed_short["predicted_trigger"] == 1
                    and changed_long["predicted_trigger"] == 1
                )
            )

            stable = roles["relevant_stable"]
            changed = changed_short
            semantic_separation.append(
                float(stable["predicted_trigger"] == 0 and changed["predicted_trigger"] == 1)
            )
            semantic_joint.append(float(stable["joint_success"] and changed["joint_success"]))
            irrelevant = roles["irrelevant_change"]
            irrelevant_separation.append(
                float(irrelevant["predicted_trigger"] == 0 and changed["predicted_trigger"] == 1)
            )
        elif "preserve_short" in roles:
            short, long = roles["preserve_short"], roles["preserve_long"]
            time_consistency.append(
                float(short["predicted_trigger"] == 0 and long["predicted_trigger"] == 0)
            )
            time_joint.append(float(short["joint_success"] and long["joint_success"]))
            negative, changed = roles["irrelevant_change"], roles["relevant_change"]
            semantic_separation.append(
                float(negative["predicted_trigger"] == 0 and changed["predicted_trigger"] == 1)
            )
            semantic_joint.append(float(negative["joint_success"] and changed["joint_success"]))
        elif "temporal_before_long" in roles:
            # ``before_long`` is the near-boundary counterfactual.  ``before_short``
            # is near the initial time and is not the matched boundary contrast.
            before, crossing = roles["temporal_before_long"], roles["temporal_crossing"]
            temporal_boundary.append(
                float(before["predicted_trigger"] == 0 and crossing["predicted_trigger"] == 1)
            )
            temporal_boundary_joint.append(
                float(before["joint_success"] and crossing["joint_success"])
            )
        group_joint = float(all(x["joint_success"] for x in roles.values()))
        protocol_group_joint = float(
            all(x["protocol_joint_success"] for x in roles.values())
        )
        joint_group_success.append(group_joint)
        protocol_joint_group_success.append(protocol_group_joint)
        group_domain = str(next(iter(roles.values()))["domain"])
        joint_group_success_by_domain[group_domain].append(group_joint)
        if "inert_short" in roles:
            group_type = "threshold"
        elif "temporal_before_long" in roles:
            group_type = "temporal"
        else:
            group_type = "legacy_v0"
        joint_group_success_by_type[group_type].append(group_joint)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    grouped_records = [list(roles.values()) for roles in by_group.values()]
    trigger_hits = [x["trigger_correct"] for x in records]
    answer_hits = [x["answer_correct"] for x in records]
    strict_answer_hits = [x["strict_answer_correct"] for x in records]
    compatible_answer_hits = [x["time_compatible_answer_correct"] for x in records]
    think_lengths = [x["think_characters"] for x in records]

    record_mean = lambda field: lambda items: float(np.mean([x[field] for x in items]))
    ci95 = {
        "resampling_unit": "group_id",
        "bootstrap_reps": reps,
        "trigger_accuracy": bootstrap_clusters(
            grouped_records, record_mean("trigger_correct"), rng, reps
        ),
        "trigger_precision": bootstrap_clusters(
            grouped_records,
            lambda items: confusion_rate(items, "precision"),
            rng,
            reps,
        ),
        "trigger_recall": bootstrap_clusters(
            grouped_records, lambda items: confusion_rate(items, "recall"), rng, reps
        ),
        "trigger_f1": bootstrap_clusters(grouped_records, trigger_f1, rng, reps),
        "false_retrigger_rate": bootstrap_clusters(
            grouped_records,
            lambda items: confusion_rate(items, "false_retrigger"),
            rng,
            reps,
        ),
        "missed_state_change_rate": bootstrap_clusters(
            grouped_records,
            lambda items: confusion_rate(items, "missed_state_change"),
            rng,
            reps,
        ),
        "answer_accuracy": bootstrap_clusters(
            grouped_records, record_mean("answer_correct"), rng, reps
        ),
        "strict_answer_accuracy": bootstrap_clusters(
            grouped_records, record_mean("strict_answer_correct"), rng, reps
        ),
        "time_compatible_answer_accuracy": bootstrap_clusters(
            grouped_records,
            record_mean("time_compatible_answer_correct"),
            rng,
            reps,
        ),
        "protocol_answer_accuracy": bootstrap_clusters(
            grouped_records, record_mean("protocol_answer_correct"), rng, reps
        ),
        "format_error_rate": bootstrap_clusters(
            grouped_records,
            lambda items: float(
                np.mean([not x["time_compatible_format_ok"] for x in items])
            ),
            rng,
            reps,
        ),
        "time_compatible_format_error_rate": bootstrap_clusters(
            grouped_records,
            lambda items: float(
                np.mean([not x["time_compatible_format_ok"] for x in items])
            ),
            rng,
            reps,
        ),
        "protocol_format_error_rate": bootstrap_clusters(
            grouped_records,
            lambda items: float(np.mean([not x["protocol_format_ok"] for x in items])),
            rng,
            reps,
        ),
        "mean_think_characters": bootstrap_clusters(
            grouped_records, record_mean("think_characters"), rng, reps
        ),
        "time_consistency": bootstrap_mean(time_consistency, rng, reps),
        "time_joint_success": bootstrap_mean(time_joint, rng, reps),
        "assumption_break_time_consistency": bootstrap_mean(
            assumption_break_time_consistency, rng, reps
        ),
        "semantic_delta_separation": bootstrap_mean(semantic_separation, rng, reps),
        "semantic_delta_joint_success": bootstrap_mean(semantic_joint, rng, reps),
        "irrelevant_delta_separation": bootstrap_mean(irrelevant_separation, rng, reps),
        "temporal_boundary_sensitivity": bootstrap_mean(temporal_boundary, rng, reps),
        "temporal_boundary_joint_success": bootstrap_mean(
            temporal_boundary_joint, rng, reps
        ),
        "joint_group_success": bootstrap_mean(joint_group_success, rng, reps),
        "protocol_joint_group_success": bootstrap_mean(
            protocol_joint_group_success, rng, reps
        ),
    }

    joint_group_mean = mean_or_none(joint_group_success)
    summary = {
        "n": len(records),
        "n_groups": len(by_group),
        "n_complete_groups": len(complete_groups),
        "n_incomplete_groups": len(incomplete_groups),
        "incomplete_group_ids": sorted(incomplete_groups),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "trigger_precision": precision,
        "trigger_recall": recall,
        "trigger_f1": safe_div(2 * precision * recall, precision + recall),
        "trigger_accuracy": float(np.mean(trigger_hits)),
        "false_retrigger_rate": safe_div(fp, fp + tn),
        "missed_state_change_rate": safe_div(fn, fn + tp),
        "counterfactual_time_consistency": mean_or_none(time_consistency),
        "counterfactual_time_joint_success": mean_or_none(time_joint),
        "assumption_break_time_consistency": mean_or_none(
            assumption_break_time_consistency
        ),
        "semantic_delta_separation_accuracy": mean_or_none(semantic_separation),
        "semantic_delta_joint_success": mean_or_none(semantic_joint),
        "irrelevant_delta_separation_accuracy": mean_or_none(irrelevant_separation),
        "temporal_boundary_sensitivity": mean_or_none(temporal_boundary),
        "temporal_boundary_joint_success": mean_or_none(temporal_boundary_joint),
        "answer_accuracy": float(np.mean(answer_hits)),
        "time_compatible_answer_accuracy": float(np.mean(compatible_answer_hits)),
        "protocol_answer_accuracy": float(np.mean(strict_answer_hits)),
        "strict_answer_accuracy": float(np.mean(strict_answer_hits)),
        "joint_group_success": joint_group_mean,
        "protocol_joint_group_success": mean_or_none(protocol_joint_group_success),
        "joint_group_success_by_type": {
            group_type: {
                "n_groups": len(values),
                "value": mean_or_none(values),
                "ci95": bootstrap_mean(values, rng, reps),
            }
            for group_type, values in sorted(joint_group_success_by_type.items())
        },
        "joint_group_success_by_domain": {
            domain: {
                "n_groups": len(values),
                "value": mean_or_none(values),
                "ci95": bootstrap_mean(values, rng, reps),
            }
            for domain, values in sorted(joint_group_success_by_domain.items())
        },
        "paired_metric_group_counts": {
            "counterfactual_time_consistency": len(time_consistency),
            "assumption_break_time_consistency": len(
                assumption_break_time_consistency
            ),
            "semantic_delta_separation": len(semantic_separation),
            "irrelevant_delta_separation": len(irrelevant_separation),
            "temporal_boundary_sensitivity": len(temporal_boundary),
            "joint_group_success": len(joint_group_success),
        },
        # Backward-compatible alias; the unit has always been group, not domain.
        "joint_domain_success": joint_group_mean,
        "mean_think_characters": float(np.mean(think_lengths)),
        "p95_think_characters": float(np.quantile(think_lengths, 0.95)),
        "format_error_rate": float(np.mean([not x["format_ok"] for x in records])),
        "time_compatible_format_error_rate": float(
            np.mean([not x["time_compatible_format_ok"] for x in records])
        ),
        "protocol_format_error_rate": float(
            np.mean([not x["protocol_format_ok"] for x in records])
        ),
        "by_role": {
            role: {
                "n": len(items),
                "trigger_rate": float(np.mean([x["predicted_trigger"] for x in items])),
                "answer_accuracy": float(np.mean([x["answer_correct"] for x in items])),
                "strict_answer_accuracy": float(
                    np.mean([x["strict_answer_correct"] for x in items])
                ),
                "time_compatible_answer_accuracy": float(
                    np.mean([x["time_compatible_answer_correct"] for x in items])
                ),
                "protocol_answer_accuracy": float(
                    np.mean([x["protocol_answer_correct"] for x in items])
                ),
                "format_error_rate": float(
                    np.mean([not x["time_compatible_format_ok"] for x in items])
                ),
                "time_compatible_format_error_rate": float(
                    np.mean([not x["time_compatible_format_ok"] for x in items])
                ),
                "protocol_format_error_rate": float(
                    np.mean([not x["protocol_format_ok"] for x in items])
                ),
            }
            for role, items in sorted(by_role.items())
        },
        "by_domain": {
            domain: {
                "n": len(items),
                "n_groups": len({str(x.get("group_id") or x.get("pair_id")) for x in items}),
                "trigger_precision": confusion_rate(items, "precision"),
                "trigger_recall": confusion_rate(items, "recall"),
                "trigger_f1": trigger_f1(items),
                "false_retrigger_rate": confusion_rate(items, "false_retrigger"),
                "missed_state_change_rate": confusion_rate(items, "missed_state_change"),
                "answer_accuracy": float(np.mean([x["answer_correct"] for x in items])),
                "strict_answer_accuracy": float(
                    np.mean([x["strict_answer_correct"] for x in items])
                ),
                "time_compatible_answer_accuracy": float(
                    np.mean([x["time_compatible_answer_correct"] for x in items])
                ),
                "protocol_answer_accuracy": float(
                    np.mean([x["protocol_answer_correct"] for x in items])
                ),
                "format_error_rate": float(
                    np.mean([not x["time_compatible_format_ok"] for x in items])
                ),
                "time_compatible_format_error_rate": float(
                    np.mean([not x["time_compatible_format_ok"] for x in items])
                ),
                "protocol_format_error_rate": float(
                    np.mean([not x["protocol_format_ok"] for x in items])
                ),
                "mean_think_characters": float(
                    np.mean([x["think_characters"] for x in items])
                ),
            }
            for domain, items in sorted(by_domain.items())
        },
        "ci95": ci95,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--scored-output")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Permit partial runs (for monitoring only). Paired metrics and joint "
            "group success exclude incomplete groups."
        ),
    )
    args = parser.parse_args()
    records = json.loads(Path(args.input).read_text())
    summary = score(
        records,
        args.bootstrap_reps,
        args.seed,
        allow_incomplete=args.allow_incomplete,
    )
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if args.scored_output:
        Path(args.scored_output).write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
