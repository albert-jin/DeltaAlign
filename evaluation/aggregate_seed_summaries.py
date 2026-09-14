#!/usr/bin/env python3
"""Aggregate DeltaTIMEBench summary JSON files across random seeds.

Inputs may be literal paths or glob patterns.  Files that are currently being
written, malformed, incomplete, or missing required metrics are reported under
``skipped_files`` and do not abort the aggregation.  Configuration and seed are
inferred from filenames containing ``_seed<id>``.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_METRICS = (
    "trigger_f1",
    "false_retrigger_rate",
    "missed_state_change_rate",
    "answer_accuracy",
    "semantic_delta_joint_success",
    "temporal_boundary_joint_success",
    "joint_group_success",
    "mean_think_characters",
)

SEED_RE = re.compile(r"^(?P<prefix>.*)_seed(?P<seed>[^_]+)(?P<suffix>.*)$")
SUMMARY_SUFFIX = "_summary"


def expand_inputs(values: list[str]) -> tuple[list[Path], list[dict]]:
    """Expand paths/globs while retaining unmatched patterns as skip records."""
    paths: list[Path] = []
    skipped: list[dict] = []
    seen: set[str] = set()
    for value in values:
        matches = sorted(glob.glob(value))
        if not matches and Path(value).exists():
            matches = [value]
        if not matches:
            skipped.append({"path": value, "reason": "no_match"})
            continue
        for match in matches:
            path = Path(match)
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                paths.append(path)
    return paths, skipped


def infer_config_seed(path: Path) -> tuple[str, str] | None:
    stem = path.stem
    if stem.endswith(SUMMARY_SUFFIX):
        stem = stem[: -len(SUMMARY_SUFFIX)]
    match = SEED_RE.match(stem)
    if not match:
        return None
    config = match.group("prefix") + match.group("suffix")
    config = config.strip("_")
    if not config:
        return None
    return config, match.group("seed")


def finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def load_summary(path: Path, metrics: tuple[str, ...]) -> tuple[dict | None, str | None]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"unreadable_or_unfinished_json: {type(exc).__name__}: {exc}"
    if not isinstance(value, dict):
        return None, "summary_is_not_an_object"

    n_groups = value.get("n_groups")
    n_complete = value.get("n_complete_groups")
    n_incomplete = value.get("n_incomplete_groups", 0)
    if (finite_number(n_incomplete) and int(n_incomplete) > 0) or (
        finite_number(n_groups)
        and finite_number(n_complete)
        and int(n_complete) < int(n_groups)
    ):
        return None, (
            "incomplete_summary: "
            f"complete={n_complete}, groups={n_groups}, incomplete={n_incomplete}"
        )

    missing = [name for name in metrics if not finite_number(value.get(name))]
    if missing:
        return None, f"missing_or_nonfinite_metrics: {', '.join(missing)}"
    return value, None


def metric_stats(values: list[float]) -> dict:
    if not values:
        raise ValueError("metric_stats requires at least one value")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        # Across-seed reporting convention: sample SD.  A one-seed interim
        # aggregate uses 0.0 rather than NaN so the JSON remains portable.
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate_files(paths: list[Path], metrics: tuple[str, ...]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    skipped: list[dict] = []
    occupied: dict[tuple[str, str], dict] = {}

    for path in paths:
        identity = infer_config_seed(path)
        if identity is None:
            skipped.append({"path": str(path), "reason": "cannot_infer_config_or_seed"})
            continue
        config, seed = identity
        summary, reason = load_summary(path, metrics)
        if summary is None:
            skipped.append({"path": str(path), "reason": reason})
            continue
        row = {
            "seed": seed,
            "path": str(path.resolve()),
            "metrics": {name: float(summary[name]) for name in metrics},
        }
        key = (config, seed)
        prior = occupied.get(key)
        if prior is not None:
            if prior["metrics"] == row["metrics"]:
                reason = f"duplicate_config_seed_identical: config={config}, seed={seed}"
            else:
                reason = f"duplicate_config_seed_conflict: config={config}, seed={seed}"
            skipped.append({"path": str(path), "reason": reason})
            continue
        occupied[key] = row
        grouped[config].append(row)

    configs = {}
    for config, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (seed_sort_key(row["seed"]), row["path"]))
        configs[config] = {
            "n_seeds": len(rows),
            "seeds": [row["seed"] for row in rows],
            "per_seed": rows,
            "aggregate": {
                metric: metric_stats([row["metrics"][metric] for row in rows])
                for metric in metrics
            },
        }
    return {
        "schema_version": "deltatime-seed-aggregate-0.1",
        "metrics": list(metrics),
        "std_definition": "sample standard deviation (ddof=1); 0.0 when n=1",
        "n_loaded_files": sum(len(value["per_seed"]) for value in configs.values()),
        "n_skipped_files": len(skipped),
        "skipped_files": skipped,
        "configs": configs,
    }


def seed_sort_key(seed: str) -> tuple[int, int | str]:
    try:
        return 0, int(seed)
    except ValueError:
        return 1, seed


def markdown_report(result: dict) -> str:
    metrics = result["metrics"]
    lines = [
        "# DeltaTIMEBench multi-seed aggregate",
        "",
        f"Loaded files: {result['n_loaded_files']}; skipped files: {result['n_skipped_files']}.",
        "",
    ]
    for config, value in result["configs"].items():
        lines.extend(
            [
                f"## {config}",
                "",
                f"Seeds: {', '.join(value['seeds'])}",
                "",
                "### Aggregate",
                "",
                "| Metric | N | Mean | Std | Min | Max |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for metric in metrics:
            stats = value["aggregate"][metric]
            lines.append(
                f"| {metric} | {stats['n']} | {stats['mean']:.6g} | "
                f"{stats['std']:.6g} | {stats['min']:.6g} | {stats['max']:.6g} |"
            )
        lines.extend(
            [
                "",
                "### Per seed",
                "",
                "| Seed | " + " | ".join(metrics) + " | Source |",
                "|---:|" + "---:|" * len(metrics) + "---|",
            ]
        )
        for row in value["per_seed"]:
            metric_values = " | ".join(
                f"{row['metrics'][metric]:.6g}" for metric in metrics
            )
            lines.append(
                f"| {row['seed']} | {metric_values} | `{row['path']}` |"
            )
        lines.append("")

    if result["skipped_files"]:
        lines.extend(["## Skipped inputs", "", "| Path | Reason |", "|---|---|"])
        for item in result["skipped_files"]:
            lines.append(f"| `{item['path']}` | {item['reason']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Summary JSON paths or glob patterns; quote globs to tolerate unfinished runs.",
    )
    parser.add_argument("--output", required=True, help="Machine-readable aggregate JSON.")
    parser.add_argument("--markdown", help="Optional Markdown aggregate and per-seed tables.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Summary fields to aggregate.",
    )
    args = parser.parse_args()
    metrics = tuple(dict.fromkeys(args.metrics))
    paths, expansion_skips = expand_inputs(args.inputs)
    result = aggregate_files(paths, metrics)
    result["skipped_files"] = expansion_skips + result["skipped_files"]
    result["n_skipped_files"] = len(result["skipped_files"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if args.markdown:
        markdown = Path(args.markdown)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(markdown_report(result) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "markdown": args.markdown,
                "configs": {k: v["seeds"] for k, v in result["configs"].items()},
                "n_loaded_files": result["n_loaded_files"],
                "n_skipped_files": result["n_skipped_files"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
