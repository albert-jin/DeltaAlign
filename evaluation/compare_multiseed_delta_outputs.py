#!/usr/bin/env python3
"""Paper-oriented paired comparison across model-training seeds.

The DeltaTIMEBench rows are *not* independent replicates.  Counterfactual
siblings are clustered by ``group_id``, and outputs from different training
seeds are evaluated on the same test groups.  This script therefore reports
three deliberately separate views of uncertainty:

1. Per-seed model metrics and paired effects (B - A).
2. Across-seed mean, sample SD, paired t interval, and an exact seed-level
   sign-flip test.  Here the independent replication unit is the training
   seed, not a record.
3. Paired group-cluster percentile bootstrap intervals, with shared group IDs
   resampled synchronously across seeds.  One interval is conditional on the
   observed seeds; a crossed seed x group bootstrap also resamples seeds and
   is supplied as a sensitivity analysis.

Models A and B always remain paired within both seed and group.  Metrics are
computed within each resampled seed and then averaged with equal seed weight;
the script never treats ``number of records x number of seeds`` as an
independent sample size.  With only three seeds, all seed-generalization
inference should be described as low-powered.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

import compare_delta_outputs
import score_delta_outputs
from compare_delta_outputs import (
    LOWER_IS_BETTER,
    align_groups,
    differences,
    load_and_score,
)


DEFAULT_METRICS = (
    "trigger_f1",
    "false_retrigger_rate",
    "missed_state_change_rate",
    "answer_accuracy",
    "semantic_delta_separation_accuracy",
    "irrelevant_delta_separation_accuracy",
    "temporal_boundary_sensitivity",
    "joint_group_success",
    "mean_think_characters",
    "format_error_rate",
)


@dataclass(frozen=True)
class PairData:
    seed: str
    file_a: Path
    file_b: Path
    sha256_a: str
    sha256_b: str
    n_records: int
    groups_by_id: dict[str, dict[str, dict[str, dict]]]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_sort_key(seed: str) -> tuple[int, int | str]:
    try:
        return 0, int(seed)
    except ValueError:
        return 1, seed


def group_id_for_pair(pair: dict[str, dict[str, dict]]) -> str:
    group_ids = {
        str(item["group_id"])
        for side in ("a", "b")
        for item in pair[side].values()
    }
    if len(group_ids) != 1:
        raise ValueError(f"aligned pair contains multiple group IDs: {sorted(group_ids)}")
    return next(iter(group_ids))


def index_groups(
    groups: list[dict[str, dict[str, dict]]],
) -> dict[str, dict[str, dict[str, dict]]]:
    indexed: dict[str, dict[str, dict[str, dict]]] = {}
    for pair in groups:
        group_id = group_id_for_pair(pair)
        if group_id in indexed:
            raise ValueError(f"duplicate aligned group_id: {group_id}")
        indexed[group_id] = pair
    return indexed


def make_pair_data(
    *,
    seed: str,
    records_a: list[dict],
    records_b: list[dict],
    file_a: Path = Path("model_a.json"),
    file_b: Path = Path("model_b.json"),
    sha256_a: str = "synthetic",
    sha256_b: str = "synthetic",
) -> PairData:
    """Construct validated pair data; exposed for CPU-only synthetic tests."""
    groups = align_groups(records_a, records_b)
    return PairData(
        seed=str(seed),
        file_a=file_a,
        file_b=file_b,
        sha256_a=sha256_a,
        sha256_b=sha256_b,
        n_records=len(records_a),
        groups_by_id=index_groups(groups),
    )


def load_pair(seed: str, file_a: Path, file_b: Path) -> PairData:
    records_a = load_and_score(file_a)
    records_b = load_and_score(file_b)
    return make_pair_data(
        seed=seed,
        records_a=records_a,
        records_b=records_b,
        file_a=file_a.resolve(),
        file_b=file_b.resolve(),
        sha256_a=file_sha256(file_a),
        sha256_b=file_sha256(file_b),
    )


def metric_stats(values: list[float]) -> dict:
    if not values:
        raise ValueError("metric_stats requires at least one value")
    return {
        "n_seeds": len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def paired_seed_inference(effects: list[float]) -> dict:
    """Inference whose sole independent unit is the model-training seed."""
    n_seeds = len(effects)
    mean_effect = statistics.fmean(effects)
    result = {
        **metric_stats(effects),
        "standard_error": None,
        "ci95_paired_t": None,
        "exact_sign_flip_p_two_sided": None,
        "sign_flip_enumerations": None,
    }
    if n_seeds < 2:
        return result

    sample_sd = statistics.stdev(effects)
    standard_error = sample_sd / math.sqrt(n_seeds)
    t_critical = float(scipy_stats.t.ppf(0.975, df=n_seeds - 1))
    result["standard_error"] = standard_error
    result["ci95_paired_t"] = [
        mean_effect - t_critical * standard_error,
        mean_effect + t_critical * standard_error,
    ]

    # Enumerating 2^n signs is exact and transparent for the intended n=3.
    # Avoid an accidental exponential run if the utility is reused at scale.
    if n_seeds <= 20:
        null_effects = [
            statistics.fmean(sign * effect for sign, effect in zip(signs, effects))
            for signs in itertools.product((-1.0, 1.0), repeat=n_seeds)
        ]
        tolerance = 1e-15
        result["exact_sign_flip_p_two_sided"] = sum(
            abs(value) >= abs(mean_effect) - tolerance for value in null_effects
        ) / len(null_effects)
        result["sign_flip_enumerations"] = len(null_effects)
    return result


def holm_adjust(p_values: dict[str, float | None]) -> dict[str, float | None]:
    usable = {key: value for key, value in p_values.items() if value is not None}
    ordered = sorted(usable, key=usable.get)
    adjusted: dict[str, float | None] = {key: None for key in p_values}
    running = 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (count - rank) * float(usable[key]))
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def validate_crossed_design(pairs: list[PairData]) -> list[str]:
    if not pairs:
        raise ValueError("at least one complete seed pair is required")
    seeds = [pair.seed for pair in pairs]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate seeds are not allowed: {seeds}")
    reference = set(pairs[0].groups_by_id)
    for pair in pairs[1:]:
        current = set(pair.groups_by_id)
        if current != reference:
            missing = sorted(reference - current)[:5]
            extra = sorted(current - reference)[:5]
            raise ValueError(
                f"group_id sets differ for seed {pair.seed}: "
                f"missing={missing}, extra={extra}"
            )
        for group_id in reference:
            reference_roles = set(pairs[0].groups_by_id[group_id]["a"])
            current_roles = set(pair.groups_by_id[group_id]["a"])
            if current_roles != reference_roles:
                raise ValueError(
                    f"role sets differ for seed={pair.seed}, group_id={group_id}: "
                    f"reference={sorted(reference_roles)}, current={sorted(current_roles)}"
                )
            # A shared group_id must denote the same benchmark cluster in every
            # seed. Otherwise a synchronous group bootstrap would silently pair
            # unlike scenarios across seeds.
            metadata = (
                "scenario_id",
                "group_id",
                "domain",
                "pair_role",
                "oracle_trigger",
                "expected_answer",
            )
            for role in reference_roles:
                reference_item = pairs[0].groups_by_id[group_id]["a"][role]
                current_item = pair.groups_by_id[group_id]["a"][role]
                mismatch = {
                    key: (reference_item.get(key), current_item.get(key))
                    for key in metadata
                    if reference_item.get(key) != current_item.get(key)
                }
                if mismatch:
                    raise ValueError(
                        f"benchmark metadata differs for seed={pair.seed}, "
                        f"group_id={group_id}, role={role}: {mismatch}"
                    )
    return sorted(reference)


def summarize_pair_sample(
    pair: PairData,
    group_ids: list[str],
) -> tuple[dict, dict, dict]:
    groups = [pair.groups_by_id[group_id] for group_id in group_ids]
    return differences(groups)


def bootstrap_effects(
    pairs: list[PairData],
    *,
    metrics: tuple[str, ...],
    group_ids: list[str],
    reps: int,
    rng: np.random.Generator,
    resample_seeds: bool,
) -> dict[str, np.ndarray]:
    """Crossed bootstrap with paired labels and shared group draws.

    ``group_id`` is crossed with seed because every training seed is evaluated
    on the same benchmark scenarios.  A single group-index draw is therefore
    applied synchronously to every sampled seed, preserving cross-seed
    dependence induced by test-group difficulty.
    """
    n_seeds = len(pairs)
    n_groups = len(group_ids)
    samples = {metric: np.empty(reps, dtype=np.float64) for metric in metrics}
    for rep in range(reps):
        sampled_group_indices = rng.integers(0, n_groups, size=n_groups)
        sampled_group_ids = [group_ids[int(index)] for index in sampled_group_indices]
        if resample_seeds:
            sampled_seed_indices = rng.integers(0, n_seeds, size=n_seeds)
        else:
            sampled_seed_indices = np.arange(n_seeds)
        per_seed_effects = {metric: [] for metric in metrics}
        for seed_index in sampled_seed_indices:
            _, _, effect = summarize_pair_sample(
                pairs[int(seed_index)], sampled_group_ids
            )
            for metric in metrics:
                per_seed_effects[metric].append(float(effect[metric]))
        for metric in metrics:
            samples[metric][rep] = statistics.fmean(per_seed_effects[metric])
    return samples


def interval_summary(values: np.ndarray, preferred_direction: str) -> dict:
    lower, upper = np.quantile(values, [0.025, 0.975])
    improvement_values = -values if preferred_direction == "lower" else values
    return {
        "ci95_percentile": [float(lower), float(upper)],
        "probability_b_better": float(np.mean(improvement_values > 0.0)),
    }


def analyze_pairs(
    pairs: list[PairData],
    *,
    expected_seeds: list[str],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    bootstrap_reps: int = 10000,
    bootstrap_seed: int = 3407,
) -> dict:
    if bootstrap_reps < 1:
        raise ValueError("bootstrap_reps must be positive")
    pairs = sorted(pairs, key=lambda pair: seed_sort_key(pair.seed))
    group_ids = validate_crossed_design(pairs)
    expected_seeds = list(dict.fromkeys(str(seed) for seed in expected_seeds))
    observed_seeds = [pair.seed for pair in pairs]
    missing_seeds = [seed for seed in expected_seeds if seed not in observed_seeds]
    unexpected_seeds = [seed for seed in observed_seeds if seed not in expected_seeds]

    per_seed_raw: dict[str, tuple[dict, dict, dict]] = {}
    for pair in pairs:
        per_seed_raw[pair.seed] = summarize_pair_sample(pair, group_ids)
    available_metrics = set(next(iter(per_seed_raw.values()))[2])
    missing_metrics = [metric for metric in metrics if metric not in available_metrics]
    if missing_metrics:
        raise ValueError(f"unknown or unavailable metrics: {missing_metrics}")
    for seed, (_, _, effect) in per_seed_raw.items():
        nonfinite = [
            metric
            for metric in metrics
            if effect[metric] is None or not math.isfinite(float(effect[metric]))
        ]
        if nonfinite:
            raise ValueError(f"non-finite metrics for seed {seed}: {nonfinite}")

    seed_sequence = np.random.SeedSequence(bootstrap_seed)
    conditional_rng, crossed_rng = [
        np.random.default_rng(sequence) for sequence in seed_sequence.spawn(2)
    ]
    conditional_bootstrap = bootstrap_effects(
        pairs,
        metrics=metrics,
        group_ids=group_ids,
        reps=bootstrap_reps,
        rng=conditional_rng,
        resample_seeds=False,
    )
    crossed_bootstrap = bootstrap_effects(
        pairs,
        metrics=metrics,
        group_ids=group_ids,
        reps=bootstrap_reps,
        rng=crossed_rng,
        resample_seeds=True,
    )

    per_seed = []
    for pair in pairs:
        model_a, model_b, effect = per_seed_raw[pair.seed]
        per_seed.append(
            {
                "seed": pair.seed,
                "n_records_per_model": pair.n_records,
                "n_group_clusters": len(pair.groups_by_id),
                "model_a": {metric: float(model_a[metric]) for metric in metrics},
                "model_b": {metric: float(model_b[metric]) for metric in metrics},
                "effect_b_minus_a": {
                    metric: float(effect[metric]) for metric in metrics
                },
                "files": {
                    "model_a": {"path": str(pair.file_a), "sha256": pair.sha256_a},
                    "model_b": {"path": str(pair.file_b), "sha256": pair.sha256_b},
                },
            }
        )

    metric_results = {}
    seed_p_values: dict[str, float | None] = {}
    for metric in metrics:
        model_a_values = [float(per_seed_raw[pair.seed][0][metric]) for pair in pairs]
        model_b_values = [float(per_seed_raw[pair.seed][1][metric]) for pair in pairs]
        effects = [float(per_seed_raw[pair.seed][2][metric]) for pair in pairs]
        preferred_direction = "lower" if metric in LOWER_IS_BETTER else "higher"
        paired = paired_seed_inference(effects)
        seed_p_values[metric] = paired["exact_sign_flip_p_two_sided"]
        metric_results[metric] = {
            "preferred_direction": preferred_direction,
            "model_a_across_seeds": metric_stats(model_a_values),
            "model_b_across_seeds": metric_stats(model_b_values),
            "paired_seed_effect_b_minus_a": paired,
            "paired_group_cluster_bootstrap_conditional_on_observed_seeds": {
                "estimand": "equal-seed-weighted mean of per-seed B-A effects",
                **interval_summary(conditional_bootstrap[metric], preferred_direction),
            },
            "crossed_seed_by_group_bootstrap_sensitivity": {
                "estimand": "equal-seed-weighted mean of per-seed B-A effects",
                **interval_summary(crossed_bootstrap[metric], preferred_direction),
            },
        }

    adjusted = holm_adjust(seed_p_values)
    for metric in metrics:
        metric_results[metric]["paired_seed_effect_b_minus_a"][
            "holm_adjusted_exact_sign_flip_p"
        ] = adjusted[metric]

    complete = not missing_seeds and not unexpected_seeds and set(observed_seeds) == set(expected_seeds)
    return {
        "schema_version": "deltatime-multiseed-paired-1.0",
        "status": "complete" if complete else "partial",
        "inference_ready": complete and len(pairs) >= 3,
        "expected_seeds": expected_seeds,
        "observed_complete_seeds": observed_seeds,
        "missing_seeds": missing_seeds,
        "unexpected_seeds": unexpected_seeds,
        "analysis": {
            "estimand": "model_b_minus_model_a, computed per seed then equally averaged",
            "independent_replication_unit_for_seed_inference": "model-training seed",
            "test_cluster_unit": "group_id (all sibling records move together)",
            "seed_group_relation": "crossed: each seed is evaluated on the same group_ids",
            "record_level_independence_assumed": False,
            "n_independent_seed_replicates": len(pairs),
            "n_shared_group_clusters": len(group_ids),
            "n_records_per_seed_per_model": {pair.seed: pair.n_records for pair in pairs},
            "across_seed_sd": "sample standard deviation (ddof=1); null when n=1",
            "paired_seed_ci": "two-sided 95% Student-t interval over per-seed paired effects",
            "paired_seed_test": (
                "exact two-sided sign-flip over per-seed paired effects; Holm correction "
                "across the requested metric family"
            ),
            "conditional_group_ci": (
                "paired percentile bootstrap of shared group_id clusters, resampled "
                "synchronously across seeds; conditional on the observed training seeds"
            ),
            "crossed_bootstrap_ci": (
                "sensitivity analysis independently resampling training seeds and shared "
                "group_id clusters, while retaining A/B pairing; only three seed clusters"
            ),
            "bootstrap_reps": bootstrap_reps,
            "bootstrap_seed": bootstrap_seed,
            "small_n_warning": (
                "Three training seeds provide low-resolution sign-flip p-values and "
                "unstable seed-generalization intervals; group bootstrap intervals do "
                "not turn benchmark records into independent training replicates."
            ),
        },
        "metrics": list(metrics),
        "per_seed": per_seed,
        "aggregate": metric_results,
    }


def load_requested_pairs(
    specs: list[list[str]],
) -> tuple[list[PairData], list[dict]]:
    pairs: list[PairData] = []
    unavailable: list[dict] = []
    seen: set[str] = set()
    for seed, raw_a, raw_b in specs:
        seed = str(seed)
        if seed in seen:
            raise ValueError(f"duplicate --pair seed: {seed}")
        seen.add(seed)
        file_a, file_b = Path(raw_a), Path(raw_b)
        missing = [
            str(path.resolve())
            for path in (file_a, file_b)
            if not path.is_file()
        ]
        if missing:
            unavailable.append(
                {"seed": seed, "reason": "missing_file", "missing_paths": missing}
            )
            continue
        try:
            pairs.append(load_pair(seed, file_a, file_b))
        except Exception as exc:
            unavailable.append(
                {
                    "seed": seed,
                    "reason": "load_or_validation_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "file_a": str(file_a.resolve()),
                    "file_b": str(file_b.resolve()),
                }
            )
    return pairs, unavailable


def fmt_number(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "NA"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def fmt_ci(value: list[float] | None) -> str:
    if value is None:
        return "NA"
    return f"[{value[0]:+.4f}, {value[1]:+.4f}]"


def markdown_report(result: dict) -> str:
    analysis = result["analysis"]
    lines = [
        f"# {result['comparison_id']}",
        "",
        f"Status: **{result['status']}**; inference ready: **{str(result['inference_ready']).lower()}**.",
        "",
        f"Model A: `{result['model_a_label']}`. Model B: `{result['model_b_label']}`. "
        "Every effect is B - A.",
        "",
        (
            f"Independent training replicates: {analysis['n_independent_seed_replicates']} "
            f"seeds ({', '.join(result['observed_complete_seeds'])}); shared benchmark "
            f"clusters: {analysis['n_shared_group_clusters']} `group_id`s. Individual "
            "records are not treated as independent replicates."
        ),
        "",
    ]
    if result["missing_seeds"]:
        lines.extend(
            [
                f"Missing expected seeds: {', '.join(result['missing_seeds'])}. "
                "All one-seed statistics below are interim/descriptive only.",
                "",
            ]
        )

    lines.extend(
        [
            "## Per-seed results",
            "",
            "| Metric | Seed | A | B | B - A |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in result["metrics"]:
        for row in result["per_seed"]:
            lines.append(
                f"| {metric} | {row['seed']} | {row['model_a'][metric]:.4f} | "
                f"{row['model_b'][metric]:.4f} | "
                f"{row['effect_b_minus_a'][metric]:+.4f} |"
            )

    lines.extend(
        [
            "",
            "## Across-seed paired summary",
            "",
            (
                "The first CI uses seed as the sole independent replication unit. The "
                "conditional group CI resamples shared `group_id` clusters synchronously "
                "across the observed seeds. The crossed sensitivity CI resamples both "
                "seeds and shared groups."
            ),
            "",
            "| Metric | A mean ± sample SD | B mean ± sample SD | Paired B-A mean ± sample SD | Seed t 95% CI | Conditional group-cluster 95% CI | Crossed seed×group 95% CI | Raw seed p | Holm seed p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in result["metrics"]:
        item = result["aggregate"][metric]
        a = item["model_a_across_seeds"]
        b = item["model_b_across_seeds"]
        paired = item["paired_seed_effect_b_minus_a"]
        conditional = item[
            "paired_group_cluster_bootstrap_conditional_on_observed_seeds"
        ]["ci95_percentile"]
        crossed = item["crossed_seed_by_group_bootstrap_sensitivity"][
            "ci95_percentile"
        ]
        lines.append(
            f"| {metric} | {fmt_number(a['mean'])} ± {fmt_number(a['sample_sd'])} | "
            f"{fmt_number(b['mean'])} ± {fmt_number(b['sample_sd'])} | "
            f"{fmt_number(paired['mean'], signed=True)} ± "
            f"{fmt_number(paired['sample_sd'])} | {fmt_ci(paired['ci95_paired_t'])} | "
            f"{fmt_ci(conditional)} | {fmt_ci(crossed)} | "
            f"{fmt_number(paired['exact_sign_flip_p_two_sided'])} | "
            f"{fmt_number(paired['holm_adjusted_exact_sign_flip_p'])} |"
        )

    lines.extend(
        [
            "",
            "## Statistical interpretation",
            "",
            f"- Seed-level estimand: {analysis['estimand']}.",
            f"- Conditional group CI: {analysis['conditional_group_ci']}.",
            f"- Crossed sensitivity CI: {analysis['crossed_bootstrap_ci']}.",
            f"- Caution: {analysis['small_n_warning']}",
            "",
        ]
    )
    if result.get("unavailable_pairs"):
        lines.extend(["## Unavailable requested pairs", ""])
        for item in result["unavailable_pairs"]:
            lines.append(f"- Seed {item['seed']}: `{item['reason']}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        required=True,
        metavar=("SEED", "FILE_A", "FILE_B"),
        help="Repeat once per seed. Missing paths are recorded, allowing safe reruns.",
    )
    parser.add_argument(
        "--expected-seeds",
        nargs="+",
        default=["3407", "1234", "2025"],
    )
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=3407)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Write the auditable partial result, then exit nonzero if any pair is unavailable.",
    )
    args = parser.parse_args()
    metrics = tuple(dict.fromkeys(args.metrics))

    pairs, unavailable = load_requested_pairs(args.pair)
    if not pairs:
        detail = "; ".join(f"seed {x['seed']}: {x['reason']}" for x in unavailable)
        parser.error(f"no complete seed pairs could be loaded ({detail})")
    result = analyze_pairs(
        pairs,
        expected_seeds=[str(seed) for seed in args.expected_seeds],
        metrics=metrics,
        bootstrap_reps=args.bootstrap_reps,
        bootstrap_seed=args.bootstrap_seed,
    )
    result.update(
        {
            "comparison_id": args.comparison_id,
            "model_a_label": args.label_a,
            "model_b_label": args.label_b,
            "unavailable_pairs": unavailable,
            "software": {
                "script": str(Path(__file__).resolve()),
                "script_sha256": file_sha256(Path(__file__).resolve()),
                "single_seed_comparison_module": str(
                    Path(compare_delta_outputs.__file__).resolve()
                ),
                "single_seed_comparison_module_sha256": file_sha256(
                    Path(compare_delta_outputs.__file__).resolve()
                ),
                "scorer": str(Path(score_delta_outputs.__file__).resolve()),
                "scorer_sha256": file_sha256(Path(score_delta_outputs.__file__).resolve()),
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "scipy_version": scipy_stats.__version__
                if hasattr(scipy_stats, "__version__")
                else __import__("scipy").__version__,
            },
        }
    )
    # A load error is also a missing expected pair even if a different path for
    # that seed happened to be supplied elsewhere (duplicate seeds are rejected).
    unavailable_expected = {
        item["seed"] for item in unavailable if item["seed"] in result["expected_seeds"]
    }
    if unavailable_expected:
        result["status"] = "partial"
        result["inference_ready"] = False

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    markdown_path = args.markdown or args.output.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown_report(result) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "markdown": str(markdown_path),
                "status": result["status"],
                "inference_ready": result["inference_ready"],
                "observed_complete_seeds": result["observed_complete_seeds"],
                "missing_seeds": result["missing_seeds"],
                "unavailable_pairs": unavailable,
            },
            indent=2,
        )
    )
    if args.require_complete and result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
