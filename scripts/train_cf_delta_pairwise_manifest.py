#!/usr/bin/env python3
"""Run the existing PairCal objective from an explicit endpoint manifest.

The audited trainer's only data-shaping limitation is its hard-coded
``make_pairs`` function, which rejects leave-one-role-out record files.  This
wrapper validates and reconstructs the already selected matched pairs, then
delegates tokenization, loss computation, optimization, and artifact writing
to the unchanged ``train_cf_delta_pairwise.py`` implementation.

Importing this wrapper is CPU-safe.  The GPU training stack is imported only
inside ``main`` after CLI validation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


PAIR_FIELDS = {
    "positive_group_id",
    "negative_group_id",
    "domain",
    "positive_role",
    "negative_role",
    "positive_scenario_id",
    "negative_scenario_id",
    "positive_peer_group_id",
    "positive_peer_domain",
    "positive_peer_role",
    "positive_peer_scenario_id",
    "positive_consistency_mask",
}
def run_gpu_policy_gate(
    policy_script: Path,
    policy_python: str,
    expected_additional_gib: float,
) -> None:
    """Run an optional site-specific GPU preflight before importing the model."""
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not gpu or "," in gpu:
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES must name exactly one physical GPU when "
            "--gpu-policy-script is used"
        )
    subprocess.run(
        [
            policy_python,
            str(policy_script),
            "--expected-additional-gib",
            str(expected_additional_gib),
        ],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
    )


def load_pairs(records: list[dict], manifest: list[dict]) -> list[dict]:
    """Validate schema-v2 endpoint rows and restore the trainer pair objects."""
    by_id: dict[str, dict] = {}
    for record in records:
        scenario_id = record.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("every record must have a nonempty scenario_id")
        if scenario_id in by_id:
            raise ValueError(f"duplicate scenario_id {scenario_id!r}")
        by_id[scenario_id] = record

    pairs: list[dict] = []
    seen_rows: set[tuple[str, str, str]] = set()
    for index, row in enumerate(manifest):
        if set(row) != PAIR_FIELDS:
            raise ValueError(
                f"pair row {index} fields differ from schema v2: "
                f"missing={sorted(PAIR_FIELDS - set(row))}, "
                f"unexpected={sorted(set(row) - PAIR_FIELDS)}"
            )
        ids = (
            row["positive_scenario_id"],
            row["negative_scenario_id"],
            row["positive_peer_scenario_id"],
        )
        missing_ids = [scenario_id for scenario_id in ids if scenario_id not in by_id]
        if missing_ids:
            raise ValueError(f"pair row {index} references absent records: {missing_ids}")
        positive, negative, positive_peer = (by_id[scenario_id] for scenario_id in ids)
        expected = {
            "positive_group_id": positive["group_id"],
            "negative_group_id": negative["group_id"],
            "domain": positive["domain"],
            "positive_role": positive["pair_role"],
            "negative_role": negative["pair_role"],
            "positive_peer_group_id": positive_peer["group_id"],
            "positive_peer_domain": positive_peer["domain"],
            "positive_peer_role": positive_peer["pair_role"],
        }
        disagreements = {
            key: (row[key], value)
            for key, value in expected.items()
            if row[key] != value
        }
        if disagreements:
            raise ValueError(f"pair row {index} endpoint metadata mismatch: {disagreements}")
        if positive["domain"] != negative["domain"]:
            raise ValueError(f"pair row {index} crosses domains")
        if not isinstance(row["positive_consistency_mask"], bool):
            raise ValueError(f"pair row {index} consistency mask is not boolean")
        if row["positive_consistency_mask"] and (
            positive["pair_role"] != "assumption_break_short"
            or positive_peer["pair_role"] != "assumption_break_long"
        ):
            raise ValueError(f"pair row {index} has an invalid active consistency mapping")
        identity = ids
        if identity in seen_rows:
            raise ValueError(f"duplicate pair endpoint row at index {index}: {identity}")
        seen_rows.add(identity)
        pairs.append(
            {
                "group_id": positive["group_id"],
                "positive": positive,
                "positive_peer": positive_peer,
                "positive_consistency_mask": row["positive_consistency_mask"],
                "negative": negative,
            }
        )
    if not pairs:
        raise ValueError("pair manifest is empty; the PairCal objective is undefined")
    return pairs


def wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument(
        "--gpu-policy-script",
        type=Path,
        help="Optional site-specific GPU preflight script to run before training.",
    )
    parser.add_argument("--gpu-policy-python", default=sys.executable)
    parser.add_argument("--expected-additional-gib", type=float, default=16.0)
    cfg, remaining = parser.parse_known_args(argv)
    cfg.pair_manifest = cfg.pair_manifest.resolve()
    if cfg.gpu_policy_script is not None:
        cfg.gpu_policy_script = cfg.gpu_policy_script.resolve()
    return cfg, remaining


def main() -> None:
    wrapper_cfg, trainer_argv = wrapper_args(sys.argv[1:])
    pair_manifest_path = wrapper_cfg.pair_manifest
    manifest = json.loads(pair_manifest_path.read_text())
    if not isinstance(manifest, list):
        raise ValueError("--pair-manifest must contain a JSON list")
    if not manifest:
        raise ValueError("pair manifest is empty; the PairCal objective is undefined")

    if wrapper_cfg.gpu_policy_script is not None:
        run_gpu_policy_gate(
            wrapper_cfg.gpu_policy_script,
            wrapper_cfg.gpu_policy_python,
            wrapper_cfg.expected_additional_gib,
        )

    trainer_path = Path(__file__).with_name("train_cf_delta_pairwise.py")
    spec = importlib.util.spec_from_file_location(
        "train_cf_delta_pairwise_base", trainer_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import audited trainer from {trainer_path}")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)

    original_get_args = base.get_args

    def get_args(argv: list[str] | None = None):
        cfg = original_get_args(trainer_argv if argv is None else argv)
        cfg.input_pair_manifest = str(pair_manifest_path)
        return cfg

    def make_pairs(records: list[dict]) -> list[dict]:
        return load_pairs(records, manifest)

    base.get_args = get_args
    base.make_pairs = make_pairs
    base.main()


if __name__ == "__main__":
    main()
