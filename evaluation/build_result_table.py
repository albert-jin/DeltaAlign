#!/usr/bin/env python3
"""Build a compact Markdown/CSV table from DeltaTIMEBench summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "trigger_precision", "trigger_recall", "trigger_f1", "false_retrigger_rate",
    "missed_state_change_rate", "counterfactual_time_consistency",
    "assumption_break_time_consistency",
    "semantic_delta_separation_accuracy", "temporal_boundary_sensitivity",
    "answer_accuracy", "time_compatible_answer_accuracy", "protocol_answer_accuracy",
    "joint_group_success", "protocol_joint_group_success", "mean_think_characters",
    "time_compatible_format_error_rate", "protocol_format_error_rate",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    rows = []
    for value in args.summaries:
        path = Path(value)
        data = json.loads(path.read_text())
        row = {"model": path.name.removesuffix("_summary.json")}
        row.update({field: data.get(field) for field in FIELDS})
        rows.append(row)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", *FIELDS])
        writer.writeheader()
        writer.writerows(rows)
    header = ["model", "Trig F1", "FTR", "Miss", "Inert time", "Break time", "Semantic pair", "Time boundary", "Answer", "TIME-compatible answer", "Protocol answer", "Group joint", "Protocol group joint", "TIME format err", "Protocol format err", "Think chars"]
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        def pct(field: str) -> str:
            value = row.get(field)
            return "—" if value is None else f"{100 * value:.1f}"
        md.append("| " + " | ".join([
            row["model"], pct("trigger_f1"), pct("false_retrigger_rate"),
            pct("missed_state_change_rate"), pct("counterfactual_time_consistency"),
            pct("assumption_break_time_consistency"),
            pct("semantic_delta_separation_accuracy"), pct("temporal_boundary_sensitivity"),
            pct("answer_accuracy"), pct("time_compatible_answer_accuracy"),
            pct("protocol_answer_accuracy"), pct("joint_group_success"),
            pct("protocol_joint_group_success"),
            pct("time_compatible_format_error_rate"), pct("protocol_format_error_rate"),
            f"{row['mean_think_characters']:.1f}",
        ]) + " |")
    Path(args.markdown).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
