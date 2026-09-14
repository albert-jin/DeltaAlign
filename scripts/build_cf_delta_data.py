#!/usr/bin/env python3
"""Build rule-grounded paired-intervention data for CF-DeltaTIME Phase 5."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


SYSTEM_PROMPTS = [
    (
        "Follow the conversation and answer the latest question. You may use a <think> "
        "block when useful. End with exactly one <answer>VALUE</answer>."
    ),
    (
        "Give the correct current decision from the dialogue. Optional reasoning uses "
        "<think>...</think>; the final decision must be <answer>VALUE</answer>."
    ),
    (
        "Respond to the last user turn and finish with one machine-readable decision in "
        "the exact form <answer>VALUE</answer>."
    ),
]


@dataclass(frozen=True)
class Domain:
    name: str
    noun: str
    unit: str
    low_label: str
    high_label: str
    initial_templates: tuple[str, ...]
    update_templates: tuple[str, ...]
    irrelevant_templates: tuple[str, ...]


DOMAINS = (
    Domain(
        "server_capacity", "request load", "requests/s", "A", "B",
        (
            "Server A supports up to {threshold} requests/s and Server B handles higher load. Current load is {initial}. Use the smallest sufficient server: A or B?",
            "Choose server A when load is at most {threshold}; otherwise choose B. Monitoring shows {initial} requests/s. Which server?",
        ),
        ("Monitoring now reports {value} requests/s. Choose A or B.", "The request load changed to {value} requests/s. Which server is required?"),
        ("The dashboard logo changed, while the request load remains {initial}. Choose A or B.", "The service was renamed, but load is still {initial} requests/s. Choose A or B."),
    ),
    Domain(
        "freezer_status", "temperature", "C", "SAFE", "ALERT",
        (
            "A freezer is SAFE at or below {threshold}C and ALERT above it. The reading is {initial}C. What is the status?",
            "Policy: readings no higher than {threshold}C are SAFE; warmer readings are ALERT. Current reading: {initial}C. Status?",
        ),
        ("The new temperature reading is {value}C. Status: SAFE or ALERT?", "The sensor now reports {value}C. What is the status?"),
        ("The sensor display theme changed; the reading is still {initial}C. Status?", "The freezer was relabeled, with temperature unchanged at {initial}C. Status?"),
    ),
    Domain(
        "inventory_reorder", "inventory", "units", "REORDER", "HOLD",
        (
            "Reorder when inventory is below {threshold} units; otherwise hold. Current inventory is {initial}. Answer REORDER or HOLD.",
            "Policy requires REORDER for stock under {threshold}, and HOLD otherwise. Stock is {initial}. What action?",
        ),
        ("Inventory is now {value} units. Answer REORDER or HOLD.", "A recount shows {value} units. What action does the policy require?"),
        ("The warehouse lights changed; inventory remains {initial}. What action?", "The item label changed, but stock is still {initial}. REORDER or HOLD?"),
    ),
    Domain(
        "battery_route", "battery charge", "%", "A", "B",
        (
            "Use route B when battery is at least {threshold}%; otherwise use route A. Battery is {initial}%. Choose A or B.",
            "Route B requires {threshold}% battery; route A is the fallback. Current charge is {initial}%. Which route?",
        ),
        ("Battery is now {value}%. Choose route A or B.", "The latest charge reading is {value}%. Which route should be used?"),
        ("The robot was renamed, with battery unchanged at {initial}%. Which route?", "Its paint color changed; charge remains {initial}%. Choose A or B."),
    ),
    Domain(
        "room_capacity", "attendance", "people", "A", "B",
        (
            "Room A holds at most {threshold} people and Room B is larger. Attendance is {initial}; choose the smallest sufficient room: A or B.",
            "Choose room A for groups up to {threshold}, otherwise room B. The group has {initial} people. Which room?",
        ),
        ("Attendance is now {value}. Choose room A or B.", "The participant count changed to {value}. Which room is sufficient?"),
        ("The meeting title changed, but attendance stays {initial}. Which room?", "The agenda changed; the group is still {initial} people. Choose A or B."),
    ),
    Domain(
        "quality_gate", "quality score", "points", "REJECT", "ACCEPT",
        (
            "Accept a build at quality score {threshold} or higher; otherwise reject. Its score is {initial}. Answer ACCEPT or REJECT.",
            "The release gate is {threshold} points. A build scoring below it is REJECT, otherwise ACCEPT. Score: {initial}. Decision?",
        ),
        ("The quality score is now {value}. ACCEPT or REJECT?", "A new test gives a score of {value}. What is the gate decision?"),
        ("The build codename changed; score remains {initial}. Decision?", "The report font changed, with score still {initial}. ACCEPT or REJECT?"),
    ),
    Domain(
        "credit_limit", "purchase amount", "$", "ALLOW", "BLOCK",
        (
            "Allow purchases up to ${threshold}; block larger ones. This purchase is ${initial}. Answer ALLOW or BLOCK.",
            "The card limit is ${threshold}. A purchase above it is BLOCK, otherwise ALLOW. Amount: ${initial}. Decision?",
        ),
        ("The purchase amount is now ${value}. ALLOW or BLOCK?", "The merchant updated the charge to ${value}. What is the decision?"),
        ("The merchant logo changed; amount remains ${initial}. Decision?", "The receipt color changed, but the purchase is still ${initial}. ALLOW or BLOCK?"),
    ),
    Domain(
        "network_latency", "latency", "ms", "NORMAL", "FAILOVER",
        (
            "Use NORMAL routing at latency up to {threshold} ms; use FAILOVER above it. Current latency is {initial} ms. Which mode?",
            "FAILOVER is required only when latency exceeds {threshold} ms. Monitoring reads {initial} ms. NORMAL or FAILOVER?",
        ),
        ("Latency is now {value} ms. NORMAL or FAILOVER?", "The latest network latency is {value} ms. Which mode is required?"),
        ("The graph color changed; latency remains {initial} ms. Which mode?", "The endpoint name changed, while latency stays {initial} ms. NORMAL or FAILOVER?"),
    ),
)


def decision(domain: Domain, value: int, threshold: int) -> str:
    # Inventory reverses the ordinary low/high naming in natural semantics,
    # but the labels are deliberately carried by the domain definition.
    return domain.low_label if value <= threshold else domain.high_label


def think_text(domain: Domain, old: int, new: int, threshold: int, old_answer: str, new_answer: str) -> str:
    return (
        f"The prior {old_answer} decision depended on {domain.noun}={old}{domain.unit}. "
        f"It is now {new}{domain.unit}, crossing the {threshold}{domain.unit} boundary, "
        f"so the valid decision changes to {new_answer}."
    )


def make_record(
    *, group_id: str, domain: Domain, role: str, timestamp: datetime,
    initial: int, threshold: int, final_text: str, expected: str,
    trigger: bool, target_think: str | None, system_prompt: str,
) -> dict:
    initial_answer = decision(domain, initial, threshold)
    target = f"<answer>{expected}</answer>"
    if target_think:
        target = f"<think>{target_think}</think>\n{target}"
    return {
        "schema_version": "cf-deltatime-0.2",
        "group_id": group_id,
        "scenario_id": f"{group_id}__{role}",
        "domain": domain.name,
        "pair_role": role,
        "oracle_trigger": int(trigger),
        "expected_answer": expected,
        "state": {"initial": initial, "threshold": threshold},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "<time>2026-01-01T09:00:00</time>\n"
                + domain.initial_templates[sum(map(ord, group_id)) % len(domain.initial_templates)].format(
                    threshold=threshold, initial=initial
                ),
            },
            {"role": "assistant", "content": f"<answer>{initial_answer}</answer>"},
            {"role": "user", "content": f"<time>{timestamp.isoformat()}</time>\n{final_text}"},
        ],
        "target_response": target,
    }


def build_threshold_group(rng: random.Random, domain: Domain, group_index: int) -> list[dict]:
    threshold = rng.randint(20, 80)
    start_low = rng.random() < 0.5
    margin = rng.randint(5, 15)
    initial = threshold - margin if start_low else threshold + margin
    stable_delta = rng.randint(1, max(1, margin - 1))
    stable = initial + stable_delta if start_low else initial - stable_delta
    crossing = threshold + rng.randint(2, 12) if start_low else threshold - rng.randint(2, 12)
    initial_answer = decision(domain, initial, threshold)
    changed_answer = decision(domain, crossing, threshold)
    assert initial_answer == decision(domain, stable, threshold)
    assert initial_answer != changed_answer
    base = datetime(2026, 1, 1, 9, 0, 0)
    system = rng.choice(SYSTEM_PROMPTS)
    group_id = f"{domain.name}_{group_index:06d}"
    update_template = rng.choice(domain.update_templates)
    records = [
        make_record(
            group_id=group_id, domain=domain, role="inert_short", timestamp=base + timedelta(minutes=5),
            initial=initial, threshold=threshold,
            final_text=f"Nothing relevant changed; {domain.noun} remains {initial}{domain.unit}. Give the decision again.",
            expected=initial_answer, trigger=False, target_think=None, system_prompt=system,
        ),
        make_record(
            group_id=group_id, domain=domain, role="inert_long", timestamp=base + timedelta(days=180),
            initial=initial, threshold=threshold,
            final_text=f"Nothing relevant changed; {domain.noun} remains {initial}{domain.unit}. Give the decision again.",
            expected=initial_answer, trigger=False, target_think=None, system_prompt=system,
        ),
        make_record(
            group_id=group_id, domain=domain, role="irrelevant_change", timestamp=base + timedelta(minutes=5),
            initial=initial, threshold=threshold,
            final_text=rng.choice(domain.irrelevant_templates).format(initial=initial),
            expected=initial_answer, trigger=False, target_think=None, system_prompt=system,
        ),
        make_record(
            group_id=group_id, domain=domain, role="relevant_stable", timestamp=base + timedelta(minutes=5),
            initial=initial, threshold=threshold,
            final_text=update_template.format(value=stable), expected=initial_answer,
            trigger=False, target_think=None, system_prompt=system,
        ),
        make_record(
            group_id=group_id, domain=domain, role="assumption_break_short", timestamp=base + timedelta(minutes=5),
            initial=initial, threshold=threshold,
            final_text=update_template.format(value=crossing), expected=changed_answer,
            trigger=True,
            target_think=think_text(domain, initial, crossing, threshold, initial_answer, changed_answer),
            system_prompt=system,
        ),
        make_record(
            group_id=group_id, domain=domain, role="assumption_break_long", timestamp=base + timedelta(days=180),
            initial=initial, threshold=threshold,
            final_text=update_template.format(value=crossing), expected=changed_answer,
            trigger=True,
            target_think=think_text(domain, initial, crossing, threshold, initial_answer, changed_answer),
            system_prompt=system,
        ),
    ]
    return records


def build_temporal_group(rng: random.Random, group_index: int) -> list[dict]:
    base = datetime(2026, 1, 1, 9, 0, 0)
    due_hours = rng.randint(4, 72)
    due = base + timedelta(hours=due_hours)
    before = due - timedelta(minutes=rng.randint(5, 60))
    after = due + timedelta(minutes=rng.randint(5, 60))
    group_id = f"temporal_deadline_{group_index:06d}"
    system = rng.choice(SYSTEM_PROMPTS)
    initial = (
        f"A job becomes OVERDUE after {due.isoformat()} and is PENDING before or at that time. "
        "What is its current status?"
    )

    def record(role: str, now: datetime, trigger: bool, expected: str) -> dict:
        target = f"<answer>{expected}</answer>"
        if trigger:
            target = (
                f"<think>The current time crossed the explicit due time {due.isoformat()}, "
                f"so the prior PENDING status is invalid and becomes OVERDUE.</think>\n{target}"
            )
        return {
            "schema_version": "cf-deltatime-0.2",
            "group_id": group_id,
            "scenario_id": f"{group_id}__{role}",
            "domain": "temporal_deadline",
            "pair_role": role,
            "oracle_trigger": int(trigger),
            "expected_answer": expected,
            "state": {"due": due.isoformat()},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"<time>{base.isoformat()}</time>\n{initial}"},
                {"role": "assistant", "content": "<answer>PENDING</answer>"},
                {"role": "user", "content": f"<time>{now.isoformat()}</time>\nThe job state has no manual updates. What is its status now?"},
            ],
            "target_response": target,
        }

    return [
        record("temporal_before_short", base + timedelta(minutes=5), False, "PENDING"),
        record("temporal_before_long", before, False, "PENDING"),
        record("temporal_crossing", after, True, "OVERDUE"),
    ]


def training_conversation(record: dict) -> list[dict]:
    messages = [dict(x) for x in record["messages"]]
    messages.append({"role": "assistant", "content": record["target_response"]})
    return messages


def build_split(rng: random.Random, domains: tuple[Domain, ...], groups: int, offset: int, temporal_fraction: float) -> list[dict]:
    records: list[dict] = []
    temporal_groups = round(groups * temporal_fraction)
    threshold_groups = groups - temporal_groups
    for idx in range(threshold_groups):
        domain = domains[idx % len(domains)]
        records.extend(build_threshold_group(rng, domain, offset + idx))
    for idx in range(temporal_groups):
        records.extend(build_temporal_group(rng, offset + threshold_groups + idx))
    return records


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="auto-res/data/cf_delta_v0_2")
    parser.add_argument("--train-groups", type=int, default=256)
    parser.add_argument("--dev-groups", type=int, default=64)
    parser.add_argument("--test-groups", type=int, default=128)
    parser.add_argument(
        "--replay-data",
        nargs="*",
        default=[],
        help="Optional TIME training JSON files sampled into a retention mixture.",
    )
    parser.add_argument(
        "--replay-count",
        type=int,
        default=346,
        help="Number of assistant-ending replay conversations; 346 gives 20%% replay in the default mixture.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    out = Path(args.output_dir)
    train_rng = random.Random(args.seed)
    dev_rng = random.Random(args.seed + 1)
    test_rng = random.Random(args.seed + 2)
    train_domains = DOMAINS[:6]
    test_domains = DOMAINS[6:]
    train = build_split(train_rng, train_domains, args.train_groups, 0, 0.20)
    dev = build_split(dev_rng, train_domains, args.dev_groups, 1_000_000, 0.20)
    test = build_split(test_rng, test_domains, args.test_groups, 2_000_000, 0.20)
    train_conversations = [training_conversation(x) for x in train]
    replay: list[list[dict]] = []
    if args.replay_data and args.replay_count:
        candidates: list[list[dict]] = []
        for path in args.replay_data:
            candidates.extend(
                x for x in json.loads(Path(path).read_text())
                if x and x[-1].get("role") == "assistant"
            )
        replay_rng = random.Random(args.seed + 10)
        replay = replay_rng.sample(candidates, min(args.replay_count, len(candidates)))
    mixed_conversations = train_conversations + replay
    random.Random(args.seed + 11).shuffle(mixed_conversations)
    write_json(out / "train_records.json", train)
    write_json(out / "dev_records.json", dev)
    write_json(out / "test_records.json", test)
    write_json(out / "phase5_train.json", train_conversations)
    write_json(out / "phase5_train_with_replay.json", mixed_conversations)
    manifest = {
        "schema_version": "cf-deltatime-0.2",
        "seed": args.seed,
        "group_counts": {"train": args.train_groups, "dev": args.dev_groups, "test": args.test_groups},
        "record_counts": {"train": len(train), "dev": len(dev), "test": len(test)},
        "training_mixture": {
            "cf_delta": len(train_conversations),
            "time_replay": len(replay),
            "replay_fraction": len(replay) / len(mixed_conversations) if mixed_conversations else 0.0,
        },
        "train_domains": [x.name for x in train_domains] + ["temporal_deadline"],
        "test_domains": [x.name for x in test_domains] + ["temporal_deadline"],
        "split_rule": "threshold domains held out; all siblings remain in one split",
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
