#!/usr/bin/env python3
"""Pair-ranked Phase-5 alignment for CF-DeltaTIME trigger policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

# Pairwise ranking reads raw next-token logits. Unsloth suppresses logits during
# labelled forwards by default, so this must be enabled before importing it.
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import unsloth  # noqa: F401 - patch before Transformers imports
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
from unsloth import FastLanguageModel

from trainer_pkg.chat_template import CHAT_TEMPLATE


PAIR_MANIFEST_SCHEMA_VERSION = 2


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2, help="Counterfactual pairs per device batch.")
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--ranking-margin", type=float, default=1.0)
    parser.add_argument(
        "--negative-pairing",
        choices=("matched", "within-domain-deranged"),
        default="matched",
        help=(
            "Use true same-group counterfactual negatives, or a deterministic "
            "within-domain/role derangement that preserves every record's "
            "multiplicity while breaking sibling identity."
        ),
    )
    parser.add_argument(
        "--calibration-weight",
        type=float,
        default=0.0,
        help=(
            "Optional absolute trigger calibration. Positive members are pushed "
            "toward <think> and negative members toward the direct-answer token, "
            "in addition to the relative pair margin."
        ),
    )
    parser.add_argument(
        "--positive-calibration-weight",
        type=float,
        default=None,
        help=(
            "Optional weight for pushing positive members toward <think>. "
            "Defaults to --calibration-weight for backward compatibility."
        ),
    )
    parser.add_argument(
        "--negative-calibration-weight",
        type=float,
        default=None,
        help=(
            "Optional weight for pushing negative members toward the direct-answer "
            "token. Defaults to --calibration-weight for backward compatibility."
        ),
    )
    parser.add_argument(
        "--positive-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Optional short-to-long positive trigger-score distillation. For "
            "threshold groups, the assumption-break-long score is a stopped-gradient "
            "teacher for assumption-break-short, discouraging elapsed-time shortcuts."
        ),
    )
    parser.add_argument(
        "--positive-consistency-pairing",
        choices=("matched", "within-domain-deranged"),
        default="matched",
        help=(
            "Use the true same-group assumption-break-long teacher, or a "
            "deterministic within-domain/role derangement that preserves every "
            "short student and long teacher multiplicity while breaking only "
            "positive-consistency peer identity."
        ),
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int)
    cfg = parser.parse_args(argv)
    if cfg.positive_calibration_weight is None:
        cfg.positive_calibration_weight = cfg.calibration_weight
    if cfg.negative_calibration_weight is None:
        cfg.negative_calibration_weight = cfg.calibration_weight
    for name in (
        "calibration_weight",
        "positive_calibration_weight",
        "negative_calibration_weight",
        "positive_consistency_weight",
    ):
        if getattr(cfg, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return cfg


def seed_all(seed: int) -> None:
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def encode(tokenizer, record: dict, max_length: int) -> dict:
    prompt = tokenizer.apply_chat_template(record["messages"], tokenize=False, add_generation_prompt=True)
    messages = [*record["messages"], {"role": "assistant", "content": record["target_response"]}]
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(f"prefix mismatch for {record['scenario_id']}")
    if len(full_ids) > max_length:
        trim = len(full_ids) - max_length
        full_ids = full_ids[trim:]
        prompt_len = max(0, len(prompt_ids) - trim)
    else:
        prompt_len = len(prompt_ids)
    if prompt_len == 0 or prompt_len >= len(full_ids):
        raise RuntimeError(f"invalid prompt/target boundary for {record['scenario_id']}")
    return {
        "input_ids": full_ids,
        "labels": [-100] * prompt_len + full_ids[prompt_len:],
        "decision_index": prompt_len - 1,
    }


def make_pairs(records: list[dict]) -> list[dict]:
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in records:
        group_id = record["group_id"]
        pair_role = record["pair_role"]
        if pair_role in groups[group_id]:
            raise ValueError(f"duplicate pair_role={pair_role!r} in group_id={group_id!r}")
        groups[group_id][pair_role] = record
    pairs = []
    for group_id, roles in groups.items():
        is_temporal = any(role.startswith("temporal_") for role in roles)
        if is_temporal:
            required = {"temporal_crossing", "temporal_before_short", "temporal_before_long"}
        else:
            required = {
                "assumption_break_short", "assumption_break_long", "inert_short",
                "inert_long", "irrelevant_change", "relevant_stable",
            }
        actual = set(roles)
        if actual != required:
            raise ValueError(
                f"invalid roles for group_id={group_id!r}: "
                f"missing={sorted(required - actual)}, unexpected={sorted(actual - required)}"
            )
        if is_temporal:
            positive_specs = [(roles["temporal_crossing"], roles["temporal_crossing"], False)]
            negatives = [roles[x] for x in ("temporal_before_short", "temporal_before_long")]
        else:
            positive_specs = [
                (roles["assumption_break_short"], roles["assumption_break_long"], True),
                (roles["assumption_break_long"], roles["assumption_break_long"], False),
            ]
            negatives = [roles[x] for x in ("inert_short", "inert_long", "irrelevant_change", "relevant_stable")]
        for positive, positive_peer, consistency_mask in positive_specs:
            for negative in negatives:
                pairs.append({
                    "group_id": group_id,
                    "positive": positive,
                    "positive_peer": positive_peer,
                    "positive_consistency_mask": consistency_mask,
                    "negative": negative,
                })
    return pairs


def derange_negative_groups(pairs: list[dict], seed: int) -> list[dict]:
    """Break sibling identity without changing record or role marginals."""
    original_positive_counts = Counter(
        pair["positive"]["scenario_id"] for pair in pairs
    )
    original_negative_counts = Counter(
        pair["negative"]["scenario_id"] for pair in pairs
    )
    original_strata = Counter(
        (
            pair["positive"]["domain"],
            pair["positive"]["pair_role"],
            pair["negative"]["pair_role"],
        )
        for pair in pairs
    )
    buckets: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        positive = pair["positive"]
        negative = pair["negative"]
        key = (positive["domain"], positive["pair_role"], negative["pair_role"])
        buckets[key].append(index)

    rng = random.Random(seed ^ 0xD37A_11A5)
    result = [dict(pair) for pair in pairs]
    for key, indices in sorted(buckets.items()):
        if len(indices) < 2:
            raise ValueError(f"cannot derange singleton negative-pairing bucket: {key!r}")
        order = list(indices)
        rng.shuffle(order)
        offset = rng.randrange(1, len(order))
        for target, source in zip(order, order[offset:] + order[:offset], strict=True):
            result[target]["negative"] = pairs[source]["negative"]

    fixed = [
        pair for pair in result
        if pair["positive"]["group_id"] == pair["negative"]["group_id"]
    ]
    if fixed:
        raise RuntimeError(f"negative derangement left {len(fixed)} matched siblings")
    if Counter(pair["positive"]["scenario_id"] for pair in result) != original_positive_counts:
        raise RuntimeError("negative derangement changed positive-example multiplicities")
    if Counter(pair["negative"]["scenario_id"] for pair in result) != original_negative_counts:
        raise RuntimeError("negative derangement changed negative-example multiplicities")
    shuffled_strata = Counter(
        (
            pair["positive"]["domain"],
            pair["positive"]["pair_role"],
            pair["negative"]["pair_role"],
        )
        for pair in result
    )
    if shuffled_strata != original_strata:
        raise RuntimeError("negative derangement changed domain/role strata")
    return result


def derange_positive_consistency_peers(pairs: list[dict], seed: int) -> list[dict]:
    """Break short-to-long teacher identity without changing other endpoints.

    Only examples whose ``positive_consistency_mask`` is true participate.  A
    single group-level derangement is performed for each domain/role stratum
    and repeated across all pair-expanded negative roles.  Thus each short
    student still has one consistent long teacher identity, while the ranking
    negative itself remains untouched.
    """
    carrier_indices = [
        index for index, pair in enumerate(pairs)
        if pair["positive_consistency_mask"]
    ]
    original_positive_counts = Counter(
        pair["positive"]["scenario_id"] for pair in pairs
    )
    original_negative_counts = Counter(
        pair["negative"]["scenario_id"] for pair in pairs
    )
    original_peer_counts = Counter(
        pairs[index]["positive_peer"]["scenario_id"] for index in carrier_indices
    )
    original_strata = Counter(
        (
            pairs[index]["positive"]["domain"],
            pairs[index]["positive"]["pair_role"],
            pairs[index]["positive_peer"]["domain"],
            pairs[index]["positive_peer"]["pair_role"],
            pairs[index]["negative"]["pair_role"],
        )
        for index in carrier_indices
    )
    buckets: dict[tuple[str, str, str, str], dict[str, dict]] = defaultdict(dict)
    for index in carrier_indices:
        pair = pairs[index]
        key = (
            pair["positive"]["domain"],
            pair["positive"]["pair_role"],
            pair["positive_peer"]["domain"],
            pair["positive_peer"]["pair_role"],
        )
        student_id = pair["positive"]["scenario_id"]
        previous_peer = buckets[key].get(student_id)
        if (
            previous_peer is not None
            and previous_peer["scenario_id"] != pair["positive_peer"]["scenario_id"]
        ):
            raise ValueError(
                "positive-consistency student has multiple teacher identities "
                f"before derangement: {student_id!r}"
            )
        buckets[key][student_id] = pair["positive_peer"]

    rng = random.Random(seed ^ 0xC051_57E1)
    replacement_by_student: dict[tuple[tuple[str, str, str, str], str], dict] = {}
    for key, peer_by_student in sorted(buckets.items()):
        students = sorted(peer_by_student)
        if len(students) < 2:
            raise ValueError(
                f"cannot derange singleton positive-consistency bucket: {key!r}"
            )
        order = list(students)
        rng.shuffle(order)
        offset = rng.randrange(1, len(order))
        for target, source in zip(order, order[offset:] + order[:offset], strict=True):
            replacement_by_student[(key, target)] = peer_by_student[source]

    result = [dict(pair) for pair in pairs]
    for index in carrier_indices:
        pair = pairs[index]
        key = (
            pair["positive"]["domain"],
            pair["positive"]["pair_role"],
            pair["positive_peer"]["domain"],
            pair["positive_peer"]["pair_role"],
        )
        student_id = pair["positive"]["scenario_id"]
        result[index]["positive_peer"] = replacement_by_student[(key, student_id)]

    fixed = [
        pair for pair in result
        if pair["positive_consistency_mask"]
        and pair["positive"]["group_id"] == pair["positive_peer"]["group_id"]
    ]
    if fixed:
        raise RuntimeError(
            f"positive-consistency derangement left {len(fixed)} matched peers"
        )
    if Counter(pair["positive"]["scenario_id"] for pair in result) != original_positive_counts:
        raise RuntimeError("positive-consistency derangement changed positive multiplicities")
    if Counter(pair["negative"]["scenario_id"] for pair in result) != original_negative_counts:
        raise RuntimeError("positive-consistency derangement changed negative multiplicities")
    shuffled_peer_counts = Counter(
        pair["positive_peer"]["scenario_id"]
        for pair in result
        if pair["positive_consistency_mask"]
    )
    if shuffled_peer_counts != original_peer_counts:
        raise RuntimeError(
            "positive-consistency derangement changed teacher multiplicities"
        )
    shuffled_strata = Counter(
        (
            pair["positive"]["domain"],
            pair["positive"]["pair_role"],
            pair["positive_peer"]["domain"],
            pair["positive_peer"]["pair_role"],
            pair["negative"]["pair_role"],
        )
        for pair in result
        if pair["positive_consistency_mask"]
    )
    if shuffled_strata != original_strata:
        raise RuntimeError(
            "positive-consistency derangement changed domain/role strata"
        )
    for original, shuffled in zip(pairs, result, strict=True):
        if (
            original["positive"]["scenario_id"] != shuffled["positive"]["scenario_id"]
            or original["negative"]["scenario_id"] != shuffled["negative"]["scenario_id"]
            or original["positive_consistency_mask"]
            != shuffled["positive_consistency_mask"]
        ):
            raise RuntimeError(
                "positive-consistency derangement changed a non-teacher endpoint"
            )
        if (
            not original["positive_consistency_mask"]
            and original["positive_peer"]["scenario_id"]
            != shuffled["positive_peer"]["scenario_id"]
        ):
            raise RuntimeError(
                "positive-consistency derangement changed an inactive peer"
            )
    return result


def pair_manifest(pairs: list[dict]) -> list[dict]:
    """Return the auditable pre-training endpoint mapping."""
    return [
        {
            "positive_group_id": pair["positive"]["group_id"],
            "negative_group_id": pair["negative"]["group_id"],
            "domain": pair["positive"]["domain"],
            "positive_role": pair["positive"]["pair_role"],
            "negative_role": pair["negative"]["pair_role"],
            "positive_scenario_id": pair["positive"]["scenario_id"],
            "negative_scenario_id": pair["negative"]["scenario_id"],
            "positive_peer_group_id": pair["positive_peer"]["group_id"],
            "positive_peer_domain": pair["positive_peer"]["domain"],
            "positive_peer_role": pair["positive_peer"]["pair_role"],
            "positive_peer_scenario_id": pair["positive_peer"]["scenario_id"],
            "positive_consistency_mask": pair["positive_consistency_mask"],
        }
        for pair in pairs
    ]


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def select_training_pairs(pairs: list[dict], seed: int, limit: int | None) -> list[dict]:
    """Apply the deterministic training-order shuffle and optional prefix limit."""
    selected = list(pairs)
    random.Random(seed).shuffle(selected)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer")
        selected = selected[:limit]
    if not selected:
        raise ValueError("no counterfactual pairs remain after loading and applying --limit")
    return selected


def pairing_statistics(pairs: list[dict]) -> dict[str, int | float | None]:
    """Summarize exactly the (possibly limited) pair list used for training."""
    matched_negative = sum(
        pair["positive"]["group_id"] == pair["negative"]["group_id"]
        for pair in pairs
    )
    carriers = [pair for pair in pairs if pair["positive_consistency_mask"]]
    matched_peers = sum(
        pair["positive"]["group_id"] == pair["positive_peer"]["group_id"]
        for pair in carriers
    )
    return {
        "pair_examples": len(pairs),
        "matched_positive_negative_pairs": matched_negative,
        "cross_group_positive_negative_pairs": len(pairs) - matched_negative,
        "positive_negative_group_match_rate": matched_negative / len(pairs),
        "positive_consistency_carriers": len(carriers),
        "matched_positive_consistency_peers": matched_peers,
        "cross_group_positive_consistency_peers": len(carriers) - matched_peers,
        "positive_consistency_peer_group_match_rate": (
            matched_peers / len(carriers) if carriers else None
        ),
    }


class PairCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def pad_side(self, features: list[dict], side: str) -> dict[str, torch.Tensor]:
        sequences = [x[side]["input_ids"] for x in features]
        labels = [x[side]["labels"] for x in features]
        length = max(map(len, sequences))
        pad_id = self.tokenizer.pad_token_id
        padded_ids = [x + [pad_id] * (length - len(x)) for x in sequences]
        padded_labels = [x + [-100] * (length - len(x)) for x in labels]
        masks = [[1] * len(x) + [0] * (length - len(x)) for x in sequences]
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "decision_index": torch.tensor([x[side]["decision_index"] for x in features], dtype=torch.long),
        }

    def __call__(self, features: list[dict]) -> dict:
        batch = {
            "positive": self.pad_side(features, "positive"),
            "negative": self.pad_side(features, "negative"),
        }
        if "positive_peer" in features[0]:
            batch["positive_peer"] = self.pad_side(features, "positive_peer")
            batch["positive_consistency_mask"] = torch.tensor(
                [x["positive_consistency_mask"] for x in features], dtype=torch.bool
            )
        return batch


class PairwiseTrainer(Trainer):
    def __init__(
        self,
        *args,
        think_id: int,
        direct_id: int,
        ranking_weight: float,
        ranking_margin: float,
        positive_calibration_weight: float,
        negative_calibration_weight: float,
        positive_consistency_weight: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.think_id = think_id
        self.direct_id = direct_id
        self.ranking_weight = ranking_weight
        self.ranking_margin = ranking_margin
        self.positive_calibration_weight = positive_calibration_weight
        self.negative_calibration_weight = negative_calibration_weight
        self.positive_consistency_weight = positive_consistency_weight

    @staticmethod
    def move(batch: dict, device: torch.device) -> tuple[dict, torch.Tensor]:
        decision_index = batch["decision_index"].to(device)
        model_batch = {k: v.to(device) for k, v in batch.items() if k != "decision_index"}
        return model_batch, decision_index

    def trigger_score(self, logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(logits.shape[0], device=logits.device)
        next_logits = logits[batch, indices]
        return next_logits[:, self.think_id] - next_logits[:, self.direct_id]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        positive, pos_index = self.move(inputs["positive"], model.device)
        negative, neg_index = self.move(inputs["negative"], model.device)
        pos_out = model(**positive)
        neg_out = model(**negative)
        lm_loss = 0.5 * (pos_out.loss + neg_out.loss)
        pos_score = self.trigger_score(pos_out.logits, pos_index)
        neg_score = self.trigger_score(neg_out.logits, neg_index)
        ranking_loss = F.softplus(self.ranking_margin - (pos_score - neg_score)).mean()
        positive_calibration_loss = F.softplus(-pos_score).mean()
        negative_calibration_loss = F.softplus(neg_score).mean()
        calibration_loss = 0.5 * (
            positive_calibration_loss + negative_calibration_loss
        )
        weighted_calibration_loss = 0.5 * (
            self.positive_calibration_weight * positive_calibration_loss
            + self.negative_calibration_weight * negative_calibration_loss
        )
        positive_consistency_loss = pos_score.new_zeros(())
        peer_out = None
        if self.positive_consistency_weight > 0:
            consistency_mask = inputs["positive_consistency_mask"].to(model.device)
            if consistency_mask.any().item():
                positive_peer, peer_index = self.move(inputs["positive_peer"], model.device)
                positive_peer.pop("labels", None)
                peer_out = model(**positive_peer)
                peer_score = self.trigger_score(peer_out.logits, peer_index)
                consistency_target = torch.maximum(
                    pos_score[consistency_mask].detach(),
                    peer_score[consistency_mask].detach(),
                )
                positive_consistency_loss = F.smooth_l1_loss(
                    pos_score[consistency_mask], consistency_target
                )
        loss = (
            lm_loss
            + self.ranking_weight * ranking_loss
            + weighted_calibration_loss
            + self.positive_consistency_weight * positive_consistency_loss
        )
        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            self.log({
                "lm_loss_component": float(lm_loss.detach()),
                "ranking_loss_component": float(ranking_loss.detach()),
                "calibration_loss_component": float(calibration_loss.detach()),
                "weighted_calibration_loss_component": float(weighted_calibration_loss.detach()),
                "positive_calibration_loss_component": float(positive_calibration_loss.detach()),
                "negative_calibration_loss_component": float(negative_calibration_loss.detach()),
                "positive_consistency_loss_component": float(positive_consistency_loss.detach()),
                "pair_margin": float((pos_score - neg_score).mean().detach()),
                "positive_trigger_score": float(pos_score.mean().detach()),
                "negative_trigger_score": float(neg_score.mean().detach()),
            })
        outputs = {"positive": pos_out, "negative": neg_out, "positive_peer": peer_out}
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    cfg = get_args()
    seed_all(cfg.seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.chat_template = CHAT_TEMPLATE
    records = json.loads(Path(cfg.train_records).read_text())
    pairs = make_pairs(records)
    if cfg.negative_pairing == "within-domain-deranged":
        pairs = derange_negative_groups(pairs, cfg.seed)
    if cfg.positive_consistency_pairing == "within-domain-deranged":
        pairs = derange_positive_consistency_peers(pairs, cfg.seed)
    endpoint_mapping_sha256 = stable_json_sha256(pair_manifest(pairs))
    pairs = select_training_pairs(pairs, cfg.seed, cfg.limit)
    training_manifest = pair_manifest(pairs)
    training_manifest_sha256 = stable_json_sha256(training_manifest)
    (output / "pair_manifest.json").write_text(
        json.dumps(training_manifest, indent=2) + "\n"
    )
    pairing_stats = pairing_statistics(pairs)
    think_ids = tokenizer("<think>", add_special_tokens=False)["input_ids"]
    direct_ids = tokenizer("<answer>", add_special_tokens=False)["input_ids"]
    if len(think_ids) != 1:
        raise RuntimeError(f"expected atomic <think>, got {think_ids}")
    if not direct_ids:
        raise RuntimeError("<answer> tokenized to an empty sequence")
    encoded = []
    for pair in pairs:
        positive = encode(tokenizer, pair["positive"], cfg.max_length)
        negative = encode(tokenizer, pair["negative"], cfg.max_length)
        positive_first = positive["labels"][positive["decision_index"] + 1]
        negative_first = negative["labels"][negative["decision_index"] + 1]
        if positive_first != think_ids[0] or negative_first != direct_ids[0]:
            raise RuntimeError(
                f"unexpected decision tokens for group_id={pair['group_id']!r}: "
                f"positive={positive_first}, negative={negative_first}"
            )
        example = {"positive": positive, "negative": negative}
        if cfg.positive_consistency_weight > 0:
            positive_peer = encode(tokenizer, pair["positive_peer"], cfg.max_length)
            peer_first = positive_peer["labels"][positive_peer["decision_index"] + 1]
            if peer_first != think_ids[0]:
                raise RuntimeError(
                    f"unexpected positive peer token for group_id={pair['group_id']!r}: "
                    f"positive_peer={peer_first}"
                )
            example["positive_peer"] = positive_peer
            example["positive_consistency_mask"] = pair["positive_consistency_mask"]
        encoded.append(example)
    dataset = Dataset.from_list(encoded)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model,
        max_seq_length=cfg.max_length,
        load_in_4bit=True,
        full_finetuning=False,
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=cfg.seed,
    )
    resolved = vars(cfg) | pairing_stats | {
        "pair_manifest_schema_version": PAIR_MANIFEST_SCHEMA_VERSION,
        "endpoint_mapping_sha256_before_order_shuffle": endpoint_mapping_sha256,
        "pair_manifest_sha256": training_manifest_sha256,
        "training_pair_order_sha256": training_manifest_sha256,
        "think_token_id": think_ids[0],
        "direct_first_token_id": direct_ids[0],
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (output / "resolved_config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    trainer = PairwiseTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output / "trainer"),
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            warmup_ratio=0.05,
            logging_steps=5,
            save_strategy="epoch",
            save_total_limit=2,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            weight_decay=0.01,
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
            seed=cfg.seed,
        ),
        train_dataset=dataset,
        data_collator=PairCollator(tokenizer),
        processing_class=tokenizer,
        think_id=think_ids[0],
        direct_id=direct_ids[0],
        ranking_weight=cfg.ranking_weight,
        ranking_margin=cfg.ranking_margin,
        positive_calibration_weight=cfg.positive_calibration_weight,
        negative_calibration_weight=cfg.negative_calibration_weight,
        positive_consistency_weight=cfg.positive_consistency_weight,
    )
    stats = trainer.train()
    adapter = output / "final_adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    metrics = dict(stats.metrics)
    metrics["peak_vram_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    metrics["effective_pair_batch"] = cfg.batch_size * cfg.gradient_accumulation
    metrics["estimated_optimizer_steps"] = math.ceil(len(dataset) / metrics["effective_pair_batch"] * cfg.epochs)
    (output / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
