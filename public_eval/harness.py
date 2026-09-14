#!/usr/bin/env python3
"""Reproducible public-task evaluation for TIME versus one DeltaTIME method.

The evaluator deliberately supports only two experiment roles: ``time_baseline``
and ``method``.  It talks to an already-running OpenAI-compatible server; it
never imports torch or starts a model/GPU itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "deltatime-public-eval-1.0"
SELECTION_SALT = "deltatime-public-eval-v1:3407"
SYSTEM_PROMPT = (
    "Answer the user's question. Follow the requested final-answer format exactly."
)

TASK_SPECS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "license": "MIT",
        "default_count": 256,
    },
    "arc_challenge": {
        "dataset": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "license": "CC BY-SA 4.0",
        "default_count": 256,
    },
    "boolq": {
        "dataset": "google/boolq",
        "config": None,
        "split": "validation",
        "revision": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
        "license": "CC BY-SA 3.0",
        "default_count": 256,
    },
}

PROTOCOL = {
    "schema_version": SCHEMA_VERSION,
    "tasks": list(TASK_SPECS),
    "system_prompt": SYSTEM_PROMPT,
    "answer_contract": {
        "location": "the only 'Final answer:' occurrence must be the final non-empty line",
        "gsm8k": "Final answer: <base-10 number without commas or units>",
        "arc_challenge": "Final answer: <one exact displayed choice label>",
        "boolq": "Final answer: <yes or no>",
    },
    "scoring": {
        "accuracy": "strict exact match after task-specific normalization; format failures score 0",
        "relaxed_accuracy": (
            "use strict result when strict format parses; otherwise accept exactly one balanced, "
            "non-nested <answer>...</answer> containing only a task-normalized atomic answer"
        ),
        "think_incidence": "at least one non-empty, complete <think>...</think> block",
        "reasoning_tokens": "sum of local-tokenizer token counts inside complete think blocks",
        "output_tokens": "local-tokenizer token count of the complete generated response",
        "non_reasoning_output_tokens": "local-tokenizer count after removing complete think blocks",
        "format_failure": "answer contract violated",
        "think_format_failure": "unbalanced or nested think tags",
    },
    "paired_bootstrap": {
        "unit": "paired example",
        "statistic": "mean(method - TIME)",
        "interval": "95% percentile",
        "rng": "Python random.Random with sampling by paired example index",
        "default_replications": 10000,
        "default_seed": 3407,
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


PROTOCOL_SHA256 = value_sha256(PROTOCOL)


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


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def selection_rank(task: str, source_index: int, salt: str = SELECTION_SALT) -> str:
    material = f"{salt}\0{task}\0{source_index}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def select_indices(task: str, size: int, count: int, salt: str) -> list[int]:
    if count < 1 or count > size:
        raise ValueError(f"invalid count {count} for {task} split of size {size}")
    ranked = sorted(range(size), key=lambda index: (selection_rank(task, index, salt), index))
    return sorted(ranked[:count])


def _gsm_gold(source: dict[str, Any]) -> str:
    match = re.search(r"####\s*(.+?)\s*$", source["answer"])
    if not match:
        raise ValueError("GSM8K answer has no #### final answer")
    return normalize_number(match.group(1))


def build_frozen_record(task: str, source_index: int, source: dict[str, Any]) -> dict[str, Any]:
    source_hash = value_sha256(source)
    if task == "gsm8k":
        prompt = (
            f"Problem:\n{source['question']}\n\n"
            "Solve the problem. End with exactly one final line in this form:\n"
            "Final answer: <base-10 number without commas or units>"
        )
        gold = _gsm_gold(source)
        source_id = str(source_index)
        answer_type = "number"
    elif task == "arc_challenge":
        labels = [str(item) for item in source["choices"]["label"]]
        choices = source["choices"]["text"]
        rendered = "\n".join(f"{label}. {text}" for label, text in zip(labels, choices, strict=True))
        prompt = (
            f"Question:\n{source['question']}\n\nChoices:\n{rendered}\n\n"
            f"Choose one of these labels: {', '.join(labels)}. "
            "End with exactly one final line in this form:\n"
            "Final answer: <choice label>"
        )
        gold = str(source["answerKey"])
        if gold not in labels:
            raise ValueError(f"ARC answer key {gold!r} is not among {labels!r}")
        source_id = str(source["id"])
        answer_type = "choice"
    elif task == "boolq":
        prompt = (
            f"Passage:\n{source['passage']}\n\nQuestion:\n{source['question']}\n\n"
            "Answer yes or no. End with exactly one final line in this form:\n"
            "Final answer: <yes or no>"
        )
        gold = "yes" if bool(source["answer"]) else "no"
        source_id = str(source_index)
        answer_type = "boolean"
    else:
        raise ValueError(f"unknown task: {task}")

    return {
        "record_id": f"{task}:{source_id}",
        "task": task,
        "source_index": source_index,
        "source_id": source_id,
        "source_record_sha256": source_hash,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "gold": gold,
        "answer_type": answer_type,
        "choice_labels": labels if task == "arc_challenge" else None,
    }


def _jsonl_text(records: Iterable[dict[str, Any]]) -> str:
    return "".join(canonical_json(record) + "\n" for record in records)


def default_manifest_path(subset_path: Path) -> Path:
    return subset_path.with_suffix(".manifest.json")


def prepare_subset(output: Path, manifest_path: Path, per_task: int, salt: str) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "the prepare command needs datasets; install public_eval/requirements-prepare.txt"
        ) from error

    frozen: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    for task, spec in TASK_SPECS.items():
        dataset = load_dataset(
            spec["dataset"],
            spec["config"],
            revision=spec["revision"],
            split=spec["split"],
        )
        indices = select_indices(task, len(dataset), per_task, salt)
        selected = [build_frozen_record(task, index, dataset[index]) for index in indices]
        frozen.extend(selected)
        split_hash = hashlib.sha256()
        for source in dataset:
            split_hash.update(canonical_json(source).encode("utf-8"))
            split_hash.update(b"\n")
        sources[task] = {
            **spec,
            "split_size": len(dataset),
            "datasets_fingerprint": dataset._fingerprint,
            "canonical_split_sha256": split_hash.hexdigest(),
            "selected_count": len(selected),
            "selected_indices": indices,
            "selected_record_ids": [item["record_id"] for item in selected],
        }

    text = _jsonl_text(frozen)
    atomic_write_text(output, text)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "frozen_public_subset",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "protocol_sha256": PROTOCOL_SHA256,
        "selection": {
            "algorithm": "lowest SHA256(salt\\0task\\0source_index), then source-index order",
            "salt": salt,
            "per_task": per_task,
        },
        "sources": sources,
        "record_count": len(frozen),
        "task_counts": dict(Counter(item["task"] for item in frozen)),
        "subset_path": str(output.resolve()),
        "subset_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "software": {
            "python": sys.version.split()[0],
            "datasets": package_version("datasets"),
            "huggingface_hub": package_version("huggingface-hub"),
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def load_jsonl(path: Path, *, tolerate_truncated_last_line: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if tolerate_truncated_last_line and index == len(lines) - 1:
                print(f"warning: ignoring incomplete final line in {path}", file=sys.stderr)
                break
            raise ValueError(f"invalid JSON on {path}:{index + 1}") from None
        if not isinstance(value, dict):
            raise ValueError(f"JSONL value on {path}:{index + 1} is not an object")
        records.append(value)
    return records


def load_and_verify_subset(subset: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_hash = file_sha256(subset)
    if actual_hash != manifest.get("subset_sha256"):
        raise ValueError(
            f"subset hash mismatch: manifest={manifest.get('subset_sha256')} actual={actual_hash}"
        )
    if manifest.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError(
            "frozen subset uses a different prompt/scoring protocol; regenerate it explicitly"
        )
    records = load_jsonl(subset)
    if len(records) != manifest.get("record_count"):
        raise ValueError("subset record count differs from manifest")
    seen: set[str] = set()
    required = {"record_id", "task", "messages", "gold", "answer_type"}
    for item in records:
        missing = required - item.keys()
        if missing:
            raise ValueError(f"{item.get('record_id', '<unknown>')} missing {sorted(missing)}")
        if item["record_id"] in seen:
            raise ValueError(f"duplicate record_id: {item['record_id']}")
        if item["task"] not in TASK_SPECS:
            raise ValueError(f"unknown task in subset: {item['task']}")
        seen.add(item["record_id"])
    return records, manifest


def normalize_number(value: Any) -> str:
    text = str(value).strip().replace(",", "")
    if text.startswith("$"):
        text = text[1:]
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"not a base-10 number: {value!r}") from error
    if not number.is_finite():
        raise ValueError(f"non-finite number: {value!r}")
    if number == 0:
        return "0"
    normalized = format(number.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


FINAL_PREFIX_RE = re.compile(r"(?im)^\s*Final answer\s*:")
THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
ANSWER_OPEN_RE = re.compile(r"<answer>", re.IGNORECASE)
ANSWER_CLOSE_RE = re.compile(r"</answer>", re.IGNORECASE)
ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def normalize_atomic_answer(record: dict[str, Any], candidate: str) -> str | None:
    """Normalize a candidate only when its entire text is a valid task atom."""
    text = candidate.strip()
    task = record["task"]
    if task == "gsm8k":
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
            return None
        return normalize_number(text)
    if task == "arc_challenge":
        labels = [str(item) for item in record.get("choice_labels") or []]
        matches = [label for label in labels if text.casefold() == label.casefold()]
        return matches[0].upper() if len(matches) == 1 else None
    if task == "boolq":
        return text.lower() if text.lower() in {"yes", "no"} else None
    raise ValueError(f"unknown task: {task}")


def expected_normalized_answer(record: dict[str, Any]) -> str:
    task = record["task"]
    gold = str(record["gold"])
    if task == "gsm8k":
        return normalize_number(gold)
    if task == "boolq":
        return gold.lower()
    if task == "arc_challenge":
        return gold.upper()
    raise ValueError(f"unknown task: {task}")


def extract_unique_answer_block(response: str) -> dict[str, Any]:
    """Return one answer block only when tags are balanced, ordered, and non-nested."""
    opens = list(ANSWER_OPEN_RE.finditer(response))
    closes = list(ANSWER_CLOSE_RE.finditer(response))
    events = sorted(
        [(match.start(), "open") for match in opens]
        + [(match.start(), "close") for match in closes]
    )
    format_ok = (
        len(opens) == 1
        and len(closes) == 1
        and [kind for _, kind in events] == ["open", "close"]
    )
    match = ANSWER_BLOCK_RE.search(response) if format_ok else None
    extracted = match.group(1).strip() if match else None
    return {
        "answer_wrapper_format_ok": bool(match),
        "answer_wrapper_extracted": extracted,
    }


def parse_final_answer(record: dict[str, Any], response: str) -> dict[str, Any]:
    occurrences = len(FINAL_PREFIX_RE.findall(response))
    nonempty = [line.strip() for line in response.splitlines() if line.strip()]
    final_line = nonempty[-1] if nonempty else ""
    task = record["task"]
    if task == "gsm8k":
        pattern = r"Final answer: ([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    elif task == "arc_challenge":
        labels = [re.escape(str(item)) for item in record.get("choice_labels") or []]
        pattern = rf"Final answer: ({'|'.join(labels)})" if labels else r"(?!)"
    elif task == "boolq":
        pattern = r"Final answer: (yes|no)"
    else:
        raise ValueError(f"unknown task: {task}")
    match = re.fullmatch(pattern, final_line, flags=re.IGNORECASE)
    format_ok = bool(match and occurrences == 1)
    extracted = match.group(1) if match else None
    normalized = normalize_atomic_answer(record, extracted) if extracted is not None else None
    expected = expected_normalized_answer(record)
    strict_correct = bool(format_ok and normalized == expected)
    wrapper = extract_unique_answer_block(response)
    wrapper_normalized = (
        normalize_atomic_answer(record, wrapper["answer_wrapper_extracted"])
        if wrapper["answer_wrapper_extracted"] is not None
        else None
    )
    if format_ok:
        relaxed_correct = strict_correct
        relaxed_source = "strict_final"
        relaxed_extracted = extracted
        relaxed_normalized = normalized
    else:
        relaxed_correct = bool(
            wrapper["answer_wrapper_format_ok"] and wrapper_normalized == expected
        )
        relaxed_source = "answer_block" if wrapper["answer_wrapper_format_ok"] else None
        relaxed_extracted = wrapper["answer_wrapper_extracted"]
        relaxed_normalized = wrapper_normalized
    return {
        "answer_extracted": extracted,
        "answer_normalized": normalized,
        "answer_format_ok": format_ok,
        "format_failure": not format_ok,
        "answer_correct": strict_correct,
        **wrapper,
        "answer_wrapper_normalized": wrapper_normalized,
        "relaxed_answer_extracted": relaxed_extracted,
        "relaxed_answer_normalized": relaxed_normalized,
        "relaxed_answer_source": relaxed_source,
        "relaxed_answer_correct": relaxed_correct,
    }


def analyze_thinking(response: str, tokenizer: Any) -> dict[str, Any]:
    blocks = [match.group(1).strip() for match in THINK_BLOCK_RE.finditer(response)]
    nonempty_blocks = [item for item in blocks if item]
    open_count = len(THINK_OPEN_RE.findall(response))
    close_count = len(THINK_CLOSE_RE.findall(response))
    events = re.findall(r"</?think>", response, flags=re.IGNORECASE)
    depth = 0
    nested_or_misordered = False
    for event in events:
        if event.lower() == "<think>":
            depth += 1
            if depth > 1:
                nested_or_misordered = True
        else:
            depth -= 1
            if depth < 0:
                nested_or_misordered = True
                depth = 0
    malformed = open_count != close_count or depth != 0 or nested_or_misordered
    reasoning_tokens = sum(
        len(tokenizer.encode(block, add_special_tokens=False)) for block in nonempty_blocks
    )
    outside = THINK_BLOCK_RE.sub("", response)
    completion_tokens = len(tokenizer.encode(response, add_special_tokens=False))
    non_reasoning_output_tokens = len(tokenizer.encode(outside, add_special_tokens=False))
    return {
        "think_incidence": bool(nonempty_blocks),
        "think_block_count": len(nonempty_blocks),
        "think_open_tag_count": open_count,
        "think_close_tag_count": close_count,
        "think_format_failure": malformed,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens": completion_tokens,
        "non_reasoning_output_tokens": non_reasoning_output_tokens,
        "completion_tokens_local": completion_tokens,
    }


def directory_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"checkpoint must be a local directory: {resolved}")
    files = [item for item in sorted(resolved.rglob("*")) if item.is_file()]
    if not files:
        raise ValueError(f"checkpoint directory is empty: {resolved}")
    entries = []
    for item in files:
        entries.append(
            {
                "path": item.relative_to(resolved).as_posix(),
                "size_bytes": item.stat().st_size,
                "sha256": file_sha256(item),
            }
        )
    return {
        "path": str(resolved),
        "file_count": len(entries),
        "total_bytes": sum(item["size_bytes"] for item in entries),
        "inventory_sha256": value_sha256(entries),
        "files": entries,
    }


class APIClient:
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        import requests

        self.requests = requests
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def verify_model(self, model: str) -> dict[str, Any]:
        response = self.requests.get(
            f"{self.base_url}/models", headers=self.headers, timeout=self.timeout
        )
        response.raise_for_status()
        payload = response.json()
        advertised = [str(item.get("id")) for item in payload.get("data", [])]
        if model not in advertised:
            raise ValueError(f"served model {model!r} not advertised by server: {advertised}")
        return payload

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


def per_record_seed(master_seed: int, record_id: str) -> int:
    digest = hashlib.sha256(f"{master_seed}\0{record_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def load_tokenizer(reference: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(reference.resolve()), local_files_only=True, trust_remote_code=False
    )


def run_config(
    args: argparse.Namespace,
    subset_manifest: dict[str, Any],
    checkpoint: dict[str, Any],
    selected_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": args.role,
        "model_label": args.model_label,
        "served_model": args.served_model,
        "checkpoint": checkpoint,
        "tokenizer_path": str(args.tokenizer.resolve()),
        "subset_sha256": subset_manifest["subset_sha256"],
        "protocol_sha256": PROTOCOL_SHA256,
        "harness_sha256": file_sha256(Path(__file__)),
        "selected_ids_sha256": value_sha256(list(selected_ids)),
        "selected_count": len(selected_ids),
        "task": args.task,
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": args.max_tokens,
            "master_seed": args.seed,
        },
        "endpoint": args.api_base.rstrip("/"),
    }


def validate_resume_manifest(path: Path, expected_config: dict[str, Any]) -> dict[str, Any]:
    expected_hash = value_sha256(expected_config)
    if not path.exists():
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "public_eval_run",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_config": expected_config,
            "run_config_sha256": expected_hash,
            "software": {
                "python": sys.version.split()[0],
                "requests": package_version("requests"),
                "transformers": package_version("transformers"),
            },
        }
        atomic_write_json(path, manifest)
        return manifest
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("run_config_sha256") != expected_hash or manifest.get("run_config") != expected_config:
        raise ValueError("existing run manifest does not match this run; use a new output path")
    return manifest


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def repair_truncated_jsonl(path: Path) -> bool:
    """Atomically drop only an invalid, non-newline-terminated final fragment."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return False
    final_newline = data.rfind(b"\n")
    prefix = data[: final_newline + 1] if final_newline >= 0 else b""
    tail = data[final_newline + 1 :]
    try:
        decoded = tail.decode("utf-8")
        value = json.loads(decoded)
        if not isinstance(value, dict):
            raise ValueError("final JSONL value is not an object")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        temporary = path.with_name(path.name + ".repair.tmp")
        with temporary.open("wb") as handle:
            handle.write(prefix)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return True
    with path.open("ab") as handle:
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return False


def latest_results(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    for item in load_jsonl(path, tolerate_truncated_last_line=True):
        record_id = item.get("record_id")
        if not record_id:
            raise ValueError(f"result in {path} lacks record_id")
        latest[str(record_id)] = item
    return latest


def run_evaluation(args: argparse.Namespace) -> None:
    subset_manifest_path = args.subset_manifest or default_manifest_path(args.subset)
    records, subset_manifest = load_and_verify_subset(args.subset, subset_manifest_path)
    records = [item for item in records if item["task"] == args.task]
    if not records:
        raise ValueError(f"frozen subset contains no records for task {args.task}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        records = records[: args.limit]
    selected_ids = [str(item["record_id"]) for item in records]
    tokenizer = load_tokenizer(args.tokenizer)
    checkpoint = directory_fingerprint(args.checkpoint)
    config = run_config(args, subset_manifest, checkpoint, selected_ids)
    run_manifest_path = Path(str(args.output) + ".manifest.json")
    if args.output.exists() and not args.resume:
        raise ValueError(f"output exists; pass --resume or choose a new path: {args.output}")
    manifest = validate_resume_manifest(run_manifest_path, config)
    config_hash = manifest["run_config_sha256"]

    client = APIClient(args.api_base, args.api_key, args.timeout)
    client.verify_model(args.served_model)
    if repair_truncated_jsonl(args.output):
        print(f"warning: repaired incomplete final JSONL fragment in {args.output}", file=sys.stderr)
    previous = latest_results(args.output)
    foreign = [
        record_id
        for record_id, result in previous.items()
        if result.get("run_config_sha256") != config_hash
        or result.get("task") != args.task
        or result.get("role") != args.role
    ]
    if foreign:
        raise ValueError(f"existing output contains foreign records: {foreign[:5]}")
    completed = {
        record_id
        for record_id, result in previous.items()
        if result.get("status") == "ok" and result.get("run_config_sha256") == config_hash
    }
    pending = [item for item in records if item["record_id"] not in completed]
    print(f"completed={len(completed)} pending={len(pending)} total={len(records)}", flush=True)

    for position, record in enumerate(pending, start=1):
        seed = per_record_seed(args.seed, record["record_id"])
        payload = {
            "model": args.served_model,
            "messages": record["messages"],
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": args.max_tokens,
            "seed": seed,
            "stream": False,
        }
        error: Exception | None = None
        response_payload: dict[str, Any] | None = None
        for attempt in range(args.retries + 1):
            try:
                response_payload = client.complete(payload)
                error = None
                break
            except Exception as caught:  # HTTP and transport errors are recorded for resume.
                error = caught
                if attempt < args.retries:
                    time.sleep(min(2**attempt, 8))
        common = {
            "schema_version": SCHEMA_VERSION,
            "record_id": record["record_id"],
            "task": record["task"],
            "source_index": record.get("source_index"),
            "role": args.role,
            "model_label": args.model_label,
            "served_model": args.served_model,
            "run_config_sha256": config_hash,
            "subset_sha256": subset_manifest["subset_sha256"],
            "seed": seed,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
        }
        if error is not None or response_payload is None:
            result = {**common, "status": "api_error", "error": repr(error)}
        else:
            choices = response_payload.get("choices") or []
            if not choices or not isinstance(choices[0].get("message", {}).get("content"), str):
                result = {**common, "status": "api_error", "error": "missing response content"}
            else:
                response_text = choices[0]["message"]["content"]
                result = {
                    **common,
                    "status": "ok",
                    "response": response_text,
                    "finish_reason": choices[0].get("finish_reason"),
                    "usage": response_payload.get("usage", {}),
                    **parse_final_answer(record, response_text),
                    **analyze_thinking(response_text, tokenizer),
                }
                result["hit_max_tokens"] = result["finish_reason"] == "length"
        append_jsonl(args.output, result)
        print(
            f"[{position}/{len(pending)}] {record['record_id']} status={result['status']}"
            + (f" correct={int(result['answer_correct'])}" if result["status"] == "ok" else ""),
            flush=True,
        )


METRIC_FIELDS = (
    "answer_correct",
    "relaxed_answer_correct",
    "think_incidence",
    "reasoning_tokens",
    "output_tokens",
    "format_failure",
    "think_format_failure",
    "hit_max_tokens",
)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap_ci(
    time_rows: dict[str, dict[str, Any]],
    method_rows: dict[str, dict[str, Any]],
    ids: Sequence[str],
    *,
    replications: int,
    seed: int,
) -> dict[str, list[float] | None]:
    """Bootstrap paired examples and return percentile CIs for mean deltas."""
    if not ids:
        return {field: None for field in METRIC_FIELDS}
    deltas = {
        field: [
            float(method_rows[item][field]) - float(time_rows[item][field])
            for item in ids
        ]
        for field in METRIC_FIELDS
    }
    rng = random.Random(seed)
    samples = {field: [] for field in METRIC_FIELDS}
    n = len(ids)
    for _ in range(replications):
        selected = [rng.randrange(n) for _ in range(n)]
        for field in METRIC_FIELDS:
            values = deltas[field]
            samples[field].append(sum(values[index] for index in selected) / n)
    return {
        field: [percentile(values, 0.025), percentile(values, 0.975)]
        for field, values in samples.items()
    }


def summarize_group(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(records)}
    for field in METRIC_FIELDS:
        values = [float(item[field]) for item in records]
        result[field + ("_rate" if field not in {"reasoning_tokens", "output_tokens"} else "_mean")] = mean(values)
        if field in {"reasoning_tokens", "output_tokens"}:
            result[field + "_median"] = statistics.median(values) if values else None
    return result


def summarize_run(records_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ok = [item for item in records_by_id.values() if item.get("status") == "ok"]
    observed_tasks = [
        task
        for task in TASK_SPECS
        if any(item.get("task") == task for item in records_by_id.values())
    ]
    tasks = {
        task: summarize_group([item for item in ok if item["task"] == task])
        for task in observed_tasks
    }
    macro: dict[str, float | None] = {}
    metric_keys = [key for key in next(iter(tasks.values())).keys() if key != "n"]
    for key in metric_keys:
        present = [float(stats[key]) for stats in tasks.values() if stats[key] is not None]
        macro[key] = mean(present)
    return {
        "attempted_unique": len(records_by_id),
        "successful": len(ok),
        "api_errors": sum(item.get("status") != "ok" for item in records_by_id.values()),
        "overall_micro": summarize_group(ok),
        "overall_macro_task": {"n_tasks": len(observed_tasks), **macro},
        "per_task": tasks,
    }


def _load_run_for_summary(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = Path(str(path) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    latest = latest_results(path)
    expected_hash = manifest["run_config_sha256"]
    for item in latest.values():
        if item.get("run_config_sha256") != expected_hash:
            raise ValueError(f"mixed run_config hashes in {path}")
    return latest, manifest


def compare_runs(args: argparse.Namespace) -> dict[str, Any]:
    time_rows, time_manifest = _load_run_for_summary(args.time_run)
    method_rows, method_manifest = _load_run_for_summary(args.method_run)
    time_config = time_manifest["run_config"]
    method_config = method_manifest["run_config"]
    if time_config["role"] != "time_baseline" or method_config["role"] != "method":
        raise ValueError("comparison requires --time-run role=time_baseline and --method-run role=method")
    for key in (
        "subset_sha256",
        "protocol_sha256",
        "harness_sha256",
        "selected_ids_sha256",
        "generation",
        "task",
    ):
        if time_config[key] != method_config[key]:
            raise ValueError(f"runs are not paired: {key} differs")
    selected_count = int(time_config["selected_count"])
    time_ok = {key: value for key, value in time_rows.items() if value.get("status") == "ok"}
    method_ok = {key: value for key, value in method_rows.items() if value.get("status") == "ok"}
    common_ids = sorted(set(time_ok) & set(method_ok))
    if not args.allow_incomplete:
        if len(time_ok) != selected_count or len(method_ok) != selected_count:
            raise ValueError(
                f"incomplete runs: expected {selected_count}, TIME={len(time_ok)}, method={len(method_ok)}"
            )
        if set(time_ok) != set(method_ok):
            raise ValueError("successful record IDs differ between runs")

    def paired_block(ids: Sequence[str]) -> dict[str, Any]:
        metric_delta = {}
        for field in METRIC_FIELDS:
            values = [float(method_ok[item][field]) - float(time_ok[item][field]) for item in ids]
            metric_delta[field] = mean(values)
        accuracy_pairs = Counter(
            (bool(time_ok[item]["answer_correct"]), bool(method_ok[item]["answer_correct"]))
            for item in ids
        )
        return {
            "n": len(ids),
            "delta_method_minus_time": metric_delta,
            "delta_method_minus_time_95_ci": paired_bootstrap_ci(
                time_ok,
                method_ok,
                ids,
                replications=args.bootstrap_reps,
                seed=args.bootstrap_seed,
            ),
            "accuracy_pair_counts": {
                "both_correct": accuracy_pairs[(True, True)],
                "method_only_correct": accuracy_pairs[(False, True)],
                "time_only_correct": accuracy_pairs[(True, False)],
                "both_incorrect": accuracy_pairs[(False, False)],
            },
        }

    comparison = {
        "schema_version": SCHEMA_VERSION,
        "kind": "time_vs_method_public_eval",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subset_sha256": time_config["subset_sha256"],
        "protocol_sha256": time_config["protocol_sha256"],
        "paired_bootstrap": {
            "unit": "paired example",
            "statistic": "mean(method - TIME)",
            "interval": "95% percentile",
            "replications": args.bootstrap_reps,
            "seed": args.bootstrap_seed,
        },
        "time_baseline": {
            "run": str(args.time_run.resolve()),
            "run_config_sha256": time_manifest["run_config_sha256"],
            "summary": summarize_run(time_rows),
        },
        "method": {
            "run": str(args.method_run.resolve()),
            "run_config_sha256": method_manifest["run_config_sha256"],
            "summary": summarize_run(method_rows),
        },
        "paired": {
            "overall": paired_block(common_ids),
            "per_task": {
                task: paired_block([item for item in common_ids if time_ok[item]["task"] == task])
                for task in [time_config["task"]]
            },
        },
    }
    atomic_write_json(args.output, comparison)
    if args.csv_output:
        write_comparison_csv(args.csv_output, comparison)
    return comparison


def write_comparison_csv(path: Path, comparison: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task", "role", "n", "accuracy", "relaxed_accuracy",
                "think_incidence", "reasoning_tokens_mean", "output_tokens_mean", "format_failure",
                "think_format_failure", "hit_max_tokens",
            ]
        )
        for task in ["overall_micro", *comparison["paired"]["per_task"]]:
            for role in ("time_baseline", "method"):
                summary = comparison[role]["summary"]
                stats = summary["overall_micro"] if task == "overall_micro" else summary["per_task"][task]
                writer.writerow(
                    [
                        task, role, stats["n"], stats["answer_correct_rate"],
                        stats["relaxed_answer_correct_rate"], stats["think_incidence_rate"],
                        stats["reasoning_tokens_mean"],
                        stats["output_tokens_mean"], stats["format_failure_rate"],
                        stats["think_format_failure_rate"], stats["hit_max_tokens_rate"],
                    ]
                )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="download pinned data and freeze a subset")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path)
    prepare.add_argument("--per-task", type=int, default=256)
    prepare.add_argument("--selection-salt", default=SELECTION_SALT)

    verify = subparsers.add_parser("verify", help="verify a frozen subset and manifest")
    verify.add_argument("--subset", type=Path, required=True)
    verify.add_argument("--manifest", type=Path)

    run = subparsers.add_parser("run", help="evaluate one of the two permitted experiment roles")
    run.add_argument("--role", required=True, choices=("time_baseline", "method"))
    run.add_argument("--task", required=True, choices=tuple(TASK_SPECS))
    run.add_argument("--model-label", required=True)
    run.add_argument("--served-model", required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--tokenizer", type=Path, required=True)
    run.add_argument("--subset", type=Path, required=True)
    run.add_argument("--subset-manifest", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    run.add_argument("--api-key", default=os.environ.get("API_KEY", "some_random_key"))
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument("--seed", type=int, default=3407)
    run.add_argument("--timeout", type=float, default=300.0)
    run.add_argument("--retries", type=int, default=2)
    run.add_argument("--limit", type=int, help="diagnostic prefix only; recorded in run manifest")
    run.add_argument("--resume", action="store_true")

    summarize = subparsers.add_parser("summarize", help="compare exactly TIME baseline and method")
    summarize.add_argument("--time-run", type=Path, required=True)
    summarize.add_argument("--method-run", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--csv-output", type=Path)
    summarize.add_argument("--allow-incomplete", action="store_true")
    summarize.add_argument("--bootstrap-reps", type=int, default=10000)
    summarize.add_argument("--bootstrap-seed", type=int, default=3407)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            manifest_path = args.manifest or default_manifest_path(args.output)
            if args.output.exists() or manifest_path.exists():
                raise ValueError("refusing to overwrite frozen data or manifest; choose a new path")
            manifest = prepare_subset(args.output, manifest_path, args.per_task, args.selection_salt)
            print(
                f"wrote {manifest['record_count']} records; subset_sha256={manifest['subset_sha256']}"
            )
        elif args.command == "verify":
            manifest_path = args.manifest or default_manifest_path(args.subset)
            records, manifest = load_and_verify_subset(args.subset, manifest_path)
            print(
                f"verified {len(records)} records; subset_sha256={manifest['subset_sha256']} "
                f"protocol_sha256={manifest['protocol_sha256']}"
            )
        elif args.command == "run":
            if args.max_tokens < 1 or args.retries < 0 or args.timeout <= 0:
                raise ValueError("max-tokens/timeout must be positive and retries non-negative")
            run_evaluation(args)
        elif args.command == "summarize":
            if args.bootstrap_reps < 1:
                raise ValueError("bootstrap-reps must be positive")
            result = compare_runs(args)
            print(json.dumps(result["paired"], indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
