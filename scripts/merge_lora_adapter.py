#!/usr/bin/env python3
"""Safely merge a local LoRA adapter into a local causal-LM checkpoint.

The script is intentionally conservative because its output is intended for
paper evaluation and vLLM serving.  It refuses a non-empty destination, checks
that the adapter declares the requested base, merges in a staging directory,
validates the saved full-model layout, records SHA-256 provenance, and only
then atomically publishes the destination.

Use ``--dry-run`` for an import/config/tokenizer/layout preflight.  Dry-run does
not load model tensors and does not create the output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "deltatime-lora-merge-0.1"
WEIGHT_SUFFIXES = (".safetensors", ".bin")
TOKENIZER_MARKERS = (
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "spiece.model",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_inventory(path: Path, *, include_sha256: bool = True) -> list[dict]:
    """Return a stable recursive inventory without following directory links."""
    root = path.resolve(strict=True)
    rows = []
    for item in sorted(root.rglob("*")):
        if not item.is_file():
            continue
        row = {
            "path": item.relative_to(root).as_posix(),
            "size_bytes": item.stat().st_size,
        }
        if include_sha256:
            row["sha256"] = file_sha256(item)
        rows.append(row)
    return rows


def inventory_digest(rows: list[dict]) -> str:
    """Hash the canonical inventory (including per-file hashes when present)."""
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def require_local_directory(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} directory does not exist: {path}") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}")
    return resolved


def same_local_reference(declared: str, requested: Path) -> bool:
    if not declared:
        return False
    candidate = Path(declared).expanduser()
    if not candidate.is_absolute():
        return False
    try:
        return candidate.resolve(strict=True) == requested.resolve(strict=True)
    except FileNotFoundError:
        return False


def validate_static_layout(
    base: Path,
    adapter: Path,
    *,
    allow_base_mismatch: bool,
) -> dict:
    base_config_path = base / "config.json"
    adapter_config_path = adapter / "adapter_config.json"
    if not base_config_path.is_file():
        raise ValueError(f"base lacks config.json: {base}")
    if not adapter_config_path.is_file():
        raise ValueError(f"adapter lacks adapter_config.json: {adapter}")
    base_config = read_json(base_config_path)
    adapter_config = read_json(adapter_config_path)

    peft_type = str(adapter_config.get("peft_type", "")).upper()
    task_type = str(adapter_config.get("task_type", "")).upper()
    if peft_type != "LORA":
        raise ValueError(f"only LoRA adapters are supported, got peft_type={peft_type!r}")
    if task_type != "CAUSAL_LM":
        raise ValueError(f"expected task_type='CAUSAL_LM', got {task_type!r}")
    if adapter_config.get("modules_to_save"):
        raise ValueError(
            "modules_to_save is non-empty; this merger only validates pure LoRA adapters"
        )
    declared_base = str(adapter_config.get("base_model_name_or_path") or "")
    matches = same_local_reference(declared_base, base)
    if not matches and not allow_base_mismatch:
        raise ValueError(
            "adapter base mismatch: adapter declares "
            f"{declared_base!r}, requested base is {str(base)!r}; "
            "use --allow-base-mismatch only after an explicit compatibility audit"
        )

    architectures = base_config.get("architectures") or []
    declared_class = (adapter_config.get("auto_mapping") or {}).get(
        "base_model_class"
    )
    if declared_class and architectures and declared_class not in architectures:
        raise ValueError(
            f"adapter expects {declared_class!r}, base architectures are {architectures!r}"
        )
    if base_config.get("quantization_config"):
        raise ValueError(
            "base config contains quantization_config; merge from an unquantized full "
            "checkpoint to produce a portable vLLM model"
        )

    adapter_weights = [
        item
        for item in (adapter / "adapter_model.safetensors", adapter / "adapter_model.bin")
        if item.is_file() and item.stat().st_size > 0
    ]
    if len(adapter_weights) != 1:
        raise ValueError(
            "adapter must contain exactly one non-empty adapter_model.safetensors or "
            f"adapter_model.bin, found {[x.name for x in adapter_weights]}"
        )
    if not any(
        item.is_file() and item.stat().st_size > 0
        for item in base.iterdir()
        if item.name.endswith(WEIGHT_SUFFIXES)
    ):
        raise ValueError(f"base contains no non-empty model weight file: {base}")

    return {
        "base_model_type": base_config.get("model_type"),
        "base_architectures": architectures,
        "adapter_peft_type": peft_type,
        "adapter_task_type": task_type,
        "adapter_declared_base": declared_base,
        "adapter_declared_base_matches": matches,
        "adapter_base_model_class": declared_class,
        "adapter_weight_file": adapter_weights[0].name,
        "target_modules": sorted(adapter_config.get("target_modules") or []),
        "lora_rank": adapter_config.get("r"),
        "lora_alpha": adapter_config.get("lora_alpha"),
    }


def import_versions() -> tuple[dict, dict[str, Any]]:
    """Import the exact runtime used for merge and return modules plus versions."""
    import peft
    import safetensors
    import torch
    import transformers
    from peft import PeftConfig, PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    modules = {
        "torch": torch,
        "PeftConfig": PeftConfig,
        "PeftModel": PeftModel,
        "AutoConfig": AutoConfig,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "safetensors": safetensors.__version__,
        "platform": platform.platform(),
    }
    return versions, modules


def runtime_preflight(
    base: Path,
    adapter: Path,
    *,
    trust_remote_code: bool,
    require_chat_template: bool,
) -> tuple[dict, Any, dict]:
    versions, modules = import_versions()
    common = {
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
    }
    base_config = modules["AutoConfig"].from_pretrained(base, **common)
    peft_config = modules["PeftConfig"].from_pretrained(adapter, **common)

    tokenizer_source = adapter
    try:
        tokenizer = modules["AutoTokenizer"].from_pretrained(adapter, **common)
    except (OSError, ValueError):
        tokenizer_source = base
        tokenizer = modules["AutoTokenizer"].from_pretrained(base, **common)
    if require_chat_template and not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"tokenizer loaded from {tokenizer_source} has no chat_template; "
            "vLLM chat-completions serving would be protocol-ambiguous"
        )
    if not hasattr(modules["PeftModel"], "from_pretrained"):
        raise RuntimeError("installed peft.PeftModel lacks from_pretrained")

    # In PEFT 0.18 merge_and_unload is delegated from the instantiated tuner,
    # so it is intentionally not present on the bare PeftModel class.
    from peft.tuners.lora.model import LoraModel

    if not hasattr(LoraModel, "merge_and_unload"):
        raise RuntimeError("installed PEFT LoRA implementation lacks merge_and_unload")

    report = {
        "base_config_class": type(base_config).__name__,
        "base_model_type": getattr(base_config, "model_type", None),
        "base_architectures": getattr(base_config, "architectures", None),
        "peft_config_class": type(peft_config).__name__,
        "peft_type": str(getattr(peft_config, "peft_type", None)),
        "task_type": str(getattr(peft_config, "task_type", None)),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_source": str(tokenizer_source),
        "tokenizer_size": len(tokenizer),
        "chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
        "chat_template_sha256": (
            hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
            if getattr(tokenizer, "chat_template", None)
            else None
        ),
        "peft_merge_api": "peft.tuners.lora.model.LoraModel.merge_and_unload",
    }
    return versions, tokenizer, {"report": report, "modules": modules}


def validate_weight_index(output: Path) -> dict:
    index_files = sorted(output.glob("*.index.json"))
    weight_files = sorted(
        item
        for item in output.iterdir()
        if item.is_file()
        and item.name.endswith(WEIGHT_SUFFIXES)
        and not item.name.startswith("adapter_model")
        and item.stat().st_size > 0
    )
    if not weight_files:
        raise ValueError("merged output has no non-empty full-model weight files")

    indexed_shards: list[str] = []
    if index_files:
        if len(index_files) != 1:
            raise ValueError(f"expected at most one weight index, found {index_files}")
        index = read_json(index_files[0])
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid or empty weight_map in {index_files[0]}")
        indexed_shards = sorted(set(map(str, weight_map.values())))
        missing = [name for name in indexed_shards if not (output / name).is_file()]
        if missing:
            raise ValueError(f"weight index references missing shards: {missing[:5]}")

    return {
        "weight_files": [item.name for item in weight_files],
        "weight_index": index_files[0].name if index_files else None,
        "indexed_shards": indexed_shards,
    }


def validate_merged_output(
    output: Path,
    *,
    modules: dict[str, Any],
    trust_remote_code: bool,
    require_chat_template: bool,
) -> dict:
    if (output / "adapter_config.json").exists() or any(
        output.glob("adapter_model.*")
    ):
        raise ValueError("merged output still contains PEFT adapter artifacts")
    if not (output / "config.json").is_file():
        raise ValueError("merged output lacks config.json")

    weights = validate_weight_index(output)
    common = {"local_files_only": True, "trust_remote_code": trust_remote_code}
    config = modules["AutoConfig"].from_pretrained(output, **common)
    tokenizer = modules["AutoTokenizer"].from_pretrained(output, **common)
    if require_chat_template and not getattr(tokenizer, "chat_template", None):
        raise ValueError("saved tokenizer lost its chat_template")
    if not any((output / marker).is_file() for marker in TOKENIZER_MARKERS):
        raise ValueError("merged output lacks tokenizer vocabulary/model assets")

    return {
        **weights,
        "config_class": type(config).__name__,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_size": len(tokenizer),
        "chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
        "vllm_layout_ready": True,
    }


def torch_dtype(name: str, torch_module: Any) -> Any:
    mapping = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    return mapping[name]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-base-mismatch", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--allow-missing-chat-template",
        action="store_true",
        help="Disable the default vLLM chat-template requirement.",
    )
    parser.add_argument(
        "--skip-input-hashes",
        action="store_true",
        help="Debug only: omit expensive input SHA-256 values from the manifest.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = require_local_directory(args.base, "base")
    adapter = require_local_directory(args.adapter, "adapter")
    output = args.output.expanduser().resolve(strict=False)
    if output == base or output == adapter:
        raise ValueError("output must differ from both base and adapter")
    if output.exists():
        raise ValueError(
            f"refusing existing output path {output}; choose a fresh destination"
        )

    static = validate_static_layout(
        base, adapter, allow_base_mismatch=args.allow_base_mismatch
    )
    require_chat_template = not args.allow_missing_chat_template
    versions, tokenizer, runtime = runtime_preflight(
        base,
        adapter,
        trust_remote_code=args.trust_remote_code,
        require_chat_template=require_chat_template,
    )
    preflight = {
        "schema_version": SCHEMA_VERSION,
        "mode": "dry_run" if args.dry_run else "merge",
        "base": str(base),
        "adapter": str(adapter),
        "output": str(output),
        "merge": {
            "dtype": args.dtype,
            "max_shard_size": args.max_shard_size,
            "safe_serialization": True,
            "safe_merge": True,
            "device_map": "cpu",
            "local_files_only": True,
            "trust_remote_code": args.trust_remote_code,
            "require_chat_template": require_chat_template,
        },
        "static_validation": static,
        "runtime_validation": runtime["report"],
        "software": versions,
    }
    if args.dry_run:
        print(json.dumps(preflight, indent=2))
        return

    include_hashes = not args.skip_input_hashes
    base_inventory = directory_inventory(base, include_sha256=include_hashes)
    adapter_inventory = directory_inventory(adapter, include_sha256=include_hashes)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.merge-", dir=str(output.parent))
    )
    published = False
    try:
        modules = runtime["modules"]
        dtype = torch_dtype(args.dtype, modules["torch"])
        model = modules["AutoModelForCausalLM"].from_pretrained(
            base,
            dtype=dtype,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=args.trust_remote_code,
        )
        peft_model = modules["PeftModel"].from_pretrained(
            model,
            adapter,
            is_trainable=False,
            local_files_only=True,
        )
        if not hasattr(peft_model, "merge_and_unload"):
            raise RuntimeError("instantiated PEFT model lacks merge_and_unload")
        merged = peft_model.merge_and_unload(progressbar=True, safe_merge=True)
        merged.save_pretrained(
            staging,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        tokenizer.save_pretrained(staging)
        validation = validate_merged_output(
            staging,
            modules=modules,
            trust_remote_code=args.trust_remote_code,
            require_chat_template=require_chat_template,
        )
        output_inventory = directory_inventory(staging, include_sha256=True)
        manifest = {
            **preflight,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "environment": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "inputs": {
                "base": {
                    "path": str(base),
                    "inventory_sha256": inventory_digest(base_inventory),
                    "files": base_inventory,
                },
                "adapter": {
                    "path": str(adapter),
                    "inventory_sha256": inventory_digest(adapter_inventory),
                    "files": adapter_inventory,
                },
            },
            "output_validation": validation,
            # The manifest is deliberately excluded to avoid a circular hash.
            "output_before_manifest": {
                "inventory_sha256": inventory_digest(output_inventory),
                "files": output_inventory,
            },
        }
        (staging / "merge_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        staging.rename(output)
        published = True
        print(json.dumps({"output": str(output), "manifest": str(output / "merge_manifest.json"), "validation": validation}, indent=2))
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
