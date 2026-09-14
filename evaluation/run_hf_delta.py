#!/usr/bin/env python3
"""Run DeltaTIMEBench directly with a Hugging Face causal LM."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_template(path: str) -> str:
    spec = importlib.util.spec_from_file_location("time_chat_template", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import chat template from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CHAT_TEMPLATE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", help="Optional PEFT adapter to load on top of --model.")
    parser.add_argument("--input", default="auto-res/data/delta_scenarios_v0.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-chat-template")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=False to chat-template rendering (Qwen3 baseline).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit raw records; may split counterfactual groups. Prefer --limit-groups.",
    )
    parser.add_argument(
        "--limit-groups",
        type=int,
        help="Evaluate the first N complete group_id clusters (preferred for smoke runs).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.limit is not None and args.limit_groups is not None:
        parser.error("--limit and --limit-groups are mutually exclusive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if args.time_chat_template:
        tokenizer.chat_template = load_template(args.time_chat_template)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).eval()

    records = json.loads(Path(args.input).read_text())
    if args.limit_groups is not None:
        if args.limit_groups < 1:
            parser.error("--limit-groups must be positive")
        selected_groups = []
        for item in records:
            group_id = item.get("group_id") or item.get("pair_id") or item.get("domain")
            if group_id not in selected_groups:
                selected_groups.append(group_id)
            if len(selected_groups) == args.limit_groups:
                break
        selected = set(selected_groups)
        records = [
            item
            for item in records
            if (item.get("group_id") or item.get("pair_id") or item.get("domain"))
            in selected
        ]
    if args.limit:
        records = records[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prior = {}
    if args.resume and output.exists():
        prior = {x["scenario_id"]: x for x in json.loads(output.read_text())}

    completed = [prior[x["scenario_id"]] for x in records if x["scenario_id"] in prior]
    pending = [x for x in records if x["scenario_id"] not in prior]
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        prompts = []
        for item in batch:
            template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if args.disable_thinking:
                template_kwargs["enable_thinking"] = False
            prompts.append(tokenizer.apply_chat_template(item["messages"], **template_kwargs))
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            kwargs["temperature"] = args.temperature
        with torch.inference_mode():
            generated = model.generate(**inputs, **kwargs)
        responses = tokenizer.batch_decode(
            generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        for item, response in zip(batch, responses, strict=True):
            result = dict(item)
            result["response"] = response
            result["model"] = args.model
            if args.adapter:
                result["adapter"] = args.adapter
            completed.append(result)
        output.write_text(json.dumps(completed, indent=2, ensure_ascii=False) + "\n")
        end = min(start + len(batch), len(pending))
        print(f"[{end}/{len(pending)}] {batch[-1]['scenario_id']}: {responses[-1][:160]!r}", flush=True)


if __name__ == "__main__":
    main()
