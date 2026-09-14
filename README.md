# DeltaAlign

Official code release for **Think on Change, Not Time: Counterfactual Trigger Alignment for Selective Explicit Reasoning**.

> **Paper status:** under review at IEEE ICASSP 2027.

DeltaAlign extends [TIME](https://github.com/The-Coherence-Initiative/TIME) with matched counterfactual supervision for selective explicit reasoning. Instead of treating elapsed time itself as a reason to emit a `<think>` block, the method aligns the first decision token with answer-relevant state changes while suppressing re-triggering for inert, irrelevant, or decision-preserving updates.

This repository contains the compact source release used by our experiments. Model checkpoints, adapters, generated datasets, raw generations, and logs are intentionally excluded.

## Method

Each counterfactual group shares the same base dialogue and decision rule, then varies only the update:

- `inert_short` / `inert_long`: time passes without a state change;
- `irrelevant_change`: a contextual detail changes but the decision does not;
- `relevant_stable`: a decision variable changes without crossing its threshold;
- `assumption_break_short` / `assumption_break_long`: the answer-relevant state changes and explicit reasoning should re-trigger; and
- `temporal_before_*` / `temporal_crossing`: a deadline remains unbroken or is crossed.

The canonical PairCal objective combines assistant-token language modeling with:

1. a matched positive-vs-negative first-token ranking loss;
2. separate positive and negative trigger-calibration losses; and
3. an optional short/long positive-consistency loss.

The trigger score is the first-decision-token logit difference
`logit(<think>) - logit(<answer>)`. The implementation is in `PairwiseTrainer.compute_loss` in [`scripts/train_cf_delta_pairwise.py`](scripts/train_cf_delta_pairwise.py).

## Repository layout

```text
.
├── scripts/
│   ├── build_cf_delta_data.py          # matched counterfactual data builder
│   ├── train_cf_delta_pairwise.py      # canonical PairCal trainer
│   ├── train_cf_delta.py               # ordinary SFT control
│   ├── train_cf_delta_pairwise_manifest.py
│   └── merge_lora_adapter.py
├── trainer_pkg/chat_template.py        # TIME-compatible chat template
├── evaluation/
│   ├── run_hf_delta.py                 # local generation
│   ├── score_delta_outputs.py          # group-aware scoring
│   ├── compare_delta_outputs.py        # paired single-seed comparison
│   └── compare_multiseed_delta_outputs.py
├── public_eval/                         # frozen GSM8K / ARC-C / BoolQ harness
└── requirements.txt
```

## Installation

Python 3.12 and a CUDA-capable environment are recommended for training.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The frozen public-task subset can be prepared with the additional lightweight dependencies:

```bash
python -m pip install -r public_eval/requirements-prepare.txt
```

## Quick start

### 1. Build matched counterfactual data

```bash
python scripts/build_cf_delta_data.py \
  --output-dir data/cf_delta_v0_2 \
  --train-groups 256 \
  --dev-groups 64 \
  --test-groups 128 \
  --seed 3407
```

The default build writes grouped train/dev/test records, SFT conversations, and a compact manifest. All siblings from a group remain in the same split, and threshold domains are held out between train and evaluation.

To mix in TIME retention data, pass one or more assistant-ending TIME JSON files through `--replay-data`.

### 2. Train the primary DeltaAlign adapter

`--model` should point to a merged TIME checkpoint. The primary configuration used in the paper is:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_cf_delta_pairwise.py \
  --model /path/to/TIME-4B \
  --train-records data/cf_delta_v0_2/train_records.json \
  --output-dir outputs/deltaalign-primary-seed3407 \
  --max-length 512 \
  --epochs 1 \
  --learning-rate 2e-5 \
  --batch-size 2 \
  --gradient-accumulation 8 \
  --lora-r 16 \
  --lora-alpha 16 \
  --ranking-weight 0.25 \
  --ranking-margin 1.0 \
  --positive-calibration-weight 0.375 \
  --negative-calibration-weight 0.25 \
  --positive-consistency-weight 0.0 \
  --seed 3407
```

Set `--positive-consistency-weight 0.015` for the consistency extension. The trainer writes the resolved configuration, the exact pair manifest, training metrics, and the final LoRA adapter under the selected output directory.

### 3. Generate and score held-out outputs

```bash
python evaluation/run_hf_delta.py \
  --model /path/to/TIME-4B \
  --adapter outputs/deltaalign-primary-seed3407/final_adapter \
  --input data/cf_delta_v0_2/test_records.json \
  --time-chat-template trainer_pkg/chat_template.py \
  --output outputs/deltaalign-primary-seed3407-test.json

python evaluation/score_delta_outputs.py \
  --input outputs/deltaalign-primary-seed3407-test.json \
  --summary outputs/deltaalign-primary-seed3407-summary.json \
  --scored-output outputs/deltaalign-primary-seed3407-scored.json \
  --bootstrap-reps 10000 \
  --seed 3407
```

The scorer keeps counterfactual siblings together during bootstrap resampling and reports trigger behavior, answer accuracy, format validity, reasoning length, and complete-group joint success.

For a paired comparison against TIME:

```bash
python evaluation/compare_delta_outputs.py \
  --file-a outputs/time-test.json \
  --file-b outputs/deltaalign-primary-seed3407-test.json \
  --label-a TIME-4B \
  --label-b DeltaAlign \
  --output outputs/time-vs-deltaalign.json \
  --markdown outputs/time-vs-deltaalign.md
```

## Public-task evaluation

`public_eval/harness.py` freezes deterministic 256-example subsets of GSM8K, ARC-Challenge, and BoolQ, evaluates an OpenAI-compatible endpoint, and computes strict/relaxed answer accuracy together with reasoning and format metrics.

```bash
python public_eval/harness.py prepare --output public_eval/data/frozen_v1.jsonl
python public_eval/harness.py verify --subset public_eval/data/frozen_v1.jsonl
```

Use `python public_eval/harness.py run --help` for endpoint options and `summarize` for paired TIME-vs-DeltaAlign comparisons. `public_eval/aggregate_matrix.py` validates and aggregates the complete TIME / Primary / Consistency matrix.

CPU-only tests for the public evaluator can be run with:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest public_eval/tests
```

## Reproducibility notes

- Adapter seeds used in the main experiments: `1234`, `2025`, and `3407`.
- Effective training batch: 16 counterfactual pairs.
- Data construction and pair ordering are deterministic for a fixed seed.
- Pair manifests and resolved configurations are written by the trainer for provenance.
- Generated data, model weights, adapters, raw outputs, caches, and experiment logs are excluded from Git.
- The public `train_cf_delta_pairwise_manifest.py` only replaces the experiment server's hard-coded GPU-policy path with optional `--gpu-policy-*` flags; its pair validation and training semantics are unchanged.
- Historical experiments used TIME commit [`3be5f44`](https://github.com/The-Coherence-Initiative/TIME/tree/3be5f4441207a7c9b62860966f3563c723bdb614) and TIMEBench commit [`ba3527f`](https://github.com/The-Coherence-Initiative/TIMEBench/tree/ba3527f701df09002bcea04d69e78f18aa15bf4d).

## Acknowledgements

DeltaAlign is built on TIME and uses a TIME-compatible Qwen3 chat template. We thank the authors of TIME and TIMEBench for releasing their training and evaluation code.

## Citation

Citation metadata will be added after the ICASSP review process.
