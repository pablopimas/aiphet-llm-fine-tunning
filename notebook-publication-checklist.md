# Notebook Publication Checklist

Date (UTC): 2026-04-08
Notebook: `lab/aiprom-fhir-finetune.ipynb`
Model alias: `mlx-community-qwen2.5-coder-3b-instruct-bf16`

This file records an archived evaluated snapshot for the 3B run.
It is not the single source of truth for the repository's supported model/configuration set.
The raw files under `lab/artifacts/` are local runtime artifacts ignored by git; this checklist records the intended local snapshot and should be read together with the checked-in reports under `tunes/`, which are the publication-facing evidence.

## 1) Documentation and paths

- [x] README artifact paths aligned with real output layout (shared vs model-scoped).

## 2) Core artifacts (present)

- [x] `lab/artifacts/split_manifest.json`
- [x] `lab/artifacts/train.jsonl`
- [x] `lab/artifacts/val.jsonl`
- [x] `lab/artifacts/valid.jsonl`
- [x] `lab/artifacts/dataset_manifest.json`
- [x] `lab/artifacts/mlx-community-qwen2.5-coder-3b-instruct-bf16/training_config_stable.json`
- [x] `lab/artifacts/mlx-community-qwen2.5-coder-3b-instruct-bf16/checkpoints_stable/adapters.safetensors`
- [x] `lab/artifacts/mlx-community-qwen2.5-coder-3b-instruct-bf16/fused-noquant/aiphet-qwen2.5-3b-fhir-f16.gguf`

## 3) Reporting artifacts (present)

- [x] `tunes/models/mlx-community-qwen2.5-coder-3b-instruct-bf16.md`
- [x] `tunes/leaderboard.md`
- [x] `tunes/leaderboard.csv`

## 4) Runtime logs (excluded from publication snapshot)

- [x] `lab/artifacts/training_run_log.json`
- [x] `lab/artifacts/energy/emissions.csv`
- [x] `lab/artifacts/energy/*_energy.json`
- [x] `lab/artifacts/*/gguf_export_log.json`

## 5) Evaluation/decision artifacts (present)

- [x] `lab/artifacts/ab_rule_eval.json`
- [x] `lab/artifacts/ab_rule_eval_analysis.json`
- [x] `lab/artifacts/adapter_go_no_go.json`

## 6) Release decision (current)

- Status: **NOT READY** for final model publication as a successful adapter release.
- Reason: A/B evaluation artifacts exist, but the recorded decision is **NO_GO** for `mlx-community-qwen2.5-coder-3b-instruct-bf16`.
- Status: **READY** for publication as a reproducibility snapshot that documents a negative evaluation result.

## 7) Pending actions (without running full notebook)

1. Keep `lab/aiprom-fhir-finetune.ipynb` with execution flags disabled by default for publication-safe sharing.
2. Re-enable `AIPROM_RUN_EVALUATION=1` and `AIPROM_UPDATE_TUNES_REPORTS=1` only when intentionally regenerating A/B artifacts.
3. Decide whether to publish the current 3B run as a negative-result reproducibility snapshot or rerun after improving adapter quality.
4. Keep the published notebook in a clean, output-free state unless intentionally sharing a fully curated executed snapshot.
