# Notebook Publication Checklist

Date (UTC): 2026-04-09
Notebook: `lab/aiprom-fhir-finetune.ipynb`
Model alias: `mlx-community-qwen2.5-coder-7b-instruct-8bit`

This file records the archived executed snapshot for the current 7B 8-bit release candidate.
It is not the single source of truth for the repository's supported model/configuration set.
The raw files under `lab/artifacts/` are local runtime artifacts ignored by git; this checklist records the intended executed snapshot and should be read together with the checked-in reports under `tunes/`, which are the publication-facing evidence.

## 1) Documentation and paths

- [x] README artifact paths aligned with real output layout (shared vs model-scoped).

## 2) Core artifacts (present)

- [x] `lab/artifacts/split_manifest.json`
- [x] `lab/artifacts/train.jsonl`
- [x] `lab/artifacts/val.jsonl`
- [x] `lab/artifacts/valid.jsonl`
- [x] `lab/artifacts/dataset_manifest.json`
- [x] `lab/artifacts/mlx-community-qwen2.5-coder-7b-instruct-8bit/training_config_stable.json`
- [x] `lab/artifacts/mlx-community-qwen2.5-coder-7b-instruct-8bit/checkpoints_stable/adapters.safetensors`
- [ ] GGUF export artifact for the 7B 8-bit path

## 3) Reporting artifacts (present)

- [x] `tunes/models/mlx-community-qwen2.5-coder-7b-instruct-8bit.md`
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

- Status: **READY** for publication as the current best evaluated adapter snapshot.
- Reason: A/B evaluation artifacts exist, and the recorded decision is **GO** for `mlx-community-qwen2.5-coder-7b-instruct-8bit`.
- Constraint: GGUF export is not part of the validated 7B 8-bit release path because the current fuse-plus-convert workflow is not compatible with that quantized base.

## 7) Pending actions (without running full notebook)

1. Keep `lab/aiprom-fhir-finetune.ipynb` aligned to the executed 7B 8-bit snapshot used for the checked-in reports.
2. Re-run `AIPROM_RUN_EVALUATION=1` and `AIPROM_UPDATE_TUNES_REPORTS=1` only when intentionally regenerating the published A/B evidence.
3. Keep `AIPROM_RUN_EXPORT=0` on the 7B 8-bit path unless you intentionally switch to a non-quantized export workflow.
4. Treat the publication notebook as a curated executed snapshot, not as a mandatory clean, output-free notebook.
