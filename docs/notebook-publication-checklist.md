# Notebook Publication Checklist

Date (UTC): 2026-03-08
Notebook: `lab/aiprom-fhir-finetune.ipynb`
Model alias: `mlx-community-qwen2.5-coder-3b-instruct-bf16`

## 1) Documentation and paths

- [x] README artifact paths aligned with real output layout (shared vs model-scoped).

## 2) Core artifacts (present)

- [x] `lab/artifacts/split_manifest.json`
- [x] `lab/artifacts/train.jsonl`
- [x] `lab/artifacts/val.jsonl`
- [x] `lab/artifacts/valid.jsonl`
- [x] `lab/artifacts/dataset_manifest.json`
- [x] `lab/artifacts/training_run_log.json`
- [x] `lab/artifacts/energy/emissions.csv`
- [x] `lab/artifacts/mlx-community-qwen2.5-coder-3b-instruct-bf16/training_config_stable.json`
- [x] `lab/artifacts/mlx-community-qwen2.5-coder-3b-instruct-bf16/checkpoints_stable/adapters.safetensors`
- [x] `lab/artifacts/mlx-community-qwen2.5-coder-3b-instruct-bf16/fused-noquant/aiphet-qwen2.5-3b-fhir-f16.gguf`

## 3) Reporting artifacts (present)

- [x] `tunes/models/mlx-community-qwen2.5-coder-3b-instruct-bf16.md`
- [x] `tunes/leaderboard.md`
- [x] `tunes/leaderboard.csv`

## 4) Evaluation/decision artifacts (missing)

- [ ] `lab/artifacts/ab_rule_eval.json`
- [ ] `lab/artifacts/adapter_go_no_go.json`

## 5) Release decision (current)

- Status: **NOT READY** for final publication if strict release requires A/B evaluation + GO/NO-GO artifacts.
- Status: **READY** for publication as training + export reproducibility report only.

## 6) Pending actions (without running full notebook)

1. Generate missing evaluation artifacts by running only the evaluation/go-no-go section cells.
2. Re-run `scripts/update_tunes_reports.py` if you want leaderboard fields to include evaluation metrics.
3. Freeze notebook outputs and clear any transient runtime warnings before final share.
