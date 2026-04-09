<!--
Author: Pablo Pimàs
Email: pablo@pimas.cat
Date: 2026-02-22
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
SPDX-License-Identifier: CC-BY-4.0
Reference: https://creativecommons.org/licenses/by/4.0/
-->

# aiprom-llm

This repository contains the dataset, notebook workflow, and utility scripts used to fine-tune MLX-compatible Qwen models for **FHIR R4 Questionnaire** generation.

The documented workflow is aligned with the checked-in assets in this repository:

- dataset: `data/synthetic-aiprom-1500-firh4.jsonl`
- notebook: `lab/aiprom-fhir-finetune.ipynb`
- supported configs: `configs/Qwen2.5-Coder-7B-Instruct-8bit.yaml`, `configs/Qwen2.5-Coder-7B-Instruct-bf16.yaml`, and `configs/Qwen2.5-Coder-3B-Instruct-bf16.yaml`
- reporting: `tunes/leaderboard.md` and `tunes/models/*.md`

## Artifact policy

The repository documents workflows that produce local experiment artifacts under `lab/artifacts/`, but that directory is intentionally ignored by git.

- `lab/artifacts/` is treated as local runtime state, not as versioned source.
- Published repository snapshots keep the notebook, configs, scripts, tune reports, and packaged assets under `tunes/`.
- Documents such as `notebook-publication-checklist.md` describe an archived executed snapshot and should be read together with the checked-in reports under `tunes/`.

## Current evaluated snapshots

| Model | Alias | Status | Report | Updated at (UTC) |
|---|---|---|---|---|
| mlx-community/Qwen2.5-Coder-7B-Instruct-8bit | `mlx-community-qwen2.5-coder-7b-instruct-8bit` | `GO` | `tunes/models/mlx-community-qwen2.5-coder-7b-instruct-8bit.md` | 2026-04-09T20:00:42.654667+00:00 |
| mlx-community/Qwen2.5-Coder-3B-Instruct-bf16 | `mlx-community-qwen2.5-coder-3b-instruct-bf16` | `NO_GO` | `tunes/models/mlx-community-qwen2.5-coder-3b-instruct-bf16.md` | 2026-04-08T13:40:35.972163+00:00 |
| mlx-community/Qwen2.5-Coder-7B-Instruct-bf16 | `mlx-community-qwen2.5-coder-7b-instruct-bf16` | `NO_GO` | `tunes/models/mlx-community-qwen2.5-coder-7b-instruct-bf16.md` | 2026-04-07T11:55:33.717215+00:00 |

These rows summarize the latest checked-in published reports. The current best checked-in snapshot is the 7B 8-bit run, and the publication checklist should be read as an executed snapshot record rather than as a clean-notebook requirement.

## Documents

- [Datasets](datasets.md)
- [Notebook Publication Checklist](notebook-publication-checklist.md)

## What is in this repo

- Prompt/completion SFT dataset under `data/`
- Reproducible end-to-end notebook workflow under `lab/`
- Training/config helpers under `configs/`
- Validation and reporting scripts under `scripts/`
- Tune reports and packaged release assets under `tunes/`

## Quick start

Create and install the Poetry environment:

```bash
poetry env use 3.11
poetry install
```

Run an interactive shell in the project environment:

```bash
poetry shell
```

Validate the current committed dataset:

```bash
poetry run python scripts/validate_prompt_completion_dataset.py data/synthetic-aiprom-1500-firh4.jsonl
```

Run the notebook workflow from the repository root:

```bash
poetry run jupyter lab lab/aiprom-fhir-finetune.ipynb
```

If you prefer the CLI training entrypoint, pass the checked-in YAML directly:

```bash
poetry run python -m mlx_lm lora -c configs/Qwen2.5-Coder-7B-Instruct-8bit.yaml
```

## Rebuild published summaries

Use this sequence when you want to rebuild the publication-facing summaries from local runtime artifacts:

1. Validate the dataset:

```bash
poetry run python scripts/validate_prompt_completion_dataset.py data/synthetic-aiprom-1500-firh4.jsonl
```

2. Open the notebook from the repository root:

```bash
poetry run jupyter lab lab/aiprom-fhir-finetune.ipynb
```

3. Regenerate A/B evaluation artifacts locally from the notebook only when you intentionally enable `AIPROM_RUN_EVALUATION=1`.

4. Refresh the checked-in tune reports from the latest local artifacts:

```bash
poetry run python scripts/update_tunes_reports.py --repo-root .
```

To refresh a single model row/report:

```bash
poetry run python scripts/update_tunes_reports.py --repo-root . --model-name "mlx-community/Qwen2.5-Coder-7B-Instruct-8bit"
```

## Current workflow target

This repository currently documents and ships a workflow for generating **complete FHIR R4 Questionnaire JSON** objects from synthetic prompt/completion pairs.

It does not currently ship the historical 150-example QuestionnaireItem dataset that older drafts of the docs referenced.

## Training notebook (end-to-end)

Use the notebook [lab/aiprom-fhir-finetune.ipynb](lab/aiprom-fhir-finetune.ipynb) for the full reproducible workflow:

- environment and reproducibility snapshot,
- dataset parsing and FHIR structural validation,
- deterministic train/validation split,
- train/val artifact materialization,
- MLX-LM LoRA training command execution,
- quantitative and qualitative evaluation sections.

### Model selection (single source of truth)

The notebook now reads model selection from the first configuration cell and supports per-model artifact isolation.

- `AIPROM_MODEL_BACKEND` (currently supported by this notebook: `mlx_lm`)
- `AIPROM_CONFIG` (recommended: path to one of the checked-in YAML configs)
- `AIPROM_MODEL_NAME` (optional override; normally derived from config)
- `AIPROM_MODEL_ALIAS` (optional override for artifact folder naming)

If `AIPROM_MODEL_ALIAS` is not provided, it is derived from `AIPROM_MODEL_NAME`.

Current supported configs:

- `configs/Qwen2.5-Coder-7B-Instruct-8bit.yaml`
- `configs/Qwen2.5-Coder-7B-Instruct-bf16.yaml`
- `configs/Qwen2.5-Coder-3B-Instruct-bf16.yaml`

### What artifacts does it generate?

The notebook writes artifacts in two scopes:

- Shared dataset/run artifacts in `lab/artifacts/`
- Model-specific artifacts in `lab/artifacts/<MODEL_ALIAS>/`

Typical files include:

- Shared (`lab/artifacts/`):
	- `split_manifest.json`
	- `train.jsonl`
	- `val.jsonl`
	- `valid.jsonl`
	- `dataset_manifest.json`
- Model-specific (`lab/artifacts/<MODEL_ALIAS>/`):
	- `training_config_stable.json`
	- checkpoints/adapters in `checkpoints_stable/`
	- fused exports under `fused/` and `fused-noquant/`

Transient runtime logs such as `training_run_log.json`, `gguf_export_log.json`, `emissions.csv`, and `*_energy.json` are generated locally during experiments but are not part of the published repository snapshot.

Checked-in reports under `tunes/` are the publication-facing summaries derived from those local artifacts.

### Does it produce a specialized model?

Yes. Running the training cell produces a **specialized LoRA adapter** over the selected base Qwen model for FHIR Questionnaire generation. It does not replace the base model weights; it creates adapter weights that are loaded on top of the base model.

### Quick inference with the trained adapter

After training, you can run an inference test with MLX-LM using the adapter directory:

```bash
poetry run python -m mlx_lm generate \
	--ignore-chat-template \
	--model mlx-community/Qwen2.5-Coder-7B-Instruct-8bit \
	--adapter-path lab/artifacts/<MODEL_ALIAS>/checkpoints_stable \
	--prompt "<|im_start|>system\nYou are a FHIR R4 expert. Return only valid JSON for a complete FHIR Questionnaire resource.\n<|im_end|>\n<|im_start|>user\nGenerate a complete FHIR Questionnaire for PHQ-9 depression assessment\n<|im_end|>\n<|im_start|>assistant\n" \
	--max-tokens 900 \
	--temp 0.0 \
	--top-p 1.0 \
	--verbose F \
	--extra-eos-token "<|im_end|>"
```

Tip: keep prompts in ChatML format to match training conditions.
For the current checked-in winner, use the 7B 8-bit config for training and evaluation. Keep export disabled on that path unless you intentionally switch to a non-quantized base for GGUF conversion.

### Use the specialized model in LM Studio (GGUF)

For this project (`Qwen2.5` family), the direct MLX flag `--export-gguf` may fail with:

`Model type qwen2 not supported for GGUF conversion.`

Also, fusing on top of certain quantized bases can produce conversion errors in `llama.cpp` (for example unsupported quantization metadata or tensor mapping issues).

Use this validated workflow instead when you need a GGUF deliverable (fuse with a non-quantized base, then convert):

1) Fuse base model + trained LoRA adapter (MLX, non-4bit base):

```bash
poetry run python -m mlx_lm fuse \
	--model mlx-community/Qwen2.5-Coder-7B-Instruct-bf16 \
	--adapter-path lab/artifacts/<MODEL_ALIAS>/checkpoints_stable \
	--save-path lab/artifacts/<MODEL_ALIAS>/fused-noquant
```

2) Convert fused HF model to GGUF with `llama.cpp`:

```bash
poetry run python tools/llama.cpp/convert_hf_to_gguf.py \
	lab/artifacts/<MODEL_ALIAS>/fused-noquant \
	--outfile lab/artifacts/<MODEL_ALIAS>/fused-noquant/model-fused-f16.gguf \
	--outtype f16
```

3) (Optional, recommended) Quantize for lighter inference:

```bash
tools/llama.cpp/build/bin/llama-quantize \
	lab/artifacts/<MODEL_ALIAS>/fused-noquant/model-fused-f16.gguf \
	lab/artifacts/<MODEL_ALIAS>/fused/model-fused-q4_k_m.gguf \
	q4_k_m
```

4) Import the generated `.gguf` file into LM Studio (`My Models` -> `Import`).

Recommended import target for lightweight local inference:

- `lab/artifacts/<MODEL_ALIAS>/fused/model-fused-q4_k_m.gguf`

## Evaluation framing

- Evaluation compares the base model and the adapter on the same deterministic validation sample.
- Sample size is controlled from the notebook configuration cell via `AIPROM_EVAL_SAMPLES`.
- The scoring path is structural and rule-based, focused on parseability and FHIR-oriented output validity.
- Derived artifacts can include `ab_rule_eval.json`, `ab_rule_eval_analysis.json`, and `adapter_go_no_go.json` in local runtime storage.
- The current strongest checked-in result is the 7B 8-bit snapshot with a `GO` decision in the published tune reports.
- Checked-in reports preserve negative outcomes intentionally; `NO_GO` is treated as a valid reproducibility result, not filtered out.

## Key takeaways

- End-to-end reproducible MLX workflow for FHIR Questionnaire generation.
- Clear separation between local runtime artifacts and versioned publication reports.
- Negative-result reporting is preserved instead of hidden.
- Documentation includes legal and licensing constraints for downstream use of questionnaire-derived content.

## Legal & Copyright Notice

### Repository license

Unless stated otherwise in a specific file header, the original content in this repository (code, documentation, and original data/curation work) is licensed under:

- Creative Commons Attribution 4.0 International (CC BY 4.0)
- SPDX: `CC-BY-4.0`

Copyright (c) 2026 Pablo Pimàs.

### Base model copyright and license

This repository currently ships configs for third-party base models in the Qwen2.5-Coder family:

- `mlx-community/Qwen2.5-Coder-7B-Instruct-8bit`
- `mlx-community/Qwen2.5-Coder-7B-Instruct-bf16`
- `mlx-community/Qwen2.5-Coder-3B-Instruct-bf16`

Representative Hugging Face repositories:

- https://huggingface.co/mlx-community/Qwen2.5-Coder-7B-Instruct-8bit
- https://huggingface.co/mlx-community/Qwen2.5-Coder-7B-Instruct-bf16
- https://huggingface.co/mlx-community/Qwen2.5-Coder-3B-Instruct-bf16

- Declared license for these `mlx-community` distributions: **Apache-2.0**.
- The original Qwen model family is provided by its respective authors/rights holders (Qwen/Alibaba Cloud) and is governed by its own model license and terms.
- The `mlx-community` repository is a converted distribution of model weights for MLX usage and remains subject to the upstream model licensing constraints.
- LoRA adapters generated in this repository are derivative artifacts that must be used and distributed in compliance with the base model license.

Before deployment, redistribution, or commercial use, review the model card and license terms for both the upstream Qwen model and the specific `mlx-community` distribution used.

### Third-party questionnaire content

This project does not distribute official or authorized versions of any third-party questionnaires for clinical/operational use. It only contains open, structured definitions and training examples intended exclusively for model training and internal R&D (e.g., research/educational use under fair use principles where applicable).

The training dataset may include (or resemble) item text from standardized instruments. Those instruments are typically owned by third parties and may have separate terms that restrict redistribution and/or commercial use.

If you plan to distribute or commercially use any third-party questionnaire content (or anything derived from it), you must comply with the applicable copyright and licensing policies for each instrument.

This section is provided for engineering guidance and risk awareness only; it is not legal advice.

### Form licensing overview (dataset content)

- PHQ-9 — Status: Free to reproduce/distribute per published terms. Training: Typically OK for research/training. Production/commercial: Typically OK. Source: https://patient.info/doctor/mental-health/phq-9
- GAD-7 — Status: Free to use per published terms. Training: Typically OK for research/training. Production/commercial: Typically OK. Source: https://www.corc.uk.net/outcome-measures-guidance/directory-of-outcome-measures/generalised-anxiety-disorder-assessment-gad-7/
- PROMIS-29 — Status: Free with attribution (NIH/HealthMeasures). Training: Typically OK for research/training. Production/commercial: Attribution required; verify specific instrument terms. Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10674041/
- EORTC QLQ-C30 — Status: Registration / academic agreement required. Training: Typically OK under fair use in private research settings. Production/commercial: Requires agreement; commercial terms differ. Source: https://qol.eortc.org/terms-conditions/academic-user/
- EQ-5D-5L — Status: Registration required; non-commercial free in some cases. Training: Typically OK under fair use in private research settings. Production/commercial: Commercial restrictions apply; registration required. Source: https://euroqol.org/register/obtain-eq-5d/how-to-obtain-eq-5d/
- SF-12 — Status: Commercial license required. Training: Typically OK under fair use in private research settings. Production/commercial: Not OK without a license. Source: https://optumce.com/about/eula/
- FACT-G — Status: Commercial license required. Training: Typically OK under fair use in private research settings. Production/commercial: Not OK without a license. Source: https://www.facit.org/measures/fact-gp
- HADS — Status: Restricted / licensing required for distribution. Training: Typically OK under fair use in private research settings. Production/commercial: Not OK without a license. Source: https://www.gl-assessment.co.uk/products/hospital-anxiety-depression-scale/

### Practical implications

- Training (private R&D): often treated as fair use for research/education, but still depends on jurisdiction, distribution, and the exact content used.
- Production / distribution: you may need registration, agreements, attribution, and/or paid licenses depending on the instrument.

### Recommended filtered subsets

If you want a lower-risk subset for commercial use, consider excluding instruments with registration/licensing restrictions and keeping only clearly permissive instruments plus custom/original fields.

For the current FHIR workflow, use `data/synthetic-aiprom-1500-firh4.jsonl` as the source dataset and create filtered derivatives with your own inclusion/exclusion rules per instrument licensing.

Current full dataset size: **1500 examples** (see [datasets.md](datasets.md)).

Tip: keep filtered outputs as separate files and document the filtering criteria used.
