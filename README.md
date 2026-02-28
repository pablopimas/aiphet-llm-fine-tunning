<!--
Author: Pablo Pimàs
Email: pablo@pimas.cat
Date: 2026-02-22
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
SPDX-License-Identifier: CC-BY-4.0
Reference: https://creativecommons.org/licenses/by/4.0/
-->

# aiprom-llm

This repository contains dataset(s) and utilities for supervised fine-tuning (SFT) an LLM (currently **mlx-community/Qwen2.5-7B-Instruct-4bit**) to generate strict **FHIR R4 QuestionnaireItem JSON** objects.

## Documents

- [Datasets](docs/datasets.md)

## What is in this repo

- SFT datasets under `data/` (JSONL with ChatML stored in a `text` string)
- A deterministic FHIR dataset normalizer/validator: `scripts/normalize_dataset.py`
- Training/config helpers under `configs/` (currently minimal)

## Quick start

Normalize (and validate) the FHIR dataset in-place:

```bash
python3 scripts/normalize_dataset.py data/aiprom-items-dataset-fhir-4-150.jsonl
```

This will:

- Parse each JSONL ChatML row and validate the assistant payload as QuestionnaireItem-like JSON
- Apply safe FHIR normalization (`type`, `required`, `code`) and reuse canonical `answerOption`/`code` on duplicates by `(linkId, type)`
- Exit non-zero if blocking issues remain (e.g., invalid `type`, missing `linkId`, invalid `code`, missing `answerOption` for `choice/open-choice`)

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
- `AIPROM_MODEL_NAME` (example: `mlx-community/Qwen2.5-7B-Instruct-4bit`)
- `AIPROM_MODEL_ALIAS` (optional override for artifact folder naming)

If `AIPROM_MODEL_ALIAS` is not provided, it is derived from `AIPROM_MODEL_NAME`.

### What artifacts does it generate?

The notebook writes reproducibility and training artifacts under:

- `lab/artifacts/<MODEL_ALIAS>/`

Typical files include:

- `lab/artifacts/<MODEL_ALIAS>/split_manifest.json`
- `lab/artifacts/<MODEL_ALIAS>/train.jsonl`
- `lab/artifacts/<MODEL_ALIAS>/val.jsonl`
- `lab/artifacts/<MODEL_ALIAS>/valid.jsonl`
- `lab/artifacts/<MODEL_ALIAS>/dataset_manifest.json`
- `lab/artifacts/<MODEL_ALIAS>/training_config_stable.json`
- `lab/artifacts/<MODEL_ALIAS>/training_run_log.json`
- checkpoints/adapters in `lab/artifacts/<MODEL_ALIAS>/checkpoints_stable`

### Does it produce a specialized model?

Yes. Running the training cell produces a **specialized LoRA adapter** over the base Qwen model for FHIR QuestionnaireItem-style generation. It does not replace the base model weights; it creates adapter weights that are loaded on top of the base model.

### Quick inference with the trained adapter

After training, you can run an inference test with MLX-LM using the adapter directory:

```bash
python -m mlx_lm.generate \
	--model mlx-community/Qwen2.5-7B-Instruct-4bit \
	--adapter-path lab/artifacts/<MODEL_ALIAS>/checkpoints_stable \
	--prompt "<|im_start|>system\nYou are a FHIR R4 expert.\n<|im_end|><|im_start|>user\nGenerate a QuestionnaireItem for PHQ-9 depressed mood\n<|im_end|><|im_start|>assistant\n" \
	--max-tokens 400
```

Tip: keep prompts in ChatML format to match training conditions.

### Use the specialized model in LM Studio (GGUF)

For this project (`Qwen2.5` family), the direct MLX flag `--export-gguf` may fail with:

`Model type qwen2 not supported for GGUF conversion.`

Also, fusing on top of a `4bit` base can produce conversion errors in `llama.cpp` (for example unsupported quantization metadata or tensor mapping issues).

Use this validated workflow instead (fuse with the non-quantized base, then convert):

1) Fuse base model + trained LoRA adapter (MLX, non-4bit base):

```bash
python -m mlx_lm fuse \
	--model Qwen/Qwen2.5-7B-Instruct \
	--adapter-path lab/artifacts/<MODEL_ALIAS>/checkpoints_stable \
	--save-path lab/artifacts/<MODEL_ALIAS>-fused-fp
```

2) Convert fused HF model to GGUF with `llama.cpp`:

```bash
python lab/artifacts/llama.cpp/convert_hf_to_gguf.py \
	lab/artifacts/<MODEL_ALIAS>-fused-fp \
	--outfile lab/artifacts/<MODEL_ALIAS>-fused-fp/model-fused-f16.gguf \
	--outtype f16
```

3) (Optional, recommended) Quantize for lighter inference:

```bash
lab/artifacts/llama.cpp/build/bin/llama-quantize \
	lab/artifacts/<MODEL_ALIAS>-fused-fp/model-fused-f16.gguf \
	lab/artifacts/<MODEL_ALIAS>-fused-fp/model-fused-q4_k_m.gguf \
	q4_k_m
```

4) Import the generated `.gguf` file into LM Studio (`My Models` -> `Import`).

Recommended import target for lightweight local inference:

- `lab/artifacts/<MODEL_ALIAS>-fused-fp/model-fused-q4_k_m.gguf`

## Legal & Copyright Notice

### Repository license

Unless stated otherwise in a specific file header, the original content in this repository (code, documentation, and original data/curation work) is licensed under:

- Creative Commons Attribution 4.0 International (CC BY 4.0)
- SPDX: `CC-BY-4.0`

Copyright (c) 2026 Pablo Pimàs.

### Base model copyright and license

This project fine-tunes the third-party base model `mlx-community/Qwen2.5-7B-Instruct-4bit`.

Hugging Face repository (base model used in this project):

- https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit

- Declared license for `mlx-community/Qwen2.5-7B-Instruct-4bit`: **Apache-2.0**.
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

For the current FHIR workflow, use `data/aiprom-items-dataset-fhir-4-150.jsonl` as the source dataset and create filtered derivatives with your own inclusion/exclusion rules per instrument licensing.

Current full dataset size: **150 examples** (see [docs/datasets.md](docs/datasets.md)).

Tip: keep filtered outputs as separate files (e.g., `data/aiprom-items-dataset-fhir-4-150-commercial.jsonl`) and document the filtering criteria used.
