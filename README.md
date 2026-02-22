<!--
Author: Pablo Pimàs
Email: pablo@pimas.cat
Date: 2026-02-22
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
SPDX-License-Identifier: CC-BY-4.0
Reference: https://creativecommons.org/licenses/by/4.0/
-->

# aiprom-llm

This repository contains dataset(s) and utilities for supervised fine-tuning (SFT) an LLM (e.g., Qwen family) to generate strict **AIPROM FieldTemplate JSON** objects for:

- `POST /api/forms/templates`

## Documents

- [Datasets](docs/datasets.md)

## What is in this repo

- SFT datasets under `data/` (JSONL with ChatML stored in a `text` string)
- A deterministic dataset normalizer: `scripts/normalize_dataset.py`
- Training/config helpers under `configs/` (currently minimal)

## Quick start

Normalize (and validate) the expanded dataset in-place:

```bash
python3 scripts/normalize_dataset.py data/aiprom-train-dataset-150.jsonl
```

This will:

- Ensure `is_active: true` exists on every assistant FieldTemplate JSON
- Upgrade low-quality duplicates by copying canonical fields already present in the dataset
- Exit non-zero if option-based templates are still missing `options`

## Legal & Copyright Notice

### Repository license

Unless stated otherwise in a specific file header, the original content in this repository (code, documentation, and original data/curation work) is licensed under:

- Creative Commons Attribution 4.0 International (CC BY 4.0)
- SPDX: `CC-BY-4.0`

Copyright (c) 2026 Pablo Pimàs.

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

Example subsets (based on the counts documented in [docs/datasets.md](docs/datasets.md)):

Generate these subsets with:

```bash
python3 scripts/filter_dataset.py --preset commercial-focused \
  data/aiprom-train-dataset-150.jsonl \
  data/aiprom-train-dataset-commercial-focused.jsonl
```

- Commercial-focused (avoid restricted instruments):
	- PHQ-9: 13
	- GAD-7: 10
	- PROMIS: 12
	- Custom clinical + demographics + vitals: 24
	- Total: 59
- Academic-only (still verify terms):
	- PHQ-9: 13
	- GAD-7: 10
	- EORTC: 35
	- EQ-5D: 9
	- PROMIS: 12
	- Custom clinical + demographics + vitals: 24
	- Total: 103

Note: These totals are for filtered subsets, not the full dataset.

The expanded dataset totals 150 examples (see [docs/datasets.md](docs/datasets.md)), composed of:

- EORTC QLQ-C30: 35
- PHQ-9: 13
- HADS: 18
- GAD-7: 10
- FACT-G: 20
- PROMIS-29: 12
- EQ-5D-5L: 9
- SF-12: 9
- Baseline demographics: 10
- Vital signs: 5
- Custom clinical: 9
