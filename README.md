<!--
Author: Pablo Pimàs
Email: pablo@pimas.cat
Date: 2026-02-22
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
SPDX-License-Identifier: CC-BY-4.0
Reference: https://creativecommons.org/licenses/by/4.0/
-->

# aiprom-llm

This repository contains dataset(s) and utilities for supervised fine-tuning (SFT) an LLM (e.g., Qwen family) to generate strict **FHIR R4 QuestionnaireItem JSON** objects.

## Documents

- [Datasets](docs/datasets.md)

## What is in this repo

- SFT datasets under `data/` (JSONL with ChatML stored in a `text` string)
- A deterministic FHIR dataset normalizer/validator: `scripts/normalize_dataset.py`
- Training/config helpers under `configs/` (currently minimal)

## Quick start

Normalize (and validate) the FHIR dataset in-place:

```bash
python3 scripts/normalize_dataset.py data/aiprom-dataset-fhir-4-150.jsonl
```

This will:

- Parse each JSONL ChatML row and validate the assistant payload as QuestionnaireItem-like JSON
- Apply safe FHIR normalization (`type`, `required`, `code`) and reuse canonical `answerOption`/`code` on duplicates by `(linkId, type)`
- Exit non-zero if blocking issues remain (e.g., invalid `type`, missing `linkId`, invalid `code`, missing `answerOption` for `choice/open-choice`)

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

For the current FHIR workflow, use `data/aiprom-dataset-fhir-4-150.jsonl` as the source dataset and create filtered derivatives with your own inclusion/exclusion rules per instrument licensing.

Current full dataset size: **150 examples** (see [docs/datasets.md](docs/datasets.md)).

Tip: keep filtered outputs as separate files (e.g., `data/aiprom-dataset-fhir-4-150-commercial.jsonl`) and document the filtering criteria used.
