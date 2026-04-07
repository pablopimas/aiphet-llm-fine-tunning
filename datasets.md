<!--
Author: Pablo Pimàs
Email: pablo@pimas.cat
Date: 2026-02-23
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
SPDX-License-Identifier: CC-BY-4.0
Reference: https://creativecommons.org/licenses/by/4.0/
-->

# Datasets

- **Dataset Purpose**: Fair Use for research/educational purposes
- **Questionnaire / Item Content**: See Legal & Copyright Notice section

This repository currently ships the dataset used to train an LLM (for example, a Qwen-family model under MLX-LM) to generate **FHIR R4 Questionnaire JSON** objects (complete questionnaires).

Training examples are stored as **JSONL** (one JSON object per line).

## Current committed dataset

Datasets live under `data/`:

- `data/synthetic-aiprom-1500-firh4.jsonl`: **Synthetic AIPROM Questionnaires** dataset. 1500 examples.

The older QuestionnaireItem-oriented dataset sometimes referenced in historical notes is **not included in the current repository snapshot**. The checked-in notebook, configs, and reports all target the 1500-example Questionnaire dataset above.

## Primary dataset: Synthetic AIPROM Questionnaires (FHIR R4)

File: `data/synthetic-aiprom-1500-firh4.jsonl`

Current snapshot (2026-02-23):

- Total examples: **1500**
- Format: JSONL, one object per line, with fields:
	- `prompt`: string
	- `completion`: string (a JSON-encoded FHIR resource)
- Target output: **FHIR R4 `Questionnaire`** resources (complete questionnaires)
- Parsing status: 1500/1500 rows valid JSONL and 1500/1500 completion blocks valid JSON

### Record format (JSONL)

Each line is a JSON object with at least:

```json
{
	"prompt": "Generate complete FHIR R4 Questionnaire for PHQ-9 Depression Assessment",
	"completion": "{\"resourceType\":\"Questionnaire\",...}"
}
```

Notes:

- `completion` is a **string** that must be parsed as JSON to obtain the FHIR object.
- In this dataset, the completion resource type is always `Questionnaire`.

### Target output: FHIR R4 Questionnaire JSON

The completion JSON is a FHIR `Questionnaire` object. Common fields observed (present on 1500/1500 rows in the current snapshot):

- `resourceType`: "Questionnaire"
- `id`: string
- `title`: string
- `status`: string (e.g., "active")
- `subjectType`: array (e.g., ["Patient"])
- `date`: string (date)
- `code`: array of codings
- `item`: array of `Questionnaire.item`

### Item-level characteristics (current snapshot)

- Items per questionnaire:
	- min: **5**
	- max: **9**
	- average: **5.8**
	- distribution: 5 items (900), 6 items (300), 7 items (150), 9 items (150)
- Item `type` distribution across all `Questionnaire.item` entries:
	- `integer`: **3450**
	- `choice`: **3300**
	- `boolean`: **1350**
	- `decimal`: **450**
	- `string`: **150**

The checked-in `lab/artifacts/split_manifest.json` reports only the **dominant item type per questionnaire** used for split stratification, not the full nested item-type inventory. That is why its current train/validation/test summaries only show `boolean`, `choice`, and `integer`, while the dataset-wide item census above also includes `decimal` and `string`.

### Validation rules (practical contract)

For this dataset, a row is considered valid when:

- Line is a JSON object with non-empty string `prompt`.
- `completion` is a non-empty string and parses to a JSON object.
- `completion.resourceType == "Questionnaire"`.
- `completion.item` is a non-empty array.
- For each element in `completion.item`:
	- `linkId`: non-empty string
	- `type`: valid FHIR Questionnaire item type
	- `text`: non-empty string when `type` is not `group`/`display`
	- `answerOption`: non-empty array when `type` is `choice`/`open-choice`

### Quick validation

Run the validator script included in this repo:

```bash
python3 scripts/validate_prompt_completion_dataset.py data/synthetic-aiprom-1500-firh4.jsonl
```

## Coverage (what kinds of prompts exist)

The current dataset mixes prompts for complete clinical questionnaires covering PROM/PREM-style instruments and synthetic custom forms.

- Standardized instrument-inspired forms (e.g. PHQ-9, GAD-7, EQ-5D-5L, PROMIS-style variants)
- Custom clinical-style forms and mixed item trees

## Scope

- This project currently handles **FHIR-only** dataset cases.
- The source dataset for the checked-in workflow is `data/synthetic-aiprom-1500-firh4.jsonl`.
- Item-only workflows are not part of the current committed artifact set.

## Extending the dataset

When adding new examples to `data/synthetic-aiprom-1500-firh4.jsonl` (or similar):

1. Keep **one JSON object per line**.
2. Include a non-empty string `prompt`.
3. Include a non-empty string `completion` that parses as JSON.
4. Ensure `completion` JSON is a **FHIR R4 `Questionnaire`** object:
	 - `resourceType == "Questionnaire"`
	 - `item` is a non-empty array
	 - each `item` has `linkId` and `type`
	 - if `type` is `choice/open-choice`, include non-empty `answerOption`

## Quick validation checklist

Recommended checks before training:

- JSONL parseable: every line is valid JSON
- Each row has non-empty `prompt` and `completion`
- `completion` parses as JSON and `resourceType == "Questionnaire"`
- `item` exists and is non-empty

Run:

```bash
python3 scripts/validate_prompt_completion_dataset.py data/synthetic-aiprom-1500-firh4.jsonl
```

Keep validators strict (fail fast), because small format errors can silently degrade training quality.

## Legal & Copyright Notice

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

### Scope note for this document

- This document tracks the current committed FHIR dataset: `data/synthetic-aiprom-1500-firh4.jsonl`.
- Licensing constraints above still apply regardless of downstream serialization or packaging format.




