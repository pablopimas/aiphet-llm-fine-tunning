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

This repository contains Supervised Fine-Tuning (SFT) datasets for training an LLM (e.g., Qwen family) to generate **FHIR R4 Questionnaire JSON** objects (complete questionnaires) and **FHIR R4 QuestionnaireItem JSON** objects (questionnaire items).

Training examples are stored as **JSONL** (one JSON object per line). This repo currently contains two dataset formats:

- **Prompt/Completion JSONL**: a `prompt` string + a `completion` string containing a JSON-encoded FHIR resource.
- **ChatML-in-a-string JSONL**: a single `text` field that contains a ChatML transcript (system/user/assistant), where the assistant emits JSON.

## Files

Datasets live under `data/`:

- `data/synthetic-aiprom-1500-firh4.jsonl`: **Synthetic AIPROM Questionnaires** dataset (current for Questionnaire training). 1500 examples.
- `data/aiprom-items-dataset-fhir-4-150.jsonl`: **AIPROM Items** dataset (QuestionnaireItem-like fragments). 150 examples.

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

## Secondary dataset: AIPROM Items (FHIR R4 QuestionnaireItem-like)

File: `data/aiprom-items-dataset-fhir-4-150.jsonl`


### data/aiprom-items-dataset-fhir-4-150.jsonl coverage

Current snapshot (2026-02-23):

- Total examples: **150**
- Format: JSONL, one object per line, with ChatML-in-a-string under `text`
- Target output: FHIR R4 `Questionnaire.item`-compatible JSON fragments (QuestionnaireItem-like objects)
- Parsing status: 150/150 rows valid JSONL and 150/150 assistant blocks valid JSON

#### FHIR R4 Questionnaire item types (verified)

| Type | Count |
|------|------:|
| group | 5 |
| display | 5 |
| boolean | 8 |
| decimal | 10 |
| integer | 17 |
| date | 8 |
| dateTime | 6 |
| time | 6 |
| string | 8 |
| text | 8 |
| url | 6 |
| choice | 52 |
| open-choice | 11 |

#### Content coverage

- Items derived from (or inspired by) standardized instruments: EORTC QLQ-C30, PHQ-9, GAD-7, EQ-5D-5L, PROMIS, WPAI, HADS
- Clinical scales present in prompts/items: pain, fatigue, anxiety, depression, quality of life

## Record format (JSONL)

Each line is a JSON object with at least:

```json
{
	"text": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n{...json...}<|im_end|>"
}
```

Notes:

- **One example per line**: the `text` value includes escaped newlines (`\n`), so each example remains a single JSON line.
- The **system prompt is repeated** on most/all examples and includes formatting rules and a schema outline.
- The **assistant turn is expected to be a single JSON object** (FHIR QuestionnaireItem-like fragment).

## ChatML structure inside `text`

The `text` field follows ChatML tokens:

- `<|im_start|>system ... <|im_end|>`
- `<|im_start|>user ... <|im_end|>`
- `<|im_start|>assistant ... <|im_end|>`

Training objective: given the system + user prompt, generate the assistant content: a **valid FHIR R4 QuestionnaireItem JSON**.

## Target output: FHIR R4 QuestionnaireItem JSON

The FHIR-oriented system prompt in `data/aiprom-items-dataset-fhir-4-150.jsonl` describes output compatible with FHIR R4 Questionnaire items, commonly including:

- `linkId`: string (stable identifier for item)
- `text`: question text shown to patient/user
- `type`: FHIR item type (e.g. `choice`, `integer`, `boolean`, `string`, `date`, `quantity`)
- `required`: boolean
- `code`: list of codings (`system` + `code`) for tagging/classification
- `answerOption`: for `choice` items, list with options (commonly `valueInteger` + `label` in this dataset)
- `minValue` / `maxValue`: for numeric constraints when applicable

Important: examples are intentionally minimal QuestionnaireItem-like JSON fragments for training. They are not full `Questionnaire` resources.

### FHIR item types observed

From inspection of `data/aiprom-items-dataset-fhir-4-150.jsonl`, all 13 Questionnaire item types are represented:

- `group`, `display`, `boolean`, `decimal`, `integer`, `date`, `dateTime`, `time`, `string`, `text`, `url`, `choice`, `open-choice`

### Validation rules patterns

Common FHIR-oriented patterns in this file:

- `choice` items usually include `answerOption`
- `required` is expected to be boolean
- `code` is expected to be a non-empty array of codings
- Numeric items may include `minValue` and `maxValue`

## Coverage (what kinds of prompts exist)

The FHIR dataset mixes:

- Standardized questionnaire items (e.g., PHQ-9, GAD-7, HADS, EORTC QLQ-C30, FACT-G, PROMIS-29, EQ-5D-5L, SF-12)
- Custom clinical-style fields

The assistant JSON usually encodes instrument + item intent through `linkId`, `text`, `type` and `code` tags.

## Scope

- This project currently handles **FHIR-only** dataset cases.
- For complete questionnaire generation, the source dataset is `data/synthetic-aiprom-1500-firh4.jsonl`.
- For item-only (QuestionnaireItem-like) generation, the source dataset is `data/aiprom-items-dataset-fhir-4-150.jsonl`.

## Extending the dataset

This repo supports extending both dataset formats.

### Prompt/Completion JSONL (Questionnaire generation)

When adding new examples to `data/synthetic-aiprom-1500-firh4.jsonl` (or similar):

1. Keep **one JSON object per line**.
2. Include a non-empty string `prompt`.
3. Include a non-empty string `completion` that parses as JSON.
4. Ensure `completion` JSON is a **FHIR R4 `Questionnaire`** object:
	 - `resourceType == "Questionnaire"`
	 - `item` is a non-empty array
	 - each `item` has `linkId` and `type`
	 - if `type` is `choice/open-choice`, include non-empty `answerOption`

### ChatML-in-a-string JSONL (QuestionnaireItem-like generation)

When adding new examples to `data/aiprom-items-dataset-fhir-4-150.jsonl` (or similar):

1. Keep **one JSON object per line**.
2. Keep `text` in **ChatML** with `system`, `user`, `assistant` turns.
3. Ensure the assistant content is:
	 - A **single JSON object** (no trailing commentary)
	 - Valid JSON
	 - Includes core FHIR item keys (`linkId`, `text`, `type`, `required`, `code`) when applicable
4. If `type` is `choice`, include non-empty `answerOption` with consistent option encoding.

## Quick validation checklist

Recommended checks before training depend on the dataset format.

### Prompt/Completion JSONL (Questionnaire)

- JSONL parseable: every line is valid JSON
- Each row has non-empty `prompt` and `completion`
- `completion` parses as JSON and `resourceType == "Questionnaire"`
- `item` exists and is non-empty

Run:

```bash
python3 scripts/validate_prompt_completion_dataset.py data/synthetic-aiprom-1500-firh4.jsonl
```

### ChatML-in-a-string JSONL (QuestionnaireItem-like)

- JSONL parseable: every line is valid JSON
- `text` contains the three ChatML roles in order: system → user → assistant
- Extracted assistant segment is valid JSON
- Assistant JSON contains at least: `linkId`, `text`, `type`, `required`, `code`
- If `type == choice`: verify `answerOption` exists and is non-empty

Run:

```bash
python3 scripts/normalize_dataset.py data/aiprom-items-dataset-fhir-4-150.jsonl
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

- This document tracks the current FHIR datasets (`data/synthetic-aiprom-1500-firh4.jsonl` and `data/aiprom-items-dataset-fhir-4-150.jsonl`).
- Licensing constraints above still apply regardless of serialization format (FieldTemplate or FHIR).




