<!--
Author: Pablo Pimàs
Email: pablo@pimas.cat
Date: 2026-02-22
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
SPDX-License-Identifier: CC-BY-4.0
Reference: https://creativecommons.org/licenses/by/4.0/
-->

# Datasets

- **Dataset Purpose**: Fair Use for research/educational purposes
- **Questionnaire Content**: See Legal & Copyright Notice section

This repository contains Supervised Fine-Tuning (SFT) datasets for training an LLM (e.g., Qwen family) to generate **FHIR R4 QuestionnaireItem JSON** objects.

The training examples are stored as **JSONL** (one JSON object per line). Each example uses **ChatML-in-a-string** (a single `text` field containing system/user/assistant turns).

## Files

Datasets live under `data/`:

- `data/aiprom-dataset-fhir-4-150.jsonl`: FHIR-focused dataset (current). 150 examples (1 JSON object per line).


### data/aiprom-dataset-fhir-4-150.jsonl coverage

Current snapshot (2026-02-22):

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

- Validated PROMs questionnaires: EORTC QLQ-C30, PHQ-9, GAD-7, EQ-5D-5L, PROMIS, WPAI, HADS
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

The FHIR-oriented system prompt in `data/aiprom-dataset-fhir-4-150.jsonl` describes output compatible with FHIR R4 Questionnaire items, commonly including:

- `linkId`: string (stable identifier for item)
- `text`: question text shown to patient/user
- `type`: FHIR item type (e.g. `choice`, `integer`, `boolean`, `string`, `date`, `quantity`)
- `required`: boolean
- `code`: list of codings (`system` + `code`) for tagging/classification
- `answerOption`: for `choice` items, list with options (commonly `valueInteger` + `label` in this dataset)
- `minValue` / `maxValue`: for numeric constraints when applicable

Important: examples are intentionally minimal QuestionnaireItem-like JSON fragments for training. They are not full `Questionnaire` resources.

### FHIR item types observed

From inspection of `data/aiprom-dataset-fhir-4-150.jsonl`, all 13 Questionnaire item types are represented:

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
- The source of truth for training is `data/aiprom-dataset-fhir-4-150.jsonl`.

## Extending the dataset

When adding new examples:

1. Keep **one JSON object per line**.
2. Keep `text` in **ChatML** with `system`, `user`, `assistant` turns.
3. Ensure the assistant content is:
	 - A **single JSON object** (no trailing commentary)
	 - Valid JSON
	 - Includes core FHIR item keys (`linkId`, `text`, `type`, `required`, `code`) when applicable
4. If `type` is `choice`, include non-empty `answerOption` with consistent option encoding.

## Quick validation checklist

Recommended checks before training:

- JSONL parseable: every line is valid JSON
- `text` contains the three ChatML roles in order: system → user → assistant
- Extracted assistant segment is valid JSON
- Assistant JSON contains at least: `linkId`, `text`, `type`, `required`, `code`
- If `type == choice`: verify `answerOption` exists and is non-empty
- If `type == integer`: verify numeric constraints (`minValue`/`maxValue`) when required by your prompt contract

If you want a lightweight validator, you can implement a script that:

- Reads `data/*.jsonl`
- Parses each line JSON
- Splits `text` on `<|im_start|>assistant` and `<|im_end|>`
- Attempts `json.loads()` on the assistant content

Keep the validator strict (fail fast), because small format errors can silently degrade training quality.

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

- This document tracks the current FHIR dataset (`data/aiprom-dataset-fhir-4-150.jsonl`).
- Licensing constraints above still apply regardless of serialization format (FieldTemplate or FHIR).




