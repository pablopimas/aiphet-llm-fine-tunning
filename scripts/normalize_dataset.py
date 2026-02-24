#!/usr/bin/env python3
"""Normalize FHIR R4 QuestionnaireItem SFT JSONL datasets.

Author: Pablo Pimàs
Email: pablo@pimas.cat
Date: 2026-02-22
License: CC BY 4.0
SPDX-License-Identifier: CC-BY-4.0
Reference: https://creativecommons.org/licenses/by/4.0/

This script:
- Parses each JSONL record with a ChatML transcript in the `text` field.
- Extracts assistant JSON (FHIR QuestionnaireItem-like object).
- Applies safe normalization for FHIR fields (`type`, `required`, `code`).
- Reuses canonical `answerOption`/`code` from duplicates by `(linkId, type)`.
- Warns on schema problems and exits non-zero for blocking validation issues.

It rewrites the file in-place and saves a `.bak` backup next to it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast


ASSISTANT_MARKER = "<|im_start|>assistant\n"
END_MARKER = "<|im_end|>"

FHIR_TYPE_VALUES = {
    "group",
    "display",
    "boolean",
    "decimal",
    "integer",
    "date",
    "dateTime",
    "time",
    "string",
    "text",
    "url",
    "choice",
    "open-choice",
}

FHIRType = Literal[
    "group",
    "display",
    "boolean",
    "decimal",
    "integer",
    "date",
    "dateTime",
    "time",
    "string",
    "text",
    "url",
    "choice",
    "open-choice",
]

JsonDict = dict[str, Any]


class DatasetFormatError(RuntimeError):
    pass


@dataclass(frozen=True)
class Canonical:
    obj: JsonDict
    score: int


def _extract_assistant_json(text: str) -> tuple[JsonDict, int, int]:
    start = text.rfind(ASSISTANT_MARKER)
    if start < 0:
        raise DatasetFormatError("Missing assistant marker")

    json_start = start + len(ASSISTANT_MARKER)
    json_end = text.find(END_MARKER, json_start)
    if json_end < 0:
        raise DatasetFormatError("Missing end marker after assistant")

    assistant_raw = text[json_start:json_end].strip()
    try:
        assistant_obj_raw = json.loads(assistant_raw)
    except json.JSONDecodeError as e:
        raise DatasetFormatError(f"Assistant JSON decode error: {e}") from e

    if not isinstance(assistant_obj_raw, dict):
        raise DatasetFormatError("Assistant JSON is not an object")

    return cast(JsonDict, assistant_obj_raw), json_start, json_end


def _dump_assistant_json(obj: JsonDict) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _norm_type(obj: JsonDict) -> tuple[FHIRType | None, bool]:
    raw = obj.get("type")
    if not isinstance(raw, str):
        return None, False

    cleaned = raw.strip()
    mapping = {
        "datetime": "dateTime",
        "date-time": "dateTime",
        "open_choice": "open-choice",
        "openchoice": "open-choice",
    }
    normalized = mapping.get(cleaned.lower(), cleaned)

    changed = False
    if normalized != raw:
        obj["type"] = normalized
        changed = True

    if normalized in FHIR_TYPE_VALUES:
        return cast(FHIRType, normalized), changed
    return None, changed


def _norm_required(obj: JsonDict) -> bool:
    if "required" not in obj:
        return False
    value = obj.get("required")
    if isinstance(value, bool):
        return False

    if isinstance(value, (int, float)) and value in (0, 1):
        obj["required"] = bool(value)
        return True

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            obj["required"] = True
            return True
        if v in {"false", "0", "no", "n"}:
            obj["required"] = False
            return True
    return False


def _norm_code(obj: JsonDict) -> bool:
    code = obj.get("code")
    if code is None:
        return False
    if isinstance(code, dict):
        obj["code"] = [code]
        return True
    return False


def _valid_code(obj: JsonDict) -> bool:
    code = obj.get("code")
    if not isinstance(code, list) or not code:
        return False
    code_list = cast(list[Any], code)
    for item in code_list:
        if not isinstance(item, dict):
            return False
        coding = cast(dict[str, Any], item)
        system = coding.get("system")
        code_value = coding.get("code")
        if not isinstance(system, str) or not system.strip():
            return False
        if not isinstance(code_value, str) or not code_value.strip():
            return False
    return True


def _answer_option_list(obj: JsonDict) -> list[JsonDict] | None:
    answer_option = obj.get("answerOption")
    if not isinstance(answer_option, list) or not answer_option:
        return None
    answer_option_list = cast(list[Any], answer_option)
    if not all(isinstance(x, dict) for x in answer_option_list):
        return None
    return cast(list[JsonDict], answer_option_list)


def _has_choice_options(obj: JsonDict) -> bool:
    options = _answer_option_list(obj)
    if not options:
        return False
    for item in options:
        if not any(key.startswith("value") for key in item.keys()):
            return False
    return True


def _canonical_score(obj: JsonDict) -> int:
    score = 0
    if _valid_code(obj):
        score += 20
    options = _answer_option_list(obj)
    if options:
        score += min(len(options), 10)
    text = obj.get("text")
    if isinstance(text, str) and text.strip():
        score += 3
    required = obj.get("required")
    if isinstance(required, bool):
        score += 2
    return score


def _canonical_key(obj: JsonDict) -> tuple[str, str] | None:
    link_id = obj.get("linkId")
    type_value = obj.get("type")
    if not isinstance(link_id, str) or not link_id.strip():
        return None
    if not isinstance(type_value, str) or not type_value.strip():
        return None
    return (link_id.strip(), type_value.strip())


def _choose_canonical(existing: Canonical | None, candidate_obj: JsonDict) -> Canonical:
    candidate = Canonical(candidate_obj, _canonical_score(candidate_obj))
    if existing is None or candidate.score > existing.score:
        return candidate
    return existing


def _upgrade_from_canonical(obj: JsonDict, canonical: JsonDict) -> bool:
    changed = False
    type_value = obj.get("type")
    if type_value in {"choice", "open-choice"} and not _has_choice_options(obj):
        canonical_answer_option = _answer_option_list(canonical)
        if canonical_answer_option:
            obj["answerOption"] = canonical_answer_option
            changed = True

    if not _valid_code(obj) and _valid_code(canonical):
        obj["code"] = canonical["code"]
        changed = True

    return changed


def normalize_jsonl(path: Path, *, in_place: bool = True) -> tuple[int, int, list[str]]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    records: list[JsonDict] = []
    assistant_objs: list[JsonDict] = []
    spans: list[tuple[int, int]] = []
    warnings: list[str] = []

    canon_by_key: dict[tuple[str, str], Canonical] = {}

    for idx, line in enumerate(raw_lines, start=1):
        if not line.strip():
            warnings.append(f"Line {idx}: empty line")
            continue

        try:
            record_raw = json.loads(line)
        except json.JSONDecodeError as e:
            raise DatasetFormatError(f"Line {idx}: JSON decode error: {e}") from e

        if not isinstance(record_raw, dict) or "text" not in record_raw:
            raise DatasetFormatError(f"Line {idx}: record must be an object with 'text'")

        record = cast(JsonDict, record_raw)
        text = record.get("text")
        if not isinstance(text, str):
            raise DatasetFormatError(f"Line {idx}: 'text' must be a string")

        assistant_obj, json_start, json_end = _extract_assistant_json(text)
        _norm_type(assistant_obj)
        key = _canonical_key(assistant_obj)
        if key:
            canon_by_key[key] = _choose_canonical(canon_by_key.get(key), assistant_obj)

        records.append(record)
        assistant_objs.append(assistant_obj)
        spans.append((json_start, json_end))

    changed_count = 0
    out_lines: list[str] = []

    for idx, (record, assistant_obj, (json_start, json_end)) in enumerate(
        zip(records, assistant_objs, spans),
        start=1,
    ):
        changed = False
        normalized_type, changed_type = _norm_type(assistant_obj)
        changed |= changed_type
        changed |= _norm_required(assistant_obj)
        changed |= _norm_code(assistant_obj)

        key = _canonical_key(assistant_obj)
        if key and key in canon_by_key:
            canonical_obj = canon_by_key[key].obj
            if canonical_obj is not assistant_obj:
                changed |= _upgrade_from_canonical(assistant_obj, canonical_obj)

        link_id = assistant_obj.get("linkId")
        if not isinstance(link_id, str) or not link_id.strip():
            warnings.append(f"Line {idx}: missing/invalid linkId")
        if normalized_type is None:
            warnings.append(f"Line {idx}: missing/invalid type")
        if normalized_type not in {"group", "display"}:
            if not isinstance(assistant_obj.get("text"), str) or not cast(str, assistant_obj.get("text", "")).strip():
                warnings.append(f"Line {idx}: missing/invalid text")
        if "required" in assistant_obj and not isinstance(assistant_obj.get("required"), bool):
            warnings.append(f"Line {idx}: required must be boolean")
        if normalized_type not in {"group", "display"} and not _valid_code(assistant_obj):
            warnings.append(f"Line {idx}: missing/invalid code array")
        if normalized_type in {"choice", "open-choice"} and not _has_choice_options(assistant_obj):
            warnings.append(f"Line {idx}: missing/invalid answerOption for type={normalized_type}")

        if changed:
            changed_count += 1
            text = cast(str, record["text"])
            new_assistant = _dump_assistant_json(assistant_obj)
            record["text"] = text[:json_start] + new_assistant + text[json_end:]

        out_lines.append(json.dumps(record, ensure_ascii=False))

    if in_place:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    return len(records), changed_count, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize FHIR QuestionnaireItem SFT JSONL")
    parser.add_argument(
        "path",
        nargs="?",
        default="data/aiprom-items-dataset-fhir-4-150.jsonl",
        help="Path to JSONL dataset (default: data/aiprom-items-dataset-fhir-4-150.jsonl)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    total, changed, warnings = normalize_jsonl(path, in_place=True)
    print(f"Normalized {path}: changed {changed}/{total} records")
    if warnings:
        print("Warnings:")
        for warning in warnings[:80]:
            print(f"- {warning}")
        if len(warnings) > 80:
            print(f"- ... and {len(warnings) - 80} more")

    has_blocking_issues = any(
        "missing/invalid type" in w
        or "missing/invalid linkId" in w
        or "missing/invalid code array" in w
        or "missing/invalid answerOption" in w
        or "required must be boolean" in w
        for w in warnings
    )
    return 2 if has_blocking_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
