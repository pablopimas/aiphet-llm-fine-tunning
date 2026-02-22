#!/usr/bin/env python3
"""Normalize aiprom SFT JSONL datasets.

Author: Pablo Pimàs
Email: pablo@pimas.cat
Date: 2026-02-22
License: CC BY 4.0
SPDX-License-Identifier: CC-BY-4.0
Reference: https://creativecommons.org/licenses/by/4.0/

This script:
- Parses each JSONL record with a ChatML transcript in the `text` field.
- Extracts the assistant JSON (a FieldTemplate).
- Ensures `is_active: true` is present.
- Builds a per-`name` canonical template (prefers rich tags + non-generic options).
- Fixes low-quality duplicates by copying canonical `options`, `field_type`,
  `validation_rules`, `tags`, and `description` when available.

It rewrites the file in-place and saves a `.bak` backup next to it.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NotRequired, Required, TypedDict, cast


ASSISTANT_MARKER = "<|im_start|>assistant\n"
END_MARKER = "<|im_end|>"


FieldType = Literal["likert", "text", "number", "select", "radio", "checkbox"]


class I18nText(TypedDict):
    """Fixed i18n structure used across the dataset."""

    en: str
    es: str
    ca: str


class FieldOption(TypedDict, total=False):
    """Option entry for likert/select/radio templates."""

    value: int | float | str
    label: I18nText


class FieldTemplate(TypedDict, total=False):
    """AIPROM FieldTemplate JSON schema as used by the SFT dataset.

    This is intentionally permissive (many keys are optional in the dataset).
    """

    name: Required[str]
    field_type: Required[FieldType]
    version: NotRequired[str]
    is_active: NotRequired[bool]

    question: NotRequired[I18nText]
    label: NotRequired[I18nText]
    placeholder: NotRequired[I18nText]
    hint: NotRequired[I18nText]

    options: NotRequired[list[FieldOption]]
    tags: NotRequired[list[str]]
    errors: NotRequired[dict[str, I18nText]]
    validation_rules: NotRequired[dict[str, Any]]
    description: NotRequired[str]


class DatasetRecord(TypedDict, total=False):
    """Single JSONL record wrapper around a ChatML transcript."""

    text: Required[str]


# NOTE: The dataset is user/model-generated. When validating/parsing JSON coming
# from the dataset, we intentionally operate on plain dicts (untrusted input)
# rather than TypedDicts (trusted shapes). TypedDicts above document the target
# schema and are useful for readers, but strict type checkers will otherwise
# (incorrectly) assume keys are always present and correctly typed.
JsonDict = dict[str, Any]


class DatasetFormatError(RuntimeError):
    pass


def _extract_assistant_json(text: str) -> tuple[JsonDict, int, int]:
    """Return (assistant_obj, json_start_index, json_end_index) in `text`.

    json_start_index points at the first character of the assistant JSON.
    json_end_index points at the start of the END_MARKER.
    """

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
    """Serialize assistant FieldTemplate JSON in a stable, readable format."""
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _get_name(obj: JsonDict) -> str | None:
    """Return the template `name` if present and non-empty, else None."""
    name = obj.get("name")
    return name if isinstance(name, str) and name.strip() else None


def _field_type(obj: JsonDict) -> FieldType | None:
    """Return `field_type` if present and one of the allowed values, else None."""
    ft = obj.get("field_type")
    if isinstance(ft, str) and ft in {"likert", "text", "number", "select", "radio", "checkbox"}:
        return cast(FieldType, ft)
    return None


def _options(obj: JsonDict) -> list[FieldOption] | None:
    """Return `options` as a list of dicts when well-formed, else None."""
    opts = obj.get("options")
    if opts is None:
        return None
    if isinstance(opts, list):
        opts_list = cast(list[Any], opts)
        if all(isinstance(x, dict) for x in opts_list):
            return cast(list[FieldOption], opts_list)
    return None


def _tags(obj: JsonDict) -> list[str] | None:
    """Return `tags` as a list of strings when well-formed, else None."""
    tags = obj.get("tags")
    if isinstance(tags, list):
        tags_list = cast(list[Any], tags)
        if all(isinstance(x, str) for x in tags_list):
            return cast(list[str], tags_list)
    return None


_GENERIC_LEVEL_RE = re.compile(r"^Level\s+\d+$", re.IGNORECASE)
_GENERIC_OPTION_RE = re.compile(r"^Option\s+\d+$", re.IGNORECASE)


def _is_generic_option_label(label: Any) -> bool:
    """Heuristic: True if label.en looks like a placeholder (e.g. 'Level 1')."""
    if not isinstance(label, dict):
        return False
    label_dict = cast(dict[str, Any], label)
    en = label_dict.get("en")
    if not isinstance(en, str):
        return False
    en = en.strip()
    return bool(_GENERIC_LEVEL_RE.match(en) or _GENERIC_OPTION_RE.match(en))


def _options_look_generic(obj: JsonDict) -> bool:
    """Return True if all option labels look generic/placeholder-ish."""
    opts = _options(obj)
    if not opts:
        return False
    labels = [o.get("label") for o in opts]
    if not labels:
        return False
    return all(_is_generic_option_label(l) for l in labels)


def _needs_options(obj: JsonDict) -> bool:
    """Return True if this field_type should have `options` in the dataset."""
    return _field_type(obj) in {"likert", "radio", "select"}


def _has_valid_options(obj: JsonDict) -> bool:
    """Return True if options are present when required (else False)."""
    if not _needs_options(obj):
        return True
    opts = _options(obj)
    return bool(opts)


def _ensure_is_active(obj: JsonDict) -> bool:
    """Ensure `is_active` exists; returns True if the object was modified."""
    if "is_active" in obj:
        return False
    obj["is_active"] = True
    return True


@dataclass(frozen=True)
class Canonical:
    obj: JsonDict
    score: int


def _canonical_score(obj: JsonDict) -> int:
    """Score a template for canonical selection among duplicates with same name."""
    score = 0
    if _has_valid_options(obj):
        score += 50
    if _options(obj):
        score += 10
    if _options_look_generic(obj):
        score -= 10
    tags = _tags(obj) or []
    score += min(len(tags), 10)
    # Prefer newer versions when present (lexicographically works for these datasets)
    version = obj.get("version")
    if isinstance(version, str):
        if version.strip() == "2.0.0":
            score += 2
        elif version.strip() == "1.0.0":
            score += 1
    return score


def _choose_canonical(existing: Canonical | None, candidate_obj: JsonDict) -> Canonical:
    """Pick the higher-scoring canonical template between existing and candidate."""
    cand = Canonical(candidate_obj, _canonical_score(candidate_obj))
    if existing is None or cand.score > existing.score:
        return cand
    return existing


def _should_upgrade_to_canonical(obj: JsonDict, canonical: JsonDict) -> bool:
    """Decide whether to upgrade `obj` by copying fields from `canonical`."""
    # Upgrade if options are missing for option-requiring field types.
    if _needs_options(obj) and not _options(obj) and _options(canonical):
        return True

    # Upgrade if options look generic but canonical has non-generic options.
    if _options_look_generic(obj) and _options(canonical) and not _options_look_generic(canonical):
        return True

    # Upgrade if tags are too minimal and canonical has richer tags.
    tags = _tags(obj) or []
    canonical_tags = _tags(canonical) or []
    if len(tags) <= 2 and len(canonical_tags) >= 4:
        return True

    return False


def _upgrade_from_canonical(obj: JsonDict, canonical: JsonDict) -> bool:
    """Copy selected fields from canonical into obj; returns True if modified."""
    changed = False
    for key in ("field_type", "options", "validation_rules", "tags", "description"):
        if key in canonical and obj.get(key) != canonical.get(key):
            obj[key] = canonical[key]
            changed = True
    return changed


def normalize_jsonl(path: Path, *, in_place: bool = True) -> tuple[int, int, list[str]]:
    """Normalize dataset and return (records_total, records_changed, warnings)."""

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    records: list[JsonDict] = []
    assistant_objs: list[JsonDict] = []
    spans: list[tuple[int, int]] = []
    warnings: list[str] = []

    # Pass 1: parse and build canonical map.
    canon_by_name: dict[str, Canonical] = {}

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
        name = _get_name(assistant_obj)
        if name:
            canon_by_name[name] = _choose_canonical(canon_by_name.get(name), assistant_obj)
        else:
            warnings.append(f"Line {idx}: assistant JSON missing/invalid name")

        records.append(record)
        assistant_objs.append(assistant_obj)
        spans.append((json_start, json_end))

    # Pass 2: normalize.
    changed_count = 0
    out_lines: list[str] = []

    for idx, (record, assistant_obj, (json_start, json_end)) in enumerate(
        zip(records, assistant_objs, spans),
        start=1,
    ):
        changed = False
        changed |= _ensure_is_active(assistant_obj)

        name = _get_name(assistant_obj)
        if name and name in canon_by_name:
            canonical_obj = canon_by_name[name].obj
            if canonical_obj is not assistant_obj and _should_upgrade_to_canonical(assistant_obj, canonical_obj):
                changed |= _upgrade_from_canonical(assistant_obj, canonical_obj)

        # Final safety: if options required but missing, warn (we don't invent scales).
        if _needs_options(assistant_obj) and not _options(assistant_obj):
            warnings.append(
                f"Line {idx}: '{name or 'UNKNOWN'}' still missing options for field_type={_field_type(assistant_obj)!r}"
            )

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
    """CLI entrypoint for normalizing a JSONL dataset file."""
    parser = argparse.ArgumentParser(description="Normalize aiprom SFT dataset JSONL")
    parser.add_argument(
        "path",
        nargs="?",
        default="data/aiprom-train-dataset-150.jsonl",
        help="Path to JSONL dataset (default: data/aiprom-train-dataset-150.jsonl)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    total, changed, warnings = normalize_jsonl(path, in_place=True)
    print(f"Normalized {path}: changed {changed}/{total} records")
    if warnings:
        print("Warnings:")
        for w in warnings[:50]:
            print(f"- {w}")
        if len(warnings) > 50:
            print(f"- ... and {len(warnings) - 50} more")

    # Treat remaining missing-options as an error exit code.
    has_missing_options = any("still missing options" in w for w in warnings)
    return 2 if has_missing_options else 0


if __name__ == "__main__":
    raise SystemExit(main())
