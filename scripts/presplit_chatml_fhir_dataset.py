#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any, cast

SYSTEM_MARKER = "<|im_start|>system\n"
USER_MARKER = "<|im_start|>user\n"
ASSISTANT_MARKER = "<|im_start|>assistant\n"
END_MARKER = "<|im_end|>"

JsonDict = dict[str, Any]


class DatasetFormatError(RuntimeError):
    pass


def _find_span(text: str, marker: str, start: int = 0) -> tuple[int, int]:
    marker_start = text.find(marker, start)
    if marker_start < 0:
        raise DatasetFormatError(f"Missing marker: {marker!r}")
    content_start = marker_start + len(marker)
    content_end = text.find(END_MARKER, content_start)
    if content_end < 0:
        raise DatasetFormatError(f"Missing end marker for {marker!r}")
    return content_start, content_end


def _extract_assistant_obj(text: str) -> tuple[JsonDict, int, int]:
    start = text.rfind(ASSISTANT_MARKER)
    if start < 0:
        raise DatasetFormatError("Missing assistant marker")

    json_start = start + len(ASSISTANT_MARKER)
    json_end = text.find(END_MARKER, json_start)
    if json_end < 0:
        raise DatasetFormatError("Missing end marker after assistant")

    payload = text[json_start:json_end].strip()
    obj_any = json.loads(payload)
    if not isinstance(obj_any, dict):
        raise DatasetFormatError("Assistant payload must be a JSON object")

    return cast(JsonDict, obj_any), json_start, json_end


def _replace_span(text: str, start: int, end: int, replacement: str) -> str:
    return text[:start] + replacement + text[end:]


def _chunk_items(items: list[Any], chunk_size: int) -> list[list[Any]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def _split_record(text: str, max_items_per_sample: int) -> list[str]:
    assistant_obj, json_start, json_end = _extract_assistant_obj(text)
    items_any = assistant_obj.get("item")

    if not isinstance(items_any, list) or len(items_any) <= max_items_per_sample:
        return [text]

    item_chunks = _chunk_items(items_any, max_items_per_sample)
    total_parts = len(item_chunks)

    user_start, user_end = _find_span(text, USER_MARKER)
    original_user = text[user_start:user_end]

    base_id = assistant_obj.get("id") if isinstance(assistant_obj.get("id"), str) else None

    out: list[str] = []
    for idx, chunk in enumerate(item_chunks, start=1):
        obj_part = copy.deepcopy(assistant_obj)
        obj_part["item"] = chunk

        if base_id:
            obj_part["id"] = f"{base_id}-p{idx:02d}"

        json_payload = json.dumps(obj_part, ensure_ascii=False, separators=(",", ":"))

        user_suffix = f" (part {idx}/{total_parts})"
        user_text = original_user
        if user_suffix not in original_user:
            user_text = f"{original_user}{user_suffix}"

        record_text = _replace_span(text, json_start, json_end, json_payload)
        user_start_new, user_end_new = _find_span(record_text, USER_MARKER)
        record_text = _replace_span(record_text, user_start_new, user_end_new, user_text)
        out.append(record_text)

    return out


def presplit_jsonl(path: Path, *, max_items_per_sample: int, in_place: bool) -> tuple[int, int]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()

    expanded_lines: list[str] = []
    input_rows = 0
    output_rows = 0

    for line_no, line in enumerate(raw_lines, start=1):
        if not line.strip():
            continue
        input_rows += 1

        try:
            row_any = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetFormatError(f"Line {line_no}: invalid JSON ({exc})") from exc

        if not isinstance(row_any, dict):
            raise DatasetFormatError(f"Line {line_no}: row must be JSON object")

        row = cast(JsonDict, row_any)
        text = row.get("text")
        if not isinstance(text, str):
            raise DatasetFormatError(f"Line {line_no}: 'text' must be string")

        split_texts = _split_record(text, max_items_per_sample=max_items_per_sample)

        for chunk_text in split_texts:
            row_out = dict(row)
            row_out["text"] = chunk_text
            expanded_lines.append(json.dumps(row_out, ensure_ascii=False))
            output_rows += 1

    if in_place:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)

    path.write_text("\n".join(expanded_lines) + "\n", encoding="utf-8")
    return input_rows, output_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-split ChatML FHIR Questionnaire rows by Questionnaire.item chunks "
            "to reduce sequence length and avoid truncation warnings."
        )
    )
    parser.add_argument("paths", nargs="+", help="JSONL files with {'text': ChatML}")
    parser.add_argument(
        "--max-items-per-sample",
        type=int,
        default=4,
        help="Maximum number of Questionnaire.item entries per output sample (default: 4)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite files in place and create .bak backups",
    )

    args = parser.parse_args()
    if args.max_items_per_sample < 1:
        raise SystemExit("--max-items-per-sample must be >= 1")

    total_in = 0
    total_out = 0

    for p in args.paths:
        path = Path(p)
        if not path.exists():
            raise SystemExit(f"File not found: {path}")

        in_rows, out_rows = presplit_jsonl(
            path,
            max_items_per_sample=args.max_items_per_sample,
            in_place=args.in_place,
        )
        total_in += in_rows
        total_out += out_rows
        ratio = out_rows / max(in_rows, 1)
        print(f"{path}: {in_rows} -> {out_rows} rows (x{ratio:.2f})")

    inflation = total_out / max(total_in, 1)
    print(f"TOTAL: {total_in} -> {total_out} rows (x{inflation:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
