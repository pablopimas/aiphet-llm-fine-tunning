#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

ASSISTANT = "<|im_start|>assistant\n"
END = "<|im_end|>"

FILES = [
    Path("lab/artifacts/train.jsonl"),
    Path("lab/artifacts/val.jsonl"),
    Path("lab/artifacts/valid.jsonl"),
]

FHIR_TYPES = {
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
    "attachment",
    "reference",
    "quantity",
}


def audit(path: Path) -> None:
    total = 0
    bad_jsonl = 0
    bad_text = 0
    missing_assistant = 0
    bad_assistant_json = 0
    not_questionnaire = 0
    bad_item_array = 0
    bad_item = 0
    examples: list[tuple[int, str]] = []

    if not path.exists():
        print(f"\n{path}\n  MISSING")
        return

    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        total += 1

        try:
            row = json.loads(line)
        except Exception as e:
            bad_jsonl += 1
            if len(examples) < 8:
                examples.append((i, f"jsonl parse error: {e}"))
            continue

        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            bad_text += 1
            if len(examples) < 8:
                examples.append((i, "missing/invalid text"))
            continue

        text = row["text"]
        s = text.rfind(ASSISTANT)
        if s < 0:
            missing_assistant += 1
            if len(examples) < 8:
                examples.append((i, "missing assistant marker"))
            continue

        j0 = s + len(ASSISTANT)
        j1 = text.find(END, j0)
        if j1 < 0:
            missing_assistant += 1
            if len(examples) < 8:
                examples.append((i, "missing end marker after assistant"))
            continue

        payload = text[j0:j1].strip()
        try:
            obj = json.loads(payload)
        except Exception as e:
            bad_assistant_json += 1
            if len(examples) < 8:
                examples.append((i, f"assistant JSON parse error: {e}"))
            continue

        if not isinstance(obj, dict) or obj.get("resourceType") != "Questionnaire":
            not_questionnaire += 1
            if len(examples) < 8:
                examples.append((i, f"resourceType={obj.get('resourceType')!r}"))
            continue

        items = obj.get("item")
        if not isinstance(items, list) or not items:
            bad_item_array += 1
            if len(examples) < 8:
                examples.append((i, "missing/empty item array"))
            continue

        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                bad_item += 1
                if len(examples) < 8:
                    examples.append((i, f"item[{idx}] not object"))
                break

            item_type = item.get("type")
            link_id = item.get("linkId")
            if not isinstance(link_id, str) or not link_id.strip() or not isinstance(item_type, str) or item_type not in FHIR_TYPES:
                bad_item += 1
                if len(examples) < 8:
                    examples.append((i, f"item[{idx}] invalid linkId/type"))
                break

            if item_type not in {"group", "display"}:
                text_value = item.get("text")
                if not isinstance(text_value, str) or not text_value.strip():
                    bad_item += 1
                    if len(examples) < 8:
                        examples.append((i, f"item[{idx}] missing text"))
                    break

            if item_type in {"choice", "open-choice"}:
                answer_option = item.get("answerOption")
                if not isinstance(answer_option, list) or not answer_option:
                    bad_item += 1
                    if len(examples) < 8:
                        examples.append((i, f"item[{idx}] missing answerOption"))
                    break

    print(f"\n{path}")
    print(f"  rows={total}")
    print(f"  bad_jsonl={bad_jsonl}")
    print(f"  bad_text={bad_text}")
    print(f"  missing_assistant={missing_assistant}")
    print(f"  bad_assistant_json={bad_assistant_json}")
    print(f"  not_questionnaire={not_questionnaire}")
    print(f"  bad_item_array={bad_item_array}")
    print(f"  bad_item={bad_item}")

    if examples:
        print("  examples:")
        for ln, msg in examples:
            print(f"    line {ln}: {msg}")


if __name__ == "__main__":
    for p in FILES:
        audit(p)
