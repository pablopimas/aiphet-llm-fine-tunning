#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, cast


FHIR_QUESTIONNAIRE_ITEM_TYPES: set[str] = {
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


@dataclass(frozen=True)
class ValidationReport:
    path: Path
    total_rows: int
    json_parse_errors: int
    completion_parse_errors: int
    blocking_errors: list[str]
    warnings: list[str]
    resource_types: Counter[str]
    item_type_counts: Counter[str]

    def is_ok(self) -> bool:
        return not self.blocking_errors and self.json_parse_errors == 0 and self.completion_parse_errors == 0


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _iter_jsonl_lines(path: Path) -> Iterable[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                yield line_no, line


def validate_prompt_completion_jsonl(
    path: Path,
    *,
    expected_resource_type: str | None = "Questionnaire",
    max_errors: int = 200,
) -> ValidationReport:
    """Validate a JSONL dataset where each row is {prompt: str, completion: str}.

    The completion is expected to be a JSON-encoded FHIR resource (commonly Questionnaire).

    Validation is intentionally conservative: it checks for structural correctness and
    key FHIR invariants used by this repo, without trying to fully validate against
    the official HL7 schema.
    """

    total_rows = 0
    json_parse_errors = 0
    completion_parse_errors = 0

    blocking_errors: list[str] = []
    warnings: list[str] = []

    resource_types: Counter[str] = Counter()
    item_type_counts: Counter[str] = Counter()

    for line_no, line in _iter_jsonl_lines(path):
        total_rows += 1

        try:
            record_any = json.loads(line)
        except Exception as exc:
            json_parse_errors += 1
            if len(blocking_errors) < max_errors:
                blocking_errors.append(f"Line {line_no}: invalid JSON ({exc})")
            continue

        if not isinstance(record_any, dict):
            if len(blocking_errors) < max_errors:
                blocking_errors.append(f"Line {line_no}: expected JSON object")
            continue

        record = cast(dict[str, Any], record_any)

        prompt = record.get("prompt")
        completion = record.get("completion")

        if _as_str(prompt) is None:
            if len(blocking_errors) < max_errors:
                blocking_errors.append(f"Line {line_no}: missing/invalid prompt (expected non-empty string)")

        if not isinstance(completion, str) or not completion.strip():
            if len(blocking_errors) < max_errors:
                blocking_errors.append(f"Line {line_no}: missing/invalid completion (expected non-empty string)")
            continue

        try:
            completion_obj_any = json.loads(completion)
        except Exception as exc:
            completion_parse_errors += 1
            if len(blocking_errors) < max_errors:
                blocking_errors.append(f"Line {line_no}: completion is not valid JSON ({exc})")
            continue

        if not isinstance(completion_obj_any, dict):
            if len(blocking_errors) < max_errors:
                blocking_errors.append(f"Line {line_no}: completion JSON must be an object")
            continue

        completion_obj = cast(dict[str, Any], completion_obj_any)

        resource_type = completion_obj.get("resourceType")
        if isinstance(resource_type, str):
            resource_types[resource_type] += 1
        else:
            resource_types["<missing>"] += 1

        if expected_resource_type is not None and resource_type != expected_resource_type:
            if len(blocking_errors) < max_errors:
                blocking_errors.append(
                    f"Line {line_no}: unexpected resourceType={resource_type!r} (expected {expected_resource_type!r})"
                )

        # Minimal Questionnaire checks
        items_any = completion_obj.get("item")
        if not isinstance(items_any, list) or not items_any:
            if len(blocking_errors) < max_errors:
                blocking_errors.append(f"Line {line_no}: Questionnaire.item must be a non-empty array")
            continue

        items = cast(list[Any], items_any)
        for idx, item_any in enumerate(items, start=1):
            if not isinstance(item_any, dict):
                if len(blocking_errors) < max_errors:
                    blocking_errors.append(f"Line {line_no}: item[{idx}] must be an object")
                continue

            item = cast(dict[str, Any], item_any)

            link_id = _as_str(item.get("linkId"))
            if link_id is None and len(blocking_errors) < max_errors:
                blocking_errors.append(f"Line {line_no}: item[{idx}] missing/invalid linkId")

            item_type = item.get("type")
            if isinstance(item_type, str):
                item_type_counts[item_type] += 1
            else:
                item_type_counts["<missing>"] += 1

            if not isinstance(item_type, str) or item_type not in FHIR_QUESTIONNAIRE_ITEM_TYPES:
                if len(blocking_errors) < max_errors:
                    blocking_errors.append(
                        f"Line {line_no}: item[{idx}] missing/invalid type={item_type!r}"
                    )
                continue

            # For question items (not group/display), require a non-empty text
            if item_type not in {"group", "display"}:
                if _as_str(item.get("text")) is None and len(blocking_errors) < max_errors:
                    blocking_errors.append(f"Line {line_no}: item[{idx}] missing/invalid text")

            # For choice-like items, require answerOption
            if item_type in {"choice", "open-choice"}:
                answer_option = item.get("answerOption")
                if not isinstance(answer_option, list) or not answer_option:
                    if len(blocking_errors) < max_errors:
                        blocking_errors.append(
                            f"Line {line_no}: item[{idx}] missing/invalid answerOption for type={item_type}"
                        )

            # Optional sanity checks (warnings only)
            required = item.get("required")
            if required is not None and not isinstance(required, bool):
                if len(warnings) < max_errors:
                    warnings.append(f"Line {line_no}: item[{idx}] required should be boolean")

    return ValidationReport(
        path=path,
        total_rows=total_rows,
        json_parse_errors=json_parse_errors,
        completion_parse_errors=completion_parse_errors,
        blocking_errors=blocking_errors,
        warnings=warnings,
        resource_types=resource_types,
        item_type_counts=item_type_counts,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate prompt/completion JSONL where completion is a FHIR Questionnaire JSON string"
    )
    parser.add_argument("path", help="Path to JSONL dataset")
    parser.add_argument(
        "--resource-type",
        default="Questionnaire",
        help="Expected FHIR resourceType in completion JSON (default: Questionnaire)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=200,
        help="Maximum number of errors/warnings to collect (default: 200)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    report = validate_prompt_completion_jsonl(
        path,
        expected_resource_type=args.resource_type if args.resource_type else None,
        max_errors=args.max_errors,
    )

    print(f"Dataset: {report.path}")
    print(f"Rows: {report.total_rows}")
    print(f"JSON parse errors: {report.json_parse_errors}")
    print(f"Completion parse errors: {report.completion_parse_errors}")
    print(f"Resource types: {dict(report.resource_types)}")
    print(f"Top item types: {report.item_type_counts.most_common(15)}")

    if report.warnings:
        print("Warnings (first 20):")
        for w in report.warnings[:20]:
            print(f"- {w}")
        if len(report.warnings) > 20:
            print(f"- ... and {len(report.warnings) - 20} more")

    if report.blocking_errors:
        print("Errors (first 50):")
        for e in report.blocking_errors[:50]:
            print(f"- {e}")
        if len(report.blocking_errors) > 50:
            print(f"- ... and {len(report.blocking_errors) - 50} more")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
