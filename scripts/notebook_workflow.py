from __future__ import annotations

import json
import os
import random
import re
import shlex
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


FHIR_ALLOWED_TYPES: frozenset[str] = frozenset(
    {
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
)

SYSTEM_PROMPT = (
    "You are a FHIR R4 expert. Return only valid JSON for a complete "
    "FHIR Questionnaire resource."
)

DEFAULT_CONFIG_REL_PATH = "configs/Qwen2.5-Coder-7B-Instruct-8bit.yaml"
DEFAULT_DATASET_REL_PATH = "data/synthetic-aiprom-1500-firh4.jsonl"


@dataclass(frozen=True)
class ParsedRecord:
    row_id: int
    prompt: str
    completion_raw: str
    text: str
    assistant_obj: dict[str, Any]


@dataclass(frozen=True)
class AuditSummary:
    total_records: int
    valid_records: int
    invalid_records: int
    pass_rate: float
    avg_items_per_form: float
    type_counts: dict[str, int]
    issue_counts: dict[str, int]


@dataclass(frozen=True)
class WorkflowConfig:
    repo_root: Path
    dataset_path: Path
    config_path: Path
    model_name: str
    model_alias: str
    model_backend: str
    global_seed: int
    shared_artifacts_dir: Path
    model_artifacts_dir: Path
    run_materialization: bool
    run_training: bool
    run_export: bool
    run_adapter_inference: bool
    run_baseline_inference: bool
    run_evaluation: bool
    update_tunes_reports: bool


@dataclass(frozen=True)
class MaterializedDataset:
    train_path: Path
    val_path: Path
    valid_path: Path
    dataset_manifest_path: Path
    split_manifest_path: Path
    train_size: int
    val_size: int
    test_size: int


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_name.strip())
    return slug.strip("-._").lower() or "default-model"


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "lab").is_dir():
            return candidate
    raise FileNotFoundError("Could not resolve repository root from the current working directory.")


def resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (repo_root / path)


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return payload


def build_workflow_config(
    env: Mapping[str, str] | None = None,
    *,
    repo_root: Path | None = None,
) -> WorkflowConfig:
    environment = dict(env or os.environ)
    root = find_repo_root(repo_root)

    config_raw = environment.get("AIPROM_CONFIG", DEFAULT_CONFIG_REL_PATH).strip() or DEFAULT_CONFIG_REL_PATH
    config_path = resolve_path(root, config_raw)
    yaml_config = load_yaml_config(config_path)

    dataset_raw = environment.get("AIPROM_DATASET", DEFAULT_DATASET_REL_PATH).strip() or DEFAULT_DATASET_REL_PATH
    dataset_path = resolve_path(root, dataset_raw)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    model_name = str(environment.get("AIPROM_MODEL_NAME") or yaml_config.get("model") or "").strip()
    if not model_name:
        raise ValueError("Model name is missing. Set AIPROM_MODEL_NAME or provide model: in the YAML config.")

    model_alias = str(environment.get("AIPROM_MODEL_ALIAS") or "").strip() or slugify_model_name(model_name)
    model_backend = str(environment.get("AIPROM_MODEL_BACKEND") or "mlx_lm").strip().lower() or "mlx_lm"
    if model_backend != "mlx_lm":
        raise ValueError(f"Unsupported model backend for this notebook: {model_backend!r}")

    global_seed = int(environment.get("AIPROM_GLOBAL_SEED") or yaml_config.get("seed") or 173)
    shared_artifacts_dir = root / "lab" / "artifacts"
    model_artifacts_dir = shared_artifacts_dir / model_alias

    return WorkflowConfig(
        repo_root=root,
        dataset_path=dataset_path,
        config_path=config_path,
        model_name=model_name,
        model_alias=model_alias,
        model_backend=model_backend,
        global_seed=global_seed,
        shared_artifacts_dir=shared_artifacts_dir,
        model_artifacts_dir=model_artifacts_dir,
        run_materialization=parse_bool(environment.get("AIPROM_RUN_MATERIALIZATION"), default=False),
        run_training=parse_bool(environment.get("AIPROM_RUN_TRAINING"), default=False),
        run_export=parse_bool(environment.get("AIPROM_RUN_EXPORT"), default=False),
        run_adapter_inference=parse_bool(environment.get("AIPROM_RUN_ADAPTER_INFERENCE"), default=False),
        run_baseline_inference=parse_bool(environment.get("AIPROM_RUN_BASELINE_INFERENCE"), default=False),
        run_evaluation=parse_bool(environment.get("AIPROM_RUN_EVALUATION"), default=False),
        update_tunes_reports=parse_bool(environment.get("AIPROM_UPDATE_TUNES_REPORTS"), default=False),
    )


def workflow_summary(workflow: WorkflowConfig) -> dict[str, Any]:
    payload = asdict(workflow)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in payload.items()
    }


def build_training_text(prompt: str, completion_json: str) -> str:
    prompt_clean = prompt.strip()
    completion_clean = completion_json.strip()
    if not prompt_clean:
        raise ValueError("Prompt is empty")
    if not completion_clean:
        raise ValueError("Completion JSON is empty")
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt_clean}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{completion_clean}\n"
        "<|im_end|>"
    )


def build_generation_prompt(user_prompt: str) -> str:
    prompt_clean = user_prompt.strip()
    if not prompt_clean:
        raise ValueError("User prompt is empty")
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt_clean}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_prompt_completion_records(path: Path) -> tuple[list[ParsedRecord], list[str]]:
    records: list[ParsedRecord] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_id, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {row_id}: invalid JSON ({exc})")
                continue
            if not isinstance(record, dict):
                errors.append(f"Line {row_id}: expected a JSON object")
                continue
            prompt = record.get("prompt")
            completion = record.get("completion")
            if not isinstance(prompt, str) or not prompt.strip():
                errors.append(f"Line {row_id}: missing or invalid prompt")
                continue
            if not isinstance(completion, str) or not completion.strip():
                errors.append(f"Line {row_id}: missing or invalid completion")
                continue
            try:
                assistant_obj = json.loads(completion)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {row_id}: completion is not valid JSON ({exc})")
                continue
            if not isinstance(assistant_obj, dict):
                errors.append(f"Line {row_id}: completion JSON must be an object")
                continue
            text = build_training_text(prompt, completion)
            records.append(
                ParsedRecord(
                    row_id=row_id,
                    prompt=prompt,
                    completion_raw=completion,
                    text=text,
                    assistant_obj=assistant_obj,
                )
            )
    return records, errors


def iter_questionnaire_items(items: Any) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []

    def walk(nodes: Any) -> None:
        if not isinstance(nodes, list):
            return
        for node in nodes:
            if not isinstance(node, dict):
                continue
            flattened.append(node)
            walk(node.get("item"))

    walk(items)
    return flattened


def audit_questionnaire_item(item: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    link_id = item.get("linkId")
    item_type = item.get("type")
    if not isinstance(link_id, str) or not link_id.strip():
        issues.append("missing_or_invalid_linkId")
    if not isinstance(item_type, str) or item_type not in FHIR_ALLOWED_TYPES:
        issues.append("missing_or_invalid_type")
        item_type = None
    if item_type not in {"group", "display", None}:
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            issues.append("missing_or_invalid_text")
    if item_type in {"choice", "open-choice"}:
        options = item.get("answerOption")
        if not isinstance(options, list) or not options:
            issues.append("missing_or_invalid_answerOption")
    children = item.get("item")
    if children is not None and not isinstance(children, list):
        issues.append("invalid_nested_item_block")
    if isinstance(children, list):
        for child in children:
            if not isinstance(child, dict):
                issues.append("invalid_nested_item_block")
                continue
            issues.extend(audit_questionnaire_item(child))
    return issues


def audit_questionnaire(payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if payload.get("resourceType") != "Questionnaire":
        issues.append("invalid_resourceType")
    for field_name in ("id", "status", "title"):
        value = payload.get(field_name)
        if not isinstance(value, str) or not value.strip():
            issues.append(f"missing_or_invalid_{field_name}")
    root_items = payload.get("item")
    if not isinstance(root_items, list) or not root_items:
        issues.append("missing_root_items")
        return issues
    for item in root_items:
        if not isinstance(item, dict):
            issues.append("invalid_root_item")
            continue
        issues.extend(audit_questionnaire_item(item))
    return issues


def summarize_dataset(records: list[ParsedRecord]) -> dict[str, Any]:
    item_counter: Counter[str] = Counter()
    item_counts: list[int] = []
    for record in records:
        items = iter_questionnaire_items(record.assistant_obj.get("item"))
        item_counts.append(len(items))
        for item in items:
            item_type = item.get("type")
            if isinstance(item_type, str):
                item_counter[item_type] += 1
            else:
                item_counter["<missing>"] += 1
    total = len(records)
    return {
        "records": total,
        "min_items_per_form": min(item_counts) if item_counts else 0,
        "max_items_per_form": max(item_counts) if item_counts else 0,
        "avg_items_per_form": round(sum(item_counts) / total, 2) if total else 0.0,
        "item_type_counts": dict(sorted(item_counter.items())),
    }


def run_fhir_audit(records: list[ParsedRecord]) -> AuditSummary:
    type_counter: Counter[str] = Counter()
    issue_counter: Counter[str] = Counter()
    total_items = 0
    valid_records = 0

    for record in records:
        payload = record.assistant_obj
        flat_items = iter_questionnaire_items(payload.get("item"))
        total_items += len(flat_items)
        for item in flat_items:
            item_type = item.get("type")
            type_counter[item_type if isinstance(item_type, str) else "<missing>"] += 1
        issues = audit_questionnaire(payload)
        if not issues:
            valid_records += 1
        else:
            issue_counter.update(issues)

    total_records = len(records)
    invalid_records = total_records - valid_records
    pass_rate = round((valid_records / total_records) * 100.0, 2) if total_records else 0.0
    avg_items = round(total_items / total_records, 2) if total_records else 0.0
    return AuditSummary(
        total_records=total_records,
        valid_records=valid_records,
        invalid_records=invalid_records,
        pass_rate=pass_rate,
        avg_items_per_form=avg_items,
        type_counts=dict(sorted(type_counter.items())),
        issue_counts=dict(sorted(issue_counter.items(), key=lambda pair: (-pair[1], pair[0]))),
    )


def dominant_item_type(payload: dict[str, Any]) -> str:
    counter: Counter[str] = Counter()
    for item in iter_questionnaire_items(payload.get("item")):
        item_type = item.get("type")
        counter[item_type if isinstance(item_type, str) and item_type.strip() else "<missing>"] += 1
    if not counter:
        return "<missing>"
    return sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]


def stratified_train_val_test_split(
    records: list[ParsedRecord],
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 173,
) -> tuple[list[int], list[int], list[int]]:
    if not records:
        raise ValueError("Cannot split an empty dataset")
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("Expected 0 < val_ratio, test_ratio and val_ratio + test_ratio < 1")

    rng = random.Random(seed)
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        buckets[dominant_item_type(record.assistant_obj)].append(index)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for indices in buckets.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_test = max(1, int(round(n_total * test_ratio))) if n_total > 2 else 0
        remaining = n_total - n_test
        n_val = max(1, int(round(n_total * val_ratio))) if remaining > 1 else 0
        n_val = min(n_val, max(remaining - 1, 0))

        test_idx.extend(shuffled[:n_test])
        val_idx.extend(shuffled[n_test:n_test + n_val])
        train_idx.extend(shuffled[n_test + n_val:])

    train_idx.sort()
    val_idx.sort()
    test_idx.sort()
    return train_idx, val_idx, test_idx


def summarize_split_distribution(records: list[ParsedRecord], indices: list[int]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for index in indices:
        counter[dominant_item_type(records[index].assistant_obj)] += 1
    return dict(sorted(counter.items()))


def write_split_manifest(
    output_path: Path,
    *,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    seed: int,
    val_ratio: float,
    test_ratio: float,
    train_distribution: dict[str, int],
    val_distribution: dict[str, int],
    test_distribution: dict[str, int],
) -> Path:
    payload = {
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_test": len(test_indices),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "train_type_distribution": train_distribution,
        "val_type_distribution": val_distribution,
        "test_type_distribution": test_distribution,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def materialize_dataset(
    workflow: WorkflowConfig,
    records: list[ParsedRecord],
    *,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> MaterializedDataset:
    shared_dir = workflow.shared_artifacts_dir
    shared_dir.mkdir(parents=True, exist_ok=True)

    train_path = shared_dir / "train.jsonl"
    val_path = shared_dir / "val.jsonl"
    valid_path = shared_dir / "valid.jsonl"
    dataset_manifest_path = shared_dir / "dataset_manifest.json"
    split_manifest_path = shared_dir / "split_manifest.json"

    def write_jsonl(path: Path, indices: list[int]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for index in indices:
                handle.write(json.dumps({"text": records[index].text}, ensure_ascii=False) + "\n")

    write_jsonl(train_path, train_indices)
    write_jsonl(val_path, val_indices)
    write_jsonl(valid_path, val_indices)

    train_distribution = summarize_split_distribution(records, train_indices)
    val_distribution = summarize_split_distribution(records, val_indices)
    test_distribution = summarize_split_distribution(records, test_indices)
    write_split_manifest(
        split_manifest_path,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        seed=seed,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        train_distribution=train_distribution,
        val_distribution=val_distribution,
        test_distribution=test_distribution,
    )

    dataset_manifest = {
        "dataset_source": str(workflow.dataset_path),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "valid_path": str(valid_path),
        "split_manifest_path": str(split_manifest_path),
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_test": len(test_indices),
        "split_seed": seed,
        "global_seed": workflow.global_seed,
    }
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")

    return MaterializedDataset(
        train_path=train_path,
        val_path=val_path,
        valid_path=valid_path,
        dataset_manifest_path=dataset_manifest_path,
        split_manifest_path=split_manifest_path,
        train_size=len(train_indices),
        val_size=len(val_indices),
        test_size=len(test_indices),
    )


def build_training_snapshot(workflow: WorkflowConfig, materialized: MaterializedDataset) -> dict[str, Any]:
    yaml_config = load_yaml_config(workflow.config_path)
    snapshot = {
        "experiment_name": f"{workflow.model_alias}_fhir_questionnaire_lora",
        "model": {
            "backend": workflow.model_backend,
            "name": workflow.model_name,
        },
        "data": {
            "dataset_path": str(workflow.dataset_path),
            "train_jsonl": str(materialized.train_path),
            "val_jsonl": str(materialized.val_path),
            "valid_jsonl": str(materialized.valid_path),
            "split_manifest_path": str(materialized.split_manifest_path),
            "n_train": materialized.train_size,
            "n_val": materialized.val_size,
            "n_test": materialized.test_size,
        },
        "reproducibility": {
            "global_seed": workflow.global_seed,
            "config_path": str(workflow.config_path),
        },
        "config_yaml": yaml_config,
        "outputs": {
            "model_artifact_dir": str(workflow.model_artifacts_dir),
            "training_config_path": str(workflow.model_artifacts_dir / "training_config_stable.json"),
            "checkpoint_dir": str(workflow.model_artifacts_dir / "checkpoints_stable"),
        },
    }
    return snapshot


def persist_training_snapshot(snapshot: dict[str, Any]) -> Path:
    output_path = Path(snapshot["outputs"]["training_config_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return output_path


def build_training_command(workflow: WorkflowConfig) -> list[str]:
    rel_config = workflow.config_path
    try:
        rel_config = workflow.config_path.relative_to(workflow.repo_root)
    except ValueError:
        pass
    return [
        "poetry",
        "run",
        "python",
        "-m",
        "mlx_lm",
        "lora",
        "-c",
        str(rel_config),
    ]


def adapter_checkpoint_dir(workflow: WorkflowConfig) -> Path:
    return workflow.model_artifacts_dir / "checkpoints_stable"


def build_fuse_command(workflow: WorkflowConfig, *, fused_dir: Path | None = None) -> list[str]:
    target_dir = fused_dir or (workflow.model_artifacts_dir / "fused-noquant")
    return [
        "poetry",
        "run",
        "python",
        "-m",
        "mlx_lm",
        "fuse",
        "--model",
        workflow.model_name,
        "--adapter-path",
        str(adapter_checkpoint_dir(workflow)),
        "--save-path",
        str(target_dir),
    ]


def build_llama_cpp_convert_command(
    workflow: WorkflowConfig,
    *,
    fused_dir: Path | None = None,
    output_path: Path | None = None,
) -> list[str]:
    source_dir = fused_dir or (workflow.model_artifacts_dir / "fused-noquant")
    target_path = output_path or (workflow.model_artifacts_dir / "fused-noquant" / "model-fused-f16.gguf")
    script_path = workflow.repo_root / "tools" / "llama.cpp" / "convert_hf_to_gguf.py"
    return [
        "poetry",
        "run",
        "python",
        str(script_path),
        str(source_dir),
        "--outfile",
        str(target_path),
        "--outtype",
        "f16",
    ]


def build_generate_command(
    workflow: WorkflowConfig,
    *,
    user_prompt: str,
    max_tokens: int = 1600,
    seed: int | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    use_adapter: bool = False,
) -> list[str]:
    command = [
        "poetry",
        "run",
        "python",
        "-m",
        "mlx_lm",
        "generate",
        "--ignore-chat-template",
        "--model",
        workflow.model_name,
        "--prompt",
        build_generation_prompt(user_prompt),
        "--max-tokens",
        str(max_tokens),
        "--seed",
        str(seed if seed is not None else workflow.global_seed),
        "--temp",
        str(temperature),
        "--top-p",
        str(top_p),
        "--verbose",
        "F",
        "--extra-eos-token",
        "<|im_end|>",
    ]
    if use_adapter:
        command.extend(["--adapter-path", str(adapter_checkpoint_dir(workflow))])
    return command


def command_preview(command: list[str]) -> str:
    return shlex.join(command)


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def artifact_status(workflow: WorkflowConfig) -> dict[str, dict[str, Any]]:
    shared = workflow.shared_artifacts_dir
    model = workflow.model_artifacts_dir
    inventory = {
        "split_manifest": shared / "split_manifest.json",
        "dataset_manifest": shared / "dataset_manifest.json",
        "training_config": model / "training_config_stable.json",
        "adapter_checkpoint_dir": model / "checkpoints_stable",
        "fused_noquant_dir": model / "fused-noquant",
        "ab_rule_eval": shared / "ab_rule_eval.json",
        "ab_rule_eval_analysis": shared / "ab_rule_eval_analysis.json",
        "adapter_go_no_go": shared / "adapter_go_no_go.json",
        "tune_report": workflow.repo_root / "tunes" / "models" / f"{workflow.model_alias}.md",
    }
    return {
        name: {
            "path": str(path),
            "exists": path.exists(),
        }
        for name, path in inventory.items()
    }


def build_publication_notes(workflow: WorkflowConfig) -> list[dict[str, str]]:
    return [
        {
            "topic": "Dataset",
            "note": f"Primary dataset: {workflow.dataset_path.relative_to(workflow.repo_root)}",
        },
        {
            "topic": "Config",
            "note": f"Active YAML config: {workflow.config_path.relative_to(workflow.repo_root)}",
        },
        {
            "topic": "Artifacts",
            "note": f"Shared artifacts: {workflow.shared_artifacts_dir.relative_to(workflow.repo_root)}",
        },
        {
            "topic": "Model artifacts",
            "note": f"Model-scoped artifacts: {workflow.model_artifacts_dir.relative_to(workflow.repo_root)}",
        },
        {
            "topic": "Execution policy",
            "note": "Training, export, inference and evaluation are disabled by default.",
        },
    ]


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    value = (text or "").strip()
    if not value:
        return None, "stdout is empty"
    try:
        payload = json.loads(value)
        if isinstance(payload, dict):
            return payload, None
    except Exception:
        pass
    start = value.find("{")
    if start < 0:
        return None, "no JSON object found"
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(value[start:])
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: {exc}"
    if not isinstance(payload, dict):
        return None, "first JSON fragment is not an object"
    return payload, None


def sample_validation_records(
    records: list[ParsedRecord],
    indices: list[int],
    *,
    limit: int,
    seed: int,
) -> list[ParsedRecord]:
    if limit <= 0:
        raise ValueError("limit must be > 0")
    selected_indices = indices[:]
    rng = random.Random(seed)
    rng.shuffle(selected_indices)
    return [records[index] for index in selected_indices[: min(limit, len(selected_indices))]]


def count_questionnaire_items(payload: dict[str, Any] | None) -> int:
    if not isinstance(payload, dict):
        return 0
    return len(iter_questionnaire_items(payload.get("item")))


def item_type_set(payload: dict[str, Any] | None) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    return {
        item_type
        for item_type in (
            item.get("type")
            for item in iter_questionnaire_items(payload.get("item"))
            if isinstance(item, dict)
        )
        if isinstance(item_type, str) and item_type.strip()
    }


def has_any_initial(value: Any) -> bool:
    if isinstance(value, dict):
        if "initial" in value:
            return True
        return any(has_any_initial(child) for child in value.values())
    if isinstance(value, list):
        return any(has_any_initial(child) for child in value)
    return False


def choice_items_have_answer_options(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    for item in iter_questionnaire_items(payload.get("item")):
        item_type = item.get("type")
        if item_type in {"choice", "open-choice"}:
            options = item.get("answerOption")
            if not isinstance(options, list) or not options:
                return False
    return True


def evaluate_generation_rules(
    generated_obj: dict[str, Any] | None,
    expected_obj: dict[str, Any],
) -> dict[str, bool]:
    expected_type_set = item_type_set(expected_obj)
    generated_type_set = item_type_set(generated_obj)
    expected_item_count = count_questionnaire_items(expected_obj)
    generated_item_count = count_questionnaire_items(generated_obj)

    rules: dict[str, bool] = {
        "parseable_json": isinstance(generated_obj, dict),
        "resourceType_questionnaire": False,
        "id_present": False,
        "status_present": False,
        "title_present": False,
        "root_items_non_empty": False,
        "item_count_matches_gold": False,
        "item_types_match_gold": False,
        "choice_items_have_answer_options": False,
        "no_initial_anywhere": False,
    }

    if not isinstance(generated_obj, dict):
        return rules

    root_items = generated_obj.get("item")
    rules["resourceType_questionnaire"] = generated_obj.get("resourceType") == "Questionnaire"
    rules["id_present"] = isinstance(generated_obj.get("id"), str) and bool(generated_obj.get("id", "").strip())
    rules["status_present"] = isinstance(generated_obj.get("status"), str) and bool(generated_obj.get("status", "").strip())
    rules["title_present"] = isinstance(generated_obj.get("title"), str) and bool(generated_obj.get("title", "").strip())
    rules["root_items_non_empty"] = isinstance(root_items, list) and len(root_items) > 0
    rules["item_count_matches_gold"] = generated_item_count == expected_item_count
    rules["item_types_match_gold"] = generated_type_set == expected_type_set
    rules["choice_items_have_answer_options"] = choice_items_have_answer_options(generated_obj)
    rules["no_initial_anywhere"] = not has_any_initial(generated_obj)
    return rules


def aggregate_rule_results(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("mode") == mode]
    if not selected:
        return {
            "mode": mode,
            "n": 0,
            "all_rules_pass": 0,
            "success_rate": 0.0,
            "rule_pass_rates": {},
        }

    rule_names = list(selected[0]["rules"].keys())
    counts: Counter[str] = Counter()
    all_rules_pass = 0
    for row in selected:
        if all(bool(row["rules"].get(rule, False)) for rule in rule_names):
            all_rules_pass += 1
        for rule in rule_names:
            counts[rule] += int(bool(row["rules"].get(rule, False)))

    total = len(selected)
    return {
        "mode": mode,
        "n": total,
        "all_rules_pass": all_rules_pass,
        "success_rate": round(all_rules_pass * 100.0 / total, 2),
        "rule_pass_rates": {
            rule: round(counts[rule] * 100.0 / total, 2)
            for rule in rule_names
        },
    }


def build_analysis_payload(rule_eval_payload: dict[str, Any]) -> dict[str, Any]:
    rows = rule_eval_payload.get("sample_level", [])
    if not isinstance(rows, list):
        raise ValueError("rule_eval_payload['sample_level'] must be a list")

    strict_keys = [
        "parseable_json",
        "resourceType_questionnaire",
        "id_present",
        "status_present",
        "title_present",
        "root_items_non_empty",
        "item_count_matches_gold",
        "item_types_match_gold",
        "choice_items_have_answer_options",
        "no_initial_anywhere",
    ]
    relaxed_keys = [
        "parseable_json",
        "resourceType_questionnaire",
        "status_present",
        "title_present",
        "root_items_non_empty",
        "choice_items_have_answer_options",
    ]

    def summarize(mode: str, keys: list[str]) -> dict[str, Any]:
        selected = [row for row in rows if row.get("mode") == mode]
        total = len(selected)
        if total == 0:
            return {"mode": mode, "n": 0, "all_pass": 0, "rate": 0.0, "fail_counter": {}}
        fail_counter: Counter[str] = Counter()
        all_pass = 0
        for row in selected:
            rules = row.get("rules", {})
            if not isinstance(rules, dict):
                continue
            failures = [key for key in keys if not bool(rules.get(key, False))]
            if not failures:
                all_pass += 1
            for failure in failures:
                fail_counter[failure] += 1
        return {
            "mode": mode,
            "n": total,
            "all_pass": all_pass,
            "rate": round(all_pass * 100.0 / total, 2),
            "fail_counter": dict(fail_counter.most_common()),
        }

    return {
        "strict": {
            "baseline": summarize("baseline", strict_keys),
            "adapter": summarize("adapter", strict_keys),
        },
        "relaxed": {
            "baseline": summarize("baseline", relaxed_keys),
            "adapter": summarize("adapter", relaxed_keys),
        },
    }


def build_go_no_go_payload(
    rule_eval_payload: dict[str, Any],
    analysis_payload: dict[str, Any],
) -> dict[str, Any]:
    baseline_rates = rule_eval_payload.get("baseline", {}).get("rule_pass_rates", {})
    adapter_rates = rule_eval_payload.get("adapter", {}).get("rule_pass_rates", {})
    strict_baseline = float(analysis_payload.get("strict", {}).get("baseline", {}).get("rate", 0.0))
    strict_adapter = float(analysis_payload.get("strict", {}).get("adapter", {}).get("rate", 0.0))
    relaxed_baseline = float(analysis_payload.get("relaxed", {}).get("baseline", {}).get("rate", 0.0))
    relaxed_adapter = float(analysis_payload.get("relaxed", {}).get("adapter", {}).get("rate", 0.0))

    core_rules = [
        "parseable_json",
        "resourceType_questionnaire",
        "root_items_non_empty",
    ]
    core_regressions = []
    for rule in core_rules:
        baseline_value = float(baseline_rates.get(rule, 0.0))
        adapter_value = float(adapter_rates.get(rule, 0.0))
        if adapter_value < baseline_value:
            core_regressions.append(
                f"{rule}: adapter={adapter_value}% < baseline={baseline_value}%"
            )

    strict_delta = round(strict_adapter - strict_baseline, 2)
    relaxed_delta = round(relaxed_adapter - relaxed_baseline, 2)
    strict_ok = strict_adapter >= strict_baseline
    relaxed_ok = relaxed_delta >= 5.0
    core_ok = len(core_regressions) == 0

    if strict_ok and relaxed_ok and core_ok:
        decision = "GO"
    elif core_ok and (strict_delta >= 0 or relaxed_delta >= 0):
        decision = "CONDITIONAL_GO"
    else:
        decision = "NO_GO"

    return {
        "decision": decision,
        "strict": {
            "baseline_rate": strict_baseline,
            "adapter_rate": strict_adapter,
            "delta_pp": strict_delta,
        },
        "relaxed": {
            "baseline_rate": relaxed_baseline,
            "adapter_rate": relaxed_adapter,
            "delta_pp": relaxed_delta,
        },
        "core_rule_regressions": core_regressions,
        "criteria": {
            "strict_non_regression": strict_ok,
            "relaxed_delta_ge_5pp": relaxed_ok,
            "no_core_rule_regression": core_ok,
        },
    }


def write_json_artifact(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_generation_command(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def evaluate_record_pair(
    workflow: WorkflowConfig,
    record: ParsedRecord,
    *,
    sample_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode, use_adapter in (("baseline", False), ("adapter", True)):
        command = build_generate_command(
            workflow,
            user_prompt=record.prompt,
            use_adapter=use_adapter,
        )
        result = run_generation_command(command, cwd=workflow.repo_root)
        generated_json, parse_error = extract_json_object(result.stdout)
        rows.append(
            {
                "sample": sample_index,
                "mode": mode,
                "row_id": record.row_id,
                "prompt": record.prompt,
                "command": command,
                "return_code": result.returncode,
                "rules": evaluate_generation_rules(generated_json, record.assistant_obj),
                "expected_id": record.assistant_obj.get("id"),
                "generated_id": generated_json.get("id") if isinstance(generated_json, dict) else None,
                "generated_json": generated_json,
                "parse_error": parse_error,
                "stderr_head": result.stderr.splitlines()[:5],
            }
        )
    return rows


def build_rule_eval_payload(
    workflow: WorkflowConfig,
    *,
    sample_rows: list[dict[str, Any]],
    materialized: MaterializedDataset,
    n_samples: int,
) -> dict[str, Any]:
    baseline_summary = aggregate_rule_results(sample_rows, "baseline")
    adapter_summary = aggregate_rule_results(sample_rows, "adapter")
    delta_by_rule = {
        rule: round(
            float(adapter_summary["rule_pass_rates"].get(rule, 0.0))
            - float(baseline_summary["rule_pass_rates"].get(rule, 0.0)),
            2,
        )
        for rule in sorted(
            set(baseline_summary["rule_pass_rates"].keys())
            | set(adapter_summary["rule_pass_rates"].keys())
        )
    }
    return {
        "config": {
            "n_samples": n_samples,
            "seed": workflow.global_seed,
            "model": workflow.model_name,
            "model_alias": workflow.model_alias,
            "adapter_dir": str(adapter_checkpoint_dir(workflow)),
            "split_manifest_path": str(materialized.split_manifest_path),
            "dataset_manifest_path": str(materialized.dataset_manifest_path),
        },
        "baseline": baseline_summary,
        "adapter": adapter_summary,
        "delta_adapter_minus_baseline": {
            "all_rules_pass_rate": round(
                float(adapter_summary["success_rate"]) - float(baseline_summary["success_rate"]),
                2,
            ),
            "by_rule": delta_by_rule,
        },
        "sample_level": sample_rows,
    }


def summarize_evaluation_payloads(
    rule_eval_payload: dict[str, Any] | None,
    analysis_payload: dict[str, Any] | None,
    decision_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rule_eval = rule_eval_payload or {}
    analysis = analysis_payload or {}
    decision = decision_payload or {}
    baseline = rule_eval.get("baseline", {}) if isinstance(rule_eval.get("baseline"), dict) else {}
    adapter = rule_eval.get("adapter", {}) if isinstance(rule_eval.get("adapter"), dict) else {}
    strict = analysis.get("strict", {}) if isinstance(analysis.get("strict"), dict) else {}
    relaxed = analysis.get("relaxed", {}) if isinstance(analysis.get("relaxed"), dict) else {}
    return [
        {"metric": "baseline_samples", "value": baseline.get("n")},
        {"metric": "adapter_samples", "value": adapter.get("n")},
        {"metric": "strict_baseline_rate", "value": (strict.get("baseline") or {}).get("rate")},
        {"metric": "strict_adapter_rate", "value": (strict.get("adapter") or {}).get("rate")},
        {"metric": "relaxed_baseline_rate", "value": (relaxed.get("baseline") or {}).get("rate")},
        {"metric": "relaxed_adapter_rate", "value": (relaxed.get("adapter") or {}).get("rate")},
        {"metric": "go_no_go_decision", "value": decision.get("decision")},
    ]


def run_ab_evaluation(
    workflow: WorkflowConfig,
    *,
    validation_records: list[ParsedRecord],
    materialized: MaterializedDataset,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not adapter_checkpoint_dir(workflow).exists():
        raise FileNotFoundError(
            f"Adapter checkpoints were not found at {adapter_checkpoint_dir(workflow)}"
        )

    sample_rows: list[dict[str, Any]] = []
    for sample_index, record in enumerate(validation_records, start=1):
        sample_rows.extend(
            evaluate_record_pair(
                workflow,
                record,
                sample_index=sample_index,
            )
        )

    rule_eval_payload = build_rule_eval_payload(
        workflow,
        sample_rows=sample_rows,
        materialized=materialized,
        n_samples=len(validation_records),
    )
    analysis_payload = build_analysis_payload(rule_eval_payload)
    decision_payload = build_go_no_go_payload(rule_eval_payload, analysis_payload)
    return rule_eval_payload, analysis_payload, decision_payload