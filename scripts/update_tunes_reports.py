#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ModelTuneSummary:
    model_name: str
    model_alias: str
    artifact_root: Path
    generated_at_utc: str
    train_return_code: int | None = None
    train_duration_sec: float | None = None
    emissions_kg_co2eq: float | None = None
    adapter_inference_return_code: int | None = None
    baseline_inference_return_code: int | None = None
    adapter_parseable_json_rate: float | None = None
    baseline_parseable_json_rate: float | None = None
    adapter_relaxed_rate: float | None = None
    baseline_relaxed_rate: float | None = None
    relaxed_delta_pp: float | None = None
    strict_adapter_rate: float | None = None
    strict_baseline_rate: float | None = None
    strict_delta_pp: float | None = None
    go_no_go_decision: str | None = None
    n_eval_samples: int | None = None


def _safe_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def slugify_model_name(model_name: str) -> str:
    return (
        model_name.strip().lower().replace("/", "-").replace(" ", "-")
        if model_name
        else "unknown-model"
    )


def _fmt_num(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _extract_train_summary(training_payload: dict[str, Any]) -> tuple[int | None, float | None, float | None]:
    final_return_code = training_payload.get("final_return_code")
    attempts = training_payload.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return (
            int(final_return_code) if isinstance(final_return_code, int) else None,
            None,
            None,
        )

    first_attempt = _as_dict(attempts[0] if attempts else None)
    energy = _as_dict(first_attempt.get("energy"))

    return (
        int(final_return_code) if isinstance(final_return_code, int) else None,
        float(energy.get("duration_sec")) if isinstance(energy.get("duration_sec"), (int, float)) else None,
        float(energy.get("emissions_kg_co2eq"))
        if isinstance(energy.get("emissions_kg_co2eq"), (int, float))
        else None,
    )


def _extract_eval_summary(
    eval_payload: dict[str, Any] | None,
    analysis_payload: dict[str, Any] | None,
    go_no_go_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "adapter_parseable_json_rate": None,
        "baseline_parseable_json_rate": None,
        "strict_adapter_rate": None,
        "strict_baseline_rate": None,
        "strict_delta_pp": None,
        "adapter_relaxed_rate": None,
        "baseline_relaxed_rate": None,
        "relaxed_delta_pp": None,
        "go_no_go_decision": None,
        "n_eval_samples": None,
    }

    if eval_payload:
        baseline = _as_dict(eval_payload.get("baseline"))
        adapter = _as_dict(eval_payload.get("adapter"))
        baseline_rates = _as_dict(baseline.get("rule_pass_rates"))
        adapter_rates = _as_dict(adapter.get("rule_pass_rates"))
        out["baseline_parseable_json_rate"] = baseline_rates.get("parseable_json")
        out["adapter_parseable_json_rate"] = adapter_rates.get("parseable_json")
        if isinstance(adapter.get("n"), int):
            out["n_eval_samples"] = adapter.get("n")

    if analysis_payload:
        strict = _as_dict(analysis_payload.get("strict"))
        relaxed = _as_dict(analysis_payload.get("relaxed"))

        strict_base = _as_dict(strict.get("baseline"))
        strict_adp = _as_dict(strict.get("adapter"))
        relaxed_base = _as_dict(relaxed.get("baseline"))
        relaxed_adp = _as_dict(relaxed.get("adapter"))

        if isinstance(strict_base.get("rate"), (int, float)):
            out["strict_baseline_rate"] = float(strict_base["rate"])
        if isinstance(strict_adp.get("rate"), (int, float)):
            out["strict_adapter_rate"] = float(strict_adp["rate"])
        if isinstance(out["strict_adapter_rate"], float) and isinstance(out["strict_baseline_rate"], float):
            out["strict_delta_pp"] = out["strict_adapter_rate"] - out["strict_baseline_rate"]

        if isinstance(relaxed_base.get("rate"), (int, float)):
            out["baseline_relaxed_rate"] = float(relaxed_base["rate"])
        if isinstance(relaxed_adp.get("rate"), (int, float)):
            out["adapter_relaxed_rate"] = float(relaxed_adp["rate"])
        if isinstance(out["adapter_relaxed_rate"], float) and isinstance(out["baseline_relaxed_rate"], float):
            out["relaxed_delta_pp"] = out["adapter_relaxed_rate"] - out["baseline_relaxed_rate"]

    if go_no_go_payload and isinstance(go_no_go_payload.get("decision"), str):
        out["go_no_go_decision"] = str(go_no_go_payload["decision"])

    return out


def collect_model_summary(repo_root: Path, model_artifact_root: Path) -> ModelTuneSummary | None:
    cfg = _safe_json(model_artifact_root / "training_config_stable.json")
    if not cfg:
        return None

    model_cfg = _as_dict(cfg.get("model"))
    model_name = str(model_cfg.get("name", "")).strip() or model_artifact_root.name
    model_alias = slugify_model_name(model_name)

    now = datetime.now(UTC).isoformat()
    summary = ModelTuneSummary(
        model_name=model_name,
        model_alias=model_alias,
        artifact_root=model_artifact_root,
        generated_at_utc=now,
    )

    training_payload = _safe_json(model_artifact_root / "training_run_log.json") or _safe_json(repo_root / "lab" / "artifacts" / "training_run_log.json")
    if training_payload:
        trc, duration_sec, emissions_kg = _extract_train_summary(training_payload)
        summary.train_return_code = trc
        summary.train_duration_sec = duration_sec
        summary.emissions_kg_co2eq = emissions_kg

    adapter_inf = _safe_json(model_artifact_root / "inference_test_log.json")
    if adapter_inf and isinstance(adapter_inf.get("result"), dict):
        rc = adapter_inf["result"].get("return_code")
        summary.adapter_inference_return_code = int(rc) if isinstance(rc, int) else None

    baseline_inf = _safe_json(model_artifact_root / "inference_baseline_log.json")
    if baseline_inf and isinstance(baseline_inf.get("result"), dict):
        rc = baseline_inf["result"].get("return_code")
        summary.baseline_inference_return_code = int(rc) if isinstance(rc, int) else None

    global_artifacts = repo_root / "lab" / "artifacts"
    eval_payload = _safe_json(global_artifacts / "ab_rule_eval.json")
    analysis_payload = _safe_json(global_artifacts / "ab_rule_eval_analysis.json")
    go_no_go_payload = _safe_json(global_artifacts / "adapter_go_no_go.json")
    eval_summary = _extract_eval_summary(eval_payload, analysis_payload, go_no_go_payload)

    summary.adapter_parseable_json_rate = eval_summary["adapter_parseable_json_rate"]
    summary.baseline_parseable_json_rate = eval_summary["baseline_parseable_json_rate"]
    summary.strict_adapter_rate = eval_summary["strict_adapter_rate"]
    summary.strict_baseline_rate = eval_summary["strict_baseline_rate"]
    summary.strict_delta_pp = eval_summary["strict_delta_pp"]
    summary.adapter_relaxed_rate = eval_summary["adapter_relaxed_rate"]
    summary.baseline_relaxed_rate = eval_summary["baseline_relaxed_rate"]
    summary.relaxed_delta_pp = eval_summary["relaxed_delta_pp"]
    summary.go_no_go_decision = eval_summary["go_no_go_decision"]
    summary.n_eval_samples = eval_summary["n_eval_samples"]

    return summary


def render_model_markdown(summary: ModelTuneSummary) -> str:
    lines = [
        f"# Tune Report · {summary.model_name}",
        "",
        f"- Generated at (UTC): {summary.generated_at_utc}",
        f"- Model alias: `{summary.model_alias}`",
        f"- Artifact root: `{summary.artifact_root}`",
        "",
        "## Training",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Return code | {_fmt_num(summary.train_return_code)} |",
        f"| Duration (sec) | {_fmt_num(summary.train_duration_sec, 2)} |",
        f"| Emissions (kg CO2eq) | {_fmt_num(summary.emissions_kg_co2eq, 6)} |",
        "",
        "## Inference",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Adapter inference return code | {_fmt_num(summary.adapter_inference_return_code)} |",
        f"| Baseline inference return code | {_fmt_num(summary.baseline_inference_return_code)} |",
        f"| Adapter parseable_json (%) | {_fmt_num(summary.adapter_parseable_json_rate)} |",
        f"| Baseline parseable_json (%) | {_fmt_num(summary.baseline_parseable_json_rate)} |",
        "",
        "## Evaluation",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Samples | {_fmt_num(summary.n_eval_samples)} |",
        f"| Strict baseline rate (%) | {_fmt_num(summary.strict_baseline_rate)} |",
        f"| Strict adapter rate (%) | {_fmt_num(summary.strict_adapter_rate)} |",
        f"| Strict delta (pp) | {_fmt_num(summary.strict_delta_pp)} |",
        f"| Relaxed baseline rate (%) | {_fmt_num(summary.baseline_relaxed_rate)} |",
        f"| Relaxed adapter rate (%) | {_fmt_num(summary.adapter_relaxed_rate)} |",
        f"| Relaxed delta (pp) | {_fmt_num(summary.relaxed_delta_pp)} |",
        f"| GO/NO-GO | {summary.go_no_go_decision or '-'} |",
        "",
    ]
    return "\n".join(lines)


def write_model_report(tunes_dir: Path, summary: ModelTuneSummary) -> Path:
    models_dir = tunes_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / f"{summary.model_alias}.md"
    out_path.write_text(render_model_markdown(summary), encoding="utf-8")
    return out_path


def write_leaderboard(tunes_dir: Path, summaries: list[ModelTuneSummary]) -> tuple[Path, Path]:
    tunes_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tunes_dir / "leaderboard.csv"
    md_path = tunes_dir / "leaderboard.md"

    rows = sorted(
        summaries,
        key=lambda s: (
            -9999.0 if s.adapter_relaxed_rate is None else -s.adapter_relaxed_rate,
            -9999.0 if s.relaxed_delta_pp is None else -s.relaxed_delta_pp,
            s.model_name,
        ),
    )

    headers = [
        "model_name",
        "model_alias",
        "go_no_go_decision",
        "adapter_relaxed_rate",
        "baseline_relaxed_rate",
        "relaxed_delta_pp",
        "adapter_parseable_json_rate",
        "train_duration_sec",
        "emissions_kg_co2eq",
        "updated_at_utc",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for s in rows:
            writer.writerow(
                {
                    "model_name": s.model_name,
                    "model_alias": s.model_alias,
                    "go_no_go_decision": s.go_no_go_decision or "",
                    "adapter_relaxed_rate": "" if s.adapter_relaxed_rate is None else f"{s.adapter_relaxed_rate:.2f}",
                    "baseline_relaxed_rate": "" if s.baseline_relaxed_rate is None else f"{s.baseline_relaxed_rate:.2f}",
                    "relaxed_delta_pp": "" if s.relaxed_delta_pp is None else f"{s.relaxed_delta_pp:.2f}",
                    "adapter_parseable_json_rate": "" if s.adapter_parseable_json_rate is None else f"{s.adapter_parseable_json_rate:.2f}",
                    "train_duration_sec": "" if s.train_duration_sec is None else f"{s.train_duration_sec:.2f}",
                    "emissions_kg_co2eq": "" if s.emissions_kg_co2eq is None else f"{s.emissions_kg_co2eq:.6f}",
                    "updated_at_utc": s.generated_at_utc,
                }
            )

    lines = [
        "# Tunes Leaderboard",
        "",
        "Cumulative model comparison (overwritten by alias when run again).",
        "",
        "| Rank | Model | Alias | Decision | Relaxed Adapter % | Relaxed Baseline % | Delta pp | Parseable % | Duration s | Emissions kg | Updated (UTC) |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for idx, s in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {model} | `{alias}` | {decision} | {adp} | {base} | {delta} | {parseable} | {dur} | {em} | {ts} |".format(
                rank=idx,
                model=s.model_name,
                alias=s.model_alias,
                decision=s.go_no_go_decision or "-",
                adp=_fmt_num(s.adapter_relaxed_rate),
                base=_fmt_num(s.baseline_relaxed_rate),
                delta=_fmt_num(s.relaxed_delta_pp),
                parseable=_fmt_num(s.adapter_parseable_json_rate),
                dur=_fmt_num(s.train_duration_sec),
                em=_fmt_num(s.emissions_kg_co2eq, 6),
                ts=s.generated_at_utc,
            )
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, csv_path


def discover_model_dirs(artifacts_root: Path, selected_model_name: str = "") -> list[Path]:
    if selected_model_name:
        alias = slugify_model_name(selected_model_name)
        candidate = artifacts_root / alias
        return [candidate] if (candidate / "training_config_stable.json").exists() else []

    discovered: list[Path] = []
    if artifacts_root.exists():
        for child in artifacts_root.iterdir():
            if child.is_dir() and (child / "training_config_stable.json").exists():
                discovered.append(child)
    return sorted(discovered)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build per-model tune reports and a comparative leaderboard.")
    parser.add_argument("--repo-root", default=".", help="Repository root (default: current directory)")
    parser.add_argument("--model-name", default="", help="Optional model name to update only one model alias")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    artifacts_root = repo_root / "lab" / "artifacts"
    tunes_dir = repo_root / "tunes"

    model_dirs = discover_model_dirs(artifacts_root, selected_model_name=args.model_name)
    if not model_dirs:
        print("No model artifact directories with training_config_stable.json were found.")
        return 0

    summaries: list[ModelTuneSummary] = []
    report_paths: list[Path] = []

    for model_dir in model_dirs:
        summary = collect_model_summary(repo_root, model_dir)
        if summary is None:
            continue
        summaries.append(summary)
        report_paths.append(write_model_report(tunes_dir, summary))

    if not summaries:
        print("No summaries generated.")
        return 0

    md_path, csv_path = write_leaderboard(tunes_dir, summaries)

    print("Generated tune reports:")
    for p in report_paths:
        print(f"- {p}")
    print(f"- {md_path}")
    print(f"- {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
