# Tunes

This directory stores:

- per-model reports in `tunes/models/<model-alias>.md`
- packaged release assets in `tunes/releases/`
- cumulative comparison table in `tunes/leaderboard.md`
- tabular version in `tunes/leaderboard.csv`

The leaderboard is generated from the latest known per-model summaries.
When you refresh a single model, its row is updated while other existing model rows are preserved from their most recent checked-in reports.

Publication model:

- raw experiment artifacts under `lab/artifacts/` stay local and are ignored by git
- checked-in markdown/csv files under `tunes/` are the publication-facing summaries derived from those local artifacts
- the leaderboard aggregates the latest known checked-in summary for each discovered model alias

Some report fields can appear as `-` when the checked-in snapshot does not include the corresponding local runtime log, even if the higher-level evaluation summary is available. That is expected for a publication snapshot built from derived reports rather than raw artifact directories.

Current checked-in reports align with the Qwen2.5-Coder bf16 runs tracked in the repository configs and artifact folders.

Manual generation:

```bash
poetry run python scripts/update_tunes_reports.py --repo-root .
```

To update only one model:

```bash
poetry run python scripts/update_tunes_reports.py --repo-root . --model-name "mlx-community/Qwen2.5-Coder-7B-Instruct-bf16"
```
