# Tunes

This directory stores:

- per-model reports in `tunes/models/<model-alias>.md`
- packaged release assets in `tunes/releases/`
- cumulative comparison table in `tunes/leaderboard.md`
- tabular version in `tunes/leaderboard.csv`

Manual generation:

```bash
poetry run python scripts/update_tunes_reports.py --repo-root .
```

To update only one model:

```bash
poetry run python scripts/update_tunes_reports.py --repo-root . --model-name "mlx-community/Qwen2.5-Coder-7B-Instruct-bf16"
```

Current checked-in reports align with the Qwen2.5-Coder bf16 runs tracked in the repository configs and artifact folders.
