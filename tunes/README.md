# Tunes

This directory stores:

- per-model reports in `tunes/models/<model-alias>.md`
- cumulative comparison table in `tunes/leaderboard.md`
- tabular version in `tunes/leaderboard.csv`

Manual generation:

```bash
poetry run python scripts/update_tunes_reports.py --repo-root .
```

To update only one model:

```bash
poetry run python scripts/update_tunes_reports.py --repo-root . --model-name "mlx-community/Qwen2.5-7B-Instruct-4bit"
```
