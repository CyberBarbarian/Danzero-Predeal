#!/usr/bin/env bash
# Pre-create local folders and a README so cloud logs can be dropped in and analyzed.
set -euo pipefail
mkdir -p logs/raw logs/processed runs
cat > logs/README.md << 'MD'
# Logs Folder

- Put completed RLlib runs under `logs/raw/` (each run folder contains `progress.csv`).
- Example: `logs/raw/DMC_2025-09-30_12-00-00/`.

Then run:

```bash
scripts/analyze_rllib_logs.py --rllib-root logs/raw --outdir logs/processed
```

Artifacts generated:
- `logs/processed/summary.json`: per-run stats (final loss, env steps)
- `logs/processed/all_progress.csv`: concatenated CSV
- `logs/processed/loss_curves.png`: training loss plots
- `logs/processed/env_steps.png`: env steps plots
MD

echo "Local folders ready. Place RLlib runs under logs/raw and run analyze script."
