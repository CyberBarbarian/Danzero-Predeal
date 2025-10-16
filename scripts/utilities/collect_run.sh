#!/usr/bin/env bash
# Usage: scripts/collect_run.sh /path/to/ray_results/SomeRunName
set -euo pipefail
SRC=${1:-}
if [[ -z "$SRC" ]]; then
  echo "Usage: $0 /path/to/ray_results/SomeRunName"
  exit 1
fi
DST=logs/raw/$(basename "$SRC")_$(date +%Y%m%d_%H%M%S)
mkdir -p "$DST"
rsync -a "$SRC"/ "$DST"/
echo "Copied run to $DST"
