#!/usr/bin/env bash
# Full-size data build that keeps running if the terminal, editor or Claude session closes.
#
#   cd gcip && setsid nohup bash scripts/build_full_data.sh runs/full 30000 2 > runs/full_build.log 2>&1 < /dev/null &
#   tail -f runs/full_build.log          # progress
#   ls runs/full/_done/                  # finished steps (BUILD_COMPLETE when everything is done)
#
# Resumable: generation skips finished chunks, later steps skip anything with a marker in <out>/_done/.
# Memory-safe for a 14 GB laptop: 2 worker processes; data on disk (never /tmp, which is RAM here).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-runs/full}
N=${2:-30000}
WORKERS=${3:-2}
PY=.venv/bin/python
mkdir -p "$OUT/_done"
echo "$(date '+%F %T') build start: $OUT, $N customers, $WORKERS workers"

if [ ! -f "$OUT/_done/data.done" ]; then
  echo "$(date '+%F %T') generate + ingest + validate + data card"
  "$PY" -I scripts/generate_data.py --out "$OUT" --n "$N" --workers "$WORKERS" --chunk-size 1000
  date '+%F %T' > "$OUT/_done/data.done"
fi

echo "$(date '+%F %T') Customer 360, categoriser, features, labels"
"$PY" -I scripts/build_model_inputs.py --data "$OUT"

date '+%F %T' > "$OUT/_done/BUILD_COMPLETE"
echo "$(date '+%F %T') build complete"
du -sh "$OUT"
