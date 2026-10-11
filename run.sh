#!/usr/bin/env bash
# One command: set up and launch the bank dashboard and the customer app.
#
#   ./run.sh             about 2 minutes: installs 4 packages and serves the shipped results for 30,000 customers (demo/)
#   ./run.sh --quick     about 5 minutes: also runs the whole pipeline live on a 2,000-customer bank and serves that
#   ./run.sh --rebuild   about 35 minutes: full 30,000-customer build and pipeline, then serves it
#   ./run.sh --serve     serve what is already built (runs/ if present, otherwise demo/)
#   ./run.sh --port 8060
# Optional: put GEMINI_API_KEY=... in .env to enable the "Rewrite" button (Gemini Flash, guarded).
set -euo pipefail
cd "$(dirname "$0")"

MODE=demo; PORT=8050
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) MODE=quick ;;
    --rebuild) MODE=rebuild ;;
    --serve) MODE=serve ;;
    --port) PORT="$2"; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
PY=.venv/bin/python

if [ "$MODE" != serve ]; then
  say "Python environment (.venv)"
  if [ ! -x "$PY" ]; then
    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Python 3.11 or newer is needed")'
    python3 -m venv .venv
  fi
  if ! "$PY" -m pip --version >/dev/null 2>&1; then
    say "Repairing pip in .venv"
    "$PY" -m ensurepip --upgrade
  fi
  if [ "$MODE" = demo ]; then "$PY" -m pip install -q -r requirements-app.txt
  else "$PY" -m pip install -q -r requirements.txt; fi
fi

RUN=demo/pipeline; DATA=demo/data
if [ "$MODE" = quick ]; then
  RUN=runs/quick/pipeline; DATA=runs/quick/full; N=2000
  rm -rf runs/quick
elif [ "$MODE" = rebuild ]; then
  RUN=runs/pipeline; DATA=runs/full; N=30000
elif [ "$MODE" = serve ]; then
  if [ -f runs/pipeline/_checkpoints/report.done ]; then RUN=runs/pipeline; DATA=runs/full
  elif [ -f runs/quick/pipeline/_checkpoints/report.done ]; then RUN=runs/quick/pipeline; DATA=runs/quick/full; fi
fi
if [ "$MODE" = quick ] || [ "$MODE" = rebuild ]; then
  say "Synthetic bank ($N customers): 7 source systems -> FR1 ingestion + audit -> Customer 360, FR2, features, labels"
  bash scripts/build_full_data.sh "$DATA" "$N" 2
  say "Pipeline: FR3 propensity -> FR4 impact -> FR5-FR7 suggestions + explanations -> memory -> FR6 insights"
  "$PY" -I scripts/run_pipeline.py --data "$DATA" --run-dir "$RUN" \
      --stages propensity impact recommend explain impact_checks memory insights value evaluate report
fi
say "Starting the dashboards on the results in $RUN"
exec "$PY" -I scripts/serve_dashboard.py --run-dir "$RUN" --data-dir "$DATA" --port "$PORT"
