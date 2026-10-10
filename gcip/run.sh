#!/usr/bin/env bash
# One command: set up, get the data and the model, launch the bank dashboard and the customer app.
#
#   ./run.sh              download the prebuilt data and pipeline outputs (Drive) and the local LLM (Hugging Face),
#                         then serve http://localhost:8050 (bank) and http://localhost:8050/app/ (customer)
#   ./run.sh --rebuild    build everything locally instead: synthetic bank (about 25 min) and pipeline (about 8 min)
#   ./run.sh --no-llm     skip the LLM packages and model (explanations stay as templates)
#   ./run.sh --serve      only launch the dashboards on what is already in runs/
#   ./run.sh --port 8060  serve on another port
set -euo pipefail
cd "$(dirname "$0")"

REBUILD=0; LLM=1; SERVE_ONLY=0; PORT=8050
while [ $# -gt 0 ]; do
  case "$1" in
    --rebuild) REBUILD=1 ;;
    --no-llm) LLM=0 ;;
    --serve) SERVE_ONLY=1 ;;
    --port) PORT="$2"; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

PY=.venv/bin/python
if [ "$SERVE_ONLY" -eq 0 ]; then
  say "Python environment (.venv)"
  if [ ! -x "$PY" ]; then
    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Python 3.11 or newer is needed")'
    python3 -m venv .venv
  fi
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements.txt
  [ "$LLM" -eq 1 ] && "$PY" -m pip install -q -r requirements-llm.txt

  if [ ! -f runs/pipeline/_checkpoints/report.done ]; then
    if [ "$REBUILD" -eq 0 ] && "$PY" -I scripts/fetch_release.py; then
      say "Prebuilt data and outputs ready in runs/"
    else
      [ "$REBUILD" -eq 0 ] && say "Download not available: building locally instead"
      say "Synthetic bank: 7 source systems -> FR1 ingestion -> Customer 360, FR2, features, labels"
      bash scripts/build_full_data.sh runs/full 30000 2
      say "Pipeline: FR3 propensity -> FR4 impact -> FR5-FR7 recommendations -> FR6 insights"
      stages="propensity impact recommend explain impact_checks memory insights report"
      [ "$LLM" -eq 1 ] && stages="propensity impact recommend explain impact_checks memory insights explain_llm report"
      [ "$LLM" -eq 1 ] && "$PY" -I scripts/fetch_release.py --model-only
      "$PY" -I scripts/run_pipeline.py --data runs/full --run-dir runs/pipeline --stages $stages
    fi
  fi
  if [ "$LLM" -eq 1 ]; then
    say "Local LLM (Hugging Face cache)"
    "$PY" -I scripts/fetch_release.py --model-only
  fi
fi

[ -e runs/pipeline/data ] || ln -s ../full runs/pipeline/data
say "Bank dashboard:  http://localhost:$PORT      Customer app:  http://localhost:$PORT/app/"
exec "$PY" -I scripts/serve_dashboard.py --run-dir runs/pipeline --port "$PORT"
