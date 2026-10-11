# Green Customer Insights Platform (PS1)

Setup and run instructions. What each stage does and the results are in `docs/PIPELINE.md`; the design and the reasons
behind it in `docs/EXPLAINER.md`; the public data in `docs/DATA_SOURCES.md`; the synthetic bank in `docs/DATA_CARD.md`
and `docs/data_contract/`.

All customers are synthetic. No real personal or banking data is used.

## Run (one command)

Needs Python 3.11+ (Linux or macOS) and internet for `pip`.

```bash
cd gcip
./run.sh
```

About 2 minutes: creates `.venv`, installs 4 packages, and serves the shipped results for 30,000 customers (`demo/`):

- bank dashboard: http://localhost:8050
- customer app: http://localhost:8050/app/

To watch the whole pipeline run live instead (about 5 minutes):

```bash
./run.sh --quick       # builds a 2,000-customer bank (7 source systems -> FR1 -> FR2 -> features), runs FR3-FR7,
                       # memory and FR6, then serves those results
```

Other options:

```bash
./run.sh --rebuild     # the full 30,000-customer build and pipeline (about 35 minutes), then serves it
./run.sh --serve       # serve what is already built
./run.sh --port 8060
```

Optional: the "Rewrite" button on a suggestion uses Gemini Flash to reword its explanation (checked by the same guards;
only the explanation text and its facts are sent). Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Without a
key, explanations keep their standard wording.

## Run step by step

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                                          # 13 tests
.venv/bin/python scripts/build_reference.py                            # optional: rebuild data/reference from the publishers
bash scripts/build_full_data.sh runs/full 30000 2                      # synthetic bank, FR1, FR2, Customer 360, features, labels
.venv/bin/python scripts/run_pipeline.py --data runs/full --run-dir runs/pipeline   # FR3-FR7, memory, FR6, checks
.venv/bin/python scripts/serve_dashboard.py --run-dir runs/pipeline    # dashboards
.venv/bin/python scripts/make_demo.py                                  # refresh demo/ from a full run
```

The full generated bank (source-system extracts, warehouse, truth vault; 1.6 GB) is published separately:
`.venv/bin/python scripts/fetch_release.py --all` downloads it from the Drive folder in `release/manifest.json`.
Keep `runs/` on disk (not a RAM-backed /tmp). Configuration lives in `configs/`.
