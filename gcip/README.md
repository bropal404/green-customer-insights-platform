# Green Customer Insights Platform (PS1)

Setup and run instructions. What each stage does and the results are in `docs/PIPELINE.md`; the design and the reasons
behind it in `docs/EXPLAINER.md`; the public data in `docs/DATA_SOURCES.md`; the synthetic bank in `docs/DATA_CARD.md`
and `docs/data_contract/`.

All customers are synthetic. No real personal or banking data is used, and nothing is sent to an external service: the
optional LLM runs locally.

## Run (one command)

Needs Python 3.11+ on Linux or macOS, about 3 GB of disk, and internet for the downloads.

```bash
git clone <this repo> && cd gcip
./run.sh
```

This creates `.venv`, installs the requirements, downloads the prebuilt data and pipeline outputs (checked against the
SHA-256 in `release/manifest.json`) and the local LLM (Qwen/Qwen2.5-1.5B-Instruct from Hugging Face), then serves:

- bank dashboard: http://localhost:8050
- customer app: http://localhost:8050/app/

Options:

```bash
./run.sh --rebuild     # build everything locally instead of downloading: synthetic bank (~25 min) + pipeline (~8 min)
./run.sh --no-llm      # skip the LLM packages and model; explanations stay as templates
./run.sh --serve       # only launch the dashboards on what is already in runs/
./run.sh --port 8060
.venv/bin/python scripts/fetch_release.py --all        # also the full bank: source extracts, warehouse, truth vault
```

## Run step by step

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-llm.txt
.venv/bin/python -m pytest -q                                          # 13 tests
.venv/bin/python scripts/build_reference.py                            # optional: rebuild data/reference from the publishers
bash scripts/build_full_data.sh runs/full 30000 2                      # synthetic bank, FR1, FR2, Customer 360, features, labels
.venv/bin/python scripts/run_pipeline.py --data runs/full --run-dir runs/pipeline   # FR3-FR7, memory, FR6, checks
.venv/bin/python scripts/serve_dashboard.py --run-dir runs/pipeline    # dashboards
.venv/bin/python scripts/make_release.py                               # pack runs/ into release bundles (dist/)
```

Keep `runs/` on disk (not a RAM-backed /tmp). Configuration lives in `configs/`.
