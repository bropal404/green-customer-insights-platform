# Green Customer Insights Platform (PS1)

Setup and run instructions. What each stage does and the results are in `docs/PIPELINE.md`; the design and the reasons
behind it in `docs/EXPLAINER.md`; the public data in `docs/DATA_SOURCES.md`; the synthetic bank in `docs/DATA_CARD.md`
and `docs/data_contract/`. Where each of the nine hand-in items (A-I) is: `docs/SUBMISSION.md`.

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
                       # memory, FR6, business value and the evaluation report, then serves those results
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
.venv/bin/python -m pytest -q                                          # 16 tests
.venv/bin/python scripts/build_reference.py                            # optional: rebuild data/reference from the publishers
bash scripts/build_full_data.sh runs/full 30000 2                      # synthetic bank, FR1, FR2, Customer 360, features, labels
.venv/bin/python scripts/run_pipeline.py --data runs/full --run-dir runs/pipeline   # FR3-FR7, memory, FR6, checks
.venv/bin/python scripts/serve_dashboard.py --run-dir runs/pipeline    # dashboards
.venv/bin/python scripts/make_demo.py                                  # refresh demo/ from a full run
```

## Inference with the trained models

The trained models ship in `demo/pipeline/models/` (six propensity models and the categoriser); scoring needs
`requirements.txt`, not a rebuild:

```bash
.venv/bin/python scripts/predict.py propensity --data demo/data --as-of 2026-09-30 --out scores.csv   # FR3, 30,000 customers
.venv/bin/python scripts/predict.py categorise "SHELL PETROL 0423 LEEDS" "TFL TRAVEL CH"             # FR2
.venv/bin/python scripts/predict.py categorise "BRITISH GAS" --ptype DD
```

Test set predictions: `demo/pipeline/predictions/test_predictions.csv`. Evaluation report (validation and test):
`docs/EVALUATION.md`.

The full generated bank (source-system extracts, warehouse, truth vault; 1.6 GB) is hosted externally, on Google Drive:
https://drive.google.com/drive/folders/1fJV1LTqDVxyUE5GCjPFQ1mKiVNwNVs1C (`.venv/bin/python scripts/fetch_release.py
--all` downloads it). Nothing above needs it: `./run.sh --quick` and `--rebuild` generate the bank locally.
Keep `runs/` on disk (not a RAM-backed /tmp). Configuration lives in `configs/`.

## Credits and licences

Our code is MIT-licensed (`LICENSE`). Third-party software, models, APIs and data used, with their licences:

| Kind | Name | Licence |
|---|---|---|
| Python libraries | pandas, NumPy, SciPy, scikit-learn, joblib, PyTorch, lxml | BSD-3-Clause |
| | LightGBM, PyYAML, openpyxl, gdown, pytest | MIT |
| | PyArrow, XGBoost, CatBoost, Hugging Face Transformers, huggingface_hub, sentence-transformers | Apache-2.0 |
| | odfpy | Apache-2.0 (also GPL-2.0 / LGPL-2.1) |
| | tqdm | MIT and MPL-2.0 |
| Front end | Chart.js 4.4.1 (cdnjs) | MIT |
| | DM Sans, Manrope, Inter, JetBrains Mono (Google Fonts) | SIL Open Font License 1.1 |
| LLM API | Google Gemini API (gemini-3.8-flash, gemini-3.5-flash, gemini-3.1-flash-lite) | Gemini API Terms of Service |
| Models (Hugging Face) | Qwen/Qwen2.5-1.5B-Instruct (offline explanation evaluation) | Apache-2.0 |
| | sentence-transformers/all-MiniLM-L6-v2 (optional categoriser embeddings) | Apache-2.0 |
| | ProsusAI/finbert (optional categoriser embeddings) | see its model card |
| Data | ONS Census 2021 (Nomis), DESNZ (NEED, greenhouse-gas conversion factors, energy statistics), DfT, DVSA MOT results, HM Land Registry UK HPI | Open Government Licence v3.0 |
| | EU JRC PVGIS 5.3 (solar yield) | free use, EU JRC |
| | Ofgem price cap (public web pages); Bank of England database | publisher terms |
| | DoDataThings US bank transaction categories (Hugging Face) | MIT |
| | greggles/mcc-codes | The Unlicense |

Every dataset, with its publisher, licence and download link: `docs/DATA_SOURCES.md`. No real customer or banking data
is used.
