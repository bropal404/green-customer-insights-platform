# The pipeline: what runs, in what order, what it produces, and the results

This is the operational record of the platform: every stage with its command, inputs, outputs and run time, the decision
rules, and the results on the 30,000-customer build. The design reasons behind each choice are in `EXPLAINER.md`; the
public data is listed in `DATA_SOURCES.md`; the synthetic bank is described field by field in `DATA_CARD.md` and
`data_contract/`.

All customers are synthetic. Nothing is sent to an external service: the LLM runs on the local GPU.

## 1. One command

`./run.sh` sets up `.venv`, downloads the prebuilt data and pipeline outputs (release bundles, checked against the SHA-256
in `release/manifest.json`), downloads the local LLM, and serves the bank dashboard (`http://localhost:8050`) and the
customer app (`http://localhost:8050/app/`). `./run.sh --rebuild` builds everything locally instead. Measured on a laptop
(12 cores, 14 GB RAM, 6 GB GPU): data build 25 minutes, pipeline 7 minutes.

```
public data (57 files)  --scripts/build_reference.py-->  data/reference/ (in the repo)
data/reference/  --scripts/generate_data.py-->  runs/full/: 7 source systems, truth vault, FR1 warehouse, validation
runs/full/       --scripts/build_model_inputs.py-->  Customer 360 + FR2 categorised + features + labels (2 snapshots)
runs/full/       --scripts/run_pipeline.py-->  runs/pipeline/: FR3, FR4, FR5-FR7, memory, FR6, checks, summary
runs/pipeline/   --scripts/serve_dashboard.py-->  bank dashboard + customer app
```

## 2. Two dates

| Date | Role | Used by |
|---|---|---|
| 31 March 2026 | Training snapshot: Customer 360 and features use only data up to this date; labels are take-ups from 1 April to 30 September 2026 | FR3 training, calibration, test metrics, thresholds, ablations, fairness; FR4 truth-recovery check (before any take-up) |
| 30 September 2026 | Live snapshot: the same 360 and features rebuilt to this date | FR3 scoring, FR4 prices in force, recommendations, the data date in every explanation, memory cool-down clock, FR6, dashboards |

## 3. Data build (`scripts/build_full_data.sh runs/full 30000 2`)

| Step | Module | Output (in `runs/full/`) | Time |
|---|---|---|---|
| World: people, homes (DESNZ NEED), cars (MOT), banking, life events, hidden traits, take-up labels | `simulate/population.py`, `world.py`, `banking.py`, `events.py`, `labels.py` | `truth_vault/state.parquet`, `truth_vault/labels.parquet` | 1 min |
| Transactions (28,266,495) at the prices in force, rendered as each source system writes them | `simulate/transactions.py`, `render.py` | card and payments extracts, `truth_vault/transaction_truth/` | 2 min |
| Source-system extracts (7 systems, 71 files) and control file | `simulate/sources.py` | `sources/` | 1 min |
| FR1 ingestion: control checks, harmonise, crosswalks, de-duplicate, quarantine, audit | `fr1/ingest.py` | `warehouse/`, `quarantine/`, `meta/ingestion_report.json`, `meta/lineage.csv` | 1 min |
| Validation against public statistics (36 checks) and data card | `simulate/validate.py`, `datacard.py` | `meta/data_validation.json`, `meta/DATA_CARD.md` | 1 min |
| Customer 360 for both snapshots (69 fields, 9 sections) | `fr1/customer360.py` | `customer_360_<date>.parquet` | 37 s |
| FR2 categoriser: boundary rules, merchant lookup, TF-IDF model | `categorise/` | `categorised/part-*.parquet` (30 parts of 1,000 customers) | 4 min |
| Features for both snapshots (63 features, 6 sections) | `features/store.py` | `features_<date>.parquet` | 4 min |
| Take-up labels from the bank's product system | `fr1/labels.py` | `labels.parquet` | 1 s |

## 4. Pipeline (`scripts/run_pipeline.py --data runs/full --run-dir runs/pipeline`)

Each stage writes `runs/pipeline/_checkpoints/<stage>.done` and is skipped on a rerun unless `--force <stage>`.

| Stage | What it does | Output (in `runs/pipeline/`) | Time |
|---|---|---|---|
| propensity | Per product on eligible customers: LightGBM (random search), out-of-fold isotonic calibration bounded to 0.1-95%, 5-model bootstrap, confidence, TreeSHAP reason codes; rules baseline, logistic regression and the oracle for comparison; fairness slices; ablations; suggestion thresholds | `models/`, `models/thresholds.json`, `metrics/propensity_report.json`, `predictions/heldout_predictions.csv`, `scores/live_scores.parquet` | 4.7 min |
| impact | Carbon and money statement for every customer x product (DESNZ 2026 factors, prices on 30 Sep 2026, PCAF data quality 1-5) | `impact/impact.parquet` | <1 s |
| recommend | Eligibility, consent, memory, thresholds, routing, top 3, human review; one audited decision per customer x product | `recommendations/recommendations.parquet`, `recommendations/decisions.parquet` | 1 s |
| explain | Template explanation for every suggestion (body + fixed compliance sentence), checked by the guards | `explanations/explanations.parquet`, `metrics/explanations.json` | 1 s |
| impact_checks | Engine inputs against true usage in the truth vault; one-at-a-time sensitivity | `metrics/impact_truth_recovery.json`, `metrics/impact_sensitivity.csv` | <1 s |
| memory | Replay of 300 customers over 3 monthly rounds with responses drawn from the true take-up probability | `memory/replay_events.parquet`, `metrics/memory_replay.json` | 1 s |
| insights | FR6: opportunity by product and region, decision funnel, footprint, 6 segments, monthly trends | `insights/bank_insights.json`, `insights/monthly_trends.parquet` | 19 s |
| explain_llm | Local Qwen2.5-1.5B rewrites the body of 360 explanations (60 per product); guarded; passes replace templates | `explanations/eval_set.parquet` | 2.2 min |
| report | Every metric in one file | `summary.json` | <1 s |

## 5. Decision rules for a suggestion (`configs/pipeline.yaml` `recommend`)

In order; the first rule that applies is recorded as the decision:

1. Eligibility blocks (`configs/products.yaml`, e.g. no car evidence, renter, EPC A/B, solar fitted, arrears, debt over 25% of income).
2. No marketing consent: nothing is sent.
3. Customer memory: already has it; "not interested" 180-day and "dismissed" 90-day cool-downs.
4. The carbon and money impact can be estimated.
5. Calibrated propensity at least 2x the product's take-up rate among eligible training customers (threshold per product in `models/thresholds.json`).
6. Confidence at least 0.45 (not "low").
7. At least 100 kg CO2e avoided a year (Paperless exempt).
8. Green Loans steps aside when EV Finance, Green Home Loan or Solar Finance is shown.
9. Top 3 by propensity x confidence ("interested" earlier adds priority).
10. Credit products for a customer with a vulnerability flag go to human review instead of being sent.

## 6. Results (30,000 customers)

**Data and FR1.** 28,266,495 transactions from 7 source systems (71 files, all passing size and SHA-256 checks). Audit:
8/8 tables and 28,266,495/28,266,495 transactions identical to the bank's system of record. 36/36 checks against public
statistics. Bank-side labels equal the simulator's on all 180,000 customer x product rows (3,975 take-ups).

**FR2.** Test accuracy 99.0% on seen merchants and 90.4% on merchants never seen in training (macro-F1 0.925 and 0.688);
82.1% coarse accuracy on 68,000 out-of-domain US descriptions. Weakest: electricity vs gas from unseen suppliers.

**FR3, test set** (customers never seen in training):

| Product | Test take-ups | Rules AUC | Logistic AUC | LightGBM AUC | Oracle AUC | Lift@10% | ECE |
|---|---|---|---|---|---|---|---|
| GP01 EV Finance | 164 | 0.628 | 0.726 | 0.728 | 0.855 | 3.0x | 0.010 |
| GP02 Green Home Loan | 31 | 0.573 | 0.597 | 0.634 | 0.812 | 1.9x | 0.014 |
| GP03 Solar Finance | 56 | 0.623 | 0.747 | 0.726 | 0.906 | 3.4x | 0.006 |
| GP04 Paperless | 122 | 0.737 | 0.814 | 0.821 | 0.827 | 4.1x | 0.012 |
| GP05 Green Investments | 60 | 0.677 | 0.764 | 0.744 | 0.858 | 4.3x | 0.013 |
| GP06 Green Loans | 106 | 0.625 | 0.734 | 0.709 | 0.792 | 3.7x | 0.005 |

**Thresholds** (test set): take-up rate among customers above the threshold vs all eligible customers.

| Product | Take-up rate | Threshold | Share above | Their take-up rate | Lift | Take-ups captured |
|---|---|---|---|---|---|---|
| EV Finance | 4.8% | 9.5% | 15.5% | 12.9% | 2.6x | 40% |
| Green Home Loan | 4.4% | 8.9% | 2.9% | 6.5% | 2.2x | 7% |
| Solar Finance | 2.0% | 3.9% | 12.6% | 7.5% | 3.0x | 38% |
| Paperless | 12.3% | 24.5% | 17.7% | 44.1% | 3.4x | 61% |
| Green Investments | 6.9% | 13.8% | 10.3% | 28.6% | 4.2x | 43% |
| Green Loans | 3.3% | 6.6% | 7.3% | 12.3% | 4.3x | 31% |

**FR4.** Engine inputs vs true usage (visible spend only): fuel litres within 20% for 83% of drivers (r = 0.95); gas and
electricity kWh within 20% for 39% (direct debits smooth bills). Hand-worked golden case (1,000 litres of petrol: 2,656 kg
baseline, 2,170 kg avoided, GBP 873 a year less) matched exactly by an automated test. 131,656 customer x product
estimates; 48,344 marked "insufficient data" rather than guessed.

**FR5-FR7.** 5,478 suggestions for 4,824 customers; 292 credit offers routed to human review. All 5,478 template
explanations pass the guards; the local LLM passes 67.8% of 360 rewrites (EV Finance 22%), and every rejection falls back
to the template. On every suggestion: none for a blocked product or below its threshold, no solar to renters, nothing
without consent, vulnerable credit always to human review, at most 3 per customer (0 violations).

**Memory.** Replay of 300 customers over 3 rounds: 0 products re-shown inside a cool-down; 122 and then 197 held back.

**FR6.** Expected 925 take-ups, 1,237 t CO2e avoided a year and GBP 418k a year of customer savings; median visible
footprint 4.3 t CO2e a year (transport 43%).

## 7. Tests

`.venv/bin/python -m pytest -q` (13 tests, about a minute): FR1 audit, injected-issue handling, Customer 360 quality,
tamper detection (changed checksum, a one-penny change, an edited narrative), identifier collisions, account-number
normalisation, FR4 golden case, published factors and prices, the insufficient-data path, memory cool-downs and events.

## 8. Release bundles (`release/manifest.json`)

| Bundle | Size | Contents | Fetched by `./run.sh` |
|---|---|---|---|
| `gcip_outputs.tar.gz` | 8 MB | `runs/pipeline/`: models, scores, impact, recommendations, explanations, insights, metrics | yes |
| `gcip_bank_core.tar.gz` | 430 MB | Customer 360 and features (both dates), labels, categorised transactions, product holdings, truth-vault labels and state, reports | yes |
| `gcip_bank_full.tar.gz` | 1.2 GB | the 7 source-system extracts, full warehouse, transaction truth, quarantine | with `scripts/fetch_release.py --all` |

`scripts/make_release.py` rebuilds the bundles and the manifest from a finished run.
