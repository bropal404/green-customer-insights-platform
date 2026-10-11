# The pipeline: what runs, in what order, what it produces, and the results

This is the operational record of the platform: every stage with its command, inputs, outputs and run time, the decision
rules, and the results on the 30,000-customer build. The design reasons behind each choice are in `EXPLAINER.md`; the
public data is listed in `DATA_SOURCES.md`; the synthetic bank is described field by field in `DATA_CARD.md` and
`data_contract/`.

All customers are synthetic. The only external call is the optional Gemini Flash rewrite of an explanation's wording
(the explanation text and its facts only; no identifiers).

## 1. One command

`./run.sh` (about 2 minutes) installs 4 packages and serves the shipped 30,000-customer results in `demo/` (16 MB,
refreshed by `scripts/make_demo.py`): the bank dashboard at `http://localhost:8050` and the customer app at
`http://localhost:8050/app/`. `./run.sh --quick` (about 3-5 minutes) runs the whole chain live on a 2,000-customer bank
and serves it; `./run.sh --rebuild` does the full 30,000-customer build (data 25 minutes, pipeline 7 minutes on a laptop
with 12 cores and 14 GB RAM).

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
| FR2 categoriser: boundary rules, merchant lookup, TF-IDF model (FinBERT optional: `--finbert`; tested 76% vs 83% on unseen merchants, 8x slower) | `categorise/` | `categorised/part-*.parquet` (30 parts of 1,000 customers) | 4 min |
| Features for both snapshots (63 features, 6 sections) | `features/store.py` | `features_<date>.parquet` | 4 min |
| Take-up labels from the bank's product system | `fr1/labels.py` | `labels.parquet` | 1 s |

## 4. Pipeline (`scripts/run_pipeline.py --data runs/full --run-dir runs/pipeline`)

Each stage writes `runs/pipeline/_checkpoints/<stage>.done` and is skipped on a rerun unless `--force <stage>`.

| Stage | What it does | Output (in `runs/pipeline/`) | Time |
|---|---|---|---|
| propensity | Per product on eligible customers: LightGBM (random search), out-of-fold isotonic calibration bounded to 0.1-95%, 5-model bootstrap, XGBoost and CatBoost benchmarks, confidence, TreeSHAP reason codes; rules baseline, logistic regression and the oracle for comparison; fairness slices; ablations; suggestion thresholds | `models/`, `models/thresholds.json`, `metrics/propensity_report.json`, `predictions/heldout_predictions.csv`, `scores/live_scores.parquet` | 4.7 min |
| impact | Carbon and money statement for every customer x product (DESNZ 2026 factors, prices on 30 Sep 2026, PCAF data quality 1-5) | `impact/impact.parquet` | <1 s |
| recommend | Eligibility, consent, memory, thresholds, routing, top 3, human review; one audited decision per customer x product | `recommendations/recommendations.parquet`, `recommendations/decisions.parquet` | 1 s |
| explain | Template explanation for every suggestion (body + fixed compliance sentence), checked by the guards | `explanations/explanations.parquet`, `metrics/explanations.json` | 1 s |
| impact_checks | Engine inputs against true usage in the truth vault; one-at-a-time sensitivity | `metrics/impact_truth_recovery.json`, `metrics/impact_sensitivity.csv` | <1 s |
| memory | Replay of 300 customers over 3 monthly rounds with responses drawn from the true take-up probability | `memory/replay_events.parquet`, `metrics/memory_replay.json` | 1 s |
| insights | FR6: opportunity by product and region, decision funnel, footprint, 6 segments, estimate-confidence mix; trends by month: spend by category and region, green signals (customers using public EV charging, spend at energy installers, bike hire, rail, bus, heating oil), road fuel spend vs litres at that month's pump price, home energy bills, green take-ups; each customer's last 12 months of spend | `insights/bank_insights.json`, `insights/monthly_trends.parquet` | 19 s |
| value | Business value per suggestion and for the portfolio (`configs/business.yaml`): customer savings, net present value, payback, cost of carbon abatement; bank new lending, net interest income, fees, credit losses, campaign cost, ROI, carbon per £1,000 spent, financed emissions avoided; backtest of who to contact (model vs rules vs random) on the held-out test customers | `recommendations/value.parquet`, `metrics/business_value.json` | 6 s |
| explain_llm | The configured LLM rewrites the body of 360 explanations (60 per product); guarded; passes replace templates. Reported with the local Qwen2.5-1.5B (`--llm-provider local`); a free-tier Gemini key covers only about 20 requests a day per model (sample: 16 of 17 Gemini 3.8 Flash replies passed) | `explanations/eval_set.parquet`, `metrics/explanations_gemini_sample.json` | 2.2 min |
| report | Every metric in one file | `summary.json` | <1 s |

## 5. Decision rules for a suggestion (`configs/pipeline.yaml` `recommend`)

In order; the first rule that applies is recorded as the decision:

1. Eligibility blocks (`configs/products.yaml`, e.g. no car evidence, renter, EPC A/B, solar fitted, arrears, debt over 25% of income).
2. No marketing consent: nothing is sent.
3. Customer memory (every contact and response in the last 12 months, CRM and apps): already has it stops the product;
   "not interested" 180-day and "dismissed" 90-day cool-downs; 3 contacts in a row with no engagement pause it for 180
   days; a memory weight (interest up, ignoring or turning down down) multiplies the priority in step 9.
4. The carbon and money impact can be estimated.
5. Calibrated propensity at least 2x the product's take-up rate among eligible training customers (threshold per product in `models/thresholds.json`).
6. Confidence at least 0.45 (not "low").
7. At least 100 kg CO2e avoided a year (Paperless exempt).
8. Green Loans steps aside when EV Finance, Green Home Loan or Solar Finance is shown.
9. Top 3 by propensity x confidence x memory weight.
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

Benchmarks added by the team (uncalibrated, untuned, test AUC): XGBoost 0.629-0.821 and CatBoost 0.640-0.818 across the
products, within noise of LightGBM (CatBoost slightly ahead on four products; test take-ups are only 31-164 per product).
LightGBM stays the production model because the calibration, confidence, reason codes and thresholds are built on it.

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

**FR5-FR7.** 5,437 suggestions for 4,792 customers; 291 credit offers routed to human review; 657 customer x product
decisions held back by customer memory. All 5,437 template
explanations pass the guards; the local Qwen2.5-1.5B passes 89.7% of 360 rewrites (EV Finance 92%, up from 22%, once
each product got its own worked example and the evidence sentences were written in the second person with a verb), and
every rejection falls back to the template. In the dashboards, "Rewrite" uses Gemini Flash behind the same guards. On every suggestion: none for a blocked product or below its threshold, no solar to renters, nothing
without consent, vulnerable credit always to human review, at most 3 per customer (0 violations).

**Memory.** Compared with the same engine and an empty memory, memory hides 57 suggestions, brings 1 into a customer's
top 3 and reorders 43; 375 suggestions carry the memory reason EARLIER_INTEREST with its date. On 30 September, CRM responses from the June and August 2026 campaigns pause 106-154 customers per product
(dismissed) and 218-346 (not interested), stop 57-153 (already has it) and raise 442-866 (interested). Replay of 300
customers over 3 rounds: 0 products re-shown inside a pause.

**FR6.** Expected 914 take-ups, 1,242 t CO2e avoided a year and GBP 419k a year of customer savings; median visible
footprint 4.3 t CO2e a year (transport 43%).

**Business value** (`configs/business.yaml`, illustrative assumptions; expected = propensity × value if taken up).
Propensity is the chance a customer opens the product in the next six months, and in this data take-ups happen whether
or not we suggest anything, so the bank figures are value *at stake* among the customers suggested to, not profit caused
by the suggestion. What the bank can stand behind: £481k at stake for £7.7k of campaign cost, so the suggestions pay
for themselves if they cause 1.6% of those take-ups (break-even uplift); at 10% caused, £40k net (5.2×). How a take-up
is linked to a suggestion: every suggestion has an id, shown date and channel; the call to action carries the id into
the application (direct); otherwise the same product opened within 90 days is linked (window); 10% of selected
customers are held back at random, and take-ups caused = treated rate − held-back rate. Run on the backtest this
measured +0.7 points (95% interval −5.7 to +7.1): consistent with zero, as it must be, because the simulator gives
contacts no causal effect; in production this measurement replaces the scenarios.


| For customers | | For the bank | |
|---|---|---|---|
| Expected savings | £419k a year | Expected new green lending | £7.28m |
| Expected net present value (3.5%) | £1.38m | Expected net interest income | £610k |
| Suggestions with a positive NPV | 61% | Expected credit losses | £91k |
| Cost of carbon abatement | −£219 per tonne (saves money) | Campaign cost | £7.7k |
| Expected CO2e avoided | 1,242 t a year, 6,306 t over product lives | Value at stake / campaign cost | £481k / £7.7k (break-even uplift 1.6%) |
| Median payback | 10.3 years | Carbon per £1,000 spent | 815 t CO2e |
| | | Financed emissions avoided | 905 t a year (124 t per £1m lent) |

Backtest on the held-out test customers (what really happened, same 1,343 contacts): our model 221 take-ups, 1,682 t
CO2e and £117k bank value; rules baseline 159, 1,174 t, £79k; random 80, 580 t, £46k (2.8× random and 1.4× the rules
baseline on take-ups). Heat pumps (+£77/t) and green loans (+£392/t) cost the customer money per tonne at 2026 prices;
EV finance (−£299/t) and solar (−£230/t) save it.

## 7. Tests

`.venv/bin/python -m pytest -q` (13 tests, about a minute): FR1 audit, injected-issue handling, Customer 360 quality,
tamper detection (changed checksum, a one-penny change, an edited narrative), identifier collisions, account-number
normalisation, FR4 golden case, published factors and prices, the insufficient-data path, memory cool-downs and events.

## 8. Shipped results and release bundles

`demo/` (in the repo, 16 MB) holds what the dashboards read for the 30,000-customer run. The complete generated data is
published as release bundles (`release/manifest.json`), downloaded with `scripts/fetch_release.py` (`--all` for all three):

| Bundle | Size | Contents | Default download |
|---|---|---|---|
| `gcip_outputs.tar.gz` | 8 MB | `runs/pipeline/`: models, scores, impact, recommendations, explanations, insights, metrics | yes |
| `gcip_bank_core.tar.gz` | 430 MB | Customer 360 and features (both dates), labels, categorised transactions, product holdings, truth-vault labels and state, reports | yes |
| `gcip_bank_full.tar.gz` | 1.2 GB | the 7 source-system extracts, full warehouse, transaction truth, quarantine | with `scripts/fetch_release.py --all` |

`scripts/make_release.py` rebuilds the bundles and the manifest from a finished run.
