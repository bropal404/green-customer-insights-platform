# Data contract: what downstream stages read and write

Column-level schemas of the four files the pipeline reads, generated from the build by
`scripts/export_data_contract.py --data runs/full --out docs/data_contract` (add `--samples` for small sample files).
The data itself is in the release bundles (`release/manifest.json`).

- `schemas/*.csv` – one row per column: `section, column, type, allowed_values, unit, null_share_in_sample, example, description, source, used_by`.
- `schema_tables.md` – the same schemas as readable tables.

## Primary input files

Kept deliberately lean: every column is read by a model, an eligibility rule, the confidence score, the impact engine,
customer memory or the bank-level insights. The source systems still hold everything a bank would; these files are the
curated layer on top.

| File | Grain and key | Columns | What it is |
|---|---|---|---|
| `customer_360_<as_of>.parquet` | one row per customer per snapshot; key `(customer_id, as_of)` | 69 | FR1 Customer 360: profile, consent and vulnerability, products and lending values, balances and arrears, green products held, the lender's property and vehicle records, digital habits, campaign memory, transaction count, 6 area-context columns |
| `features_<as_of>.parquet` | same key | 65 (2 keys + 63 features) | Point-in-time features from categorised transactions: spend by category, fuel litres and km, electricity and gas kWh, coverage flags, and evidence flags (homeowner, EV owner, financial difficulty, and so on). Join with the 360 on `(customer_id, as_of)`; models and rules read the joined table |
| `categorised/part-*.parquet` | one row per transaction; key `transaction_id` | 13 | FR2 output: every transaction with exactly one of the eight categories, a sub-category, a confidence and the layer that decided. Used for evidence, drill-downs and monthly trends |
| `labels` | one row per customer per product; key `(customer_id, product)` | 5 | Take-up between 1 April and 30 September 2026 from the bank's own product system (a green product opened in the window), with the customer's split: fit on train, tune on validation, report on test |

Snapshots: `as_of = 2026-03-31` for training (features use data up to that date; labels come from the six months after)
and `as_of = 2026-09-30` for live scoring and the dashboard.

Configuration and reference used by downstream code (not customer data): `configs/products.yaml` (products, block
rules, reason codes), `configs/taxonomy.yaml` (categories and boundary rules), `configs/carbon.yaml` and
`data/reference/ghg_factors_2026.csv` (carbon engine), `data/reference/fuel_prices_weekly.csv` and
`ofgem_price_cap.csv` (money statement).

Deliberately correlated pairs, each with a different job: money and physical units (fuel spend, litres, km), yes/no flags
for rules and counts for models (for example `fuel_visible` and `fuel_months_12m`), and the mortgage value at origination
(PCAF attribution) next to the current loan-to-value. Each model's input list keeps one of each pair.

## Rules for using the columns

- `vulnerability_flag` is never a model feature. It routes credit offers to human review.
- `area_*` columns describe the customer's local authority, not the person. Never show them to a customer as a fact about them.
- `declared_*` fields are what the customer told the bank and can be out of date. Transaction evidence usually beats them.
- `asset_*` fields exist only for mortgages or car finance held with the bank. EPC fields also need a certificate on the register.
- Nothing from the simulator's truth vault (true usage, hidden traits, probabilities) is available downstream.

## Planned outputs (what downstream stages produce)

| Output | Key | Fields |
|---|---|---|
| eligibility | `(customer_id, product, as_of)` | `eligible` bool; `blocks` list of block ids from `products.yaml`; `human_review` bool |
| scores (FR3) | `(customer_id, product, as_of)` | `propensity` float 0-1; `confidence` float 0-1; `reason_codes` list of ids; `evidence` list of `{reason_code, feature, value, text}` |
| impact (FR4) | `(customer_id, product, as_of)` | `co2e_kg_per_year`, `co2e_kg_horizon`, `horizon_years`, `method`, `pcaf_data_quality` 1-5; money: `running_cost_change_gbp_per_year`, `upfront_cost_gbp`, `payback_years` (null for GP05); `status` (`ok` or `insufficient_data`) |
| recommendations (FR5-FR7) | `(customer_id, rank, as_of)` | `product`, `propensity`, `confidence`, `reason_codes`, `evidence`, `impact`, `explanation` text, `data_timestamp`, `human_review`, `suppressed_reason` |
| customer document | `customer_id` | one JSON per customer nesting all of the above with the Customer 360 sections |
