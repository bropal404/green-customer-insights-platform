"""Writes meta/DATA_CARD.md: sources, schema, generation process, label rules,
boundary rules, validation results, assumptions and limitations (brief slides 06 and 08).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from gcip.config import CONFIG_DIR, REFERENCE_DIR

TABLES = {
    "customers": ("One row per customer: what the bank recorded at onboarding.", {
        "customer_id": "Synthetic customer key.",
        "birth_year": "Year of birth.",
        "la_code / la_name / region": "Local authority (April 2023 boundaries) and region of the home address.",
        "declared_residential_status": "Residential status given at onboarding; about 10% is out of date.",
        "declared_employment_status": "Employment status given at onboarding.",
        "declared_income_band": "Banded gross annual income; about 12% is one band out.",
        "customer_since": "Date the relationship began.",
        "marketing_consent": "Whether the customer accepts marketing.",
        "vulnerability_flag": "Bank-recorded vulnerability (drives human review and blocks).",
        "preferred_contact_channel": "app, email or post.",
        "stated_sustainability_preference": "Investment fact-find answer (1 yes, 0 no); blank for most customers.",
        "declared_occupation": "Occupation group given at onboarding (SOC 2020 major group, or retired, student, not in work); about 8% out of date.",
        "declared_dependants": "Number of dependent children given in credit applications.",
        "declared_years_at_address": "Whole years at the current address.",
    }),
    "accounts": ("One row per account the customer holds with the bank.", {
        "account_id": "Customer id plus a suffix: CA current, SV savings, CC card, MG mortgage, ML motor loan, PL personal loan, IV investment; green products EV, GH, SF, SI, GL (N = opened in the label window).",
        "account_type / product_code": "Product family and product code (green codes: MOTOR-EV-GREEN, MTG-GREEN-FA, LOAN-SOLAR, INV-SUSTAIN-FOCUS, PL-GREEN).",
        "opened_date / closed_date / status": "Account lifecycle.",
        "statement_preference": "paper or online at extraction (history in preference_changes).",
        "credit_limit, original_amount, interest_rate, term_months": "Credit terms where relevant. Rates follow the Bank of England quoted rate for the product when it started (mortgages by fixed period and LTV, personal loans by size, cards, savings).",
        "collateral_value_at_origination": "Property or vehicle value when the loan began (PCAF attribution denominator).",
        "loan_purpose": "Stated purpose of personal loans.",
    }),
    "balances": ("Month-end balance per account, January 2025 to September 2026.", {
        "account_id, month_end": "Key.",
        "balance": "GBP; liabilities are negative.",
        "arrears_months": "Months in arrears on credit products.",
    }),
    "transactions": ("Every booked transaction on current and credit card accounts, in the UK Open Banking OBTransaction shape.", {
        "transaction_id, account_id, booking_datetime": "Keys and booking time.",
        "amount, currency, credit_debit_indicator": "Positive amount in GBP, Credit or Debit.",
        "transaction_information": "Statement narrative as the customer sees it (truncated and noisy).",
        "merchant_name": "Card payments only.",
        "merchant_category_code": "ISO 18245 MCC, card payments only (about 97% present).",
        "proprietary_bank_transaction_code": "DEB card, DD direct debit, SO standing order, FPO/FPI faster payment out/in, BGC credit, CPT cash, TFR internal transfer, CHG charge.",
        "bank_transaction_code": "ISO 20022-style family of the payment type.",
    }),
    "product_holdings": ("Products held and when, including green products and the paperless service.", {
        "customer_id, account_id, product_code, product_family": "What is held.",
        "green_product": "GP01-GP06 when the holding is one of the six green products.",
        "start_date, end_date": "Holding period.",
    }),
    "preference_changes": ("Statement preference switches (paper to online).", {
        "account_id, change_date, old_preference, new_preference": "History needed for point-in-time features.",
    }),
    "assets": ("Property and vehicle records the bank holds as lender.", {
        "property_type, property_detail, construction_period, bedrooms": "From the valuation report for mortgages with us.",
        "floor_area_m2, epc_band, epc_main_heating, solar_pv_on_record": "From the EPC register match (blank when the home has no certificate or was not matched).",
        "vehicle_fuel_type, vehicle_first_registered_year, vehicle_registration": "For motor loans with us.",
        "valuation_at_origination": "Property or vehicle value at loan origination.",
    }),
    "digital_activity": ("Monthly digital engagement per customer.", {
        "app_logins, web_logins, branch_visits, estatement_views": "Counts per month.",
        "paperless_all_accounts": "Whether every account was on online statements that month.",
    }),
    "campaign_responses": ("Earlier marketing contacts and responses (seed history for customer memory).", {
        "campaign_id, sent_date, product, channel": "What was sent.",
        "response, response_date": "no_response, viewed, clicked, dismissed, not_interested or already_have.",
    }),
}

TRUTH = {
    "state": "Every customer's true situation: hidden traits (z_*), their observable parts (zmu_*), true km, litres, kWh, heating, EPC, solar, billing visibility, life events, prior holdings, take-up dates, split.",
    "labels": "Per customer and product: true eligibility, p_true (Bayes-optimal take-up probability), took_up, takeup_date.",
    "transaction_truth": "Per transaction: true category and sub-category, merchant, local or national merchant, physical quantity (litres or kWh) and fuel type; plus the true account and a hash of the booked record (audit_account_id, audit_fingerprint) for the ingestion audit.",
    "merchant_directory": "Every payee the simulator can produce, with its true category.",
    "system_of_record": "Fingerprints (row count, order-independent hash, one hash per column) of each table as the bank's systems held it before extraction; FR1 ingestion must reproduce them exactly.",
}

DRIVER_DOCS = {
    "log_km": "log(annual petrol or diesel km / 11,000)",
    "log_km_sq": "square of log_km (very high mileage lowers take-up: range concerns)",
    "car_age": "car age in years / 10",
    "loan_ending_12m": "current car finance ends within 12 months of the cutoff",
    "offstreet": "home has off-street parking for charging",
    "income": "(income decile - 5.5) / 2.87",
    "multi_car": "household has two or more cars",
    "clean_air_charges": "pays ULEZ or clean-air-zone charges",
    "area_ev_share": "(local battery-electric share of private cars - 0.032) / 0.012",
    "epc_low": "EPC band D to G",
    "heat_intensity": "log(heat demand per m2 / 140)",
    "recent_move": "moved home in the 12 months before the cutoff",
    "diy": "log of DIY spending intensity",
    "mortgage_with_us": "mortgage is held with this bank",
    "household_size": "(household size - 2.4) / 1.3",
    "log_elec": "log(electricity demand / 3,000 kWh)",
    "solar_yield": "(regional PV yield kWh/kWp - 980) / 50",
    "ev_at_home": "household drives a battery-electric or plug-in hybrid car",
    "daytime_home": "retired or works from home",
    "detached": "lives in a detached house",
    "log_logins": "log(1 + online/app log-ins per month)",
    "app_user": "uses the mobile app",
    "age": "(age - 48) / 17",
    "paper_accounts": "number of accounts on paper statements - 2",
    "invests_with_us": "holds an investment account with this bank",
    "log_investment": "log(1 + investment balance / 1,000) / 2",
    "regular_investor": "invests every month",
    "stated_pref": "stated sustainability preference (1, 0, or 0 when unknown)",
    "green_supplier": "energy supplier is a green-leaning supplier",
    "existing_borrowing": "has a personal loan or credit card borrowing",
    "purchase_intent": "has a green purchase in mind (shows as bike and appliance shop spend)",
    "cyclist": "cycles or uses bike share",
    "owner_outright_high_gas": "owns outright and uses more gas than the local mean",
}

LIMITATIONS = [
    "England and Wales only: Census 2021 covers these nations; Scottish and Northern Irish customers are not simulated.",
    "Energy prices use Ofgem GB-average Direct Debit cap rates; regional rate differences are not modelled.",
    "Each customer is one household's main account holder; joint accounts appear only as bills paid from elsewhere.",
    "Hidden traits and label coefficients are modelling assumptions, documented here; metrics measure performance on this world, not on real customers.",
    "Insurers that sell both motor and home cover produce identical statement text for both; the true category comes from the policy type.",
    "Dual-fuel and single-fuel bills from the same supplier can read identically (rule B4); only fuel-tagged references disclose the fuel.",
    "Heating oil and LPG prices, public charging prices and water bills are assumptions, not sourced series.",
]


def _md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def _sources_section(out: Path) -> list[str]:
    """Source systems, what ingestion did with them, the audit result and column lineage."""
    dictionary = yaml.safe_load((CONFIG_DIR / "source_systems.yaml").read_text())
    lines = ["## Source systems", "",
             "The bank's data arrives as extracts from seven systems, each with its own format, keys, units, dates, codes "
             "and quality problems (configs/source_systems.yaml). FR1 ingestion (src/gcip/fr1/ingest.py) checks each file "
             "against the control totals, harmonises it, resolves keys through crosswalks, removes duplicates and "
             "quarantines records that match no customer.", ""]
    systems = pd.DataFrame([{
        "system": name,
        "files": ", ".join(spec["files"].values()),
        "format": spec["format"],
        "keys": spec["keys"],
        "dates and units": f"{spec['dates']}; {spec['units']}",
        "known issues": " ".join(spec["known_issues"]) or "-",
    } for name, spec in dictionary["systems"].items()])
    lines += [_md_table(systems), ""]
    report_path = out / "meta" / "ingestion_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        lines += ["### Ingestion result", ""] + [f"- {line}" for line in report["summary"]] + [""]
        rows = [{"file": name, "rows in": f"{info['rows_in']:,}", "rows out": f"{info['rows_out']:,}",
                 "handled": "; ".join(f"{k}: {v:,}" for k, v in info["issues"].items()) or "-"}
                for name, info in report["files"].items()]
        lines += [_md_table(pd.DataFrame(rows)), ""]
        if "audit" in report:
            audit = pd.DataFrame([{"table": name, "rows expected": f"{a['rows_expected']:,}", "rows ingested": f"{a['rows_ingested']:,}",
                                   "identical": "yes" if a["exact_match"] else "NO",
                                   "columns differing": ", ".join(a["columns_differing"]) or "-"}
                                  for name, a in report["audit"]["tables"].items()])
            lines += ["### Ingestion audit", "",
                      "Each harmonised table is fingerprinted (every cell in a canonical form, order-independent) and compared "
                      "with the fingerprint of the bank's system of record kept in the truth vault; every transaction's record "
                      "hash is compared one by one.", "", _md_table(audit), ""]
    lineage_path = out / "meta" / "lineage.csv"
    if lineage_path.exists():
        lines += ["### Column lineage", "", "Where every warehouse column comes from and how it was transformed.", "",
                  _md_table(pd.read_csv(lineage_path).fillna("")), ""]
    return lines


def write_data_card(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    meta = out / "meta"
    sim = yaml.safe_load((CONFIG_DIR / "simulation.yaml").read_text())
    taxonomy = yaml.safe_load((CONFIG_DIR / "taxonomy.yaml").read_text())
    validation = json.loads((meta / "data_validation.json").read_text())
    report = pd.read_json(meta / "label_model_report.json")
    manifest = json.loads((REFERENCE_DIR / "manifest.json").read_text())
    state = pd.read_parquet(out / "truth_vault" / "state.parquet", columns=["customer_id", "split"])
    rows = {}
    for name in TABLES:
        if name == "transactions":
            rows[name] = sum(pd.read_parquet(p, columns=["transaction_id"]).shape[0]
                             for p in sorted((out / "warehouse" / "transactions").glob("*.parquet")))
        else:
            rows[name] = pd.read_parquet(out / "warehouse" / f"{name}.parquet").shape[0]

    lines = [
        "# Data card: synthetic bank for the Green Customer Insights Platform",
        "",
        f"{len(state):,} synthetic customers, January 2025 to September 2026, seed {sim['seed']}. "
        "No real personal or banking data is used (brief slide 06): customers, accounts and transactions are generated; "
        "public UK statistics only calibrate them.",
        "",
    ]
    lines += _sources_section(out)
    lines += [
        "## Warehouse tables",
        "",
        "The harmonised tables FR1 ingestion writes. They stay separate, as a bank keeps them; "
        "FR1's Customer 360 then joins them into one point-in-time row per customer.",
        "",
    ]
    for name, (desc, cols) in TABLES.items():
        lines += [f"### {name} ({rows[name]:,} rows)", "", desc, ""]
        lines += [_md_table(pd.DataFrame({"column": list(cols.keys()), "meaning": list(cols.values())})), ""]
    lines += ["## Truth vault", "",
              "Training labels come from here for training customers only: transaction categories (FR2) and product take-up (FR3), "
              "as a bank would use a labelled history. Everything else (hidden traits, true km and kWh, p_true, validation and test "
              "labels) is read only by the evaluation harness; models, rules, the carbon engine and the LLM never see it.", ""]
    lines += [_md_table(pd.DataFrame({"table": list(TRUTH.keys()), "contents": list(TRUTH.values())})), ""]

    lines += ["## How the data is generated", "",
              "1. **Population.** Households are sampled per local authority from Census 2021 tenure, accommodation, car, heating, household-size and commute tables. The age of the household reference person matches Census RM201 (tenure by age) and economic activity matches RM133 (tenure by activity) in each authority; dependants follow TS003, occupation TS063, self-employment TS066, bedrooms TS050. Income deciles follow ONS Family Spending.",
              "2. **Physical world.** Each home is a real home from the DESNZ NEED 2025 sample (4 million homes) of the same region, type and heating: its construction period, floor-area band, EPC band, insulation and metered 2023 gas and electricity, re-anchored to the authority's DESNZ averages. Homes NEED records without an EPC have no certificate the bank can match. Cars get a fuel type from DfT local licensing shares, mileage from NTS bands and an efficiency around the DESNZ 2026 average car; solar follows DESNZ sites per household.",
              "3. **Money.** Fuel is priced at that week's DESNZ pump price; energy bills use the Ofgem cap in force each quarter; everyday spend follows ONS spend by income decile. Homes are valued at the UK House Price Index for their authority, type and purchase month; mortgage, loan, card and savings rates are the Bank of England quoted rates for the product and month.",
              "4. **Statements.** Each payment becomes a UK statement line with real merchant names, store numbers, references and truncation; about 11% of spend goes to small local businesses that are unique to each area.",
              "5. **Messiness.** Some customers pay energy from another bank, buy fuel with cash or on another bank's card, or have stale onboarding data.",
              "6. **Life events.** House moves, new cars, babies and retirements are drawn monthly and change later months.",
              "7. **Labels.** Take-up of each green product is drawn from the causal model below, only for truly eligible customers.",
              ""]

    lines += ["## Label-generation rules", "",
              "For product *p* and customer *c*: logit P(take-up between 1 April and 30 September 2026) = b0 + Σ beta × driver + Σ gamma × hidden trait.",
              "Drivers are the customer's true situation on 31 March 2026; hidden traits are standard-normal variables the platform never sees, "
              "so no rule on observed data can reproduce the labels. b0 is solved so take-up among eligible customers hits the target rate. "
              "Products are independent and may overlap: a customer can take up several.", ""]
    for _, r in report.iterrows():
        coefs = pd.DataFrame({"driver": list(r["coefficients"].keys()), "beta": [round(v, 3) for v in r["coefficients"].values()],
                              "definition": [DRIVER_DOCS.get(k, "") for k in r["coefficients"].keys()]})
        traits = ", ".join(f"{k} {v:+.2f}" for k, v in r["trait_coefficients"].items())
        lines += [f"### {r['product']} (b0 = {r['intercept']}, eligible {r['eligible']:,}, take-ups {r['takeups']:,}, rate {r['rate_among_eligible']:.3f}, target {r['target_rate']})",
                  "", _md_table(coefs), "", f"Hidden traits: {traits}.", ""]

    lines += ["## Transaction categories and boundary rules (FR2)", "",
              "Every transaction has exactly one of eight categories; sub-categories roll up to one category.", ""]
    for rule in taxonomy["boundary_rules"]:
        lines.append(f"- **{rule['id']}** {rule['rule']}")
    lines += ["", "Examples from this dataset:", ""]
    for rule, examples in validation["boundary_examples"].items():
        if examples:
            lines += [f"**{rule}**", "", _md_table(pd.DataFrame(examples)), ""]

    split_counts = state["split"].value_counts().reindex(["train", "validation", "test"]).fillna(0).astype(int)
    lines += ["## Splits", "",
              f"Customer-level split from a salted hash of the customer id: train {split_counts['train']:,}, validation "
              f"{split_counts['validation']:,}, test {split_counts['test']:,}. All of a customer's rows sit in one split. "
              "Features use data up to 31 March 2026; labels use the six months after.", ""]

    checks = pd.DataFrame(validation["checks"])[["check", "simulated", "reference", "relative_gap", "pass", "source"]]
    lines += ["## Validation against public statistics", "",
              f"{validation['passed']} of {validation['total']} checks within tolerance.", "", _md_table(checks), ""]
    for label, table in validation["spend_by_decile"].items():
        frame = pd.DataFrame(table).set_index("decile").T
        frame.insert(0, "series", frame.index)
        lines += [f"Monthly {label} spend by income decile (GBP):", "", _md_table(frame.reset_index(drop=True)), ""]
    obs = validation["observability"]
    lines += ["## Partial observability", "",
              _md_table(pd.DataFrame({"measure": [k for k in obs if k != "category_share"],
                                      "value": [obs[k] for k in obs if k != "category_share"]})), ""]
    overlap = pd.DataFrame(validation["eligibility_overlap_counts"])
    overlap.insert(0, "product", overlap.index)
    lines += ["## Product overlap", "", "Customers truly eligible for each pair of products (diagonal = eligible for that product):", "",
              _md_table(overlap.reset_index(drop=True)), ""]

    lines += ["## Assumptions", "", "Every parameter lives in configs/simulation.yaml with its source or the word 'assumption'. The main assumptions:", ""]
    for item in [
        f"Age and income given tenure: {sim['population']['age_given_tenure']['mortgage']} (mortgagors' age bands, as an example).",
        f"Energy billing: dual-fuel single direct debit {sim['homes']['billing']['dual_fuel_single_dd_share']}, prepayment {sim['homes']['billing']['prepayment_share']}, bills paid from another bank {sim['homes']['billing']['paid_from_other_bank_share']}.",
        f"Heat pump efficiency {sim['homes']['heat_pump_spf']} and boiler efficiency {sim['homes']['boiler_efficiency']} (Electrification of Heat final report).",
        f"Public charging {sim['vehicles']['public_charge_price_gbp_per_kwh']} GBP/kWh; heating oil {sim['homes']['oil_price_gbp_per_litre']} GBP/litre.",
        f"Take-up targets {sim['labels']['target_rate']}.",
        "Seasonal shapes of gas, electricity and driving (transactions.py constants).",
    ]:
        lines.append(f"- {item}")
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in LIMITATIONS]
    lines += ["", "## Sources", ""]
    for src in manifest["raw_sources"]:
        lines.append(f"- {src['description']} ({src['publisher']}, {src['licence']}): {src['url']}")
    path = meta / "DATA_CARD.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
