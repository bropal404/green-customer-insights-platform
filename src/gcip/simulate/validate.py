"""Checks the synthetic bank against the public statistics it was calibrated on.

Produces meta/data_validation.json: every check has the simulated value, the
reference value, its source and a pass/fail tolerance. The data card renders it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gcip.config import CONFIG_DIR
from gcip.io.files import write_json
from gcip.simulate.population import ACTIVITY_GROUP
from gcip.simulate.reference import Reference

MILES_TO_KM = 1.609344


SPEND_SUBS = ["road_fuel", "groceries", "electricity_bill", "electricity_prepayment", "gas_bill", "gas_prepayment",
              "dual_fuel_energy"]
BOUNDARY_PICKS = {
    "B1 Fuel vs Automotive": lambda m: m["true_sub_category"].isin(["road_fuel", "maintenance", "motor_insurance", "vehicle_tax"]),
    "B2 Public EV charging is Fuel": lambda m: m["true_sub_category"] == "ev_public_charging",
    "B3 Forecourt shop is Other": lambda m: m["true_sub_category"] == "forecourt_shop",
    "B4 Dual fuel and delivered fuels are Utilities": lambda m: m["true_sub_category"].isin(["dual_fuel_energy", "heating_oil_lpg"]),
    "B5 Prepayment top-ups follow the meter": lambda m: m["true_sub_category"].isin(["electricity_prepayment", "gas_prepayment"]),
    "B6 Clean-air charges are Automotive": lambda m: m["merchant_key"].isin(["tfl_ulez", "tfl_congestion", "gov_caz"]),
    "B7 Taxis are Other, bike share is Public Transport": lambda m: m["true_sub_category"].isin(["taxi_ridehail", "micromobility"]),
    "B8 Appliances are Home Improvement": lambda m: m["true_sub_category"] == "appliances",
    "B9 Motor vs home insurance": lambda m: m["true_sub_category"].isin(["motor_insurance", "insurance_other"]),
    "B10 Supplier credits keep the supplier's category": lambda m: (m["credit_debit_indicator"] == "Credit")
    & m["true_category"].isin(["Electricity", "Gas", "Utilities"]),
    "B11 Housing payments are Other": lambda m: m["true_sub_category"] == "housing",
}


def _stream_transactions(out: Path) -> tuple[pd.DataFrame, dict, dict]:
    """One pass over the transaction parts (each holds whole customers), keeping only small running totals."""
    spend_parts, categories = [], pd.Series(dtype=float)
    counts = {"rows": 0, "card": 0, "card_with_mcc": 0, "local": 0}
    examples = {rule: [] for rule in BOUNDARY_PICKS}
    seen = {rule: set() for rule in BOUNDARY_PICKS}
    for tx_path in sorted((out / "warehouse" / "transactions").glob("part-*.parquet")):
        tx = pd.read_parquet(tx_path, columns=["transaction_id", "amount", "credit_debit_indicator", "merchant_category_code",
                                               "proprietary_bank_transaction_code", "transaction_information"])
        truth = pd.read_parquet(out / "truth_vault" / "transaction_truth" / tx_path.name,
                                columns=["transaction_id", "customer_id", "true_category", "true_sub_category", "merchant_key",
                                         "is_local_merchant"])
        m = tx.merge(truth, on="transaction_id")
        debit = m[(m["credit_debit_indicator"] == "Debit") & m["true_sub_category"].isin(SPEND_SUBS)]
        spend_parts.append(debit.groupby(["customer_id", "true_sub_category"])["amount"].sum())
        card = m["proprietary_bank_transaction_code"] == "DEB"
        counts["rows"] += len(m)
        counts["card"] += int(card.sum())
        counts["card_with_mcc"] += int(m.loc[card, "merchant_category_code"].notna().sum())
        counts["local"] += int(m["is_local_merchant"].sum())
        categories = categories.add(m["true_category"].value_counts(), fill_value=0.0)
        for rule, pick in BOUNDARY_PICKS.items():
            if len(examples[rule]) >= 6:
                continue
            sample = m[pick(m)].drop_duplicates("merchant_key")
            sample = sample[~sample["merchant_key"].isin(seen[rule])].head(6 - len(examples[rule]))
            seen[rule].update(sample["merchant_key"])
            examples[rule] += sample[["transaction_information", "amount", "true_category", "true_sub_category"]].to_dict("records")
        del tx, truth, m, debit
    spend = pd.concat(spend_parts).groupby(level=[0, 1]).sum().unstack(fill_value=0.0)
    counts["categories"] = categories
    return spend, counts, examples


def _check(name: str, simulated: float, reference: float, source: str, tolerance: float) -> dict:
    rel = abs(simulated - reference) / abs(reference) if reference else abs(simulated)
    return {"check": name, "simulated": round(float(simulated), 4), "reference": round(float(reference), 4),
            "relative_gap": round(float(rel), 4), "tolerance": tolerance, "pass": bool(rel <= tolerance), "source": source}


def validate(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    ref = Reference.load()
    state = pd.read_parquet(out / "truth_vault" / "state.parquet")
    labels = pd.read_parquet(out / "truth_vault" / "labels.parquet")
    spend, counts, examples = _stream_transactions(out)
    months = 21  # January 2025 to September 2026
    la = ref.la
    w = la["households"]
    checks: list[dict] = []

    # Census and DESNZ / DfT marginals (household-weighted England and Wales).
    tenure_ref = {
        "owned_outright": (la["share_owned_outright"] * w).sum() / w.sum(),
        "mortgage": (la["share_mortgage"] * w).sum() / w.sum(),
        "social_rent": (la["share_social_rent"] * w).sum() / w.sum(),
        "private_rent": (la["share_private_rent"] * w).sum() / w.sum(),
    }
    for tenure, value in tenure_ref.items():
        checks.append(_check(f"tenure share: {tenure}", (state["tenure"] == tenure).mean(), value, "Census 2021 TS054", 0.10))
    checks.append(_check("households in flats", (state["accommodation"] == "flat").mean(),
                         (la["share_flat"] * w).sum() / w.sum(), "Census 2021 TS044", 0.15))
    checks.append(_check("households with no car", (state["n_cars"] == 0).mean(),
                         (la["share_cars_0"] * w).sum() / w.sum(), "Census 2021 TS045", 0.15))
    checks.append(_check("mains gas central heating", (state["main_heating"] == "gas").mean(),
                         (la["share_heat_mains_gas"] * w).sum() / w.sum(), "Census 2021 TS046", 0.10))
    cars = state[state["has_car"]]
    checks.append(_check("battery-electric share of primary cars", (cars["car_fuel"] == "bev").mean(),
                         (la["car_share_bev"] * w).sum() / w.sum(), "DfT VEH0142/VEH0105 2026 Q2", 0.35))
    checks.append(_check("homes with solar PV", state["has_solar"].mean(),
                         (la["pv_rate_per_household"] * w).sum() / w.sum(), "DESNZ renewables by LA 2025", 0.30))
    checks.append(_check("mean annual mileage, primary car (miles)", cars["annual_km_true"].mean() / MILES_TO_KM,
                         ref.nts_miles("All cars"), "NTS0901 2025", 0.10))
    gas_homes = state[state["gas_kwh_year_true"] > 0]
    checks.append(_check("mean domestic gas use, gas homes (kWh)", gas_homes["gas_kwh_year_true"].mean(),
                         (la["gas_dom_mean_kwh"] * w).sum() / w.sum(), "DESNZ sub-national gas 2024", 0.15))
    checks.append(_check("median domestic electricity use (kWh)", state["elec_kwh_year_true"].median(),
                         (la["elec_dom_median_kwh"] * w).sum() / w.sum(), "DESNZ sub-national electricity 2024", 0.20))

    # Households, work and homes against Census 2021 cross-tables, NEED and Bank of England rates.
    def weighted(column: str) -> float:
        return float((la[column] * w).sum() / w.sum())

    ta = ref.tenure_age.groupby(["tenure", "hrp_age"])["households"].sum().unstack()
    for tenure in ("owned_outright", "social_rent"):
        sim = (state.loc[state["tenure"] == tenure, "age"] >= 65).mean()
        checks.append(_check(f"household reference persons aged 65+ ({tenure.replace('_', ' ')})", sim,
                             ta.loc[tenure, "65+"] / ta.loc[tenure].sum(), "Census 2021 RM201", 0.15))
    act = ref.tenure_activity.assign(group=ref.tenure_activity["hrp_activity"].map(ACTIVITY_GROUP))
    act = act.groupby("group")["households"].sum()
    act = act / act.sum()
    working = state["employment"].isin(["employed", "self_employed"])
    checks.append(_check("household reference persons in work", working.mean(), act["work"], "Census 2021 RM133", 0.10))
    checks.append(_check("household reference persons retired", (state["employment"] == "retired").mean(), act["retired"],
                         "Census 2021 RM133", 0.15))
    checks.append(_check("self-employed share of workers", (state["employment"] == "self_employed").sum() / max(working.sum(), 1),
                         weighted("self_employed_share_of_workers"), "Census 2021 TS066", 0.20))
    checks.append(_check("households with dependent children", (state["dependants"] > 0).mean(),
                         weighted("share_hh_dependent_children"), "Census 2021 TS003", 0.10))
    checks.append(_check("homes with 4 or more bedrooms", (state["bedrooms"] >= 4).mean(), weighted("share_bedrooms_4plus"),
                         "Census 2021 TS050", 0.15))
    need = ref.need
    checks.append(_check("homes with an EPC on the register", state["epc_registered"].mean(), (need["epc"] != "No EPC").mean(),
                         "NEED 2025", 0.05))
    rated = state[state["epc_registered"]]
    checks.append(_check("rated homes in EPC band C or better", rated["epc_band"].isin(list("ABC")).mean(),
                         need.loc[need["epc"] != "No EPC", "epc"].isin(["A/B", "C"]).mean(), "NEED 2025", 0.10))
    two_year = state[state["mortgage"] & (state["mortgage_fix_years"] == 2)]
    window = ref.rates[(ref.rates["month"] >= "2023-01-01") & (ref.rates["month"] < "2025-01-01")]
    checks.append(_check("mean rate on 2-year fixed mortgages (%)", two_year["mortgage_rate"].mean() * 100,
                         window["mortgage_2y_fixed_75ltv"].mean(), "Bank of England quoted rates 2023-24", 0.10))

    # Cars against DVSA MOT 2024 (cars aged 3 or more, as MOT sees them).
    mot = ref.mot
    tested = cars[cars["car_age_years"] >= 3]
    petrol_mot = mot[mot["fuel"] == "petrol"]
    median_mot = petrol_mot.loc[(petrol_mot["cars"].cumsum() / petrol_mot["cars"].sum() >= 0.5).idxmax(), "age_year"] + 0.5
    checks.append(_check("median age of petrol cars aged 3+ (years)", tested.loc[tested["car_fuel"] == "petrol", "car_age_years"].median(),
                         median_mot, "DVSA MOT 2024", 0.15))
    def ratio(frame_ages, frame_values, weights=None):
        old = frame_values[(frame_ages >= 10) & (frame_ages < 15)]
        new = frame_values[(frame_ages >= 3) & (frame_ages < 6)]
        if weights is None:
            return old.mean() / new.mean()
        w_old, w_new = weights[(frame_ages >= 10) & (frame_ages < 15)], weights[(frame_ages >= 3) & (frame_ages < 6)]
        return np.average(old, weights=w_old) / np.average(new, weights=w_new)
    pet = tested[tested["car_fuel"] == "petrol"]
    checks.append(_check("mileage of petrol cars aged 10-14 vs 3-5 (ratio)", ratio(pet["car_age_years"].to_numpy(), pet["annual_km_true"].to_numpy()),
                         ratio(petrol_mot["age_year"].to_numpy(), petrol_mot["mean_annual_miles"].to_numpy(), petrol_mot["cars"].to_numpy()),
                         "DVSA MOT 2024", 0.10))

    # Spend by income decile against ONS Family Spending A6 (all households, monthly).
    per_sub = (spend / months).reindex(state["customer_id"]).fillna(0.0)
    decile = state.set_index("customer_id")["income_decile"]
    spend_tables = {}
    for label, subs, codes, tol in (("fuel", ["road_fuel"], ["7.2.2"], 0.30),
                                    ("food and drink", ["groceries"], ["1", "2"], 0.20),
                                    ("home energy", ["electricity_bill", "electricity_prepayment", "gas_bill", "gas_prepayment", "dual_fuel_energy"], ["4.4.1", "4.4.2"], 0.25)):
        cols = [c for c in subs if c in per_sub.columns]
        values = per_sub[cols].sum(axis=1)
        if label == "home energy":  # ONS counts every household's bills; compare where ours are visible
            visible = ~state.set_index("customer_id")["energy_paid_elsewhere"]
            values = values[visible.reindex(values.index).values]
        sim = values.groupby(decile.reindex(values.index)).mean()
        ons = sum(ref.spend_weekly(c) for c in codes) * 52 / 12
        if label == "food and drink":
            ons = ons * 0.80  # share captured on this current account (configs/simulation.yaml)
        spend_tables[label] = {"decile": list(range(1, 11)), "simulated": [round(float(sim.get(d, np.nan)), 1) for d in range(1, 11)],
                               "ons_a6": [round(float(x), 1) for x in ons]}
        checks.append(_check(f"mean monthly {label} spend (GBP)", values.mean(),
                             float(np.mean(ons)), "ONS Family Spending FYE 2025 A6", tol))
        corr = np.corrcoef([sim.get(d, np.nan) for d in range(1, 11)], ons)[0, 1]
        checks.append(_check(f"{label} spend gradient across income deciles (correlation)", corr, 1.0, "ONS Family Spending FYE 2025 A6", 0.25))

    # Labels: rates against targets, and how products overlap per customer.
    cfg = yaml.safe_load((CONFIG_DIR / "simulation.yaml").read_text())
    rates = labels[labels["eligible_true"]].groupby("product")["took_up"].mean()
    for product, target in cfg["labels"]["target_rate"].items():
        checks.append(_check(f"{product} take-up among eligible", rates.get(product, 0.0), target, "label model target", 0.35))
    wide = labels.pivot(index="customer_id", columns="product", values="took_up").astype(int)
    overlap = (wide.T @ wide).astype(int)
    eligible_wide = labels.pivot(index="customer_id", columns="product", values="eligible_true").astype(int)
    eligible_overlap = (eligible_wide.T @ eligible_wide).astype(int)
    multi = (eligible_wide.sum(axis=1) >= 2).mean()

    # Observability and transaction mix.
    observability = {
        "transactions_per_customer_month": round(counts["rows"] / len(state) / months, 1),
        "category_share": (counts["categories"] / counts["rows"]).sort_values(ascending=False).round(4).to_dict(),
        "card_share": round(counts["card"] / counts["rows"], 4),
        "mcc_present_on_card": round(counts["card_with_mcc"] / max(counts["card"], 1), 4),
        "local_merchant_share": round(counts["local"] / counts["rows"], 4),
        "customers_with_energy_bills_visible": round(float(1 - state["energy_paid_elsewhere"].mean()), 4),
        "drivers_buying_some_fuel_in_cash": round(float((cars["cash_fuel_share"] > 0).mean()), 4),
        "drivers_fuel_on_external_card": round(float(cars["fuel_on_external_card"].mean()), 4),
        "mortgages_visible": round(float(1 - state.loc[state["mortgage"], "mortgage_paid_elsewhere"].mean()), 4),
        "customers_eligible_for_two_or_more_products": round(float(multi), 4),
    }

    # Boundary rules: real examples from the generated statements (collected while streaming).

    report = {
        "checks": checks,
        "passed": int(sum(c["pass"] for c in checks)),
        "total": len(checks),
        "spend_by_decile": spend_tables,
        "takeup_overlap_counts": overlap.to_dict(),
        "eligibility_overlap_counts": eligible_overlap.to_dict(),
        "observability": observability,
        "boundary_examples": examples,
    }
    write_json(out / "meta" / "data_validation.json", report)
    return report
