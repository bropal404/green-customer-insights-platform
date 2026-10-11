"""Point-in-time features (as of a date) from categorised transactions + FR1 base view.

Everything here uses only data dated on or before the as-of date and only
PREDICTED categories (never the truth vault), exactly as a production system
would. Money is converted to physical quantities with the price in force when
each payment was made: fuel at that week's DESNZ pump price, energy at that
quarter's Ofgem cap net of standing charges. Those conversions are the
"price-normalised features" the ablation switches off.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gcip.config import CONFIG_DIR
from gcip.simulate.reference import Reference

GREEN_SUPPLIERS = re.compile(r"GOOD ENERGY|ECOTRICITY|OCTOPUS|SO ENERGY|OVO")
CLEAN_AIR = re.compile(r"ULEZ|CONGESTION|TFL CC|CLEAN AIR")
MORTGAGE = re.compile(r"MORTGAGE|\bMTG\b|\bB/S\b|BUILDING S|\bBS\b|MORTGAG")
RENT = re.compile(r"LETTINGS|RENT\b|OPENRENT|PROPERTY MGMT|HOUSING TRUST|HOUSING\b|CLARION|PEABODY|SANCTUARY|PLACES FOR PEOPLE")
BIKE = re.compile(r"CYCLES|BIKE|DECATHLON")
REMOVALS = re.compile(r"REMOVALS")
CHILD_BENEFIT = re.compile(r"CHILD BENEFIT")
PENSION = re.compile(r"PENSION")
SEG = re.compile(r"\bSEG\b|EXPORT")

# The curated features stored per customer and as-of date (with customer_id and as_of), in sections like the 360.
FEATURE_SECTIONS = {
    "history": {
        "question": "How much history we can see",
        "used_by": "confidence",
        "columns": ["months_of_history"],
    },
    "spend_by_category": {
        "question": "Where their money goes, by the FR2 categories a green product changes",
        "used_by": "bank trends, models, evidence",
        "columns": ["spend_12m_fuel", "spend_12m_electricity", "spend_12m_gas", "spend_12m_home_improvement",
                    "spend_12m_automotive", "spend_12m_utilities"],
    },
    "how_they_travel": {
        "question": "How they travel: fuel bought, distance, charging, car costs",
        "used_by": "GP01 and GP06 models and reasons, eligibility rules, carbon engine (litres, km)",
        "columns": ["fuel_spend_12m", "fuel_months_12m", "fuel_litres_12m", "fuel_spend_trend", "est_annual_km", "fuel_visible",
                    "ev_public_kwh_12m", "pt_micromobility_trips_12m", "clean_air_charges_12m", "vehicle_finance_months_12m",
                    "vehicle_finance_monthly", "vehicle_tax_payments_12m", "motor_insurance_months_12m",
                    "car_maintenance_spend_12m", "vehicle_evidence", "ev_owner_evidence", "ev_owner"],
    },
    "home_energy": {
        "question": "How they heat and power their home",
        "used_by": "GP02, GP03 and GP05 models and reasons, carbon engine (kWh, litres)",
        "columns": ["elec_kwh_12m", "gas_kwh_12m", "oil_litres_12m", "heat_kwh_12m", "gas_kwh_per_m2", "energy_months_12m",
                    "energy_bills_visible", "solar_evidence", "green_supplier"],
    },
    "home_situation": {
        "question": "Whether they own, rent or recently moved, and what they spend on their home",
        "used_by": "eligibility rules, GP02, GP03 and GP06 models and reasons",
        "columns": ["mortgage_payment_months_12m", "has_mortgage_evidence", "renter_evidence",
                    "homeowner_evidence", "owner_no_mortgage", "flat_known", "house_known", "epc_low_known", "epc_high_known",
                    "diy_spend_12m", "diy_spend_3m", "trades_spend_12m", "appliance_spend_12m", "bike_shop_spend_12m",
                    "removals_payments_12m", "recent_move_evidence"],
    },
    "income_and_finances": {
        "question": "Their income, investing, borrowing and financial health",
        "used_by": "eligibility rules, affordability, GP04, GP05 and GP06 models and reasons",
        "columns": ["income_monthly", "pension_income_months_12m", "daytime_home", "investment_transfer_months_12m",
                    "investment_transfer_total_12m", "investor_evidence", "loan_repayments_monthly",
                    "external_credit_repayments_monthly", "debt_to_income_est", "existing_borrowing",
                    "in_arrears", "financial_difficulty", "paper_statements",
                    "digitally_inactive"],
    },
}
FEATURE_COLUMNS = [c for section in FEATURE_SECTIONS.values() for c in section["columns"]]
PHYSICAL_FEATURES = ["fuel_litres_12m", "est_annual_km", "ev_public_kwh_12m", "elec_kwh_12m", "gas_kwh_12m",
                     "oil_litres_12m", "heat_kwh_12m", "gas_kwh_per_m2"]
COVERAGE_FEATURES = ["fuel_months_12m", "energy_months_12m", "energy_bills_visible", "fuel_visible", "months_of_history",
                     "mortgage_payment_months_12m"]


def _cap_rates(ref: Reference, dates: np.ndarray) -> dict[str, np.ndarray]:
    return ref.cap_rates(dates.astype("datetime64[D]"))


def part_features(part: pd.DataFrame, as_of: pd.Timestamp, ref: Reference, carbon_cfg: dict) -> pd.DataFrame:
    """Per-customer features from one categorised part (each part holds whole customers)."""
    end = as_of + pd.Timedelta(days=1)
    start = end - pd.DateOffset(months=12)
    recent_start = end - pd.DateOffset(months=3)
    six_start = end - pd.DateOffset(months=6)
    history_months = ((as_of - part.groupby("customer_id")["booking_datetime"].min()).dt.days / 30.44).clip(upper=36)
    p = part[(part["booking_datetime"] < end) & (part["booking_datetime"] >= start)].copy()
    # Missing data is explicit: a row with no date (excluded above) or no amount carries no evidence, and a row with no
    # category counts as Other, so it never adds to fuel litres or energy kWh.
    p = p[p["amount"].notna()]
    p["pred_sub_category"] = p["pred_sub_category"].fillna("")
    p["pred_category"] = p["pred_category"].fillna("Other")
    p["month"] = p["booking_datetime"].dt.to_period("M")
    debit = p["credit_debit_indicator"] == "Debit"
    credit = ~debit
    cust = pd.Index(part["customer_id"].unique(), name="customer_id")
    out = pd.DataFrame(index=cust)
    out["months_of_history"] = history_months.reindex(cust).fillna(0).round(1)

    def agg(mask, value="amount", how="sum", frame=p):
        return frame[mask].groupby("customer_id")[value].agg(how).reindex(cust).fillna(0.0)

    def months(mask, frame=p):
        return frame[mask].groupby("customer_id")["month"].nunique().reindex(cust).fillna(0).astype(int)

    sub = p["pred_sub_category"]
    cat = p["pred_category"]
    text = p["transaction_information"].astype(str)

    # Spend by category (net of refunds within the category).
    for c in ["Fuel", "Electricity", "Gas", "Public Transport", "Home Improvement", "Automotive", "Utilities", "Other"]:
        key = c.lower().replace(" ", "_")
        out[f"spend_12m_{key}"] = (agg(debit & (cat == c)) - agg(credit & (cat == c))).round(2)
        out[f"txn_12m_{key}"] = agg(debit & (cat == c), how="size")
    out["spend_12m_total"] = agg(debit).round(2)

    # Road fuel: litres at the pump price in force that week (petrol unless the bank knows otherwise).
    fuel = debit & (sub == "road_fuel")
    price = ref.fuel_price_ppl(p.loc[fuel, "booking_datetime"].values.astype("datetime64[D]"),
                               np.full(fuel.sum(), "petrol")) / 100.0
    p.loc[fuel, "litres"] = p.loc[fuel, "amount"] / price
    fuel_months = months(fuel)
    litres = agg(fuel, "litres")
    out["fuel_spend_12m"] = agg(fuel).round(2)
    out["fuel_months_12m"] = fuel_months
    out["fuel_litres_12m"] = np.where((fuel_months >= 3) & (fuel_months < 10), litres * 12.0 / fuel_months, litres)
    out["fuel_fill_count_12m"] = agg(fuel, how="size")
    out["fuel_round_fill_share"] = (agg(fuel & (np.abs(p["amount"] % 5) < 0.01), how="size") / out["fuel_fill_count_12m"].replace(0, np.nan)).fillna(0)
    out["fuel_spend_trend"] = (agg(fuel & (p["booking_datetime"] >= recent_start)) * 4.0 / out["fuel_spend_12m"].replace(0, np.nan)).fillna(0).round(3)
    out["est_annual_km"] = (out["fuel_litres_12m"] * 12.85).round(0)

    ev = debit & (sub == "ev_public_charging")
    out["ev_public_spend_12m"] = agg(ev).round(2)
    out["ev_public_kwh_12m"] = (out["ev_public_spend_12m"] / carbon_cfg["assumptions"]["public_charge_price_gbp_per_kwh"]["value"]).round(0)
    out["ev_charging_sessions_12m"] = agg(ev, how="size")

    # Home energy: monthly payments -> kWh at that month's cap, net of standing charges.
    rates = _cap_rates(ref, p["booking_datetime"].values)
    p["e_unit"], p["e_sc"], p["g_unit"], p["g_sc"] = rates["elec_unit"] / 100, rates["elec_sc"] / 100, rates["gas_unit"] / 100, rates["gas_sc"] / 100
    tdcv = carbon_cfg["assumptions"]["dual_fuel_split_kwh"]
    elec_mask = cat == "Electricity"
    gas_mask = cat == "Gas"
    dual_mask = sub == "dual_fuel_energy"
    seg_mask = credit & elec_mask & text.str.contains(SEG)
    signed = np.where(debit, p["amount"], -p["amount"])
    p["signed"] = signed
    month_days = 30.44

    def monthly_kwh(mask, unit_col, sc_col, share=None):
        frame = p[mask & ~seg_mask]
        if frame.empty:
            return pd.Series(0.0, index=cust), pd.Series(0, index=cust)
        values = frame["signed"] if share is None else frame["signed"] * share[frame.index]
        g = pd.DataFrame({"customer_id": frame["customer_id"], "month": frame["month"], "spend": values,
                          "unit": frame[unit_col], "sc": frame[sc_col]}).groupby(["customer_id", "month"]).agg(
            spend=("spend", "sum"), unit=("unit", "mean"), sc=("sc", "mean"))
        g["kwh"] = ((g["spend"] - month_days * g["sc"]).clip(lower=0) / g["unit"])
        total = g.groupby(level=0)["kwh"].sum().reindex(cust).fillna(0.0)
        n_months = g.groupby(level=0).size().reindex(cust).fillna(0).astype(int)
        return total, n_months

    e_cost = tdcv["electricity"] * p["e_unit"] + 365 * p["e_sc"]
    g_cost = tdcv["gas"] * p["g_unit"] + 365 * p["g_sc"]
    elec_share = e_cost / (e_cost + g_cost)
    e_kwh, e_months = monthly_kwh(elec_mask, "e_unit", "e_sc")
    g_kwh, g_months = monthly_kwh(gas_mask, "g_unit", "g_sc")
    de_kwh, d_months = monthly_kwh(dual_mask, "e_unit", "e_sc", elec_share)
    dg_kwh, _ = monthly_kwh(dual_mask, "g_unit", "g_sc", 1 - elec_share)
    energy_months = pd.concat([e_months, g_months, d_months], axis=1).max(axis=1)

    def annual(months):
        """Each supply is scaled to a year from its own months seen (3-11), so a gap in one fuel's bills is not hidden
        by the other fuel's complete months."""
        return np.where((months >= 3) & (months < 12), 12.0 / months.clip(lower=1), 1.0)

    out["elec_kwh_12m"] = (e_kwh * annual(e_months) + de_kwh * annual(d_months)).round(0)
    out["gas_kwh_12m"] = (g_kwh * annual(g_months) + dg_kwh * annual(d_months)).round(0)
    out["energy_months_12m"] = energy_months
    out["energy_bills_visible"] = (energy_months >= 10).astype(int)
    out["dual_fuel_billing"] = (d_months > 0).astype(int)
    out["prepayment_meter"] = (agg(debit & sub.isin(["electricity_prepayment", "gas_prepayment"]), how="size") > 0).astype(int)
    oil = debit & (sub == "heating_oil_lpg")
    out["oil_litres_12m"] = (agg(oil) / carbon_cfg["assumptions"]["heating_oil_gbp_per_litre"]["value"]).round(0)
    out["heat_kwh_12m"] = (out["gas_kwh_12m"] + out["oil_litres_12m"] * 10.35).round(0)
    out["seg_credits_12m"] = agg(seg_mask).round(2)
    out["has_seg_export"] = (out["seg_credits_12m"] > 0).astype(int)
    out["water_spend_12m"] = agg(debit & (sub == "water")).round(2)
    supplier_text = p.loc[debit & (elec_mask | gas_mask | dual_mask), ["customer_id"]].assign(t=text[debit & (elec_mask | gas_mask | dual_mask)])
    out["green_supplier"] = supplier_text.assign(g=supplier_text["t"].str.contains(GREEN_SUPPLIERS)).groupby("customer_id")["g"].max().reindex(cust).fillna(False).astype(int)

    # Transport.
    for s in ["rail", "metro_tram", "bus_coach", "micromobility"]:
        out[f"pt_{s}_trips_12m"] = agg(debit & (sub == s), how="size")
    out["public_transport_spend_12m"] = out["spend_12m_public_transport"]
    out["taxi_spend_12m"] = agg(debit & (sub == "taxi_ridehail")).round(2)

    # Vehicle evidence (Automotive sub-categories) and clean-air charges.
    out["vehicle_finance_months_12m"] = months(debit & (sub == "vehicle_finance"))
    out["vehicle_finance_monthly"] = (agg(debit & (sub == "vehicle_finance")) / out["vehicle_finance_months_12m"].replace(0, np.nan)).fillna(0).round(2)
    out["vehicle_tax_payments_12m"] = agg(debit & (sub == "vehicle_tax"), how="size")
    out["motor_insurance_months_12m"] = months(debit & (sub == "motor_insurance"))
    out["car_maintenance_spend_12m"] = agg(debit & (sub == "maintenance")).round(2)
    out["parking_spend_12m"] = agg(debit & (sub == "parking_charges") & ~text.str.contains(CLEAN_AIR)).round(2)
    out["clean_air_charges_12m"] = agg(debit & text.str.contains(CLEAN_AIR), how="size")
    out["vehicle_evidence"] = ((out["fuel_months_12m"] > 0) | (out["ev_charging_sessions_12m"] > 0) | (out["vehicle_tax_payments_12m"] > 0)
                               | (out["motor_insurance_months_12m"] > 0) | (out["vehicle_finance_months_12m"] > 0)).astype(int)
    out["ev_owner_evidence"] = ((out["ev_public_kwh_12m"] > 150) & (out["fuel_litres_12m"] < 120)).astype(int)
    out["fuel_visible"] = (out["fuel_months_12m"] >= 10).astype(int)

    # Home and housing.
    housing = debit & (sub == "housing")
    out["mortgage_payment_months_12m"] = months(housing & text.str.contains(MORTGAGE))
    out["rent_payment_months_12m"] = months(housing & ~text.str.contains(MORTGAGE) & (text.str.contains(RENT) | (p["payment_type"] == "SO")))
    out["housing_monthly_payment"] = (agg(housing) / months(housing).replace(0, np.nan)).fillna(0).round(2)
    out["council_tax_months_12m"] = months(debit & (sub == "council_tax"))
    out["diy_spend_12m"] = agg(debit & (sub == "diy_materials")).round(2)
    out["diy_spend_3m"] = agg(debit & (sub == "diy_materials") & (p["booking_datetime"] >= recent_start)).round(2)
    out["trades_spend_12m"] = agg(debit & (sub == "trades")).round(2)
    out["appliance_spend_12m"] = agg(debit & (sub == "appliances")).round(2)
    out["energy_installer_spend_12m"] = agg(debit & (sub == "energy_installers")).round(2)
    out["removals_payments_12m"] = agg(debit & text.str.contains(REMOVALS), how="size")
    out["bike_shop_spend_12m"] = agg(debit & text.str.contains(BIKE)).round(2)
    out["second_hand_charity_spend_12m"] = agg(debit & (sub == "charity_second_hand")).round(2)

    # Income, investing and financial health.
    income = credit & (sub == "income")
    out["income_monthly"] = (agg(income) / 12.0).round(2)
    out["pension_income_months_12m"] = months(income & text.str.contains(PENSION))
    out["child_benefit_months_12m"] = months(income & text.str.contains(CHILD_BENEFIT))
    invest = debit & (sub == "investments")
    out["investment_transfers_12m"] = agg(invest, how="size")
    out["investment_transfer_months_12m"] = months(invest)
    out["investment_transfer_total_12m"] = agg(invest).round(2)
    out["loan_repayments_monthly"] = (agg(debit & (sub == "finance") & text.str.contains("LOAN")) / 12.0).round(2)
    out["external_credit_repayments_monthly"] = (agg(debit & (sub == "finance") & (p["payment_type"] == "DD")) / 12.0).round(2)
    six = p["booking_datetime"] >= six_start
    out["overdraft_interest_months_6m"] = months(debit & six & text.str.contains("OVERDRAFT|OD INTEREST"))
    out["unpaid_dd_6m"] = agg(debit & six & text.str.contains("UNPAID DD|RETURNED DIRECT"), how="size")
    out["cash_withdrawals_12m"] = agg(debit & (sub == "cash"), how="size")
    out["debt_to_income_est"] = ((out["loan_repayments_monthly"] + out["external_credit_repayments_monthly"])
                                 / out["income_monthly"].clip(lower=300)).round(3)
    return out.reset_index()


def build_features(categorised_dir: Path, base_view: pd.DataFrame, as_of: str, ref: Reference | None = None) -> pd.DataFrame:
    ref = ref or Reference.load()
    carbon_cfg = yaml.safe_load((CONFIG_DIR / "carbon.yaml").read_text())
    as_of_ts = pd.Timestamp(as_of)
    frames = []
    columns = ["customer_id", "booking_datetime", "amount", "credit_debit_indicator", "transaction_information",
               "payment_type", "pred_category", "pred_sub_category"]
    for path in sorted(categorised_dir.glob("part-*.parquet")):
        frames.append(part_features(pd.read_parquet(path, columns=columns), as_of_ts, ref, carbon_cfg))
    tx = pd.concat(frames, ignore_index=True)
    out = base_view.merge(tx, on="customer_id", how="left")

    # Area enrichment from the reference tables.
    la = ref.la.set_index("la_code")
    for column in ["car_share_bev", "pv_rate_per_household", "pv_kwh_per_kwp_year", "gas_dom_mean_kwh",
                   "elec_dom_median_kwh", "share_flat", "share_cars_0", "share_owned_outright", "share_mortgage"]:
        if f"area_{column}" not in out:  # Customer 360 normally provides the area context already
            out[f"area_{column}"] = out["la_code"].map(la[column])

    # Signals combining several tables (used by eligibility rules, models and reason codes).
    declared = out["declared_residential_status"]
    out["has_mortgage_evidence"] = ((out["mortgage_payment_months_12m"] >= 3) | (out["n_mortgage_accounts"] > 0)).astype(int)
    out["renter_evidence"] = ((out["rent_payment_months_12m"] >= 3) | ((declared == "tenant") & (out["has_mortgage_evidence"] == 0))).astype(int)
    out["homeowner_evidence"] = ((out["has_mortgage_evidence"] == 1)
                                 | (declared.isin(["homeowner", "homeowner_mortgage"]) & (out["rent_payment_months_12m"] < 3))).astype(int)
    out["owner_no_mortgage"] = ((out["homeowner_evidence"] == 1) & (out["has_mortgage_evidence"] == 0)).astype(int)
    out["flat_known"] = (out["asset_property_type"] == "flat").astype(int)
    out["house_known"] = out["asset_property_type"].isin(["detached", "semi", "terraced"]).astype(int)
    out["epc_low_known"] = out["asset_epc_band"].isin(list("DEFG")).astype(int)
    out["epc_high_known"] = out["asset_epc_band"].isin(list("AB")).astype(int)
    out["solar_evidence"] = ((out["has_seg_export"] == 1)
                             | out["asset_solar_pv_on_record"].astype("boolean").fillna(False).astype(bool)).astype(int)
    out["gas_kwh_per_m2"] = (out["gas_kwh_12m"] / out["asset_floor_area_m2"]).replace([np.inf, -np.inf], np.nan)
    out["bev_on_loan_book"] = (out["asset_vehicle_fuel_type"] == "bev").astype(int)
    out["ev_owner"] = ((out["ev_owner_evidence"] == 1) | (out["bev_on_loan_book"] == 1)).astype(int)
    out["paper_statements"] = (out["paper_accounts"] > 0).astype(int)
    out["digitally_inactive"] = (out["months_with_logins_6m"] == 0).astype(int)
    out["in_arrears"] = ((out["max_arrears_months_6m"] > 0) | (out["unpaid_dd_6m"] > 0)).astype(int)
    out["financial_difficulty"] = ((out["months_overdrawn_6m"] >= 4) | (out["overdraft_interest_months_6m"] >= 3)).astype(int)
    out["investor_evidence"] = ((out["n_investment_accounts"] > 0) | (out["investment_transfers_12m"] > 0)).astype(int)
    out["existing_borrowing"] = ((out["n_personal_loan_accounts"] > 0) | (out["balance_credit_card"] < -100)
                                 | (out["external_credit_repayments_monthly"] > 0)).astype(int)
    out["daytime_home"] = ((out["pension_income_months_12m"] >= 3) | (out["declared_employment_status"] == "retired")).astype(int)
    out["recent_move_evidence"] = ((out["removals_payments_12m"] > 0) | (out["declared_years_at_address"] < 1)).astype(int)
    out["as_of"] = as_of_ts
    return out[list(base_view.columns) + [c for c in FEATURE_COLUMNS if c not in base_view.columns]]


def feature_file(frame: pd.DataFrame) -> pd.DataFrame:
    """The stored features: keys plus FEATURE_COLUMNS (join with the Customer 360 on customer_id and as_of)."""
    return frame[["customer_id", "as_of"] + FEATURE_COLUMNS]
