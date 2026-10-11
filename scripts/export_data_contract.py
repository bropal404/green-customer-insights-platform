#!/usr/bin/env python3
"""Export the data contract that downstream work builds against: one schema per primary file
(CSV + Markdown) and real sample files.

    python scripts/export_data_contract.py --data runs/contract --out docs/data_contract

--data is a built data directory (scripts/generate_data.py) that also holds customer_360_<as_of>.parquet and,
when the FR2 and feature stages have run, categorised/ and features_<as_of>.parquet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PRODUCTS = {"GP01": "EV Finance", "GP02": "Green Home Loan", "GP03": "Solar Finance", "GP04": "Paperless Banking",
            "GP05": "Green Investments", "GP06": "Green Loans"}
CATEGORIES = ["Fuel", "Electricity", "Gas", "Public Transport", "Home Improvement", "Automotive", "Utilities", "Other"]
ENUMS = {
    "region": ["North East", "North West", "Yorkshire and The Humber", "East Midlands", "West Midlands", "East", "London",
               "South East", "South West", "Wales"],
    "declared_residential_status": ["homeowner", "homeowner_mortgage", "tenant", "living_with_family"],
    "declared_employment_status": ["employed", "self_employed", "retired", "student", "not_working"],
    "declared_income_band": ["under_15k", "15k_25k", "25k_40k", "40k_60k", "60k_100k", "100k_plus"],
    "preferred_contact_channel": ["app", "email", "post"],
    "declared_occupation": ["managers_directors", "professional", "associate_professional", "administrative", "skilled_trades",
                            "caring_leisure", "sales_customer_service", "process_plant", "elementary", "retired", "student",
                            "not_in_work"],
    "motor_loan_product_code": ["MOTOR-HP", "MOTOR-EV-GREEN"],
    "personal_loan_purpose": ["car", "home_improvement", "debt_consolidation", "wedding_or_holiday", "other"],
    "asset_property_type": ["detached", "semi", "terraced", "flat", "caravan"],
    "asset_property_detail": ["Detached", "Bungalow", "Semi detached", "Mid terrace", "End terrace", "Flat", "Caravan or mobile home"],
    "asset_construction_period": ["pre-1930", "1930-1972", "1973-1999", "2000-or-later"],
    "asset_epc_band": list("ABCDEFG"),
    "asset_epc_main_heating": ["gas", "electric", "oil", "lpg", "heat_pump", "other"],
    "asset_vehicle_fuel_type": ["petrol", "diesel", "hev", "phev", "bev"],
    "credit_debit_indicator": ["Credit", "Debit"],
    "payment_type": ["DEB", "DD", "SO", "FPO", "FPI", "BGC", "CPT", "TFR", "CHG"],
    "pred_category": CATEGORIES,
    "product": list(PRODUCTS),
    "split": ["train", "validation", "test"],
}
for _gp in PRODUCTS:
    ENUMS[f"last_response_{_gp}"] = ["none", "no_response", "viewed", "clicked", "dismissed", "not_interested", "already_have"]

DESCRIPTIONS = {
    "customer_id": ("Customer key (primary key together with as_of).", "", "core banking"),
    "age": ("Age in whole years on the as-of date.", "years", "core banking (birth year)"),
    "la_code": ("Local authority code of the home address (April 2023 boundaries).", "", "core banking"),
    "la_name": ("Local authority name.", "", "core banking"),
    "region": ("Region of the home address.", "", "core banking"),
    "declared_residential_status": ("Residential status given at onboarding; about 10% out of date.", "", "core banking"),
    "declared_employment_status": ("Employment status given at onboarding.", "", "core banking"),
    "declared_income_band": ("Gross annual income band given at onboarding; about 12% one band out.", "", "core banking"),
    "customer_tenure_years": ("Years since the relationship began.", "years", "core banking"),
    "marketing_consent": ("Customer accepts marketing.", "", "core banking"),
    "vulnerability_flag": ("Bank-recorded vulnerability. Never a model feature; routes credit offers to human review.", "",
                           "core banking"),
    "preferred_contact_channel": ("Preferred contact channel.", "", "core banking"),
    "stated_sustainability_preference": ("Investment fact-find answer: 1 yes, 0 no, null when never asked (most customers).",
                                         "", "core banking"),
    "declared_occupation": ("Occupation group from onboarding (SOC 2020 major group) or status; about 8% out of date.", "",
                            "core banking"),
    "declared_dependants": ("Dependent children declared in credit applications.", "count", "core banking"),
    "declared_years_at_address": ("Whole years at the current address.", "years", "core banking"),
    "n_products": ("Accounts open on the as-of date.", "count", "core banking"),
    "paper_accounts": ("Accounts still on paper statements on the as-of date.", "count", "core banking + product system"),
    "mortgage_original_amount": ("Original mortgage advance (mortgages held with the bank).", "GBP", "core banking"),
    "mortgage_property_value_at_origination": ("Property valuation when the mortgage began.", "GBP", "core banking"),
    "mortgage_indexed_property_value": ("Valuation indexed to the as-of month with the UK House Price Index (lender-derived).",
                                        "GBP", "derived: core banking + UK HPI"),
    "mortgage_current_ltv": ("Mortgage balance divided by the indexed property value.", "ratio", "derived"),
    "motor_loan_months_to_end": ("Months until the latest motor loan held with the bank ends (negative once ended).", "months",
                                 "core banking"),
    "motor_loan_vehicle_value_at_origination": ("Vehicle value when that motor loan began.", "GBP", "core banking"),
    "motor_loan_product_code": ("Motor loan product.", "", "core banking"),
    "personal_loan_purpose": ("Stated purpose of the personal loan.", "", "core banking"),
    "months_overdrawn_6m": ("Month-ends with the current account overdrawn, last 6 months.", "count", "core banking"),
    "max_arrears_months_6m": ("Worst arrears on any credit account, last 6 months.", "months", "core banking"),
    "asset_property_type": ("Property type from the valuation report (mortgages held with the bank).", "", "lending assets"),
    "asset_property_detail": ("Detailed property type from the valuation report.", "", "lending assets"),
    "asset_construction_period": ("Construction period from the valuation report.", "", "lending assets"),
    "asset_bedrooms": ("Bedrooms from the valuation report.", "count", "lending assets"),
    "asset_floor_area_m2": ("Floor area from the EPC register match; null without a certificate or match.", "m2",
                            "lending assets (EPC register)"),
    "asset_epc_band": ("EPC band from the register match.", "", "lending assets (EPC register)"),
    "asset_epc_main_heating": ("Main heating from the register match.", "", "lending assets (EPC register)"),
    "asset_solar_pv_on_record": ("Solar PV on the EPC record.", "", "lending assets (EPC register)"),
    "asset_vehicle_fuel_type": ("Fuel of the vehicle on motor finance with the bank.", "", "lending assets"),
    "asset_vehicle_age_years": ("Age of that vehicle.", "years", "lending assets"),
    "logins_per_month_6m": ("App and web logins per month, last 6 months.", "per month", "digital analytics"),
    "app_login_share_6m": ("Share of logins made in the app.", "ratio", "digital analytics"),
    "branch_visits_6m": ("Branch visits, last 6 months.", "count", "digital analytics"),
    "estatement_views_6m": ("E-statement views, last 6 months.", "count", "digital analytics"),
    "months_with_logins_6m": ("Months with at least one login, last 6 months.", "count", "digital analytics"),
    "n_transactions_12m": ("Transactions in the 12 months to the as-of date.", "count", "card processor + payments hub"),
    "months_active_12m": ("Months with any transaction, last 12 months.", "count", "card processor + payments hub"),
    "debit_total_12m": ("Money out, last 12 months.", "GBP", "card processor + payments hub"),
    "credit_total_12m": ("Money in, last 12 months.", "GBP", "card processor + payments hub"),
    "area_avg_house_price": ("Average house price in the authority, latest month on or before the as-of date.", "GBP",
                             "UK House Price Index"),
    "area_house_price_change_12m": ("12-month change in that average price.", "ratio", "UK House Price Index"),
    "area_share_commute_public_transport": ("Share of the authority's workers commuting by train, metro/tram or bus.", "ratio",
                                            "Census 2021 TS061"),
    "as_of": ("Snapshot date (primary key with customer_id): 2026-03-31 for training, 2026-09-30 for live scoring.", "",
              "pipeline"),
    "transaction_id": ("Transaction key.", "", "card processor / payments hub"),
    "account_id": ("Account key '<customer_id>-<account type>'.", "", "harmonised in FR1"),
    "booking_datetime": ("Booking date and time.", "", "card processor / payments hub"),
    "amount": ("Amount, always positive.", "GBP", "card processor / payments hub"),
    "credit_debit_indicator": ("Money in (Credit) or out (Debit).", "", "card processor / payments hub"),
    "transaction_information": ("Statement narrative as the customer sees it.", "", "card processor / payments hub"),
    "norm_text": ("Normalised narrative used by the categoriser.", "", "FR2"),
    "merchant_category_code": ("ISO 18245 merchant category code; card payments only.", "", "card processor"),
    "payment_type": ("DEB card, DD direct debit, SO standing order, FPO/FPI faster payment out/in, BGC credit, CPT cash, "
                     "TFR transfer, CHG charge.", "", "harmonised in FR1"),
    "pred_category": ("Predicted category: exactly one of the eight.", "", "FR2"),
    "pred_sub_category": ("Predicted sub-category (rolls up to pred_category).", "", "FR2"),
    "cat_confidence": ("Confidence of the predicted category.", "probability", "FR2"),
    "layer": ("Which layer decided the category (rule, lookup or model).", "", "FR2"),
    "product": ("Green product code.", "", "labels"),
    "took_up": ("Opened the green product between 1 April and 30 September 2026.", "",
                "product system (green holding started in the window)"),
    "takeup_date": ("Date the green product was opened.", "", "product system"),
    "split": ("Customer-level split from a salted hash of customer_id (70/15/15): fit on train, tune on validation, "
              "report on test.", "", "pipeline"),
}
FEATURES = {  # feature store columns: (description, unit); all from categorised transactions up to the as-of date
    "months_of_history": ("Months between the first transaction and the as-of date (capped at 36).", "months"),
    "spend_12m_total": ("All money out, last 12 months.", "GBP"),
    "fuel_spend_12m": ("Road-fuel spend, last 12 months.", "GBP"),
    "fuel_months_12m": ("Months with a road-fuel purchase, last 12 months.", "months"),
    "fuel_litres_12m": ("Litres bought: each purchase divided by that week's DESNZ pump price; scaled to a year when only "
                        "3-9 months are visible.", "litres"),
    "fuel_fill_count_12m": ("Road-fuel purchases, last 12 months.", "count"),
    "fuel_round_fill_share": ("Share of fuel purchases in round GBP 5 amounts.", "ratio"),
    "fuel_spend_trend": ("Last 3 months' fuel spend x 4 / 12-month fuel spend (1 = steady).", "ratio"),
    "est_annual_km": ("Estimated km per year: fuel litres x 12.85 km/l (DESNZ 2026 average petrol car).", "km"),
    "ev_public_spend_12m": ("Spend at public EV chargers, last 12 months.", "GBP"),
    "ev_public_kwh_12m": ("kWh bought at public chargers (spend / GBP 0.70 per kWh, assumption).", "kWh"),
    "ev_charging_sessions_12m": ("Public charging sessions, last 12 months.", "count"),
    "elec_kwh_12m": ("Electricity kWh from bills at the Ofgem cap in force, net of standing charges (dual-fuel bills split "
                     "by Ofgem typical consumption); scaled to a year when 3-11 months are visible.", "kWh"),
    "gas_kwh_12m": ("Gas kWh, same method as electricity.", "kWh"),
    "energy_months_12m": ("Months with energy bills visible, last 12 months.", "months"),
    "energy_bills_visible": ("1 if energy bills are seen in 10+ of the last 12 months.", "flag"),
    "dual_fuel_billing": ("1 if a dual-fuel bill was seen.", "flag"),
    "prepayment_meter": ("1 if prepayment top-ups were seen.", "flag"),
    "oil_litres_12m": ("Heating oil or LPG litres (spend / GBP 0.66 per litre, assumption).", "litres"),
    "heat_kwh_12m": ("Heating energy: gas kWh + oil litres x 10.35 kWh per litre.", "kWh"),
    "seg_credits_12m": ("Smart Export Guarantee (solar export) payments received, last 12 months.", "GBP"),
    "has_seg_export": ("1 if any solar export payment was received.", "flag"),
    "water_spend_12m": ("Water bill spend, last 12 months.", "GBP"),
    "green_supplier": ("1 if the energy supplier is a green-leaning supplier (by name).", "flag"),
    "public_transport_spend_12m": ("Public transport spend, last 12 months.", "GBP"),
    "taxi_spend_12m": ("Taxi and ride-hailing spend, last 12 months.", "GBP"),
    "vehicle_finance_months_12m": ("Months with a car-finance payment (any lender), last 12 months.", "months"),
    "vehicle_finance_monthly": ("Average monthly car-finance payment.", "GBP"),
    "vehicle_tax_payments_12m": ("Vehicle tax payments, last 12 months.", "count"),
    "motor_insurance_months_12m": ("Months with a motor-insurance payment, last 12 months.", "months"),
    "car_maintenance_spend_12m": ("Servicing and repair spend, last 12 months.", "GBP"),
    "parking_spend_12m": ("Parking spend excluding clean-air charges, last 12 months.", "GBP"),
    "clean_air_charges_12m": ("ULEZ, congestion and clean-air-zone charges, last 12 months.", "count"),
    "vehicle_evidence": ("1 if any fuel, charging, vehicle tax, motor insurance or car finance is seen.", "flag"),
    "ev_owner_evidence": ("1 if public charging exceeds 150 kWh and fuel is under 120 litres a year.", "flag"),
    "fuel_visible": ("1 if fuel is bought in 10+ of the last 12 months.", "flag"),
    "mortgage_payment_months_12m": ("Months with a mortgage payment to any lender, last 12 months.", "months"),
    "rent_payment_months_12m": ("Months with a rent payment, last 12 months.", "months"),
    "housing_monthly_payment": ("Average monthly housing payment (mortgage or rent).", "GBP"),
    "council_tax_months_12m": ("Months with a council tax payment, last 12 months.", "months"),
    "diy_spend_12m": ("DIY and building-materials spend, last 12 months.", "GBP"),
    "diy_spend_3m": ("DIY and building-materials spend, last 3 months.", "GBP"),
    "trades_spend_12m": ("Payments to tradespeople, last 12 months.", "GBP"),
    "appliance_spend_12m": ("Appliance spend, last 12 months.", "GBP"),
    "energy_installer_spend_12m": ("Payments to energy installers (solar, heat pumps, insulation), last 12 months.", "GBP"),
    "removals_payments_12m": ("Payments to removals firms (a home move), last 12 months.", "count"),
    "bike_shop_spend_12m": ("Bike shop spend, last 12 months.", "GBP"),
    "second_hand_charity_spend_12m": ("Second-hand and charity shop spend, last 12 months.", "GBP"),
    "income_monthly": ("Average monthly income credits.", "GBP"),
    "pension_income_months_12m": ("Months with pension income, last 12 months.", "months"),
    "child_benefit_months_12m": ("Months with Child Benefit, last 12 months.", "months"),
    "investment_transfers_12m": ("Transfers to investment accounts or platforms, last 12 months.", "count"),
    "investment_transfer_months_12m": ("Months with an investment transfer, last 12 months.", "months"),
    "investment_transfer_total_12m": ("Total transferred to investments, last 12 months.", "GBP"),
    "loan_repayments_monthly": ("Average monthly loan repayments.", "GBP"),
    "external_credit_repayments_monthly": ("Average monthly direct-debit repayments to other lenders.", "GBP"),
    "overdraft_interest_months_6m": ("Months with overdraft interest, last 6 months.", "months"),
    "unpaid_dd_6m": ("Unpaid or returned direct debits, last 6 months.", "count"),
    "cash_withdrawals_12m": ("Cash withdrawals, last 12 months.", "count"),
    "debt_to_income_est": ("(Loan + other-lender repayments) / monthly income (income floored at GBP 300).", "ratio"),
    "has_mortgage_evidence": ("1 if mortgage payments in 3+ months or a mortgage held with the bank.", "flag"),
    "renter_evidence": ("1 if rent in 3+ months, or declared tenant with no mortgage evidence.", "flag"),
    "homeowner_evidence": ("1 if mortgage evidence, or declared owner who pays no rent.", "flag"),
    "owner_no_mortgage": ("1 if homeowner evidence without mortgage evidence.", "flag"),
    "flat_known": ("1 if the lender's property record says flat.", "flag"),
    "house_known": ("1 if the lender's property record says detached, semi or terraced.", "flag"),
    "epc_low_known": ("1 if the EPC on record is D to G.", "flag"),
    "epc_high_known": ("1 if the EPC on record is A or B.", "flag"),
    "solar_evidence": ("1 if solar export payments are received or solar is on the EPC record.", "flag"),
    "gas_kwh_per_m2": ("Gas kWh per square metre of floor area on record.", "kWh per m2"),
    "bev_on_loan_book": ("1 if the vehicle on motor finance with the bank is battery-electric.", "flag"),
    "ev_owner": ("1 if EV-owner evidence or a battery-electric car on the loan book.", "flag"),
    "paper_statements": ("1 if any account is on paper statements.", "flag"),
    "digitally_inactive": ("1 if no app or web login in the last 6 months.", "flag"),
    "in_arrears": ("1 if any arrears or unpaid direct debit in the last 6 months.", "flag"),
    "financial_difficulty": ("1 if overdrawn at 4+ month-ends or overdraft interest in 3+ months (last 6 months).", "flag"),
    "investor_evidence": ("1 if an investment account with the bank or transfers to investments.", "flag"),
    "existing_borrowing": ("1 if a personal loan, a card balance over GBP 100 or repayments to other lenders.", "flag"),
    "daytime_home": ("1 if pension income in 3+ months or declared retired.", "flag"),
    "recent_move_evidence": ("1 if a removals payment in the last 12 months.", "flag"),
}
BALANCE = {"current": "current account", "savings": "savings", "credit_card": "credit card (negative = owed)",
           "mortgage": "mortgage (negative = owed)", "motor_loan": "motor loan (negative = owed)",
           "personal_loan": "personal loan (negative = owed)", "investment": "investments"}
FEATURE_SOURCE = "feature store (from categorised transactions, point in time)"


def describe(column: str) -> tuple[str, str, str]:
    if column in DESCRIPTIONS:
        return DESCRIPTIONS[column]
    if column in FEATURES:
        return FEATURES[column] + (FEATURE_SOURCE,)
    if column.startswith("n_") and column.endswith("_accounts"):
        return (f"Open {column[2:-9].replace('_', ' ')} accounts on the as-of date.", "count", "core banking")
    if column.startswith("balance_"):
        return (f"Latest month-end balance on or before the as-of date: {BALANCE.get(column[8:], column[8:])}.", "GBP",
                "core banking")
    if column.startswith("holds_"):
        gp = column[6:]
        return (f"Already holds {gp} {PRODUCTS.get(gp, '')} on the as-of date.", "", "product system")
    if column.startswith("last_response_date_"):
        return (f"Date of the last campaign response for {column[19:]}.", "", "CRM")
    if column.startswith("last_response_"):
        return (f"Last campaign response for {column[14:]} before the as-of date ('none' if never contacted).", "", "CRM")
    if column.startswith("area_"):
        name = column[5:]
        unit = "kWh" if name.endswith("_kwh") else ("kWh per kWp per year" if "kwp" in name else "ratio")
        return (f"Area statistic for the customer's authority (describes the area, not the person): {name.replace('_', ' ')}.",
                unit, "public data (Census 2021, DESNZ, DfT, PVGIS)")
    if column.startswith("spend_12m_"):
        return (f"Net spend in category {column[10:].replace('_', ' ')}, last 12 months (debits minus refunds).", "GBP",
                FEATURE_SOURCE)
    if column.startswith("txn_12m_"):
        return (f"Debit transactions in category {column[8:].replace('_', ' ')}, last 12 months.", "count", FEATURE_SOURCE)
    if column.startswith("pt_") and column.endswith("_trips_12m"):
        return (f"Public transport trips ({column[3:-10].replace('_', ' ')}), last 12 months.", "count", FEATURE_SOURCE)
    unit = ("GBP" if any(k in column for k in ("spend", "monthly", "total", "credits", "payment")) else
            "litres" if "litres" in column else "kWh" if "kwh" in column else "km" if column.endswith("_km") else
            "ratio" if column.endswith(("_share", "_trend", "_est")) else "count or flag")
    return (column.replace("_", " ").capitalize() + ".", unit, FEATURE_SOURCE)


def logical_type(series: pd.Series, column: str) -> str:
    nullable = " (nullable)" if series.isna().any() else ""
    if column in ENUMS:
        return "category" + nullable
    if pd.api.types.is_bool_dtype(series):
        return "bool" + nullable
    if pd.api.types.is_datetime64_any_dtype(series):
        values = series.dropna()
        is_date = len(values) and (values.dt.normalize() == values).all()
        return ("date" if is_date else "datetime") + nullable
    if pd.api.types.is_integer_dtype(series):
        return "int" + nullable
    if pd.api.types.is_float_dtype(series):
        return "float" + nullable
    values = series.dropna()
    if len(values) and values.map(lambda v: isinstance(v, (bool, np.bool_))).all():
        return "bool" + nullable
    return "string" + nullable


def section_of(column: str) -> tuple[str, str]:
    from gcip.features.store import FEATURE_SECTIONS
    from gcip.fr1.customer360 import C360_SECTIONS
    for sections in (C360_SECTIONS, FEATURE_SECTIONS):
        for name, section in sections.items():
            if column in section["columns"]:
                return name, section["used_by"]
    return "", ""


def schema(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        series = frame[column]
        description, unit, source = describe(column)
        section, used_by = section_of(column)
        example = series.dropna().iloc[0] if series.notna().any() else ""
        if isinstance(example, pd.Timestamp):
            example = example.strftime("%Y-%m-%d %H:%M:%S") if (example.hour or example.minute) else example.strftime("%Y-%m-%d")
        elif isinstance(example, (float, np.floating)):
            example = round(float(example), 4)
        rows.append({"section": section, "column": column, "type": logical_type(series, column),
                     "allowed_values": "|".join(ENUMS[column]) if column in ENUMS else "",
                     "unit": unit, "null_share_in_sample": round(float(series.isna().mean()), 3),
                     "example": str(example), "description": description, "source": source, "used_by": used_by})
    return pd.DataFrame(rows)


def markdown(table: pd.DataFrame) -> str:
    cols = ["section", "column", "type", "unit", "description"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=str(ROOT / "docs" / "data_contract"))
    parser.add_argument("--as-of", default="2026-03-31")
    parser.add_argument("--samples", action="store_true", help="also write small sample files to <out>/samples/")
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.out)
    (out / "schemas").mkdir(parents=True, exist_ok=True)
    samples = out / "samples"
    if args.samples:
        samples.mkdir(parents=True, exist_ok=True)

    def sample(frame: pd.DataFrame, name: str) -> None:
        if args.samples:
            (frame.to_csv if name.endswith(".csv") else frame.to_parquet)(samples / name, index=False)

    files: dict[str, pd.DataFrame] = {}
    c360 = pd.read_parquet(data / f"customer_360_{args.as_of}.parquet")
    files["customer_360"] = c360
    sample(c360, f"customer_360_{args.as_of}.parquet")
    sample(c360.head(200), "customer_360_sample_200.csv")

    features_path = data / f"features_{args.as_of}.parquet"
    if features_path.exists():
        features = pd.read_parquet(features_path)
        files["features"] = features
        sample(features, f"features_{args.as_of}.parquet")
        sample(features.head(200), "features_sample_200.csv")

    parts = sorted((data / "categorised").glob("part-*.parquet"))
    if parts:
        tx = pd.read_parquet(parts[0])
        keep = tx["customer_id"].drop_duplicates().iloc[:25]
        tx = tx[tx["customer_id"].isin(keep)].reset_index(drop=True)
        files["transactions_categorised"] = tx
        sample(tx, "transactions_categorised_25_customers.parquet")
        sample(tx.head(2000), "transactions_categorised_sample_2000.csv")

    from gcip.fr1.labels import takeup_labels
    warehouse = data / "warehouse"
    labels = takeup_labels(pd.read_parquet(warehouse / "product_holdings.parquet"),
                           pd.read_parquet(warehouse / "customers.parquet", columns=["customer_id"]), "2026-04-01", "2026-09-30")
    files["labels"] = labels
    sample(labels, "labels.csv")

    sections = []
    for name, frame in files.items():
        table = schema(frame)
        table.to_csv(out / "schemas" / f"{name}.csv", index=False)
        sections.append(f"## {name} ({len(table)} columns)\n\n{markdown(table)}")
    (out / "schema_tables.md").write_text("# Schemas\n\nGenerated by scripts/export_data_contract.py.\n\n"
                                          + "\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"wrote {out}: " + ", ".join(f"{name} {frame.shape}" for name, frame in files.items()))


if __name__ == "__main__":
    main()
