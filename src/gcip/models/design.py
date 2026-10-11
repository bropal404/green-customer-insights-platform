"""Feature matrix design and the feature -> reason-code map.

Only behavioural and product features can become reason codes. Demographic
features (age, income band, region) may inform the score but are never given to
a customer as a reason, and the vulnerability flag is never a model feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_AND_TEXT = {"customer_id", "la_code", "la_name", "as_of", "motor_loan_product_code"}
NEVER_FEATURES = {"vulnerability_flag", "marketing_consent"}
CATEGORICAL = {
    "declared_residential_status": ["homeowner", "homeowner_mortgage", "tenant", "living_with_family"],
    "declared_employment_status": ["employed", "self_employed", "retired", "student", "not_working"],
    "region": ["North East", "North West", "Yorkshire and The Humber", "East Midlands", "West Midlands", "East",
               "London", "South East", "South West", "Wales"],
    "asset_property_type": ["detached", "semi", "terraced", "flat", "caravan"],
    "asset_vehicle_fuel_type": ["petrol", "diesel", "hev", "phev", "bev"],
    "preferred_contact_channel": ["app", "email", "post"],
    "personal_loan_purpose": ["car", "home_improvement", "debt_consolidation", "wedding_or_holiday", "other"],
    "asset_construction_period": ["pre-1930", "1930-1972", "1973-1999", "2000-or-later"],
}
ORDINAL = {
    "declared_income_band": {"under_15k": 0, "15k_25k": 1, "25k_40k": 2, "40k_60k": 3, "60k_100k": 4, "100k_plus": 5},
    "asset_epc_band": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7},
}
RESPONSES = ["viewed", "clicked", "dismissed", "not_interested", "already_have"]

# Reason code -> features whose SHAP contributions count toward it, per product.
REASON_FEATURES = {
    "GP01": {
        "HIGH_FUEL_SPEND": ["fuel_spend_12m", "fuel_litres_12m", "spend_12m_fuel", "fuel_spend_trend"],
        "HIGH_ANNUAL_MILEAGE": ["est_annual_km"],
        "EXISTING_VEHICLE": ["vehicle_evidence", "motor_insurance_months_12m", "vehicle_tax_payments_12m", "car_maintenance_spend_12m",
                             "spend_12m_automotive", "n_motor_loan_accounts", "vehicle_finance_months_12m"],
        "VEHICLE_LOAN_ENDING": ["motor_loan_months_to_end", "vehicle_finance_monthly"],
        "OLDER_VEHICLE": ["asset_vehicle_age_years"],
        "OFFSTREET_PARKING": ["homeowner_evidence", "house_known", "asset_property_type__detached", "asset_property_type__semi"],
        "CLEAN_AIR_ZONE_CHARGES": ["clean_air_charges_12m"],
    },
    "GP02": {
        "HOMEOWNER_MORTGAGE": ["has_mortgage_evidence", "n_mortgage_accounts", "mortgage_payment_months_12m", "balance_mortgage"],
        "HIGH_GAS_USE": ["gas_kwh_12m", "heat_kwh_12m", "gas_kwh_per_m2", "spend_12m_gas", "spend_12m_utilities"],
        "LOW_EPC_RATING": ["asset_epc_band", "epc_low_known"],
        "RECENT_HOME_MOVE": ["recent_move_evidence", "removals_payments_12m", "declared_years_at_address"],
        "HOME_IMPROVEMENT_SPEND": ["diy_spend_12m", "diy_spend_3m", "trades_spend_12m", "spend_12m_home_improvement",
                                   "appliance_spend_12m"],
    },
    "GP03": {
        "HOMEOWNER": ["homeowner_evidence", "has_mortgage_evidence", "owner_no_mortgage"],
        "HIGH_ELECTRICITY_USE": ["elec_kwh_12m", "spend_12m_electricity"],
        "HOUSE_SUITABLE_FOR_SOLAR": ["house_known", "asset_property_type__detached", "asset_property_type__semi", "asset_property_type__terraced"],
        "HIGH_SOLAR_YIELD_REGION": ["area_pv_kwh_per_kwp_year"],
        "EV_AT_HOME": ["ev_owner", "ev_owner_evidence", "ev_public_kwh_12m"],
        "DAYTIME_OCCUPANCY": ["daytime_home", "pension_income_months_12m"],
    },
    "GP04": {
        "PAPER_STATEMENTS": ["paper_accounts", "paper_statements"],
        "HIGH_DIGITAL_ACTIVITY": ["logins_per_month_6m", "months_with_logins_6m", "estatement_views_6m"],
        "MOBILE_APP_USER": ["app_login_share_6m", "preferred_contact_channel__app"],
    },
    "GP05": {
        "EXISTING_INVESTMENTS": ["investor_evidence", "n_investment_accounts", "balance_investment"],
        "REGULAR_INVESTING": ["investment_transfer_months_12m", "investment_transfer_total_12m"],
        "SUSTAINABILITY_PREFERENCE": ["stated_sustainability_preference"],
        "GREEN_ENERGY_TARIFF": ["green_supplier"],
    },
    "GP06": {
        "EXISTING_BORROWING": ["existing_borrowing", "n_personal_loan_accounts", "balance_credit_card", "external_credit_repayments_monthly",
                               "loan_repayments_monthly"],
        "GREEN_PURCHASE_INTENT": ["bike_shop_spend_12m", "appliance_spend_12m"],
        "CYCLING_ACTIVITY": ["pt_micromobility_trips_12m"],
        "OWNER_NO_MORTGAGE": ["owner_no_mortgage"],
        "HIGH_GAS_USE": ["gas_kwh_12m", "heat_kwh_12m"],
    },
}


def design_matrix(features: pd.DataFrame, product: str, drop: list[str] | None = None) -> pd.DataFrame:
    """Numeric model matrix for one product (one-hot, ordinals, own campaign history)."""
    f = features
    x = pd.DataFrame(index=f.index)
    for column in f.columns:
        if column in ID_AND_TEXT or column in NEVER_FEATURES or column in CATEGORICAL or column in ORDINAL:
            continue
        if column.startswith(("holds_", "last_response", "split", "label", "p_true")):
            continue
        series = f[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            x[column] = series.astype(float)
    for column, levels in CATEGORICAL.items():
        if column in f:
            for level in levels:
                x[f"{column}__{level}"] = (f[column] == level).astype(float)
    for column, mapping in ORDINAL.items():
        if column in f:
            x[column] = f[column].map(mapping).astype(float)
    response = f.get(f"last_response_{product}")
    if response is not None:
        for level in RESPONSES:
            x[f"own_campaign_{level}"] = (response == level).astype(float)
    if drop:
        x = x.drop(columns=[c for c in x.columns if c in set(drop)], errors="ignore")
    return x.replace([np.inf, -np.inf], np.nan)


def coverage_score(features: pd.DataFrame, product: str) -> np.ndarray:
    """Share of the evidence a product's estimate rests on that is actually observed (0-1)."""
    f = features
    history = np.clip(f["months_of_history"].fillna(0).values / 12.0, 0, 1)
    if product == "GP01":
        fuel = np.clip(f["fuel_months_12m"].fillna(0).values / 10.0, 0, 1)
        known = f["asset_vehicle_fuel_type"].notna().values.astype(float)
        score = 0.65 * fuel + 0.2 * known + 0.15 * history
    elif product in ("GP02", "GP03"):
        energy = np.clip(f["energy_months_12m"].fillna(0).values / 10.0, 0, 1)
        property_known = f["asset_property_type"].notna().values.astype(float)
        epc = f["asset_epc_band"].notna().values.astype(float)
        score = 0.55 * energy + 0.2 * property_known + 0.1 * epc + 0.15 * history
    elif product == "GP04":
        score = 0.7 * np.clip(f["months_with_logins_6m"].fillna(0).values / 6.0, 0, 1) + 0.3 * history
    elif product == "GP05":
        with_us = (f["n_investment_accounts"].fillna(0).values > 0).astype(float)
        transfers = np.clip(f["investment_transfer_months_12m"].fillna(0).values / 6.0, 0, 1)
        pref = f["stated_sustainability_preference"].notna().values.astype(float)
        score = 0.45 * np.maximum(with_us, transfers) + 0.25 * pref + 0.3 * history
    else:
        score = 0.6 * history + 0.4 * np.clip(f["n_transactions_12m"].fillna(0).values / 300.0, 0, 1)
    return np.clip(score, 0.0, 1.0)
