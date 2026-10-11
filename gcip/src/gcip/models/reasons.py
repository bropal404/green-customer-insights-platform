"""Reason codes with evidence (FR7): turn a model's per-feature contributions into fixed reason codes, each with
a plain-language fact from the customer's own data.

A reason code's contribution is the sum of the SHAP contributions of the features mapped to it in
gcip.models.design.REASON_FEATURES; only codes that push the score up are reported.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from gcip.models.design import REASON_FEATURES

KM_PER_MILE = 1.609344


def _num(row: pd.Series, column: str) -> float:
    value = row.get(column)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value


def _gbp(value: float) -> str:
    return f"£{value:,.0f}"


EVIDENCE = {  # second person with a verb: the sentence goes straight into the customer's explanation
    "HIGH_FUEL_SPEND": lambda r: f"you spent {_gbp(_num(r, 'fuel_spend_12m'))} on road fuel in the last 12 months",
    "HIGH_ANNUAL_MILEAGE": lambda r: f"you drive about {_num(r, 'est_annual_km') / KM_PER_MILE:,.0f} miles a year, estimated from the fuel you bought",
    "EXISTING_VEHICLE": lambda r: "you pay for a car (fuel, insurance, tax, servicing or finance)",
    "VEHICLE_LOAN_ENDING": lambda r: (f"your car finance with us ends in {_num(r, 'motor_loan_months_to_end'):.0f} months"
                                      if not math.isnan(_num(r, "motor_loan_months_to_end")) else "you make car finance payments"),
    "OLDER_VEHICLE": lambda r: f"your car on our books is {_num(r, 'asset_vehicle_age_years'):.0f} years old",
    "OFFSTREET_PARKING": lambda r: "you live in a house, so charging at home is likely possible",
    "CLEAN_AIR_ZONE_CHARGES": lambda r: f"you paid {_num(r, 'clean_air_charges_12m'):.0f} clean-air or congestion charges in the last 12 months",
    "HOMEOWNER_MORTGAGE": lambda r: "you pay a mortgage",
    "HIGH_GAS_USE": lambda r: f"you use about {_num(r, 'gas_kwh_12m'):,.0f} kWh of gas a year (local average {_num(r, 'area_gas_dom_mean_kwh'):,.0f})",
    "LOW_EPC_RATING": lambda r: f"your home has energy rating (EPC) {r.get('asset_epc_band')}",
    "RECENT_HOME_MOVE": lambda r: "you moved home recently",
    "HOME_IMPROVEMENT_SPEND": lambda r: (f"you spent {_gbp(_num(r, 'diy_spend_12m') + _num(r, 'trades_spend_12m'))} on DIY and "
                                         "tradespeople in the last 12 months"),
    "HOMEOWNER": lambda r: "you own your home",
    "HIGH_ELECTRICITY_USE": lambda r: f"you use about {_num(r, 'elec_kwh_12m'):,.0f} kWh of electricity a year",
    "HOUSE_SUITABLE_FOR_SOLAR": lambda r: "you live in a house with its own roof",
    "HIGH_SOLAR_YIELD_REGION": lambda r: f"panels in your region produce about {_num(r, 'area_pv_kwh_per_kwp_year'):,.0f} kWh per kW a year",
    "EV_AT_HOME": lambda r: "you charge an electric car",
    "DAYTIME_OCCUPANCY": lambda r: "you are often at home in the day, so more of the solar power would be used directly",
    "PAPER_STATEMENTS": lambda r: f"you have {_num(r, 'paper_accounts'):.0f} account(s) still on paper statements",
    "HIGH_DIGITAL_ACTIVITY": lambda r: f"you log in about {_num(r, 'logins_per_month_6m'):.0f} times a month",
    "MOBILE_APP_USER": lambda r: "you use the mobile app",
    "EXISTING_INVESTMENTS": lambda r: "you already hold investments",
    "REGULAR_INVESTING": lambda r: f"you invested in {_num(r, 'investment_transfer_months_12m'):.0f} of the last 12 months",
    "SUSTAINABILITY_PREFERENCE": lambda r: "you told us sustainability matters to how you invest",
    "GREEN_ENERGY_TARIFF": lambda r: "you buy energy from a green-leaning supplier",
    "EXISTING_BORROWING": lambda r: "you already borrow with a loan or card",
    "GREEN_PURCHASE_INTENT": lambda r: (f"you spent {_gbp(_num(r, 'bike_shop_spend_12m') + _num(r, 'appliance_spend_12m'))} at bike or "
                                        "appliance shops in the last 12 months"),
    "CYCLING_ACTIVITY": lambda r: f"you made {_num(r, 'pt_micromobility_trips_12m'):.0f} bike-hire trips in the last 12 months",
    "OWNER_NO_MORTGAGE": lambda r: "you own your home without a mortgage",
}


# Short labels for "Why you're seeing this" (customer view); the evidence sentences above give the detail.
LABELS = {
    "HIGH_FUEL_SPEND": "high fuel spend", "HIGH_ANNUAL_MILEAGE": "high annual mileage", "EXISTING_VEHICLE": "existing vehicle",
    "VEHICLE_LOAN_ENDING": "car finance ending soon", "OLDER_VEHICLE": "older car", "OFFSTREET_PARKING": "home charging likely",
    "CLEAN_AIR_ZONE_CHARGES": "clean-air zone charges", "HOMEOWNER_MORTGAGE": "mortgage holder", "HIGH_GAS_USE": "high gas use",
    "LOW_EPC_RATING": "low energy rating", "RECENT_HOME_MOVE": "recent home move", "HOME_IMPROVEMENT_SPEND": "home improvement spend",
    "HOMEOWNER": "home owner", "HIGH_ELECTRICITY_USE": "high electricity use", "HOUSE_SUITABLE_FOR_SOLAR": "house with its own roof",
    "HIGH_SOLAR_YIELD_REGION": "sunny region", "EV_AT_HOME": "electric car at home", "DAYTIME_OCCUPANCY": "home in the day",
    "PAPER_STATEMENTS": "paper statements", "HIGH_DIGITAL_ACTIVITY": "high digital activity", "MOBILE_APP_USER": "mobile app user",
    "EXISTING_INVESTMENTS": "existing investments", "REGULAR_INVESTING": "regular investing",
    "SUSTAINABILITY_PREFERENCE": "sustainability preference", "GREEN_ENERGY_TARIFF": "green energy tariff",
    "EXISTING_BORROWING": "existing borrowing", "GREEN_PURCHASE_INTENT": "bike or appliance shopping",
    "CYCLING_ACTIVITY": "cycling", "OWNER_NO_MORTGAGE": "owns home outright",
    "EARLIER_INTEREST": "you showed interest before",  # from customer memory, not the model
}


def _pos(r: pd.Series, column: str) -> bool:
    value = _num(r, column)
    return not math.isnan(value) and value > 0


def _above(r: pd.Series, column: str, reference: str) -> bool:
    value, ref = _num(r, column), _num(r, reference)
    return not math.isnan(value) and not math.isnan(ref) and value > ref


# A reason is reported only when the customer's own data supports its sentence. A feature can push a score up through
# its absence (no gas bills makes a heat pump more likely for an oil-heated home), and "HIGH_GAS_USE: about 0 kWh"
# would then be shown; such codes are skipped and the next strongest supported reason is used.
SUPPORTED = {
    "HIGH_FUEL_SPEND": lambda r: _pos(r, "fuel_spend_12m"),
    "HIGH_ANNUAL_MILEAGE": lambda r: _pos(r, "est_annual_km"),
    "EXISTING_VEHICLE": lambda r: _pos(r, "vehicle_evidence"),
    "VEHICLE_LOAN_ENDING": lambda r: _pos(r, "motor_loan_months_to_end") or _pos(r, "vehicle_finance_months_12m"),
    "OLDER_VEHICLE": lambda r: _pos(r, "asset_vehicle_age_years"),
    "OFFSTREET_PARKING": lambda r: _pos(r, "house_known"),
    "CLEAN_AIR_ZONE_CHARGES": lambda r: _pos(r, "clean_air_charges_12m"),
    "HOMEOWNER_MORTGAGE": lambda r: _pos(r, "has_mortgage_evidence"),
    "HIGH_GAS_USE": lambda r: _above(r, "gas_kwh_12m", "area_gas_dom_mean_kwh"),
    "LOW_EPC_RATING": lambda r: str(r.get("asset_epc_band")) in {"D", "E", "F", "G"},
    "RECENT_HOME_MOVE": lambda r: _pos(r, "recent_move_evidence"),
    "HOME_IMPROVEMENT_SPEND": lambda r: _pos(r, "diy_spend_12m") or _pos(r, "trades_spend_12m"),
    "HOMEOWNER": lambda r: _pos(r, "homeowner_evidence"),
    "HIGH_ELECTRICITY_USE": lambda r: _above(r, "elec_kwh_12m", "area_elec_dom_median_kwh"),
    "HOUSE_SUITABLE_FOR_SOLAR": lambda r: _pos(r, "house_known"),
    "HIGH_SOLAR_YIELD_REGION": lambda r: _pos(r, "area_pv_kwh_per_kwp_year"),
    "EV_AT_HOME": lambda r: _pos(r, "ev_owner"),
    "DAYTIME_OCCUPANCY": lambda r: _pos(r, "daytime_home"),
    "PAPER_STATEMENTS": lambda r: _pos(r, "paper_accounts"),
    "HIGH_DIGITAL_ACTIVITY": lambda r: _pos(r, "logins_per_month_6m"),
    "MOBILE_APP_USER": lambda r: _pos(r, "app_login_share_6m"),
    "EXISTING_INVESTMENTS": lambda r: _pos(r, "investor_evidence") or _pos(r, "n_investment_accounts"),
    "REGULAR_INVESTING": lambda r: _pos(r, "investment_transfer_months_12m"),
    "SUSTAINABILITY_PREFERENCE": lambda r: _pos(r, "stated_sustainability_preference"),
    "GREEN_ENERGY_TARIFF": lambda r: _pos(r, "green_supplier"),
    "EXISTING_BORROWING": lambda r: _pos(r, "existing_borrowing"),
    "GREEN_PURCHASE_INTENT": lambda r: _pos(r, "bike_shop_spend_12m") or _pos(r, "appliance_spend_12m"),
    "CYCLING_ACTIVITY": lambda r: _pos(r, "pt_micromobility_trips_12m"),
    "OWNER_NO_MORTGAGE": lambda r: _pos(r, "owner_no_mortgage"),
}


def reason_contributions(contrib: np.ndarray, columns: list[str], product: str) -> pd.DataFrame:
    """Per customer, the summed contribution (log-odds) of each reason code's features."""
    index = {c: i for i, c in enumerate(columns)}
    out = {}
    for code, features in REASON_FEATURES[product].items():
        idx = [index[f] for f in features if f in index]
        out[code] = contrib[:, idx].sum(axis=1) if idx else np.zeros(len(contrib))
    return pd.DataFrame(out)


def top_reasons(reasons: pd.DataFrame, rows: pd.DataFrame, max_codes: int = 3, min_contribution: float = 0.05) -> tuple[list, list]:
    """Reason codes that push the score up (largest first), supported by the customer's own data, with their evidence."""
    codes_out, evidence_out = [], []
    values = reasons.to_numpy()
    names = np.array(reasons.columns)
    for i in range(len(reasons)):
        row = rows.iloc[i]
        order = np.argsort(-values[i])
        chosen = []
        for j in order:
            if values[i, j] <= min_contribution or len(chosen) == max_codes:
                break
            check = SUPPORTED.get(names[j])
            if check is None or check(row):
                chosen.append(names[j])
        codes_out.append(chosen)
        evidence = []
        for code in chosen:
            try:
                evidence.append(EVIDENCE[code](row))
            except Exception:  # noqa: BLE001 - a missing value never blocks the reason code itself
                evidence.append(code.replace("_", " ").lower())
        evidence_out.append(evidence)
    return codes_out, evidence_out
