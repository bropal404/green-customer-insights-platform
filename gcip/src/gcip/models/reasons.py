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


EVIDENCE = {
    "HIGH_FUEL_SPEND": lambda r: f"{_gbp(_num(r, 'fuel_spend_12m'))} spent on road fuel in the last 12 months",
    "HIGH_ANNUAL_MILEAGE": lambda r: f"about {_num(r, 'est_annual_km') / KM_PER_MILE:,.0f} miles a year, estimated from fuel bought",
    "EXISTING_VEHICLE": lambda r: "pays for a car (fuel, insurance, tax, servicing or finance)",
    "VEHICLE_LOAN_ENDING": lambda r: (f"car finance with us ends in {_num(r, 'motor_loan_months_to_end'):.0f} months"
                                      if not math.isnan(_num(r, "motor_loan_months_to_end")) else "car finance payments seen"),
    "OLDER_VEHICLE": lambda r: f"car on our books is {_num(r, 'asset_vehicle_age_years'):.0f} years old",
    "OFFSTREET_PARKING": lambda r: "lives in a house, so home charging is likely possible",
    "CLEAN_AIR_ZONE_CHARGES": lambda r: f"{_num(r, 'clean_air_charges_12m'):.0f} clean-air or congestion charges in 12 months",
    "HOMEOWNER_MORTGAGE": lambda r: "pays a mortgage",
    "HIGH_GAS_USE": lambda r: f"about {_num(r, 'gas_kwh_12m'):,.0f} kWh of gas a year (local average {_num(r, 'area_gas_dom_mean_kwh'):,.0f})",
    "LOW_EPC_RATING": lambda r: f"home has EPC rating {r.get('asset_epc_band')}",
    "RECENT_HOME_MOVE": lambda r: "moved home recently",
    "HOME_IMPROVEMENT_SPEND": lambda r: (f"{_gbp(_num(r, 'diy_spend_12m') + _num(r, 'trades_spend_12m'))} on DIY and tradespeople "
                                         "in 12 months"),
    "HOMEOWNER": lambda r: "owns their home",
    "HIGH_ELECTRICITY_USE": lambda r: f"about {_num(r, 'elec_kwh_12m'):,.0f} kWh of electricity a year",
    "HOUSE_SUITABLE_FOR_SOLAR": lambda r: "lives in a house with its own roof",
    "HIGH_SOLAR_YIELD_REGION": lambda r: f"panels in their region produce about {_num(r, 'area_pv_kwh_per_kwp_year'):,.0f} kWh per kW a year",
    "EV_AT_HOME": lambda r: "charges an electric car",
    "DAYTIME_OCCUPANCY": lambda r: "often at home in the day, so more solar power is used directly",
    "PAPER_STATEMENTS": lambda r: f"{_num(r, 'paper_accounts'):.0f} account(s) still on paper statements",
    "HIGH_DIGITAL_ACTIVITY": lambda r: f"logs in about {_num(r, 'logins_per_month_6m'):.0f} times a month",
    "MOBILE_APP_USER": lambda r: "uses the mobile app",
    "EXISTING_INVESTMENTS": lambda r: "already holds investments",
    "REGULAR_INVESTING": lambda r: f"invested in {_num(r, 'investment_transfer_months_12m'):.0f} of the last 12 months",
    "SUSTAINABILITY_PREFERENCE": lambda r: "told us sustainability matters to how they invest",
    "GREEN_ENERGY_TARIFF": lambda r: "buys energy from a green-leaning supplier",
    "EXISTING_BORROWING": lambda r: "already borrows with a loan or card",
    "GREEN_PURCHASE_INTENT": lambda r: (f"{_gbp(_num(r, 'bike_shop_spend_12m') + _num(r, 'appliance_spend_12m'))} at bike or appliance "
                                        "shops in 12 months"),
    "CYCLING_ACTIVITY": lambda r: f"{_num(r, 'pt_micromobility_trips_12m'):.0f} bike-hire trips in 12 months",
    "OWNER_NO_MORTGAGE": lambda r: "owns their home without a mortgage",
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
    """Reason codes that push the score up (largest first) and their evidence sentences."""
    codes_out, evidence_out = [], []
    values = reasons.to_numpy()
    names = np.array(reasons.columns)
    for i in range(len(reasons)):
        order = np.argsort(-values[i])
        chosen = [names[j] for j in order[:max_codes] if values[i, j] > min_contribution]
        codes_out.append(chosen)
        row = rows.iloc[i]
        evidence = []
        for code in chosen:
            try:
                evidence.append(EVIDENCE[code](row))
            except Exception:  # noqa: BLE001 - a missing value never blocks the reason code itself
                evidence.append(code.replace("_", " ").lower())
        evidence_out.append(evidence)
    return codes_out, evidence_out
