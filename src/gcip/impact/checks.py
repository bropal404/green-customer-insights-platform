"""Checks on the FR4 impact engine that need no after-period.

truth_recovery: how well the litres and kWh the engine reads from transactions match each customer's true usage in the
    simulator's truth vault, and what that does to the baseline footprint the carbon statements start from.
sensitivity: one-at-a-time ranges on the main assumptions (configs/carbon.yaml `sensitivity`), as a tornado table.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from gcip.config import load_config
from gcip.impact.engine import METHODS, Factors, Prices


def _recovery(estimate: pd.Series, truth: pd.Series) -> dict:
    ok = estimate.notna() & truth.notna() & (truth > 0)
    e, t = estimate[ok].astype(float), truth[ok].astype(float)
    if len(t) < 10:
        return {"customers": int(len(t))}
    ratio = e / t
    return {"customers": int(len(t)), "median_ratio": round(float(ratio.median()), 3),
            "median_abs_pct_error": round(float((ratio - 1).abs().median() * 100), 1),
            "within_20pct": round(float(((ratio - 1).abs() <= 0.2).mean()), 3),
            "correlation": round(float(np.corrcoef(e, t)[0, 1]), 3)}


def truth_recovery(table: pd.DataFrame, state: pd.DataFrame) -> dict:
    """Engine inputs (training snapshot, before any take-up) against true annual usage, for customers whose spend is visible."""
    f = Factors.load(load_config("carbon"))
    d = table.merge(state, on="customer_id", how="inner", suffixes=("", "_true"))
    car = d["has_car"].fillna(False).astype(bool) & d["car_fuel"].isin(["petrol", "diesel"])
    main_litres = d["annual_km_true"] / d["km_per_litre_true"]
    extra_litres = (d["extra_km_true"] / d["extra_km_per_litre_true"]).where(d["extra_fuel"].isin(["petrol", "diesel"]), 0.0).fillna(0.0)
    true_litres = (main_litres + extra_litres).where(car)
    fuel_seen = car & (d["fuel_visible"].fillna(0).astype(int) == 1)
    energy_seen = d["energy_bills_visible"].fillna(0).astype(int) == 1
    gas_home = energy_seen & (d["gas_kwh_year_true"] > 0)
    true_transport = true_litres * np.where(d["car_fuel"] == "diesel", f.diesel_l, f.petrol_l)
    est_transport = d["fuel_litres_12m"] * np.where(d["car_fuel"] == "diesel", f.diesel_l, f.petrol_l)
    true_home = d["gas_kwh_year_true"].fillna(0) * f.gas_kwh + d["elec_kwh_year_true"].fillna(0) * f.grid_kwh
    est_home = d["gas_kwh_12m"].fillna(0) * f.gas_kwh + d["elec_kwh_12m"].fillna(0) * f.grid_kwh
    return {
        "note": "Engine inputs read from transactions at the 31 March 2026 snapshot vs the simulator's true annual usage. "
                "Only customers whose spend is visible (fuel bought on our cards, energy bills paid from our accounts).",
        "fuel_litres": _recovery(d.loc[fuel_seen, "fuel_litres_12m"], true_litres[fuel_seen]),
        "gas_kwh": _recovery(d.loc[gas_home, "gas_kwh_12m"], d.loc[gas_home, "gas_kwh_year_true"]),
        "electricity_kwh": _recovery(d.loc[energy_seen, "elec_kwh_12m"], d.loc[energy_seen, "elec_kwh_year_true"]),
        "transport_co2e_kg": _recovery(est_transport[fuel_seen], true_transport[fuel_seen]),
        "home_energy_co2e_kg": _recovery(est_home[energy_seen], true_home[energy_seen]),
        "coverage": {"customers": int(len(d)), "car_owners": int(car.sum()), "fuel_visible": int(fuel_seen.sum()),
                     "energy_bills_visible": int(energy_seen.sum())},
    }


KNOBS = {
    "pump_price": ("prices", ["petrol", "diesel"]),
    "km_per_litre": ("money", ["km_per_litre"]),
    "heat_pump_efficiency": ("assumptions", ["heat_pump_efficiency"]),
    "boiler_efficiency": ("assumptions", ["boiler_efficiency"]),
    "pv_yield": ("assumptions", ["pv_orientation_factor"]),
    "dual_fuel_split": ("assumptions", ["dual_fuel_split_kwh"]),
}


def _scale(node, factor: float):
    if isinstance(node, dict):
        return {k: (_scale(v, factor) if k in ("value", "petrol", "diesel", "electricity", "gas") or isinstance(v, dict) else v)
                for k, v in node.items()}
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return node * factor
    return node


def _summary(table: pd.DataFrame, cfg: dict, f: Factors, p: Prices, products: list[str]) -> dict:
    out = {}
    for product in products:
        rows = METHODS[product](table.reset_index(drop=True), cfg, f, p)
        rows = rows[rows["status"] == "ok"]
        out[product] = {"co2e_kg_per_year": float(rows["co2e_avoided_kg_per_year"].mean()),
                        "money_gbp_per_year": float(rows["running_cost_change_gbp_per_year"].mean())}
    return out


def sensitivity(table: pd.DataFrame, as_of: str, products: list[str] | None = None) -> pd.DataFrame:
    """Mean avoided CO2e and running-cost change per product with each assumption moved down and up by its range."""
    cfg = load_config("carbon")
    products = products or ["GP01", "GP02", "GP03", "GP06"]
    f, p = Factors.load(cfg), Prices.at(as_of)
    base = _summary(table, cfg, f, p, products)
    rows = []
    for knob, rel in cfg["sensitivity"].items():
        if knob not in KNOBS:
            continue
        where, keys = KNOBS[knob]
        results = {}
        for side, factor in (("low", 1 - rel), ("high", 1 + rel)):
            c, pr = copy.deepcopy(cfg), copy.copy(p)
            if where == "prices":
                for k in keys:
                    setattr(pr, k, getattr(pr, k) * factor)
            else:
                for k in keys:
                    c[where][k] = _scale(c[where][k], factor)
            results[side] = _summary(table, c, f, pr, products)
        for product in products:
            b = base[product]
            rows.append({"assumption": knob, "range": f"±{rel:.0%}", "product": product,
                         "co2e_base": round(b["co2e_kg_per_year"], 1),
                         "co2e_low": round(results["low"][product]["co2e_kg_per_year"], 1),
                         "co2e_high": round(results["high"][product]["co2e_kg_per_year"], 1),
                         "money_base": round(b["money_gbp_per_year"], 1),
                         "money_low": round(results["low"][product]["money_gbp_per_year"], 1),
                         "money_high": round(results["high"][product]["money_gbp_per_year"], 1)})
    out = pd.DataFrame(rows)
    out["co2e_swing_pct"] = ((out["co2e_high"] - out["co2e_low"]).abs() / out["co2e_base"].abs().replace(0, np.nan) * 100).round(1)
    out["money_swing_pct"] = ((out["money_high"] - out["money_low"]).abs() / out["money_base"].abs().replace(0, np.nan) * 100).round(1)
    return out
