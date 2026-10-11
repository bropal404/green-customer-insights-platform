"""Life events over the simulation window, drawn month by month from constant hazards.

Events change what later months look like (a new car changes fuel use, a move
brings removal and DIY spend, a baby grows the household) and some of them make
green products likelier. They also create the "changed circumstances" cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def month_starts(start: str, end: str) -> np.ndarray:
    return np.arange(np.datetime64(start, "M"), np.datetime64(end, "M") + 1).astype("datetime64[D]")


def _first_event(rng: np.random.Generator, hazard: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Date of the first event for each customer given a per-month hazard (NaT if none)."""
    n, m = len(hazard), len(months)
    draws = rng.random((n, m)) < hazard[:, None]
    any_event = draws.any(axis=1)
    first = draws.argmax(axis=1)
    day = rng.integers(1, 28, n).astype("timedelta64[D]")
    dates = months[first] + day
    return np.where(any_event, dates, np.datetime64("NaT")).astype("datetime64[D]")


def build_events(pop: pd.DataFrame, veh: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    ecfg = cfg["events"]
    months = month_starts(cfg["transactions"]["start"], cfg["transactions"]["end"])
    n = len(pop)
    age = pop["age"].values
    renter = np.isin(pop["tenure"].values, ["private_rent", "social_rent"])

    move_hazard = np.full(n, ecfg["house_move"]) * np.where(renter, 2.2, 0.8) * np.where(age < 35, 1.6, 1.0)
    car_hazard = np.where(veh["has_car"].values, ecfg["new_car"], ecfg["new_car"] * 0.15)
    car_hazard = car_hazard * np.where(veh["car_age_years"].fillna(0).values > 9, 1.6, 1.0)
    baby_hazard = np.where((age >= 22) & (age <= 42), ecfg["new_baby"], 0.0)
    retire_hazard = np.where((age >= 60) & np.isin(pop["employment"].values, ["employed", "self_employed"]),
                             ecfg["retirement_if_60_plus"], 0.0)

    out = pd.DataFrame({"customer_id": pop["customer_id"].values})
    out["move_date"] = _first_event(rng, move_hazard, months)
    out["new_car_date"] = _first_event(rng, car_hazard, months)
    out["baby_date"] = _first_event(rng, baby_hazard, months)
    out["retirement_date"] = _first_event(rng, retire_hazard, months)

    # A replacement car is electric more often for green-minded, higher-income drivers.
    z = pop["z_green"].values + 0.3 * (pop["income_decile"].values - 5.5) / 2.87
    p_ev = 1 / (1 + np.exp(-(-1.8 + 0.8 * z)))
    new_fuel = np.where(rng.random(n) < p_ev, "bev", np.where(rng.random(n) < 0.3, "hev", "petrol"))
    out["new_car_fuel"] = np.where(pd.notna(out["new_car_date"]), new_fuel, "")
    # Who financed the replacement (only matters before the label window: during it,
    # buying an EV with us is a GP01 take-up drawn in labels.py).
    out["new_car_financed_elsewhere"] = rng.random(n) < 0.45
    return out
