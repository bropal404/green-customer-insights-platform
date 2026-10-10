"""Product take-up labels from a documented causal model (no LLM, no hand labelling).

For product p and customer c:

    logit P(take-up in window) = b0_p + sum_k beta_pk * x_ck + sum_j gamma_pj * z_cj

x are the customer's TRUE drivers at the cutoff (the platform only sees noisy,
partial evidence of them in transactions), z are hidden traits the platform
never sees. Only customers who are truly eligible can take a product up.
b0_p is solved by bisection so the take-up rate among eligible customers hits
the target in configs/simulation.yaml. p_true is stored in the truth vault: it
is the Bayes-optimal score, so its AUC is the ceiling no model can beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRODUCTS = ["GP01", "GP02", "GP03", "GP04", "GP05", "GP06"]
ICE_FUELS = ["petrol", "diesel", "hev", "phev"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def state_at_cutoff(pop: pd.DataFrame, veh: pd.DataFrame, homes: pd.DataFrame, bank: pd.DataFrame,
                    events: pd.DataFrame, suppliers: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """The customer's true situation on the cutoff date, after pre-cutoff life events."""
    cutoff = np.datetime64(cfg["labels"]["cutoff"])
    s = pop.merge(veh, on="customer_id").merge(homes, on="customer_id").merge(bank, on="customer_id")
    s = s.merge(events, on="customer_id").merge(suppliers, on="customer_id").copy()  # consolidate before adding columns

    changed_car = pd.notna(s["new_car_date"]) & (s["new_car_date"].values <= cutoff)
    s["car_fuel_cutoff"] = np.where(changed_car, s["new_car_fuel"], s["car_fuel"])
    s["has_car_cutoff"] = s["has_car"] | changed_car
    new_age = (cutoff - s["new_car_date"].values).astype("timedelta64[D]").astype(float) / 365.25 + 1.5
    s["car_age_cutoff"] = np.where(changed_car, new_age, s["car_age_years"])
    s["household_size_cutoff"] = s["household_size"] + (pd.notna(s["baby_date"]) & (s["baby_date"].values <= cutoff))
    retired_now = pd.notna(s["retirement_date"]) & (s["retirement_date"].values <= cutoff)
    s["employment_cutoff"] = np.where(retired_now, "retired", s["employment"])
    s["commute_cutoff"] = np.where(retired_now, "none", s["commute_mode"])
    s["recent_move"] = pd.notna(s["move_date"]) & (s["move_date"].values <= cutoff) & \
        (s["move_date"].values > cutoff - np.timedelta64(365, "D"))
    end = s["motor_finance_end"].values
    s["loan_ending_12m"] = s["motor_finance"] & ~changed_car & pd.notna(end) & (end > cutoff) & \
        (end <= cutoff + np.timedelta64(365, "D"))
    return s


def drivers(state: pd.DataFrame, product: str, la: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """True driver matrix x and the true-eligibility mask for one product."""
    s = state
    income = (s["income_decile"].values - 5.5) / 2.87
    age = (s["age"].values - 48.0) / 17.0
    arrears = s["in_arrears"].values
    difficulty = arrears | s["overdraft_heavy"].values
    if product == "GP01":
        ice_km = np.maximum(s["annual_km_true"].values * (1 - s["ev_km_share"].values), 500.0)
        log_km = np.log(ice_km / 11000.0)
        bev_share = la.set_index("la_code").loc[s["la_code"], "car_share_bev"].values
        x = pd.DataFrame({
            "log_km": log_km, "log_km_sq": log_km ** 2,
            "car_age": np.nan_to_num(s["car_age_cutoff"].values) / 10.0,
            "loan_ending_12m": s["loan_ending_12m"].astype(float).values,
            "offstreet": s["offstreet_parking"].astype(float).values,
            "income": income,
            "multi_car": (s["n_cars"].values >= 2).astype(float),
            "clean_air_charges": (s["ulez_payer"] | s["caz_payer"]).astype(float).values,
            "area_ev_share": (bev_share - 0.032) / 0.012,
        })
        eligible = s["has_car_cutoff"].values & np.isin(s["car_fuel_cutoff"].values, ICE_FUELS) & ~difficulty
    elif product == "GP02":
        intensity = s["heat_demand_kwh_year_true"].values / np.maximum(s["floor_area_m2"].values, 20.0)
        x = pd.DataFrame({
            "epc_low": np.isin(s["epc_band"].values, list("DEFG")).astype(float),
            "heat_intensity": np.log(intensity / 140.0),
            "recent_move": s["recent_move"].astype(float).values,
            "diy": np.log(s["diy_intensity"].values),
            "income": income,
            "mortgage_with_us": s["mortgage_with_us"].astype(float).values,
            "household_size": (s["household_size_cutoff"].values - 2.4) / 1.3,
        })
        eligible = (s["tenure"].values == "mortgage") & ~np.isin(s["epc_band"].values, ["A", "B"]) & ~difficulty
    elif product == "GP03":
        x = pd.DataFrame({
            "log_elec": np.log(np.maximum(s["elec_demand_kwh_year_true"].values, 500.0) / 3000.0),
            "solar_yield": (s["pv_kwh_per_kwp_region"].values - 980.0) / 50.0,
            "ev_at_home": np.isin(s["car_fuel_cutoff"].values, ["bev", "phev"]).astype(float),
            "daytime_home": ((s["employment_cutoff"].values == "retired") | (s["commute_cutoff"].values == "wfh")).astype(float),
            "income": income,
            "detached": (s["accommodation"].values == "detached").astype(float),
        })
        owner = np.isin(s["tenure"].values, ["owned_outright", "mortgage"])
        house = np.isin(s["accommodation"].values, ["detached", "semi", "terraced"])
        eligible = owner & house & ~s["has_solar"].values & s["roof_suitable"].values & ~difficulty
    elif product == "GP04":
        paper_accounts = 1 + s["has_savings"].astype(int).values + s["has_credit_card"].astype(int).values
        x = pd.DataFrame({
            "log_logins": np.log1p(s["logins_per_month_rate"].values),
            "app_user": s["app_user"].astype(float).values,
            "age": age,
            "paper_accounts": (paper_accounts - 2.0),
        })
        eligible = s["paper_statements"].values & (s["logins_per_month_rate"].values > 0)
    elif product == "GP05":
        x = pd.DataFrame({
            "invests_with_us": s["investments_with_us"].astype(float).values,
            "log_investment": np.log1p(s["investment_balance"].values / 1000.0) / 2.0,
            "regular_investor": s["regular_investor"].astype(float).values,
            "stated_pref": np.nan_to_num(s["stated_sustainability_pref"].values, nan=0.0),
            "green_supplier": s["green_supplier"].astype(float).values,
            "income": income,
        })
        eligible = s["investor"].values & ~difficulty
    elif product == "GP06":
        borrowing = s["has_personal_loan"].values | (s["card_balance"].values > 0) | (s["external_card_repayment"].values > 0)
        la_gas = la.set_index("la_code").loc[s["la_code"], "gas_dom_mean_kwh"].values
        x = pd.DataFrame({
            "existing_borrowing": borrowing.astype(float),
            "purchase_intent": s["green_purchase_intent"].astype(float).values,
            "cyclist": s["cyclist"].astype(float).values,
            "owner_outright_high_gas": ((s["tenure"].values == "owned_outright")
                                        & (s["gas_kwh_year_true"].values > la_gas)).astype(float),
            "income": income,
            "age": age,
        })
        eligible = ~arrears & (s["debt_to_income"].values < 0.35)
    else:
        raise KeyError(product)
    return x, eligible


def _solve_intercept(base: np.ndarray, target: float) -> float:
    lo, hi = -20.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _sigmoid(mid + base).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def assign_prior_holdings(state: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Green products some customers already took before the cutoff (block ALREADY_HOLDS).

    Prior holders are drawn from customers whose situation is already consistent
    with the product: BEV drivers financed with us (GP01), mortgagors with a heat
    pump or an A-C rating (GP02), solar owners (GP03), green-minded investors
    with us (GP05) and personal-loan holders with a green purchase (GP06).
    """
    rates = cfg["labels"]["prior_holding_rate"]
    n = len(state)
    s = state
    pools = {
        "GP01": (s["car_fuel_cutoff"].values == "bev") & s["motor_finance_with_us"].values,
        "GP02": (s["tenure"].values == "mortgage") & ((s["main_heating"].values == "heat_pump")
                                                     | np.isin(s["epc_band"].values, list("ABC"))),
        "GP03": s["has_solar"].values,
        "GP04": np.zeros(n, dtype=bool),
        "GP05": s["investments_with_us"].values & (s["z_green"].values > 0.3),
        "GP06": s["has_personal_loan"].values & s["green_purchase_intent"].values,
    }
    out = pd.DataFrame({"customer_id": s["customer_id"].values})
    cutoff = np.datetime64(cfg["labels"]["cutoff"])
    for product, pool in pools.items():
        rate = rates.get(product, 0.0)
        if rate <= 0 or pool.sum() == 0:
            out[f"prior_{product}"] = False
            out[f"prior_{product}_date"] = np.datetime64("NaT")
            continue
        p_in_pool = min(rate * n / pool.sum(), 0.9)
        holds = pool & (rng.random(n) < p_in_pool)
        days = rng.integers(60, 900, n).astype("timedelta64[D]")
        out[f"prior_{product}"] = holds
        out[f"prior_{product}_date"] = np.where(holds, cutoff - days, np.datetime64("NaT")).astype("datetime64[D]")
    return out


def draw_labels(state: pd.DataFrame, prior: pd.DataFrame, la: pd.DataFrame, cfg: dict,
                rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (labels per customer and product, label model report)."""
    model = cfg["label_model"]
    targets = cfg["labels"]["target_rate"]
    window_start = np.datetime64(cfg["labels"]["window_start"])
    window_end = np.datetime64(cfg["labels"]["window_end"])
    window_days = int((window_end - window_start).astype(int)) + 1
    traits = {name: state[f"z_{name}"].values for name in ("green", "liquidity", "stay", "digital", "risk")}

    rows = []
    report = []
    for product in PRODUCTS:
        x, eligible = drivers(state, product, la)
        eligible = eligible & ~prior[f"prior_{product}"].values if f"prior_{product}" in prior else eligible
        coefs = model[product]["drivers"]
        gammas = model[product]["traits"]
        linear = np.zeros(len(state))
        for name, coef in coefs.items():
            linear += coef * x[name].values
        for name, coef in gammas.items():
            linear += coef * traits[name]
        b0 = _solve_intercept(linear[eligible], targets[product]) if eligible.any() else -10.0
        p_true = np.where(eligible, _sigmoid(b0 + linear), 0.0)
        took = eligible & (rng.random(len(state)) < p_true)
        offset = rng.integers(0, window_days, len(state)).astype("timedelta64[D]")
        date = window_start + offset
        if product == "GP01":
            ends = state["motor_finance_end"].values
            in_window = pd.notna(ends) & (ends >= window_start) & (ends <= window_end)
            jitter = rng.integers(-20, 21, len(state)).astype("timedelta64[D]")
            date = np.where(in_window, np.clip(ends + jitter, window_start, window_end), date)
        rows.append(pd.DataFrame({
            "customer_id": state["customer_id"].values,
            "product": product,
            "eligible_true": eligible,
            "p_true": p_true,
            "took_up": took,
            "takeup_date": np.where(took, date, np.datetime64("NaT")).astype("datetime64[D]"),
        }))
        report.append({
            "product": product, "intercept": round(b0, 4), "eligible": int(eligible.sum()),
            "takeups": int(took.sum()), "rate_among_eligible": round(float(took[eligible].mean()) if eligible.any() else 0.0, 4),
            "target_rate": targets[product], "coefficients": coefs, "trait_coefficients": gammas,
        })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(report)
