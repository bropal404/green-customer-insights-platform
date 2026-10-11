"""Warehouse tables (what a bank holds) and truth-vault tables (what only the simulator knows).

Warehouse tables, one file each, as a bank keeps them in separate systems:
    customers, accounts, balances, product_holdings, preference_changes,
    assets, digital_activity, campaign_responses, transactions (written per chunk).
Truth vault (read only by the evaluation harness):
    customer_truth, labels, label_model_report, transaction_truth (per chunk).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gcip.pipeline.splits import split_of  # noqa: F401  (re-exported for generate.py)

GREEN_ACCOUNTS = {
    "GP01": ("EV", "motor_loan", "MOTOR-EV-GREEN"),
    "GP02": ("GH", "mortgage_further_advance", "MTG-GREEN-FA"),
    "GP03": ("SF", "personal_loan", "LOAN-SOLAR"),
    "GP05": ("SI", "investment", "INV-SUSTAIN-FOCUS"),
    "GP06": ("GL", "personal_loan", "PL-GREEN"),
}


def income_band(annual: np.ndarray) -> np.ndarray:
    edges = [15000, 25000, 40000, 60000, 100000]
    labels = np.array(["under_15k", "15k_25k", "25k_40k", "40k_60k", "60k_100k", "100k_plus"])
    return labels[np.digitize(annual, edges)]


def build_customers(s: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    annual = s["gross_income_week"].values * 52
    band_idx = np.digitize(annual, [15000, 25000, 40000, 60000, 100000])
    stale = rng.random(len(s)) < 0.12
    band_idx = np.clip(band_idx + np.where(stale, rng.choice([-1, 1], len(s)), 0), 0, 5)
    labels = np.array(["under_15k", "15k_25k", "25k_40k", "40k_60k", "60k_100k", "100k_plus"])
    preferred = np.where(s["app_user"].values, "app", np.where(s["logins_per_month_rate"].values > 0, "email", "post"))
    return pd.DataFrame({
        "customer_id": s["customer_id"].values,
        "birth_year": (2026 - s["age"].values).astype(int),
        "la_code": s["la_code"].values,
        "la_name": s["la_name"].values,
        "region": s["region"].values,
        "declared_residential_status": s["declared_residential_status"].values,
        "declared_employment_status": s["employment"].values,
        "declared_income_band": labels[band_idx],
        "customer_since": s["customer_since"].values.astype("datetime64[D]"),
        "marketing_consent": s["marketing_consent"].values,
        "vulnerability_flag": s["vulnerable"].values,
        "preferred_contact_channel": preferred,
        # Investment fact-find answer: known only for investors with us who completed one.
        "stated_sustainability_preference": s["stated_sustainability_pref"].values,
        # Onboarding and credit-application answers (as collected; about 8% of occupations are out of date).
        "declared_occupation": declared_occupation(s, rng),
        "declared_dependants": s["dependants"].values.astype(int),
        "declared_years_at_address": np.floor(s["years_at_address"].values).astype(int),
    })


def declared_occupation(s: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    occupation = s["occupation"].values.astype(object).copy()
    working = np.isin(s["employment"].values, ["employed", "self_employed"])
    stale = working & (rng.random(len(s)) < 0.08)
    groups = np.array(["managers_directors", "professional", "associate_professional", "administrative", "skilled_trades",
                       "caring_leisure", "sales_customer_service", "process_plant", "elementary"])
    occupation[stale] = groups[rng.integers(0, len(groups), stale.sum())]
    return occupation


def build_accounts(s: pd.DataFrame, prior: pd.DataFrame, takeups: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    cid = s["customer_id"].values
    paper = s["paper_statements"].values
    pref = np.where(paper, "paper", "online")
    since = s["customer_since"].values.astype("datetime64[D]")

    def add(mask, suffix, kind, code, opened, **extra):
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return
        frame = pd.DataFrame({
            "account_id": pd.Series(cid[mask]) + f"-{suffix}",
            "customer_id": cid[mask],
            "account_type": kind,
            "product_code": code if np.isscalar(code) else np.asarray(code)[mask],
            "opened_date": np.asarray(opened).astype("datetime64[D]")[mask] if not np.isscalar(opened) else opened,
        })
        for key, value in extra.items():
            frame[key] = value if np.isscalar(value) or value is None else np.asarray(value)[mask]
        rows.append(frame)

    decile = s["income_decile"].values
    ca_code = np.where(decile <= 2, "CA-BASIC", np.where(decile >= 8, "CA-REWARD", "CA-CLASSIC"))
    add(np.ones(len(s), bool), "CA", "current", ca_code, since, statement_preference=pref)
    add(s["has_savings"].values, "SV", "savings", "SAV-EASY", since + np.timedelta64(200, "D"), statement_preference=pref,
        interest_rate=s["savings_rate"].values)
    add(s["has_credit_card"].values, "CC", "credit_card", "CC-EVERYDAY", since + np.timedelta64(400, "D"),
        statement_preference=pref, credit_limit=s["card_limit"].values, interest_rate=s["card_apr"].values)
    add(s["mortgage_with_us"].values, "MG", "mortgage", np.where(s["mortgage_fix_years"].values == 5, "MTG-FIX5", "MTG-FIX2"),
        s["mortgage_start"].values, original_amount=s["mortgage_original"].values, interest_rate=s["mortgage_rate"].values,
        term_months=s["mortgage_term_months"].values,
        collateral_value_at_origination=s["property_value_at_origination"].values)
    ev_prior = prior["prior_GP01"].values
    add(s["motor_finance_with_us"].values, "ML", "motor_loan", np.where(ev_prior, "MOTOR-EV-GREEN", "MOTOR-HP"),
        s["motor_finance_start"].values, original_amount=(s["motor_finance_outstanding"].values * 1.3).round(0),
        term_months=48, collateral_value_at_origination=s["vehicle_value_at_origination"].values,
        closed_date=s["motor_finance_end"].values.astype("datetime64[D]"))
    add(s["has_personal_loan"].values, "PL", "personal_loan", "PL-STANDARD", s["personal_loan_start"].values,
        original_amount=s["personal_loan_amount"].values, interest_rate=s["personal_loan_rate"].values,
        term_months=s["personal_loan_term_months"].values, loan_purpose=s["personal_loan_purpose"].values)
    add(s["investments_with_us"].values, "IV", "investment", np.where(rng.random(len(s)) < 0.6, "INV-ISA", "INV-GIA"),
        since + np.timedelta64(500, "D"), statement_preference=pref)
    for product, (suffix, kind, code) in GREEN_ACCOUNTS.items():
        if product == "GP01":
            continue  # prior EV finance is the motor loan above
        col = f"prior_{product}"
        add(prior[col].values, suffix, kind, code, prior[f"{col}_date"].values)
    for product, (suffix, kind, code) in GREEN_ACCOUNTS.items():
        dates = takeups[f"{product.lower()}_date"].values.astype("datetime64[D]")
        add(~np.isnat(dates), f"{suffix}N", kind, code, dates)
    accounts = pd.concat(rows, ignore_index=True)
    accounts["provider"] = "GCIP Bank"
    accounts["status"] = np.where(pd.notna(accounts.get("closed_date")) & (accounts.get("closed_date") <= np.datetime64("2026-09-30")),
                                  "closed", "open")
    return accounts


def build_balances(s: pd.DataFrame, accounts: pd.DataFrame, months: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    idx = pd.Index(s["customer_id"].values)
    pos = idx.get_indexer(accounts["customer_id"].values)
    month_end = (months + 1).astype("datetime64[D]") - 1
    m = len(months)
    a = len(accounts)
    kind = accounts["account_type"].values
    net = s["monthly_net_income"].values[pos]
    t = np.arange(m)[None, :]
    noise = rng.normal(0, 1, (a, m))
    bal = np.zeros((a, m))
    bal = np.where((kind == "current")[:, None], 0.35 * net[:, None] * np.exp(0.5 * noise)
                   - np.where(s["overdraft_heavy"].values[pos], 0.6 * net, 0.0)[:, None], bal)
    bal = np.where((kind == "savings")[:, None], s["savings_balance"].values[pos][:, None] * (1 + 0.004 * t) * np.exp(0.03 * noise), bal)
    bal = np.where((kind == "credit_card")[:, None], -np.abs(s["card_balance"].values[pos][:, None] * np.exp(0.2 * noise)), bal)
    principal = s["mortgage_balance"].values[pos]
    bal = np.where((kind == "mortgage")[:, None], -principal[:, None] * (1 - 0.0028 * t), bal)
    out_ml = s["motor_finance_outstanding"].values[pos]
    bal = np.where((kind == "motor_loan")[:, None], -np.maximum(out_ml[:, None] * (1 - t / 30.0), 0), bal)
    bal = np.where((kind == "personal_loan")[:, None], -np.maximum(s["personal_loan_balance"].values[pos][:, None] * (1 - t / 36.0), 0), bal)
    market = np.cumprod(1 + rng.normal(0.005, 0.03, m))
    bal = np.where((kind == "investment")[:, None], s["investment_balance"].values[pos][:, None] * market[None, :], bal)
    green = accounts["product_code"].isin(["MTG-GREEN-FA", "LOAN-SOLAR", "PL-GREEN"]).values
    bal = np.where(green[:, None], -rng.uniform(2000, 14000, a)[:, None] * (1 - t / 60.0), bal)
    ev_green = (accounts["product_code"].values == "MOTOR-EV-GREEN") & (kind == "motor_loan")
    bal = np.where(ev_green[:, None], -rng.uniform(18000, 36000, a)[:, None] * (1 - t / 48.0), bal)
    opened = accounts["opened_date"].values.astype("datetime64[D]")
    closed = accounts["closed_date"].values.astype("datetime64[D]") if "closed_date" in accounts else np.full(a, np.datetime64("NaT"))
    active = (opened[:, None] <= month_end[None, :]) & (np.isnat(closed)[:, None] | (closed[:, None] >= month_end[None, :]))
    arrears_cust = s["in_arrears"].values[pos]
    credit_kind = np.isin(kind, ["credit_card", "mortgage", "motor_loan", "personal_loan"])
    arrears = np.where((arrears_cust & credit_kind)[:, None] & (t >= m - 9), rng.integers(1, 4, (a, m)), 0)
    ai, mi = np.nonzero(active)
    return pd.DataFrame({
        "account_id": accounts["account_id"].values[ai],
        "month_end": month_end[mi],
        "balance": np.round(bal[ai, mi], 2) + 0.0,  # + 0.0 turns -0.0 (a paid-off loan) into 0.0
        "arrears_months": arrears[ai, mi].astype(int),
    })


def build_holdings(accounts: pd.DataFrame, s: pd.DataFrame, takeups: pd.DataFrame) -> pd.DataFrame:
    green_map = {"MOTOR-EV-GREEN": "GP01", "MTG-GREEN-FA": "GP02", "LOAN-SOLAR": "GP03", "INV-SUSTAIN-FOCUS": "GP05", "PL-GREEN": "GP06"}
    holdings = pd.DataFrame({
        "customer_id": accounts["customer_id"].values,
        "account_id": accounts["account_id"].values,
        "product_code": accounts["product_code"].values,
        "product_family": accounts["account_type"].values,
        "green_product": accounts["product_code"].map(green_map).fillna("").values,
        "start_date": accounts["opened_date"].values,
        "end_date": accounts["closed_date"].values if "closed_date" in accounts else pd.NaT,
    })
    gp04 = takeups["gp04_date"].values.astype("datetime64[D]")
    paperless = pd.DataFrame({
        "customer_id": s["customer_id"].values[~np.isnat(gp04)],
        "account_id": pd.Series(s["customer_id"].values[~np.isnat(gp04)]) + "-CA",
        "product_code": "SVC-PAPERLESS",
        "product_family": "service",
        "green_product": "GP04",
        "start_date": gp04[~np.isnat(gp04)],
        "end_date": pd.NaT,
    })
    return pd.concat([holdings, paperless], ignore_index=True)


def build_preference_changes(s: pd.DataFrame, takeups: pd.DataFrame) -> pd.DataFrame:
    gp04 = takeups["gp04_date"].values.astype("datetime64[D]")
    has = ~np.isnat(gp04)
    return pd.DataFrame({
        "account_id": pd.Series(s["customer_id"].values[has]) + "-CA",
        "change_date": gp04[has],
        "old_preference": "paper",
        "new_preference": "online",
    })


def build_assets(s: pd.DataFrame) -> pd.DataFrame:
    prop = s["mortgage_with_us"].values
    epc_known = s["epc_known_to_bank"].values
    veh = s["motor_finance_with_us"].values
    property_rows = pd.DataFrame({
        "asset_id": pd.Series(s["customer_id"].values[prop]) + "-PROP",
        "customer_id": s["customer_id"].values[prop],
        "asset_type": "property",
        "property_type": s["accommodation"].values[prop],
        # From the valuation report: always known for homes the bank lends on.
        "property_detail": s["property_detail"].values[prop],
        "construction_period": s["construction_period"].values[prop],
        "bedrooms": s["bedrooms"].values[prop].astype(float),
        "floor_area_m2": np.where(epc_known[prop], s["floor_area_m2"].values[prop], np.nan),
        "epc_band": np.where(epc_known[prop], s["epc_band"].values[prop], None),
        "epc_main_heating": np.where(epc_known[prop], s["main_heating"].values[prop], None),
        "solar_pv_on_record": np.where(epc_known[prop], s["has_solar"].values[prop], None),
        "valuation_at_origination": s["property_value_at_origination"].values[prop],
    })
    vehicle_rows = pd.DataFrame({
        "asset_id": pd.Series(s["customer_id"].values[veh]) + "-VEH",
        "customer_id": s["customer_id"].values[veh],
        "asset_type": "vehicle",
        "vehicle_fuel_type": s["car_fuel"].values[veh],
        "vehicle_first_registered_year": (2025 - np.nan_to_num(s["car_age_years"].values[veh])).round(0),
        "vehicle_registration": s["car_reg"].values[veh],
        "valuation_at_origination": s["vehicle_value_at_origination"].values[veh],
    })
    return pd.concat([property_rows, vehicle_rows], ignore_index=True)


def build_digital(s: pd.DataFrame, takeups: pd.DataFrame, months: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    c, m = len(s), len(months)
    rate = s["logins_per_month_rate"].values[:, None] * np.exp(rng.normal(0, 0.25, (c, m)))
    app_share = np.where(s["app_user"].values, rng.uniform(0.6, 0.95, c), rng.uniform(0.0, 0.15, c))[:, None]
    total = rng.poisson(rate)
    app = rng.binomial(total, app_share)
    gp04 = takeups["gp04_date"].values.astype("datetime64[M]")
    switched = ~np.isnat(gp04)[:, None] & (months[None, :] >= np.where(np.isnat(gp04), months[0], gp04)[:, None])
    paperless = ~s["paper_statements"].values[:, None] | switched
    branch = rng.poisson(np.where(s["z_digital"].values < -0.9, 0.45, 0.04)[:, None] * np.ones((1, m)))
    views = rng.poisson(np.where(paperless, 0.7, 0.05) * (total > 0))
    ci, mi = np.indices((c, m))
    return pd.DataFrame({
        "customer_id": s["customer_id"].values[ci.ravel()],
        "month": months[mi.ravel()].astype("datetime64[D]"),
        "app_logins": app.ravel(),
        "web_logins": (total - app).ravel(),
        "branch_visits": branch.ravel(),
        "estatement_views": views.ravel(),
        "paperless_all_accounts": paperless.ravel(),
    })


def build_campaigns(s: pd.DataFrame, labels: pd.DataFrame, prior: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Marketing contacts and responses: the history customer memory starts from. Three waves before the training date
    and two in the summer of 2026, so memory is in force on the live date (a dismissal in August is still paused on
    30 September). Customers who took the product up before a wave say they already have it."""
    p = labels.pivot(index="customer_id", columns="product", values="p_true").loc[s["customer_id"]].values
    taken = labels.pivot(index="customer_id", columns="product", values="takeup_date").loc[s["customer_id"]]
    products = np.array(["GP01", "GP02", "GP03", "GP04", "GP05", "GP06"])
    rows = []
    for k, sent in enumerate(["2025-05-15", "2025-10-15", "2026-02-16", "2026-06-15", "2026-08-17"]):
        targeted = rng.random(len(s)) < 0.30
        product_idx = rng.integers(0, 6, len(s))
        p_take = p[np.arange(len(s)), product_idx]
        prior_hold = np.array([prior[f"prior_{products[j]}"].values[i] if f"prior_{products[j]}" in prior else False
                               for i, j in enumerate(product_idx)])
        took = pd.to_datetime(taken.values[np.arange(len(s)), product_idx])
        prior_hold = prior_hold | (pd.notna(took) & (took < pd.Timestamp(sent)))
        clicked = rng.random(len(s)) < np.clip(0.04 + 3.0 * p_take, 0, 0.6)
        dismissed = ~clicked & (rng.random(len(s)) < 0.10)
        not_interested = ~clicked & ~dismissed & (rng.random(len(s)) < 0.05 + 0.1 * (p_take < 0.01))
        viewed = ~clicked & ~dismissed & ~not_interested & (rng.random(len(s)) < 0.35)
        response = np.select([prior_hold, clicked, dismissed, not_interested, viewed],
                             ["already_have", "clicked", "dismissed", "not_interested", "viewed"], default="no_response")
        channel = np.where(s["app_user"].values, "app", np.where(s["logins_per_month_rate"].values > 0, "email", "post"))
        sent_date = np.datetime64(sent)
        delay = rng.integers(0, 10, len(s)).astype("timedelta64[D]")
        rows.append(pd.DataFrame({
            "campaign_id": f"CMP-{k + 1:02d}",
            "customer_id": s["customer_id"].values[targeted],
            "sent_date": sent_date,
            "product": products[product_idx][targeted],
            "channel": channel[targeted],
            "response": response[targeted],
            "response_date": np.where(response[targeted] == "no_response", np.datetime64("NaT"), sent_date + delay[targeted]),
        }))
    return pd.concat(rows, ignore_index=True)
