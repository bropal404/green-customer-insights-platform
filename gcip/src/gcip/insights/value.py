"""Business value of suggestions: financial and carbon return, for the customer and for the bank.

Per suggestion (`suggestion_value`):
  customer  running-cost saving a year, upfront cost, lifetime net saving over the product's horizon, net present value
            (Green Book 3.5%), payback, CO2e avoided a year and over the horizon, cost of carbon abatement in GBP per tonne
            (negative: the customer saves money while cutting carbon) and kg CO2e cut per GBP spent upfront
  bank      new lending, lifetime net interest income, fund fees, paperless cost savings, expected credit loss, origination,
            value if taken up, cost of the suggestion (contact by the customer's channel, plus a human review where
            needed), expected value = propensity x value if taken up - cost, financed emissions avoided (PCAF)

`summary` adds these up for the portfolio and per product (ROI = expected net value / campaign cost; carbon return =
tonnes CO2e avoided per GBP 1,000 of campaign cost, and financed emissions avoided per GBP 1m of new lending).

`backtest` checks the value on what really happened: on the held-out test customers (training snapshot, take-up in the
six months after), it contacts the same number of customers chosen by the model, by the rules baseline and at random,
and adds up the take-ups, CO2e and bank value that followed. All assumptions are in configs/business.yaml.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gcip.config import load_config

LENDING = ("GP01", "GP02", "GP03", "GP06")


def _num(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).to_numpy(dtype=float) if column in frame else np.full(len(frame), default)


def suggestion_value(rows: pd.DataFrame, c360: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """One row per suggestion (needs product, propensity and the impact columns) with customer and bank value."""
    cfg = cfg or load_config("business")
    d = rows.merge(c360[["customer_id", "paper_accounts", "balance_investment", "preferred_contact_channel"]]
                   .rename(columns={"preferred_contact_channel": "channel"}), on="customer_id", how="left")
    product = d["product"].to_numpy()
    p = _num(d, "propensity")
    ok = (d["impact_status"] == "ok").to_numpy() if "impact_status" in d else np.ones(len(d), bool)

    # Customer
    saving = -_num(d, "running_cost_change_gbp_per_year")          # positive = cheaper to run
    upfront = _num(d, "upfront_cost_gbp")
    horizon = np.maximum(_num(d, "horizon_years", 1.0), 1.0)
    r = cfg["discount_rate"]
    annuity = (1 - (1 + r) ** -horizon) / r
    lifetime_net = saving * horizon - upfront
    npv = saving * annuity - upfront
    co2_year = np.where(ok, _num(d, "co2e_avoided_kg_per_year"), 0.0)
    co2_life = np.where(ok, _num(d, "co2e_avoided_kg_horizon"), 0.0)
    abatement = np.where(co2_life > 0, -npv / np.maximum(co2_life / 1000, 1e-9), np.nan)
    kg_per_gbp = np.where(upfront > 0, co2_life / np.maximum(upfront, 1e-9), np.nan)

    # Bank
    loan = np.zeros(len(d)); nii = np.zeros(len(d)); ecl = np.zeros(len(d)); orig = np.zeros(len(d))
    for code, spec in cfg["lending"].items():
        m = product == code
        amount = np.full(m.sum(), cfg["ev_price_gbp"] * spec.get("loan_to_value", 1.0)) if code == "GP01" else upfront[m]
        loan[m] = amount
        nii[m] = amount * (spec["rate"] - cfg["cost_of_funds"]) * spec["term_years"] / 2   # amortising: average balance ~ half
        ecl[m] = amount * spec["pd"] * spec["lgd"]
        orig[m] = spec["origination_gbp"]
    inv, pap = cfg["investments"], cfg["paperless"]
    gp05, gp04 = product == "GP05", product == "GP04"
    fees = np.where(gp05, _num(d, "balance_investment") * inv["switch_share"] * inv["fee_margin"] * inv["years"], 0.0)
    orig[gp05] = inv["origination_gbp"]
    savings_bank = np.where(gp04, _num(d, "paper_accounts") * pap["statements_per_account_year"] * pap["cost_per_statement_gbp"]
                            * pap["years"], 0.0)
    value_if_taken = nii + fees + savings_bank - ecl - orig
    contact = d["channel"].map(cfg["contact_cost_gbp"]).fillna(cfg["contact_cost_gbp"]["email"]).to_numpy(dtype=float)
    review = np.where(d["status"].eq("human_review").to_numpy() if "status" in d else False, cfg["human_review_cost_gbp"], 0.0)
    cost = contact + review
    financed = np.where(ok, _num(d, "financed_co2e_avoided_kg_per_year"), 0.0)
    return pd.DataFrame({
        "customer_id": d["customer_id"], "product": product, "propensity": p,
        "customer_saving_gbp_per_year": saving.round(0), "customer_upfront_gbp": upfront.round(0), "horizon_years": horizon,
        "customer_lifetime_net_gbp": lifetime_net.round(0), "customer_npv_gbp": npv.round(0),
        "payback_years": _num(d, "payback_years", np.nan), "co2e_kg_per_year": co2_year, "co2e_kg_lifetime": co2_life,
        "abatement_cost_gbp_per_t": np.round(abatement, 0), "kg_co2e_per_gbp_upfront": np.round(kg_per_gbp, 2),
        "bank_new_lending_gbp": loan.round(0), "bank_nii_gbp": nii.round(0), "bank_fees_gbp": fees.round(0),
        "bank_cost_savings_gbp": savings_bank.round(0), "bank_expected_credit_loss_gbp": ecl.round(0),
        "bank_origination_gbp": orig, "bank_value_if_taken_gbp": value_if_taken.round(0), "suggestion_cost_gbp": cost,
        "bank_expected_value_gbp": (p * value_if_taken - cost).round(2), "financed_co2e_kg_per_year": financed,
    })


def _totals(v: pd.DataFrame) -> dict:
    p = v["propensity"]
    exp = lambda col: float((p * v[col]).sum())  # noqa: E731 - expected value of a column over the suggestions
    campaign = float(v["suggestion_cost_gbp"].sum())
    lending = exp("bank_new_lending_gbp")
    co2_life_t = exp("co2e_kg_lifetime") / 1000
    net = float(v["bank_expected_value_gbp"].sum())
    has_t = v["co2e_kg_lifetime"] > 0
    return {
        "suggestions": int(len(v)), "expected_takeups": round(float(p.sum()), 1),
        "customer": {
            "expected_saving_gbp_per_year": round(exp("customer_saving_gbp_per_year"), 0),
            "expected_lifetime_net_gbp": round(exp("customer_lifetime_net_gbp"), 0),
            "expected_npv_gbp": round(exp("customer_npv_gbp"), 0),
            "median_npv_per_takeup_gbp": round(float(v["customer_npv_gbp"].median()), 0),
            "share_with_positive_npv": round(float((v["customer_npv_gbp"] > 0).mean()), 3),
            "median_payback_years": None if v["payback_years"].isna().all() else round(float(v["payback_years"].median()), 1),
            # Portfolio abatement cost: minus the summed NPV over the summed tonnes (negative = net saving per tonne)
            "abatement_cost_gbp_per_t": None if not has_t.any() else
                round(float(-(v.loc[has_t, "customer_npv_gbp"] * p[has_t]).sum() / ((v.loc[has_t, "co2e_kg_lifetime"] * p[has_t]).sum() / 1000)), 0),
            "expected_co2e_t_per_year": round(exp("co2e_kg_per_year") / 1000, 1),
            "expected_co2e_t_lifetime": round(co2_life_t, 0),
        },
        "bank": {
            "expected_new_lending_gbp": round(lending, 0), "expected_nii_gbp": round(exp("bank_nii_gbp"), 0),
            "expected_fees_gbp": round(exp("bank_fees_gbp"), 0), "expected_cost_savings_gbp": round(exp("bank_cost_savings_gbp"), 0),
            "expected_credit_loss_gbp": round(exp("bank_expected_credit_loss_gbp"), 0),
            "expected_origination_gbp": round(exp("bank_origination_gbp"), 0), "campaign_cost_gbp": round(campaign, 0),
            "expected_net_value_gbp": round(net, 0), "roi": None if campaign == 0 else round(net / campaign, 1),
            "t_co2e_lifetime_per_1k_campaign": None if campaign == 0 else round(co2_life_t / (campaign / 1000), 1),
            "expected_financed_co2e_t_per_year": round(exp("financed_co2e_kg_per_year") / 1000, 1),
            "financed_t_per_year_per_m_lending": None if lending == 0 else round(exp("financed_co2e_kg_per_year") / 1000 / (lending / 1e6), 1),
        },
    }


def grounded(values: pd.DataFrame, cfg: dict | None = None) -> dict:
    """What the bank can stand behind before measuring causation: the value at stake among the customers suggested to
    (propensity-weighted value of their likely take-ups; most would happen without us), the campaign cost, the break-even
    uplift (share of those take-ups the suggestion must cause to pay for itself) and scenarios for that share."""
    cfg = cfg or load_config("business")
    p = values["propensity"]
    at_stake = float((p * values["bank_value_if_taken_gbp"]).sum())
    takeups = float(p.sum())
    co2_life_t = float((p * values["co2e_kg_lifetime"]).sum()) / 1000
    cost = float(values["suggestion_cost_gbp"].sum())
    scenarios = []
    for share in cfg["attribution"]["incremental_share_scenarios"]:
        net = share * at_stake - cost
        scenarios.append({"share_caused": share, "incremental_takeups": round(share * takeups, 1),
                          "incremental_net_value_gbp": round(net, 0), "roi": None if cost == 0 else round(net / cost, 1),
                          "incremental_co2e_t_lifetime": round(share * co2_life_t, 0)})
    return {"value_at_stake_gbp": round(at_stake, 0), "takeups_at_stake": round(takeups, 1), "co2e_t_lifetime_at_stake": round(co2_life_t, 0),
            "campaign_cost_gbp": round(cost, 0), "break_even_share_caused": None if at_stake <= 0 else round(cost / at_stake, 4),
            "scenarios": scenarios}


def attribute(suggestions: pd.DataFrame, openings: pd.DataFrame, clicks: pd.DataFrame | None, window_days: int) -> pd.DataFrame:
    """Link each product opening to a suggestion. suggestions: suggestion_id, customer_id, product, shown_date, group
    (treated or control); openings: customer_id, product, opened_date; clicks: customer_id, product, date (the
    suggestion's call to action, which carries its id into the application). Returns the openings with the suggestion id,
    the group, and how they were linked: direct (clicked, then opened), window (opened within `window_days` of being
    shown), control (in the held-back group), or none."""
    m = openings.merge(suggestions, on=["customer_id", "product"], how="left")
    after = pd.to_datetime(m["opened_date"]) >= pd.to_datetime(m["shown_date"])
    within = (pd.to_datetime(m["opened_date"]) - pd.to_datetime(m["shown_date"])).dt.days <= window_days
    linked = m["suggestion_id"].notna() & after & within
    how = np.where(~linked, "none", np.where(m["group"] == "control", "control", "window"))
    if clicks is not None and len(clicks):
        c = clicks.groupby(["customer_id", "product"])["date"].min().rename("clicked")
        m = m.join(c, on=["customer_id", "product"])
        direct = linked & (m["group"] == "treated") & m["clicked"].notna() & (pd.to_datetime(m["clicked"]) <= pd.to_datetime(m["opened_date"]))
        how = np.where(direct, "direct", how)
    return m.assign(linked_by=how)


def _window(train_date: str, live_date: str) -> str:
    start = pd.Timestamp(train_date) + pd.Timedelta(days=1)
    return f"{start:%-d %b %Y} - {pd.Timestamp(live_date):%-d %b %Y}"


def measure_uplift(heldout: pd.DataFrame, values: pd.DataFrame, labels: pd.DataFrame, thresholds: dict,
                   cfg: dict | None = None, seed: int = 7, train_date: str = "2026-03-31", live_date: str = "2026-09-30") -> dict:
    """The measurement a bank would run, on the backtest: our selection of held-out test customers on 31 March 2026,
    a random control group held back, product openings from the product system in the six months after, linked to the
    suggestions, and the uplift (treated take-up rate minus control) with a 95% interval."""
    cfg = cfg or load_config("business")
    rng = np.random.default_rng(seed)
    test = heldout[heldout["split"] == "test"]
    chosen = pd.concat([g[g["propensity"] >= thresholds[p]["min_propensity"]] for p, g in test.groupby("product")])
    chosen = chosen.assign(group=np.where(rng.random(len(chosen)) < cfg["attribution"]["control_share"], "control", "treated"),
                           shown_date=pd.Timestamp(train_date))
    chosen["suggestion_id"] = f"S{pd.Timestamp(train_date):%Y%m%d}-" + chosen["customer_id"] + "-" + chosen["product"]
    opened = labels[labels["took_up"] == 1][["customer_id", "product", "takeup_date"]].rename(columns={"takeup_date": "opened_date"})
    links = attribute(chosen[["suggestion_id", "customer_id", "product", "shown_date", "group"]], opened, None,
                      window_days=(pd.Timestamp(live_date) - pd.Timestamp(train_date)).days)
    share = cfg["attribution"]["control_share"]
    out = {"note": f"held-out test customers selected by the model on {pd.Timestamp(train_date):%-d %b %Y}, {share:.0%} held back "
                   f"at random, openings from the product system {_window(train_date, live_date)}; the simulator gives contacts "
                   "no causal effect, so the true uplift is zero"}
    rows = {}
    for grp in ("treated", "control"):
        n = int((chosen["group"] == grp).sum())
        k = int(((links["group"] == grp) & links["linked_by"].isin(["window", "direct", "control"])).sum())
        rows[grp] = {"customers": n, "takeups": k, "rate": round(k / n, 4) if n else None}
    out |= rows
    pt, pc = rows["treated"]["rate"] or 0.0, rows["control"]["rate"] or 0.0
    nt, nc = rows["treated"]["customers"], rows["control"]["customers"]
    se = float(np.sqrt(pt * (1 - pt) / max(nt, 1) + pc * (1 - pc) / max(nc, 1)))
    uplift = pt - pc
    v = chosen.merge(values[["customer_id", "product", "bank_value_if_taken_gbp", "co2e_kg_lifetime"]], on=["customer_id", "product"], how="left")
    mean_value = float(v.loc[v["group"] == "treated", "bank_value_if_taken_gbp"].mean())
    out["uplift"] = {"points": round(uplift * 100, 2), "ci95_points": [round((uplift - 1.96 * se) * 100, 2), round((uplift + 1.96 * se) * 100, 2)],
                     "incremental_takeups": round(uplift * nt, 1), "incremental_value_gbp": round(uplift * nt * mean_value, 0)}
    out["linked_takeups_by"] = {k: int(v2) for k, v2 in links["linked_by"].value_counts().items()}
    return out


def summary(values: pd.DataFrame, names: dict) -> dict:
    out = {"portfolio": _totals(values), "grounded": grounded(values), "products": []}
    for product, g in values.groupby("product"):
        t = _totals(g)
        out["products"].append({"product": product, "name": names.get(product, product), "median_abatement_cost_gbp_per_t":
                                None if g["abatement_cost_gbp_per_t"].isna().all() else round(float(g["abatement_cost_gbp_per_t"].median()), 0)} | t)
    return out


def backtest(heldout: pd.DataFrame, values: pd.DataFrame, thresholds: dict, cfg: dict | None = None, seed: int = 7,
             train_date: str = "2026-03-31", live_date: str = "2026-09-30") -> dict:
    """Same contact budget per product (the number of test customers above our threshold), chosen three ways."""
    cfg = cfg or load_config("business")
    test = heldout[heldout["split"] == "test"].merge(
        values.drop(columns=["propensity"]), on=["customer_id", "product"], how="left")
    contact = float(np.mean(list(cfg["contact_cost_gbp"].values())))
    rng = np.random.default_rng(seed)
    rows = []
    for product, g in test.groupby("product"):
        n = int((g["propensity"] >= thresholds[product]["min_propensity"]).sum())
        if n == 0:
            continue
        picks = {"model": g.nlargest(n, "propensity"), "rules": g.nlargest(n, "rules_score") if "rules_score" in g else None,
                 "random": g.iloc[rng.choice(len(g), size=n, replace=False)]}
        for name, chosen in picks.items():
            if chosen is None:
                continue
            took = chosen["took_up"].to_numpy() == 1
            rows.append({"product": product, "strategy": name, "contacted": n, "takeups": int(took.sum()),
                         "co2e_t_lifetime": round(float(chosen.loc[took, "co2e_kg_lifetime"].fillna(0).sum()) / 1000, 1),
                         "bank_value_gbp": round(float(chosen.loc[took, "bank_value_if_taken_gbp"].fillna(0).sum()) - n * contact, 0),
                         "customer_npv_gbp": round(float(chosen.loc[took, "customer_npv_gbp"].fillna(0).sum()), 0)})
    t = pd.DataFrame(rows)
    total = t.groupby("strategy")[["contacted", "takeups", "co2e_t_lifetime", "bank_value_gbp", "customer_npv_gbp"]].sum()
    out = {"note": f"held-out test customers, training snapshot ({pd.Timestamp(train_date):%-d %b %Y}), take-up observed "
                   f"{_window(train_date, live_date)}; same number contacted per product by each strategy",
           "total": {s: {k: float(v) for k, v in row.items()} for s, row in total.iterrows()},
           "products": t.to_dict("records")}
    if "model" in out["total"] and "random" in out["total"] and out["total"]["random"]["takeups"]:
        m, r = out["total"]["model"], out["total"]["random"]
        out["uplift_vs_random"] = {k: round(m[k] / r[k], 2) if r[k] else None for k in ("takeups", "co2e_t_lifetime", "bank_value_gbp")}
    if "model" in out["total"] and "rules" in out["total"] and out["total"]["rules"]["takeups"]:
        m, r = out["total"]["model"], out["total"]["rules"]
        out["uplift_vs_rules"] = {k: round(m[k] / r[k], 2) if r[k] else None for k in ("takeups", "co2e_t_lifetime", "bank_value_gbp")}
    return out
