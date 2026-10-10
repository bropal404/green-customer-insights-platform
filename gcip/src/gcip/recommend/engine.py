"""FR5-FR7 recommendations: at most three per customer, each with why (reason codes and evidence), how sure
(confidence), what it would change (carbon and money) and the data timestamp.

Order of decisions, each recorded so a recommendation can be explained or audited:
  1. eligibility blocks rule a product out (never recommended, FR7);
  2. no marketing consent: nothing is sent;
  3. memory: 'already_have' suppresses; 'not_interested' and 'dismissed' start cool-downs; 'clicked' raises priority;
  4. thresholds (configs/pipeline.yaml `recommend`): the impact can be estimated; calibrated propensity is at least
     `min_lift` x the product's take-up rate (models/thresholds.json); confidence is not low; the avoided CO2e is worth a
     green suggestion;
  5. GP06 Green Loans steps aside when GP01, GP02 or GP03 is recommended (products.yaml routing);
  6. rank by propensity x confidence (customer view), at most `max_per_customer`; the business view (expected kg CO2e
     avoided) is kept too;
  7. credit products for customers with a vulnerability flag go to human review instead of being sent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gcip.config import load_config

CREDIT = {"GP01", "GP02", "GP03", "GP06"}
ROUTED_BY = {"GP06": {"GP01", "GP02", "GP03"}}


def recommend(live_scores: pd.DataFrame, impact: pd.DataFrame, c360: pd.DataFrame, cfg: dict,
              thresholds: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (recommendations, decisions): the shown recommendations and one audited decision row per customer x product.
    `thresholds` is models/thresholds.json ({product: {"base_rate", "min_propensity"}}) from training."""
    products = load_config("products")["products"]
    as_of = pd.Timestamp(c360["as_of"].iloc[0])
    d = live_scores.merge(impact.rename(columns={"status": "impact_status"}), on=["customer_id", "product"], how="left")
    d = d.merge(c360[["customer_id", "marketing_consent", "vulnerability_flag", "preferred_contact_channel"]
                     + [f"last_response_{p}" for p in products] + [f"last_response_date_{p}" for p in products]],
                on="customer_id", how="left")
    response = np.full(len(d), "none", dtype=object)
    response_date = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    for product in products:
        mask = (d["product"] == product).to_numpy()
        response[mask] = d.loc[mask, f"last_response_{product}"].fillna("none").to_numpy()
        response_date[mask] = pd.to_datetime(d.loc[mask, f"last_response_date_{product}"])
    days = (as_of - response_date).dt.days.to_numpy()
    cool = cfg["cooldown_days"]
    decision = np.full(len(d), "", dtype=object)
    decision[~d["eligible"].fillna(False).to_numpy()] = "blocked: " + d["blocks"].fillna("").astype(str).to_numpy()[~d["eligible"].fillna(False).to_numpy()]
    open_ = decision == ""
    no_consent = open_ & ~d["marketing_consent"].fillna(False).to_numpy()
    decision[no_consent] = "no marketing consent"
    open_ = decision == ""
    already = open_ & (response == "already_have")
    decision[already] = "memory: already has it"
    for kind, limit in cool.items():
        mask = (decision == "") & (response == kind) & (np.nan_to_num(days, nan=1e9) < limit)
        decision[mask] = f"memory: {kind.replace('_', ' ')} {int(limit)}-day cool-down"
    if cfg.get("require_impact", False):
        no_impact = (decision == "") & (d["impact_status"] != "ok").to_numpy()
        decision[no_impact] = "impact cannot be estimated"
    if thresholds:
        cut = d["product"].map({p: t["min_propensity"] for p, t in thresholds.items()}).fillna(1.0).to_numpy()
    else:
        cut = np.full(len(d), cfg.get("min_propensity", 0.0))
    d["propensity_threshold"] = cut
    low = (decision == "") & (d["propensity"].fillna(0).to_numpy() < cut)
    decision[low] = "below the propensity threshold"
    unsure = (decision == "") & (d["confidence"].fillna(0).to_numpy() < cfg.get("min_confidence", 0.0))
    decision[unsure] = "low confidence"
    small = ((decision == "") & ~d["product"].isin(cfg.get("co2e_exempt", [])).to_numpy()
             & (d["co2e_avoided_kg_per_year"].fillna(0).to_numpy() < cfg.get("min_co2e_kg_per_year", 0.0)))
    decision[small] = "impact too small"
    priority = d["propensity"].fillna(0).to_numpy() * d["confidence"].fillna(0).to_numpy()
    priority = priority + np.where(response == "clicked", cfg["clicked_priority_boost"], 0.0)
    d["priority"] = priority.round(5)
    d["expected_co2e_kg_per_year"] = (d["propensity"].fillna(0) * d["co2e_avoided_kg_per_year"].fillna(0)).round(1)
    d["decision"] = decision
    # Routing: GP06 steps aside when another credit product is recommended for the same customer.
    candidates = d[d["decision"] == ""].sort_values(["customer_id", "priority"], ascending=[True, False])
    chosen = candidates.groupby("customer_id").head(cfg["max_per_customer"] + 1)
    shown_products = chosen.groupby("customer_id")["product"].apply(set)
    for product, routers in ROUTED_BY.items():
        routed = (d["decision"] == "") & (d["product"] == product) & d["customer_id"].map(
            lambda c: bool(shown_products.get(c, set()) & routers)).to_numpy()
        d.loc[routed, "decision"] = "routed: a more specific green product is shown"
    candidates = d[d["decision"] == ""].sort_values(["customer_id", "priority"], ascending=[True, False])
    candidates = candidates.assign(rank=candidates.groupby("customer_id").cumcount() + 1)
    over = candidates["rank"] > cfg["max_per_customer"]
    d.loc[candidates.index[over], "decision"] = "lower priority than three other products"
    shown = candidates[~over].copy()
    review = shown["product"].isin(CREDIT) & shown["vulnerability_flag"].fillna(False)
    shown["status"] = np.where(review, "human_review", "recommended")
    d.loc[shown.index, "decision"] = np.where(review, "human review (vulnerability flag)", "recommended")
    shown["product_name"] = shown["product"].map({k: v["name"] for k, v in products.items()})
    shown["customer_action"] = shown["product"].map({k: v["customer_action"] for k, v in products.items()})
    shown["data_timestamp"] = as_of.strftime("%Y-%m-%d")
    keep = ["customer_id", "rank", "product", "product_name", "status", "propensity", "propensity_threshold", "confidence",
            "priority", "reason_codes",
            "evidence", "measure", "co2e_avoided_kg_per_year", "co2e_avoided_kg_horizon", "horizon_years", "data_quality",
            "financed_co2e_avoided_kg_per_year", "running_cost_change_gbp_per_year", "upfront_cost_gbp", "payback_years",
            "impact_status", "method", "note", "expected_co2e_kg_per_year", "customer_action", "preferred_contact_channel",
            "data_timestamp"]
    decisions = d[["customer_id", "product", "eligible", "propensity", "propensity_threshold", "confidence", "priority",
                   "co2e_avoided_kg_per_year", "expected_co2e_kg_per_year", "decision"]]
    return shown[[c for c in keep if c in shown]].reset_index(drop=True), decisions.reset_index(drop=True)
