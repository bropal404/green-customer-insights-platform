"""FR5-FR7 recommendations: at most three per customer, each with why (reason codes and evidence), how sure
(confidence), what it would change (carbon and money) and the data timestamp.

Order of decisions, each recorded so a recommendation can be explained or audited:
  1. eligibility blocks rule a product out (never recommended, FR7);
  2. no marketing consent: nothing is sent;
  3. memory (gcip.memory.signals, all contacts and responses in the last year): already has it stops the product;
     "dismissed" and "not interested" start cool-downs; repeated contacts with no engagement pause it; every response
     moves a memory weight that multiplies the priority (interest up, being ignored or turned down down);
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
              thresholds: dict | None = None, memory: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (recommendations, decisions): the shown recommendations and one audited decision row per customer x product.
    `thresholds` is models/thresholds.json ({product: {"base_rate", "min_propensity"}}) from training. `memory` is
    gcip.memory.signals output for these customers; without it, the Customer 360's last responses are used."""
    from gcip.memory.signals import events_from_c360, signals

    products = load_config("products")["products"]
    as_of = pd.Timestamp(c360["as_of"].iloc[0])
    if memory is None:
        memory = signals(events_from_c360(c360), as_of, load_config("pipeline")["memory"], cfg["cooldown_days"])
    d = live_scores.merge(impact.rename(columns={"status": "impact_status"}), on=["customer_id", "product"], how="left")
    d = d.merge(c360[["customer_id", "marketing_consent", "vulnerability_flag", "preferred_contact_channel"]],
                on="customer_id", how="left")
    d = d.join(memory[["weight", "state", "until", "streak", "last_clicked"]], on=["customer_id", "product"])
    d["memory_weight"] = pd.to_numeric(d["weight"], errors="coerce").fillna(1.0).astype(float).round(3)
    state = d["state"].fillna("").to_numpy()
    decision = np.full(len(d), "", dtype=object)
    decision[~d["eligible"].fillna(False).to_numpy()] = "blocked: " + d["blocks"].fillna("").astype(str).to_numpy()[~d["eligible"].fillna(False).to_numpy()]
    open_ = decision == ""
    no_consent = open_ & ~d["marketing_consent"].fillna(False).to_numpy()
    decision[no_consent] = "no marketing consent"
    held = (decision == "") & (state != "")
    until = pd.to_datetime(d["until"]).dt.strftime("%Y-%m-%d").fillna("").to_numpy()
    decision[held] = "memory: " + state[held] + np.where(until[held] != "", " (until " + until[held] + ")", "")
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
    priority = d["propensity"].fillna(0).to_numpy() * d["confidence"].fillna(0).to_numpy() * d["memory_weight"].to_numpy()
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
    # Human review comes from the eligibility rules (products.yaml `human_review`, e.g. VULNERABLE on credit products);
    # without that column, credit products for customers with a vulnerability flag.
    review = (shown["human_review"].fillna(False).astype(bool) if "human_review" in shown
              else shown["product"].isin(CREDIT) & shown["vulnerability_flag"].fillna(False))
    shown["status"] = np.where(review, "human_review", "recommended")
    d.loc[shown.index, "decision"] = np.where(review, "human review (vulnerability flag)", "recommended")
    # Memory as a reason: earlier interest in the product (it raised the priority) is stated with its date.
    clicked = pd.to_datetime(shown["last_clicked"])
    shown["reason_codes"] = [(list(rc) if rc is not None else []) + (["EARLIER_INTEREST"] if pd.notna(c) else [])
                             for rc, c in zip(shown["reason_codes"], clicked)]
    shown["evidence"] = [(list(ev) if ev is not None else []) + ([f"you showed interest in this on {c:%Y-%m-%d}"] if pd.notna(c) else [])
                         for ev, c in zip(shown["evidence"], clicked)]
    shown["product_name"] = shown["product"].map({k: v["name"] for k, v in products.items()})
    # Every suggestion has an id: the call to action carries it into the application, so a take-up can be linked back.
    shown["suggestion_id"] = "S" + as_of.strftime("%Y%m%d") + "-" + shown["customer_id"] + "-" + shown["product"]
    shown["opportunity"] = shown["product"].map({k: v.get("opportunity", k) for k, v in products.items()})
    shown["customer_action"] = shown["product"].map({k: v["customer_action"] for k, v in products.items()})
    shown["data_timestamp"] = as_of.strftime("%Y-%m-%d")
    keep = ["suggestion_id", "customer_id", "rank", "opportunity", "product", "product_name", "status", "propensity", "propensity_threshold", "confidence",
            "memory_weight", "priority", "reason_codes",
            "evidence", "measure", "co2e_avoided_kg_per_year", "co2e_avoided_kg_horizon", "horizon_years", "data_quality",
            "financed_co2e_avoided_kg_per_year", "running_cost_change_gbp_per_year", "upfront_cost_gbp", "payback_years",
            "impact_status", "method", "note", "expected_co2e_kg_per_year", "customer_action", "preferred_contact_channel",
            "data_timestamp"]
    decisions = d[["customer_id", "product", "eligible", "propensity", "propensity_threshold", "confidence", "memory_weight", "priority",
                   "co2e_avoided_kg_per_year", "expected_co2e_kg_per_year", "decision"]]
    return shown[[c for c in keep if c in shown]].reset_index(drop=True), decisions.reset_index(drop=True)
