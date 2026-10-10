"""FR6 bank insights: where the green opportunity is, who the customers are, and how spending is moving.

Everything here is an aggregate over customers (no customer-level output), built from the live snapshot, the
recommendations and their audited decisions, the impact engine and the categorised transactions:

  overview      headline numbers for the portfolio
  products      opportunity per product: eligible, shown, expected take-ups, expected CO2e avoided, money, payback
  funnel        why products were or were not shown (eligibility, consent, memory, floor, routing, review)
  regions       footprint and opportunity by region
  footprint     estimated customer footprint from visible fuel and energy (transport + home), by income band
  segments      k-means customer segments on behaviour, each named from its strongest traits
  trends        monthly spend by FR2 category, nationally and by region
  memory        what customers said to earlier campaigns
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gcip.config import load_config
from gcip.impact.engine import Factors

CATEGORY_ORDER = ["fuel", "electricity", "gas", "public_transport", "home_improvement", "automotive", "utilities", "other"]
SEGMENT_FEATURES = {
    "est_annual_km": "drive a lot",
    "heat_kwh_12m": "use a lot of heating",
    "income_monthly": "have higher incomes",
    "age": "are older",
    "homeowner_evidence": "own their home",
    "renter_evidence": "rent",
    "ev_owner": "already drive electric",
    "logins_per_month_6m": "bank digitally",
    "investor_evidence": "invest",
    "debt_to_income_est": "carry more debt",
}


def footprint(table: pd.DataFrame) -> pd.DataFrame:
    """Estimated annual footprint (kg CO2e) from what the bank can see: fuel litres, gas, electricity and oil."""
    f = Factors.load(load_config("carbon"))
    diesel = table.get("asset_vehicle_fuel_type", pd.Series(index=table.index, dtype=object)).eq("diesel").to_numpy()
    transport = table["fuel_litres_12m"].fillna(0).to_numpy() * np.where(diesel, f.diesel_l, f.petrol_l)
    home = (table["gas_kwh_12m"].fillna(0).to_numpy() * f.gas_kwh + table["elec_kwh_12m"].fillna(0).to_numpy() * f.grid_kwh
            + table["oil_litres_12m"].fillna(0).to_numpy() * f.oil_l)
    return pd.DataFrame({"customer_id": table["customer_id"].to_numpy(), "transport_kg": transport.round(0),
                         "home_kg": home.round(0), "total_kg": (transport + home).round(0)})


def _products(recs: pd.DataFrame, decisions: pd.DataFrame, impact: pd.DataFrame, names: dict) -> list[dict]:
    out = []
    for product, name in names.items():
        d = decisions[decisions["product"] == product]
        r = recs[recs["product"] == product]
        imp = impact[(impact["product"] == product) & (impact["status"] == "ok")]
        out.append({
            "product": product, "name": name,
            "eligible": int(d["eligible"].fillna(False).sum()),
            "shown": int((r["status"] == "recommended").sum()), "human_review": int((r["status"] == "human_review").sum()),
            "mean_propensity_shown": round(float(r["propensity"].mean()), 4) if len(r) else None,
            "expected_takeups": round(float(r["propensity"].sum()), 1),
            "expected_co2e_tonnes_per_year": round(float(r["expected_co2e_kg_per_year"].sum()) / 1000, 1),
            "co2e_kg_per_year_if_taken_up_median": round(float(r["co2e_avoided_kg_per_year"].median()), 0) if r["co2e_avoided_kg_per_year"].notna().any() else None,
            "money_gbp_per_year_median": round(float(r["running_cost_change_gbp_per_year"].median()), 0) if r["running_cost_change_gbp_per_year"].notna().any() else None,
            "upfront_gbp_median": round(float(r["upfront_cost_gbp"].median()), 0) if r["upfront_cost_gbp"].notna().any() else None,
            "payback_years_median": round(float(r["payback_years"].median()), 1) if r["payback_years"].notna().any() else None,
            "insufficient_data_share": round(1 - len(imp) / max(1, int((impact["product"] == product).sum())), 3),
            "mean_data_quality_shown": round(float(r["data_quality"].mean()), 2) if len(r) else None,
        })
    return out


def _funnel(decisions: pd.DataFrame) -> list[dict]:
    kind = decisions["decision"].str.split(":").str[0].str.replace(r" \(.*\)$", "", regex=True)
    kind = kind.replace({"memory": "memory cool-down or already has it", "routed": "routed to a more specific product"})
    table = pd.crosstab(decisions["product"], kind)
    return [{"product": p, **{k: int(v) for k, v in row.items()}} for p, row in table.iterrows()]


def _regions(table: pd.DataFrame, fp: pd.DataFrame, recs: pd.DataFrame) -> list[dict]:
    d = table[["customer_id", "region"]].merge(fp, on="customer_id")
    r = recs.merge(table[["customer_id", "region"]], on="customer_id")
    out = []
    for region, g in d.groupby("region"):
        rr = r[r["region"] == region]
        top = rr.groupby("product_name")["expected_co2e_kg_per_year"].sum().sort_values(ascending=False)
        out.append({"region": region, "customers": int(len(g)),
                    "footprint_kg_median": round(float(g["total_kg"].median()), 0),
                    "transport_kg_median": round(float(g["transport_kg"].median()), 0),
                    "home_kg_median": round(float(g["home_kg"].median()), 0),
                    "recommendations": int(len(rr)),
                    "expected_co2e_tonnes_per_year": round(float(rr["expected_co2e_kg_per_year"].sum()) / 1000, 1),
                    "expected_co2e_kg_per_customer": round(float(rr["expected_co2e_kg_per_year"].sum()) / max(1, len(g)), 1),
                    "top_product": top.index[0] if len(top) else None})
    return sorted(out, key=lambda x: -x["expected_co2e_kg_per_customer"])


def _segments(table: pd.DataFrame, recs: pd.DataFrame, fp: pd.DataFrame, k: int, seed: int = 7) -> tuple[list[dict], pd.Series]:
    from sklearn.cluster import KMeans
    cols = [c for c in SEGMENT_FEATURES if c in table]
    X = table[cols].astype(float).fillna(0.0)
    for c in ("est_annual_km", "heat_kwh_12m", "income_monthly", "logins_per_month_6m"):
        if c in X:
            X[c] = np.log1p(X[c].clip(lower=0))
    Z = (X - X.mean()) / X.std().replace(0, 1)
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z.to_numpy())
    seg = pd.Series(labels, index=table["customer_id"].to_numpy(), name="segment")
    out = []
    centres = Z.groupby(labels).mean()
    for s in range(k):
        members = table["customer_id"][labels == s]
        traits = centres.loc[s].sort_values(ascending=False)
        name = " and ".join(SEGMENT_FEATURES[c] for c in traits.index[:2] if traits[c] > 0.3) or "average on most measures"
        rr = recs[recs["customer_id"].isin(members)]
        top = rr.groupby("product_name").size().sort_values(ascending=False)
        g = table[table["customer_id"].isin(members)]
        out.append({"segment": int(s), "name": "Customers who " + name, "customers": int(len(members)),
                    "share": round(len(members) / len(table), 3),
                    "traits": {c: round(float(traits[c]), 2) for c in traits.index[:4]},
                    "median_income_monthly": round(float(g["income_monthly"].median()), 0),
                    "median_annual_km": round(float(g["est_annual_km"].median()), 0),
                    "footprint_kg_median": round(float(fp[fp["customer_id"].isin(members)]["total_kg"].median()), 0),
                    "recommendations_per_customer": round(len(rr) / max(1, len(members)), 2),
                    "top_products": [str(x) for x in top.index[:3]],
                    "expected_co2e_tonnes_per_year": round(float(rr["expected_co2e_kg_per_year"].sum()) / 1000, 1)})
    return sorted(out, key=lambda x: -x["customers"]), seg


def monthly_trends(categorised: Path, regions: pd.Series) -> pd.DataFrame:
    """Debit spend per month by FR2 category and region, streamed part by part."""
    parts = []
    for path in sorted(categorised.glob("part-*.parquet")):
        t = pd.read_parquet(path, columns=["customer_id", "booking_datetime", "amount", "credit_debit_indicator", "pred_category"])
        t = t[t["credit_debit_indicator"] == "Debit"]
        t["month"] = t["booking_datetime"].dt.to_period("M").astype(str)
        t["region"] = t["customer_id"].map(regions)
        t["pred_category"] = t["pred_category"].str.lower().str.replace(" ", "_")  # "Public Transport" -> public_transport
        parts.append(t.groupby(["month", "region", "pred_category"], observed=True).agg(
            spend=("amount", "sum"), customers=("customer_id", "nunique")).reset_index())
    out = pd.concat(parts).groupby(["month", "region", "pred_category"], as_index=False).sum()
    return out.rename(columns={"pred_category": "category"})


def build(table: pd.DataFrame, recs: pd.DataFrame, decisions: pd.DataFrame, impact: pd.DataFrame, categorised: Path,
          k: int = 6) -> tuple[dict, pd.DataFrame]:
    """Bank-level insights as one JSON-ready dict, plus the monthly trend table."""
    names = {p: v["name"] for p, v in load_config("products")["products"].items()}
    fp = footprint(table)
    segments, seg = _segments(table, recs, fp, k)
    trends = monthly_trends(categorised, table.set_index("customer_id")["region"])
    national = trends.groupby(["month", "category"], as_index=False)["spend"].sum()
    months = sorted(national["month"].unique())
    shown = recs[recs["status"] == "recommended"]
    income = table[["customer_id", "declared_income_band"]].merge(fp, on="customer_id")
    responses = pd.concat([table[f"last_response_{p}"].rename("response") for p in names]).value_counts()
    insights = {
        "as_of": str(pd.Timestamp(table["as_of"].iloc[0]).date()),
        "overview": {
            "customers": int(len(table)),
            "customers_with_a_recommendation": int(recs["customer_id"].nunique()),
            "recommendations": int(len(recs)), "shown": int(len(shown)),
            "human_review": int((recs["status"] == "human_review").sum()),
            "expected_takeups": round(float(recs["propensity"].sum()), 0),
            "expected_co2e_tonnes_per_year": round(float(recs["expected_co2e_kg_per_year"].sum()) / 1000, 1),
            "co2e_tonnes_per_year_if_all_shown_taken_up": round(float(recs["co2e_avoided_kg_per_year"].clip(lower=0).sum()) / 1000, 0),
            "customer_savings_gbp_per_year_expected": round(float((-recs["running_cost_change_gbp_per_year"].fillna(0)
                                                                    * recs["propensity"]).clip(lower=0).sum()), 0),
            "footprint_kg_median": round(float(fp["total_kg"].median()), 0),
            "footprint_tonnes_total": round(float(fp["total_kg"].sum()) / 1000, 0),
        },
        "products": _products(recs, decisions, impact, names),
        "funnel": _funnel(decisions),
        "regions": _regions(table, fp, recs),
        "footprint": {
            "note": "Estimated from fuel, gas, electricity and oil the bank can see in transactions (DESNZ 2026 factors); "
                    "spend paid from other banks is not visible, so these are lower bounds.",
            "deciles_kg": [round(float(x), 0) for x in fp["total_kg"].quantile(np.linspace(0.1, 0.9, 9))],
            "share_transport": round(float(fp["transport_kg"].sum() / max(1.0, fp["total_kg"].sum())), 3),
            "by_income_band": [{"income_band": str(b), "customers": int(len(g)), "footprint_kg_median": round(float(g["total_kg"].median()), 0)}
                               for b, g in income.groupby("declared_income_band")],
        },
        "segments": segments,
        "trends": {
            "months": months,
            "categories": [c for c in CATEGORY_ORDER if c in set(national["category"])],
            "national_spend": {c: [round(float(national[(national["month"] == m) & (national["category"] == c)]["spend"].sum()), 0)
                                   for m in months] for c in CATEGORY_ORDER if c in set(national["category"])},
        },
        "memory": {"responses": {str(k2): int(v) for k2, v in responses.items()}},
    }
    return insights, trends
