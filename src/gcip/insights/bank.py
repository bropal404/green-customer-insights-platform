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
            "confidence_mix": {lvl: int((r["data_quality"] == score).sum())
                               for score, lvl in ((2, "High"), (3, "Good"), (4, "Partial"), (5, "Low"))},
        })
    return out


def _funnel(decisions: pd.DataFrame) -> list[dict]:
    kind = decisions["decision"].str.split(":").str[0].str.replace(r" \(.*\)$", "", regex=True)
    kind = kind.replace({"memory": "customer memory (paused or stopped)", "routed": "routed to a more specific product"})
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


def monthly_trends(categorised: Path, regions: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Debit spend per month by FR2 category and region, and per month by sub-category (with the number of customers
    spending in it), streamed part by part (each part holds whole customers, so counts add up across parts)."""
    by_cat, by_sub = [], []
    for path in sorted(categorised.glob("part-*.parquet")):
        t = pd.read_parquet(path, columns=["customer_id", "booking_datetime", "amount", "credit_debit_indicator", "pred_category",
                                           "pred_sub_category"])
        t = t[t["credit_debit_indicator"] == "Debit"]
        t["month"] = t["booking_datetime"].dt.to_period("M").astype(str)
        t["region"] = t["customer_id"].map(regions)
        t["pred_category"] = t["pred_category"].str.lower().str.replace(" ", "_")  # "Public Transport" -> public_transport
        by_cat.append(t.groupby(["month", "region", "pred_category"], observed=True).agg(
            spend=("amount", "sum"), customers=("customer_id", "nunique")).reset_index())
        by_sub.append(t.groupby(["month", "pred_sub_category"], observed=True).agg(
            spend=("amount", "sum"), transactions=("amount", "size"), customers=("customer_id", "nunique")).reset_index())
    cat = pd.concat(by_cat).groupby(["month", "region", "pred_category"], as_index=False).sum().rename(columns={"pred_category": "category"})
    sub = pd.concat(by_sub).groupby(["month", "pred_sub_category"], as_index=False).sum().rename(columns={"pred_sub_category": "sub_category"})
    return cat, sub


GREEN_SIGNALS = {"ev_public_charging": "EV public charging", "energy_installers": "Energy installers",
                 "micromobility": "Bike and scooter hire", "rail": "Rail", "bus_coach": "Bus and coach",
                 "heating_oil_lpg": "Heating oil and LPG"}


def trend_details(sub: pd.DataFrame, months: list[str], holdings: pd.DataFrame | None, names: dict) -> dict:
    """Trend views for the bank: green spending signals, fuel spend vs litres, home energy bills, green take-ups."""
    from gcip.config import REFERENCE_DIR

    def series(code, column):
        g = sub[sub["sub_category"] == code].set_index("month")[column]
        return [round(float(g.get(m, 0.0)), 0) for m in months]

    fuel = pd.read_csv(REFERENCE_DIR / "fuel_prices_weekly.csv", parse_dates=["week_start"])
    fuel["month"] = fuel["week_start"].dt.to_period("M").astype(str)
    ppl = fuel.groupby("month")[["petrol_ppl", "diesel_ppl"]].mean()
    road = series("road_fuel", "spend")
    blended = [float(ppl.loc[m].mean()) / 100 if m in ppl.index else float("nan") for m in months]
    out = {
        "green_signals": {label: {"customers": series(code, "customers"), "spend": series(code, "spend")}
                          for code, label in GREEN_SIGNALS.items()},
        "fuel": {"spend": road, "pump_price_ppl": [round(b * 100, 1) for b in blended],
                 "litres": [round(sp / b, 0) if b == b and b > 0 else None for sp, b in zip(road, blended)]},
        "energy_bills": {"Electricity": [a + b for a, b in zip(series("electricity_bill", "spend"), series("electricity_prepayment", "spend"))],
                         "Gas": [a + b for a, b in zip(series("gas_bill", "spend"), series("gas_prepayment", "spend"))],
                         "Dual fuel": series("dual_fuel_energy", "spend")},
    }
    if holdings is not None:
        h = holdings[holdings["green_product"].fillna("") != ""].copy()
        h["month"] = pd.to_datetime(h["start_date"]).dt.to_period("M").astype(str)
        g = h[h["month"].isin(months)].groupby(["month", "green_product"]).size()
        out["takeups"] = {names.get(p, p): [int(g.get((m, p), 0)) for m in months] for p in names}
    return out


def customer_monthly(categorised: Path, as_of: pd.Timestamp, months: int = 12) -> pd.DataFrame:
    """Each customer's debit spend per month and category over the last `months` months (what the customer app shows)."""
    start = (as_of - pd.DateOffset(months=months - 1)).to_period("M").to_timestamp()
    parts = []
    for path in sorted(categorised.glob("part-*.parquet")):
        t = pd.read_parquet(path, columns=["customer_id", "booking_datetime", "amount", "credit_debit_indicator", "pred_category"])
        t = t[(t["credit_debit_indicator"] == "Debit") & (t["booking_datetime"] >= start)
              & (t["booking_datetime"] < as_of + pd.Timedelta(days=1))]
        t = t.assign(month=t["booking_datetime"].dt.to_period("M").astype(str),
                     category=t["pred_category"].str.lower().str.replace(" ", "_"))
        parts.append(t.groupby(["customer_id", "month", "category"], as_index=False)["amount"].sum())
    out = pd.concat(parts, ignore_index=True).rename(columns={"amount": "spend"})
    out["spend"] = out["spend"].round(2).astype("float32")
    return out.astype({"customer_id": "category", "month": "category", "category": "category"})


def build(table: pd.DataFrame, recs: pd.DataFrame, decisions: pd.DataFrame, impact: pd.DataFrame, categorised: Path,
          k: int = 6, holdings: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    """Bank-level insights as one JSON-ready dict, plus the monthly trend table."""
    names = {p: v["name"] for p, v in load_config("products")["products"].items()}
    fp = footprint(table)
    segments, seg = _segments(table, recs, fp, k)
    trends, by_sub = monthly_trends(categorised, table.set_index("customer_id")["region"])
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
            # net of products that cost more to run (e.g. heat pumps at 2026 prices); same as gcip.insights.value
            "customer_savings_gbp_per_year_expected": round(float((-recs["running_cost_change_gbp_per_year"].fillna(0)
                                                                    * recs["propensity"]).sum()), 0),
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
        } | trend_details(by_sub, months, holdings, names),
        "memory": {"responses": {str(k2): int(v) for k2, v in responses.items()}},
    }
    return insights, trends
