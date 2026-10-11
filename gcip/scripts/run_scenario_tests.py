#!/usr/bin/env python3
"""Scenario tests: change one thing about real customers and check the platform reacts as it should.

    python scripts/run_scenario_tests.py                  # about 20 s on the shipped 30,000-customer results (demo/)
    python scripts/run_scenario_tests.py --out runs/scenario_tests.json

Each scenario takes customers from the shipped run, perturbs their live Customer 360 / features (or their
transactions), and reruns the real pipeline on the saved models: eligibility rules -> FR3 scoring -> FR4 impact ->
FR5-FR7 suggestions with customer memory. It then checks the decision against what the brief requires. The brief's
scenario families: missing or conflicting data, products that should not be recommended, insufficient impact data,
changed customer circumstances; plus missing values in the transaction feed (null descriptions, merchant codes,
payment types, categories and amounts). Needs requirements.txt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PRODUCTS = ["GP01", "GP02", "GP03", "GP04", "GP05", "GP06"]
CREDIT = ["GP01", "GP02", "GP03", "GP06"]
VEHICLE_EVIDENCE = ["fuel_spend_12m", "fuel_months_12m", "fuel_litres_12m", "fuel_spend_trend", "fuel_visible",
                    "ev_public_kwh_12m", "vehicle_finance_months_12m", "vehicle_finance_monthly", "vehicle_tax_payments_12m",
                    "motor_insurance_months_12m", "car_maintenance_spend_12m", "vehicle_evidence", "ev_owner_evidence",
                    "ev_owner", "motor_loan_months_to_end", "est_annual_km"]


class Platform:
    """The live pipeline on any subset of customers, with the shipped models and thresholds."""

    def __init__(self, run_dir: Path, data_dir: Path):
        from gcip.config import load_config
        from gcip.models.propensity import model_table
        self.cfg = load_config("pipeline")
        self.live = str(self.cfg["as_of"]["live"])
        self.table = model_table(data_dir, self.live)
        self.models = {p: joblib.load(run_dir / "models" / f"propensity_{p}.joblib") for p in PRODUCTS}
        self.thresholds = json.loads((run_dir / "models" / "thresholds.json").read_text())
        self.decisions = pd.read_parquet(run_dir / "recommendations" / "decisions.parquet")
        events = data_dir / "warehouse" / "campaign_responses.parquet"
        from gcip.memory.signals import events_from_crm
        self.events = events_from_crm(pd.read_parquet(events), pd.Timestamp(self.live)) if events.exists() else None

    def rows(self, ids) -> pd.DataFrame:
        return self.table[self.table["customer_id"].isin(ids)].reset_index(drop=True).copy()

    def pick(self, product: str, decision: str = "recommended", n: int = 25, where=None) -> list[str]:
        d = self.decisions[(self.decisions["product"] == product) & (self.decisions["decision"] == decision)]
        ids = d["customer_id"]
        if where is not None:
            ok = set(self.table.loc[where(self.table), "customer_id"])
            ids = ids[ids.isin(ok)]
        return ids.head(n).tolist()

    def run(self, table: pd.DataFrame, extra_events: pd.DataFrame | None = None):
        from gcip.impact.engine import assess
        from gcip.memory.signals import signals
        from gcip.models.propensity import score_live
        from gcip.recommend.engine import recommend
        from gcip.rules.eligibility import evaluate_eligibility
        eligibility = evaluate_eligibility(table)
        scores = pd.concat([score_live(self.models[p], table, eligibility, p) for p in PRODUCTS], ignore_index=True)
        impact = assess(table, PRODUCTS, self.live)
        events = None
        if self.events is not None:
            events = self.events[self.events["customer_id"].isin(table["customer_id"])]
            if extra_events is not None:
                events = pd.concat([events, extra_events], ignore_index=True)
        memory = None if events is None else signals(events, pd.Timestamp(self.live), self.cfg["memory"],
                                                     self.cfg["recommend"]["cooldown_days"])
        recs, decisions = recommend(scores, impact, table, self.cfg["recommend"], self.thresholds, memory=memory)
        return recs, decisions, impact, eligibility


def _decision(decisions, cid, product) -> str:
    row = decisions[(decisions["customer_id"] == cid) & (decisions["product"] == product)]
    return str(row["decision"].iloc[0]) if len(row) else "missing"


def _shown(recs, cid, product, status="recommended") -> bool:
    return bool(((recs["customer_id"] == cid) & (recs["product"] == product) & (recs["status"] == status)).any())


def _result(name, family, expect, customers, failures, detail="") -> dict:
    return {"scenario": name, "brief_family": family, "expectation": expect, "customers": customers,
            "passed": customers - len(failures), "failed": failures[:5], "detail": detail}


# --------------------------------------------------------------------------- scenarios on customers


def baseline_matches(P: Platform) -> dict:
    """Rerunning the pipeline on unchanged customers reproduces the shipped decisions exactly."""
    ids = P.pick("GP01", n=20) + P.pick("GP02", n=10) + P.pick("GP04", n=10)
    _, decisions, _, _ = P.run(P.rows(ids))
    shipped = P.decisions[P.decisions["customer_id"].isin(ids)].set_index(["customer_id", "product"])["decision"]
    now = decisions.set_index(["customer_id", "product"])["decision"].reindex(shipped.index)
    bad = [f"{c} {p}: {a} -> {b}" for (c, p), a, b in zip(shipped.index, shipped, now) if a != b]
    return _result("Unchanged customers reproduce the shipped decisions", "reproducibility",
                   "every decision identical to demo/pipeline/recommendations/decisions.parquet", len(shipped), bad)


def missing_vehicle_record(P: Platform) -> dict:
    """The vehicle record is lost but fuel purchases are still visible: the estimate continues with the documented
    fallback (petrol, DESNZ average car) and a lower data-quality score; nothing crashes."""
    ids = P.pick("GP01", n=25, where=lambda t: t["asset_vehicle_fuel_type"].notna() & (t["fuel_visible"] > 0)
                 & (t["asset_vehicle_fuel_type"] != "bev"))
    before = P.run(P.rows(ids))[2]
    t = P.rows(ids)
    for c in [c for c in t.columns if c.startswith("asset_vehicle_")]:
        t[c] = np.nan
    recs, decisions, impact, _ = P.run(t)
    bad = []
    for cid in ids:
        a = before[(before["customer_id"] == cid) & (before["product"] == "GP01")].iloc[0]
        b = impact[(impact["customer_id"] == cid) & (impact["product"] == "GP01")].iloc[0]
        if b["status"] != "ok" or not np.isfinite(b["co2e_avoided_kg_per_year"]) or b["data_quality"] < a["data_quality"]:
            bad.append(f"{cid}: status {b['status']}, data quality {a['data_quality']} -> {b['data_quality']}")
    return _result("Missing vehicle record, fuel still visible", "missing or conflicting data",
                   "EV impact still estimated (assumed petrol), data quality no better than before, no crash",
                   len(ids), bad)


def no_vehicle_evidence(P: Platform) -> dict:
    """No vehicle record and no motoring spend at all: EV Finance is not suggested."""
    ids = P.pick("GP01", n=25)
    t = P.rows(ids)
    for c in VEHICLE_EVIDENCE + [c for c in t.columns if c.startswith("asset_vehicle_")]:
        if c in t:
            t[c] = np.nan if t[c].dtype == object else 0
    recs, decisions, impact, _ = P.run(t)
    bad = [f"{c}: {_decision(decisions, c, 'GP01')}" for c in ids
           if _shown(recs, c, "GP01") or not _decision(decisions, c, "GP01").startswith("blocked")]
    return _result("No vehicle data at all", "missing or conflicting data",
                   "EV Finance blocked (NO_VEHICLE); never suggested without evidence of a car", len(ids), bad)


def existing_ev(P: Platform) -> dict:
    """The customer already drives a recent battery-electric car (even while fuel purchases continue, e.g. a second
    car): EV Finance is blocked by EV_OWNER_NO_REPLACEMENT. An EV of 8+ years is a replacement candidate."""
    ids = P.pick("GP01", n=25)
    t = P.rows(ids)
    t["ev_owner"], t["asset_vehicle_fuel_type"], t["asset_vehicle_age_years"] = 1, "bev", 3.0
    recs, decisions, _, el = P.run(t)
    bad = [f"{c}: {_decision(decisions, c, 'GP01')}" for c in ids if _shown(recs, c, "GP01")
           or "EV_OWNER_NO_REPLACEMENT" not in str(el[(el.customer_id == c) & (el["product"] == "GP01")]["blocks"].iloc[0])]
    t["asset_vehicle_age_years"] = 10.0
    _, _, _, el_old = P.run(t)
    bad += [f"{c}: a 10-year-old EV is still blocked as recent" for c in ids
            if "EV_OWNER_NO_REPLACEMENT" in str(el_old[(el_old.customer_id == c) & (el_old["product"] == "GP01")]["blocks"].iloc[0])]
    return _result("Customer already owns a recent EV", "products that should not be recommended",
                   "EV Finance blocked (EV_OWNER_NO_REPLACEMENT) for an EV under 8 years; not for a 10-year-old EV",
                   len(ids), bad)


def insufficient_home_data(P: Platform) -> dict:
    """Energy bills disappear: the home estimate falls back to EPC or area data with a worse data-quality score. With no
    bills, no EPC and no area data, no number is given and Green Home Loan is not suggested."""
    ids = P.pick("GP02", n=25)
    before = P.run(P.rows(ids))[2]
    t = P.rows(ids)
    for c in ["gas_kwh_12m", "oil_litres_12m", "elec_kwh_12m", "energy_months_12m", "energy_bills_visible"]:
        if c in t:
            t[c] = 0
    _, _, fallback, _ = P.run(t)
    bad = []
    for cid in ids:
        a = before[(before.customer_id == cid) & (before["product"] == "GP02")].iloc[0]
        b = fallback[(fallback.customer_id == cid) & (fallback["product"] == "GP02")].iloc[0]
        if b["status"] != "ok" or b["data_quality"] < a["data_quality"]:
            bad.append(f"{cid}: bills removed -> {b['status']}, data quality {a['data_quality']} -> {b['data_quality']}")
    for c in ["asset_epc_band", "asset_floor_area_m2", "asset_property_type", "area_gas_dom_mean_kwh"]:
        if c in t:
            t[c] = np.nan
    recs, decisions, impact, _ = P.run(t)
    for cid in ids:
        b = impact[(impact.customer_id == cid) & (impact["product"] == "GP02")].iloc[0]
        if b["status"] != "insufficient_data" or np.isfinite(b["co2e_avoided_kg_per_year"]) or _shown(recs, cid, "GP02"):
            bad.append(f"{cid}: nothing known -> {b['status']}, {_decision(decisions, cid, 'GP02')}")
    return _result("Insufficient emissions data (home)", "insufficient impact data",
                   "bills gone: fallback estimate with worse data quality; nothing known: no number, not suggested",
                   len(ids), bad)


def vulnerable_customer(P: Platform) -> dict:
    """A vulnerability flag appears: credit offers go to human review (never straight to the customer); paperless
    banking and investments are blocked."""
    ids = P.pick("GP01", n=25, where=lambda t: ~t["vulnerability_flag"])
    t = P.rows(ids)
    t["vulnerability_flag"] = True
    recs, decisions, _, _ = P.run(t)
    bad = []
    for cid in ids:
        if any(_shown(recs, cid, p) for p in CREDIT):
            bad.append(f"{cid}: credit offer shown directly")
        if not _shown(recs, cid, "GP01", "human_review") and not _decision(decisions, cid, "GP01").startswith(("memory", "lower priority")):
            bad.append(f"{cid}: EV Finance -> {_decision(decisions, cid, 'GP01')}")
        for p in ("GP04", "GP05"):
            if _shown(recs, cid, p):
                bad.append(f"{cid}: {p} shown to a vulnerable customer")
    app = recs[(recs["customer_id"].isin(ids)) & (recs["status"] == "recommended")]
    detail = f"customer app would show {len(app)} credit offers (expected 0)"
    return _result("Customer becomes vulnerable", "products that should not be recommended",
                   "credit products -> human review; paperless and investments blocked; nothing credit-related in the app",
                   len(ids), bad, detail)


def financial_difficulty(P: Platform) -> dict:
    """Arrears or persistent heavy overdraft use: every credit product is blocked."""
    ids = P.pick("GP01", n=25)
    t = P.rows(ids)
    t["in_arrears"], t["financial_difficulty"] = 1, 1
    recs, decisions, _, _ = P.run(t)
    bad = [f"{c} {p}: {_decision(decisions, c, p)}" for c in ids for p in CREDIT
           if _shown(recs, c, p) or _shown(recs, c, p, "human_review")]
    return _result("Customer in financial difficulty", "products that should not be recommended",
                   "no credit product suggested (IN_ARREARS / FINANCIAL_DIFFICULTY blocks)", len(ids), bad)


def no_consent(P: Platform) -> dict:
    ids = P.pick("GP04", n=25)
    t = P.rows(ids)
    t["marketing_consent"] = False
    recs, decisions, _, _ = P.run(t)
    bad = [c for c in ids if (recs["customer_id"] == c).any()]
    return _result("Marketing consent withdrawn", "products that should not be recommended",
                   "no suggestion of any product", len(ids), bad)


def changed_circumstances(P: Platform) -> dict:
    """Since the last run the customer took EV finance (now held), or told us in the app they already have it."""
    ids = P.pick("GP01", n=24)
    half = len(ids) // 2
    t = P.rows(ids)
    t.loc[t["customer_id"].isin(ids[:half]), "holds_GP01"] = True
    told = pd.DataFrame({"customer_id": ids[half:], "product": "GP01", "response": "already_have",
                         "response_date": pd.Timestamp(P.live), "source": "app"})
    recs, decisions, _, _ = P.run(t, extra_events=told)
    bad = [f"{c}: {_decision(decisions, c, 'GP01')}" for c in ids if _shown(recs, c, "GP01")]
    bad += [f"{c}: {_decision(decisions, c, 'GP01')}" for c in ids[:half] if "ALREADY_HOLDS" not in _decision(decisions, c, "GP01")]
    bad += [f"{c}: {_decision(decisions, c, 'GP01')}" for c in ids[half:] if not _decision(decisions, c, "GP01").startswith("memory")]
    return _result("Changed circumstances: product now held / 'I already have this'", "changed customer circumstances",
                   "EV Finance stops (ALREADY_HOLDS block, or memory: already has it)", len(ids), bad)


# --------------------------------------------------------------------------- missing values in the transaction feed


def null_transaction_fields(run_dir: Path, share: float = 0.05, seed: int = 7) -> dict:
    """5% of descriptions, merchant codes and payment types set to null: no crash; a payment with neither text nor
    merchant code is Other (never Fuel, Electricity or Gas); untouched payments keep their category."""
    from gcip.categorise.model import Categoriser
    cat = Categoriser.load(run_dir / "models" / "categoriser.joblib")
    keys = pd.Series(list(cat.lookup)).sample(4000, random_state=seed).str.split("|", n=2, expand=True)
    parts = pd.DataFrame({"ptype": keys[0], "mcc": keys[1], "amount_band": 3, "round_amount": 0, "direction": "D",
                          "norm_text": keys[2]}).reset_index(drop=True)
    clean = cat.predict(parts, cat.combine(cat.model_proba(parts)))
    rng = np.random.default_rng(seed)
    hit = {c: rng.random(len(parts)) < share for c in ("norm_text", "mcc", "ptype")}
    dirty = parts.copy().astype(object)
    for c, m in hit.items():
        dirty.loc[m, c] = None
    try:
        pred = cat.predict(dirty, cat.combine(cat.model_proba(dirty)))
    except Exception as e:  # noqa: BLE001
        return _result("Null descriptions, merchant codes and payment types (5%)", "missing or conflicting data",
                       "no crash", len(parts), [f"crashed: {e!r}"])
    touched = hit["norm_text"] | hit["mcc"] | hit["ptype"]
    blank = hit["norm_text"] & (hit["mcc"] | (parts["mcc"] == "").to_numpy())
    bad = [f"row {i}: no text or merchant code -> {pred['pred_category'][i]}" for i in np.flatnonzero(blank)
           if pred["pred_category"][i] in ("Fuel", "Electricity", "Gas")]
    bad += [f"row {i}: untouched row changed" for i in np.flatnonzero(~touched)
            if pred["pred_sub_category"][i] != clean["pred_sub_category"][i]]
    detail = (f"{int(touched.sum())} rows with a null field; {int(blank.sum())} with neither text nor merchant code, all "
              f"Other; {int((~touched).sum())} untouched rows unchanged")
    return _result("Null descriptions, merchant codes and payment types (5%)", "missing or conflicting data",
                   "no crash; nothing to read -> Other, never fuel or energy; other rows unchanged", len(parts), bad, detail)


def null_categories_and_amounts(share: float = 0.05, seed: int = 7) -> dict:
    """A categorised feed where 5% of categories and 1% of amounts are null: features build without error, null
    rows add nothing to fuel litres or energy kWh, and coverage (months seen) is never overstated."""
    import yaml

    from gcip.config import CONFIG_DIR
    from gcip.features.store import Reference, part_features
    rng = np.random.default_rng(seed)
    as_of = pd.Timestamp("2026-09-30")
    rows = []
    for k in range(40):
        cid = f"S{k:04d}"
        for w in range(52):
            rows.append((cid, as_of - pd.Timedelta(days=7 * w + 1), round(float(rng.uniform(35, 70)), 2), "DEB",
                         "SHELL PETROL", "Fuel", "road_fuel"))
        for m in range(12):
            day = as_of - pd.DateOffset(months=m, days=3)
            rows.append((cid, day, round(float(rng.uniform(60, 110)), 2), "DD", "OCTOPUS ELEC", "Electricity", "electricity_bill"))
            rows.append((cid, day, round(float(rng.uniform(50, 140)), 2), "DD", "BRITISH GAS", "Gas", "gas_bill"))
    part = pd.DataFrame(rows, columns=["customer_id", "booking_datetime", "amount", "payment_type", "transaction_information",
                                       "pred_category", "pred_sub_category"])
    part["credit_debit_indicator"] = "Debit"
    ref, carbon = Reference.load(), yaml.safe_load((CONFIG_DIR / "carbon.yaml").read_text())
    clean = part_features(part, as_of, ref, carbon).set_index("customer_id")
    dirty = part.copy()
    m = rng.random(len(dirty)) < share
    dirty.loc[m, ["pred_category", "pred_sub_category"]] = None
    dirty.loc[rng.random(len(dirty)) < 0.01, "amount"] = np.nan
    try:
        out = part_features(dirty, as_of, ref, carbon).set_index("customer_id")
    except Exception as e:  # noqa: BLE001
        return _result("Null categories and amounts in the feed (5% / 1%)", "missing or conflicting data", "no crash",
                       part["customer_id"].nunique(), [f"crashed: {e!r}"])
    bad = []
    for col in ("fuel_litres_12m", "elec_kwh_12m", "gas_kwh_12m"):
        if out[col].isna().any():
            bad.append(f"{col} has missing values")
        change = (out[col] - clean[col]).abs() / clean[col].replace(0, np.nan)
        if change.max() > 0.25:
            bad.append(f"{col}: changed by up to {100 * change.max():.0f}%")
    for col in ("fuel_months_12m", "energy_months_12m"):
        if col in out and (out[col] > clean[col]).any():
            bad.append(f"{col} overstated")
    detail = (f"median change: litres {100 * ((out['fuel_litres_12m'] - clean['fuel_litres_12m']).abs() / clean['fuel_litres_12m']).median():.1f}%, "
              f"electricity {100 * ((out['elec_kwh_12m'] - clean['elec_kwh_12m']).abs() / clean['elec_kwh_12m']).median():.1f}%, "
              f"gas {100 * ((out['gas_kwh_12m'] - clean['gas_kwh_12m']).abs() / clean['gas_kwh_12m']).median():.1f}%")
    return _result("Null categories and amounts in the feed (5% / 1%)", "missing or conflicting data",
                   "features build; null rows add nothing; no missing values; months seen never overstated",
                   part["customer_id"].nunique(), bad, detail)


def run_all(run_dir: Path, data_dir: Path) -> list[dict]:
    P = Platform(run_dir, data_dir)
    results = [f(P) for f in (baseline_matches, missing_vehicle_record, no_vehicle_evidence, existing_ev,
                              insufficient_home_data, vulnerable_customer, financial_difficulty, no_consent,
                              changed_circumstances)]
    results += [null_transaction_fields(run_dir), null_categories_and_amounts()]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default="demo/pipeline")
    parser.add_argument("--data-dir", default="demo/data")
    parser.add_argument("--out", default="runs/scenario_tests.json")
    args = parser.parse_args()
    t0 = time.time()
    results = run_all(ROOT / args.run_dir, ROOT / args.data_dir)
    for r in results:
        ok = not r["failed"]
        print(f"{'PASS' if ok else 'FAIL'}  {r['scenario']:<64} {r['passed']}/{r['customers']}  {r['expectation']}")
        if r["detail"]:
            print(f"      {r['detail']}")
        for f in r["failed"]:
            print(f"      {f}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seconds": round(time.time() - t0, 1), "results": results}, indent=2, default=str))
    failed = sum(bool(r["failed"]) for r in results)
    print(f"\n{len(results) - failed} of {len(results)} scenarios pass ({time.time() - t0:.0f} s); written to {args.out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
