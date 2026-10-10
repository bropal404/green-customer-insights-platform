#!/usr/bin/env python3
"""Dashboard and API over a finished pipeline run (standard library only).

    python scripts/serve_dashboard.py --run-dir runs/pipeline [--port 8050]     then open http://localhost:8050

GET  /api/overview                         bank insights, model metrics, explanation, impact and memory checks
GET  /api/trends?region=London             monthly spend by category (national when no region)
GET  /api/customers?q=&product=&status=    customer list (paged with limit and offset)
GET  /api/customer/<id>                    staff view: the customer document (360, features, recommendations, decisions, memory)
GET  /api/me/<id>                          customer app: only what the customer sees (footprint, spending, suggestions shown
                                           to them in plain language; no scores, codes or flags)
POST /api/customer/<id>/feedback           {"product": "GP01", "response": "not_interested"}: stored in memory, then the
                                           customer's recommendations are made again
POST /api/customer/<id>/explain            {"product": "GP01"}: rephrase that explanation with the local LLM, guarded
POST /api/memory/reset                     forget the feedback given in the dashboard
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import warnings
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from gcip.config import load_config  # noqa: E402
from gcip.explain.run import build_items, compose, explain  # noqa: E402
from gcip.insights.bank import footprint  # noqa: E402
from gcip.memory.replay import RESPONSES, apply_events, empty_events  # noqa: E402
from gcip.recommend.engine import recommend  # noqa: E402
from gcip.serve.documents import clean, document  # noqa: E402

LIVE = "2026-09-30"
FEEDBACK = set(RESPONSES) | {"already_have", "viewed"}
SPEND_CATEGORIES = ["fuel", "electricity", "gas", "public_transport", "home_improvement", "automotive", "utilities", "other"]


class State:
    """Everything the dashboard reads, loaded once; feedback updates one customer at a time."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        data = run_dir / "data"
        self.cfg = load_config("pipeline")
        self.lock = threading.Lock()
        self.c360 = pd.read_parquet(data / f"customer_360_{LIVE}.parquet")
        self.features = pd.read_parquet(data / f"features_{LIVE}.parquet")
        self.scores = pd.read_parquet(run_dir / "scores" / "live_scores.parquet")
        self.impact = pd.read_parquet(run_dir / "impact" / "impact.parquet")
        self.recs = pd.read_parquet(run_dir / "recommendations" / "recommendations.parquet")
        self.decisions = pd.read_parquet(run_dir / "recommendations" / "decisions.parquet")
        self.explanations = pd.read_parquet(run_dir / "explanations" / "explanations.parquet")
        self.trends = pd.read_parquet(run_dir / "insights" / "monthly_trends.parquet")
        self.thresholds = json.loads((run_dir / "models" / "thresholds.json").read_text())
        self.parts = []  # (first customer, last customer, path): each categorised part holds whole customers
        for path in sorted((data / "categorised").glob("part-*.parquet")):
            col = pq.ParquetFile(path).schema_arrow.get_field_index("customer_id")
            stats = [rg.column(col).statistics for rg in (pq.ParquetFile(path).metadata.row_group(i)
                                                          for i in range(pq.ParquetFile(path).metadata.num_row_groups))]
            self.parts.append((min(st.min for st in stats), max(st.max for st in stats), path))
        self.events_path = run_dir / "memory" / "dashboard_events.parquet"
        self.events = pd.read_parquet(self.events_path) if self.events_path.exists() else empty_events()
        self.overview = self._overview()
        self.index = self._index()
        llm = self.cfg["llm"]
        self.explainer = None
        if llm.get("enabled", True):
            from gcip.explain.llm import LocalExplainer
            self.explainer = LocalExplainer(llm["model"], llm["max_new_tokens"], llm["dtype"])  # loads on first use
        for customer_id in self.events["customer_id"].unique():
            self._recompute(customer_id)

    def _json(self, *parts: str):
        path = self.run_dir.joinpath(*parts)
        return json.loads(path.read_text()) if path.exists() else None

    def _overview(self) -> dict:
        sens = self.run_dir / "metrics" / "impact_sensitivity.csv"
        report = self._json("metrics", "propensity_report.json") or {}
        return clean({
            "insights": self._json("insights", "bank_insights.json"),
            "models": report.get("products"), "ablations": report.get("ablations"),
            "explanations": self._json("metrics", "explanations.json"),
            "impact_truth_recovery": self._json("metrics", "impact_truth_recovery.json"),
            "impact_sensitivity": pd.read_csv(sens).to_dict("records") if sens.exists() else [],
            "memory_replay": self._json("metrics", "memory_replay.json"),
            "data": self._data_status(),
            "stages": self._json("logs", "stage_timings.json") or {},
            "done": sorted(p.stem for p in (self.run_dir / "_checkpoints").glob("*.done")),
            "regions": sorted(self.c360["region"].dropna().unique().tolist()),
            "products": {k: v["name"] for k, v in load_config("products")["products"].items()},
        })

    def _data_status(self) -> dict:
        meta = self.run_dir / "data" / "meta"
        out = {"customers": int(len(self.c360))}
        if (meta / "data_validation.json").exists():
            v = json.loads((meta / "data_validation.json").read_text())
            out["validation"] = f"{v.get('passed')}/{v.get('total')}"
        if (meta / "ingestion_report.json").exists():
            r = json.loads((meta / "ingestion_report.json").read_text())
            audit = r.get("audit", {})
            tables = audit.get("tables", {})
            out["audit_tables"] = f"{sum(t.get('exact_match', False) for t in tables.values())}/{len(tables)}"
            tx = audit.get("transactions", {})
            out["transactions"] = tx.get("rows_expected") or tx.get("expected")
            out["ingestion_summary"] = r.get("summary", [])
        return out

    def _index(self) -> pd.DataFrame:
        top = self.recs.sort_values("rank").groupby("customer_id").agg(
            recommendations=("product", "size"), top_product=("product_name", "first"),
            products=("product", lambda s: ",".join(s)), review=("status", lambda s: bool((s == "human_review").any())),
            expected_co2e=("expected_co2e_kg_per_year", "sum"))
        base = self.c360[["customer_id", "age", "region", "la_name", "declared_income_band", "customer_tenure_years"]]
        out = base.merge(top, left_on="customer_id", right_index=True, how="left")
        out["recommendations"] = out["recommendations"].fillna(0).astype(int)
        return out

    def _recompute(self, customer_id: str) -> None:
        """Make this customer's recommendations again from the memory store (after feedback)."""
        c = self.c360[self.c360["customer_id"] == customer_id]
        ev = self.events[self.events["customer_id"] == customer_id]
        recs, decisions = recommend(self.scores[self.scores["customer_id"] == customer_id],
                                    self.impact[self.impact["customer_id"] == customer_id], apply_events(c, ev), self.cfg["recommend"],
                                    self.thresholds)
        old = self.explanations[self.explanations["customer_id"] == customer_id]
        expl = explain(build_items(recs)) if len(recs) else old.iloc[0:0]
        if len(expl) and len(old):  # keep LLM rephrasings already made for products still shown
            kept = old[old["source"] == "llm"].set_index("product")
            for j, product in enumerate(expl["product"]):
                if product in kept.index:
                    expl.loc[expl.index[j], ["explanation", "source"]] = kept.loc[product, ["explanation", "source"]].to_numpy()
        self.recs = pd.concat([self.recs[self.recs["customer_id"] != customer_id], recs], ignore_index=True)
        self.decisions = pd.concat([self.decisions[self.decisions["customer_id"] != customer_id], decisions], ignore_index=True)
        self.explanations = pd.concat([self.explanations[self.explanations["customer_id"] != customer_id], expl], ignore_index=True)

    def customer(self, customer_id: str) -> dict | None:
        doc = document(customer_id, self.c360, self.features, self.recs, self.explanations, self.decisions, self.events)
        if doc is not None:
            items = {i["product"]: i["facts"] for i in build_items(self.recs[self.recs["customer_id"] == customer_id])}
            for rec in doc["recommendations"]:
                rec["facts"] = clean(items.get(rec["product"]))
            doc["llm_available"] = self.explainer is not None
        return doc

    def me(self, customer_id: str) -> dict | None:
        """The customer app: footprint, spending and the suggestions shown to them, in their words, nothing internal."""
        base = self.c360[self.c360["customer_id"] == customer_id]
        if base.empty:
            return None
        row = base.iloc[0]
        feat = self.features[self.features["customer_id"] == customer_id].merge(
            base[["customer_id", "asset_vehicle_fuel_type"]], on="customer_id")
        fp = footprint(feat).iloc[0]
        regions = {r["region"]: r for r in (self.overview["insights"] or {}).get("regions", [])}
        monthly = self.monthly(customer_id, pd.Timestamp(row["as_of"]))
        spend = {c: round(sum(v), 0) for c, v in monthly["series"].items()}
        recs = self.recs[(self.recs["customer_id"] == customer_id) & (self.recs["status"] == "recommended")].sort_values("rank")
        expl = self.explanations.set_index(["customer_id", "product"])["explanation"] if len(self.explanations) else pd.Series(dtype=object)
        facts = {i["product"]: i["facts"] for i in build_items(recs)}
        suggestions = []
        for rec in recs.to_dict("records"):
            fx = facts[rec["product"]]
            suggestions.append({
                "product": rec["product"], "name": rec["product_name"], "action": rec["customer_action"],
                "explanation": expl.get((customer_id, rec["product"]), ""),
                "why": fx["reasons"], "disclaimer": fx["disclaimer"], "measure": rec.get("measure"),
                "co2e_kg_per_year": rec.get("co2e_avoided_kg_per_year"), "co2e_kg_horizon": rec.get("co2e_avoided_kg_horizon"),
                "horizon_years": rec.get("horizon_years"), "money_gbp_per_year": rec.get("running_cost_change_gbp_per_year"),
                "upfront_gbp": rec.get("upfront_cost_gbp"), "payback_years": rec.get("payback_years")})
        return clean({"customer_id": customer_id, "as_of": row["as_of"], "region": row["region"],
                      "footprint": {"transport_kg": fp["transport_kg"], "home_kg": fp["home_kg"], "total_kg": fp["total_kg"],
                                    "region_median_kg": regions.get(row["region"], {}).get("footprint_kg_median")},
                      "spend_12m": spend, "monthly": monthly,
                      "suggestions": suggestions})

    def monthly(self, customer_id: str, as_of: pd.Timestamp) -> dict:
        """The customer's debit spend per month and category over the last 12 months."""
        path = next((p for lo, hi, p in self.parts if lo <= customer_id <= hi), None)
        if path is None:
            return {"months": [], "series": {}}
        t = pq.read_table(path, columns=["customer_id", "booking_datetime", "amount", "credit_debit_indicator", "pred_category"],
                          filters=[("customer_id", "==", customer_id)]).to_pandas()
        start = (as_of - pd.DateOffset(months=11)).to_period("M").to_timestamp()
        t = t[(t["credit_debit_indicator"] == "Debit") & (t["booking_datetime"] >= start) & (t["booking_datetime"] <= as_of + pd.Timedelta(days=1))]
        t["month"] = t["booking_datetime"].dt.to_period("M").astype(str)
        t["category"] = t["pred_category"].str.lower().str.replace(" ", "_")
        months = [str(m) for m in pd.period_range(start, as_of, freq="M")]
        g = t.groupby(["category", "month"])["amount"].sum()
        return {"months": months, "series": {c: [round(float(g.get((c, m), 0.0)), 2) for m in months] for c in SPEND_CATEGORIES}}

    def feedback(self, customer_id: str, product: str, response: str) -> dict | None:
        with self.lock:
            row = pd.DataFrame([{"customer_id": customer_id, "product": product, "response": response,
                                 "response_date": pd.Timestamp(LIVE), "source": "dashboard"}])
            self.events = pd.concat([self.events, row], ignore_index=True)
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            self.events.to_parquet(self.events_path, index=False)
            self._recompute(customer_id)
        return self.customer(customer_id)

    def reset(self) -> None:
        with self.lock:
            customers = self.events["customer_id"].unique()
            self.events = empty_events()
            if self.events_path.exists():
                self.events_path.unlink()
            for customer_id in customers:
                self._recompute(customer_id)

    def rephrase(self, customer_id: str, product: str) -> dict:
        if self.explainer is None:
            return {"error": "the local LLM is disabled (llm.enabled); the template explanation stays"}
        from gcip.explain.guards import check
        rec = self.recs[(self.recs["customer_id"] == customer_id) & (self.recs["product"] == product)]
        if rec.empty:
            return {"error": "no such recommendation"}
        item = build_items(rec)[0]
        try:
            text = compose(item, self.explainer.rephrase(item))
        except Exception as exc:  # e.g. no GPU memory: keep the template
            return {"error": f"local LLM failed: {exc}"}
        problems = check(text, item["facts"], draft=item["draft"]) if text else ["no reply from the LLM"]
        if not problems:
            with self.lock:
                hit = (self.explanations["customer_id"] == customer_id) & (self.explanations["product"] == product)
                self.explanations.loc[hit, ["explanation", "source"]] = [text, "llm"]
        return {"text": text, "passed": not problems, "problems": problems, "template": item["draft"],
                "model": self.cfg["llm"]["model"]}

    def customers(self, q: str, product: str, status: str, limit: int, offset: int) -> dict:
        d = self.index
        if q:
            ql = q.lower()
            d = d[d["customer_id"].str.lower().str.contains(ql) | d["la_name"].fillna("").str.lower().str.contains(ql)
                  | d["region"].fillna("").str.lower().str.contains(ql)]
        if product:
            d = d[d["products"].fillna("").str.contains(product)]
        if status == "with":
            d = d[d["recommendations"] > 0]
        elif status == "review":
            d = d[d["review"].fillna(False)]
        elif status == "none":
            d = d[d["recommendations"] == 0]
        d = d.sort_values(["expected_co2e", "customer_id"], ascending=[False, True], na_position="last")
        return {"total": int(len(d)), "rows": clean(d.iloc[offset:offset + limit].to_dict("records"))}

    def trend(self, region: str) -> dict:
        t = self.trends if not region else self.trends[self.trends["region"] == region]
        g = t.groupby(["month", "category"], as_index=False)[["spend", "customers"]].sum()
        months = sorted(g["month"].unique())
        cats = sorted(g["category"].unique())
        return {"region": region or "All regions", "months": months,
                "series": {c: [float(g[(g["month"] == m) & (g["category"] == c)]["spend"].sum()) for m in months] for c in cats}}


def make_handler(state: State, static: Path):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(static), **kw)

        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, payload, status=HTTPStatus.OK):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            parts = url.path.strip("/").split("/")
            if parts[0] != "api":
                return super().do_GET()
            if parts[1:] == ["overview"]:
                return self._send(state.overview)
            if parts[1:] == ["trends"]:
                return self._send(state.trend(q.get("region", "")))
            if parts[1:] == ["customers"]:
                return self._send(state.customers(q.get("q", ""), q.get("product", ""), q.get("status", ""),
                                                  min(200, int(q.get("limit", 50))), int(q.get("offset", 0))))
            if len(parts) == 3 and parts[1] == "me":
                doc = state.me(parts[2])
                return self._send(doc) if doc else self._send({"error": "not found"}, HTTPStatus.NOT_FOUND)
            if len(parts) == 3 and parts[1] == "customer":
                doc = state.customer(parts[2])
                return self._send(doc) if doc else self._send({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return self._send({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            parts = urlparse(self.path).path.strip("/").split("/")
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            if parts[1:] == ["memory", "reset"]:
                state.reset()
                return self._send({"ok": True})
            if len(parts) == 4 and parts[1] == "customer" and parts[3] == "feedback":
                if body.get("response") not in FEEDBACK or body.get("product") not in state.overview["products"]:
                    return self._send({"error": "bad feedback"}, HTTPStatus.BAD_REQUEST)
                return self._send(state.feedback(parts[2], body["product"], body["response"]))
            if len(parts) == 4 and parts[1] == "customer" and parts[3] == "explain":
                return self._send(state.rephrase(parts[2], body.get("product", "")))
            return self._send({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default="runs/pipeline")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    state = State(Path(args.run_dir).resolve())
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, ROOT / "dashboard"))
    print(f"dashboard: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
