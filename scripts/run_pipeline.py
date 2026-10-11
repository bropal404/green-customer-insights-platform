#!/usr/bin/env python3
"""End-to-end pipeline on a built data directory (scripts/build_full_data.sh writes runs/full).

    python scripts/run_pipeline.py --data runs/full --run-dir runs/pipeline            # every stage, resumable
    python scripts/run_pipeline.py --data runs/full --run-dir runs/pipeline --force explain insights

Stages (each writes <run-dir>/_checkpoints/<stage>.done and is skipped on a rerun unless forced):
  propensity     FR3: train on the 31 Mar 2026 snapshot, evaluate on test, score the 30 Sep 2026 snapshot
  impact         FR4: carbon and money statements for every customer x product on the live snapshot
  recommend      FR5-FR7: eligibility, consent, memory, routing, top 3, human review; audited decisions
  explain        FR5: template explanation for every recommendation, checked by the guards
  impact_checks  FR4: inputs vs truth (litres, kWh), sensitivity of the main assumptions
  memory         customer memory replay over three rounds
  insights       FR6: bank insights (opportunity, funnel, regions, footprint, segments, monthly trends)
  value          business value: customer and bank financial and carbon return per suggestion, portfolio totals, and a
                 backtest of targeting (model vs rules vs random) on the held-out test customers
  explain_llm    FR5: LLM rephrasing (llm.provider: Gemini API or a local Qwen) of a 360-item evaluation set (60 per
                 product), guarded; passes replace templates
  evaluate       evaluation report: every model's metrics on the validation set next to test and a baseline, and the
                 test set predictions (gcip.models.evaluate)
  report         every metric in one file: <run-dir>/summary.json
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

import pandas as pd  # noqa: E402

from gcip.pipeline.context import make_context  # noqa: E402

STAGES = ["propensity", "impact", "recommend", "explain", "impact_checks", "memory", "insights", "value", "explain_llm", "evaluate", "report"]
PRODUCTS = ["GP01", "GP02", "GP03", "GP04", "GP05", "GP06"]
_DATES = __import__("gcip.config", fromlist=["load_config"]).load_config("pipeline")["as_of"]
TRAIN, LIVE = str(_DATES["training_cutoff"]), str(_DATES["live"])


def memory_events(data: Path):
    """The CRM's campaign history up to the live date as memory events (None when the log is not available)."""
    from gcip.memory.signals import events_from_crm
    path = data / "warehouse" / "campaign_responses.parquet"
    return events_from_crm(pd.read_parquet(path), pd.Timestamp(LIVE)) if path.exists() else None


def memory_signals(ctx, data: Path):
    from gcip.memory.signals import signals
    events = memory_events(data)
    return None if events is None else signals(events, pd.Timestamp(LIVE), ctx.cfg["memory"], ctx.cfg["recommend"]["cooldown_days"])


def thresholds(ctx) -> dict:
    """Per-product propensity thresholds written by the propensity stage."""
    return json.loads((ctx.run_dir / "models" / "thresholds.json").read_text())


def stage_propensity(ctx, data: Path) -> dict:
    from gcip.models import propensity as P
    cfg = ctx.cfg["propensity"] | {"min_lift": ctx.cfg["recommend"]["min_lift"]}
    report = P.run(data, ctx.run_dir, cfg, truth_dir=data / "truth_vault", ablations=True)
    return {"products": list(report["products"])}


def stage_impact(ctx, data: Path) -> dict:
    from gcip.impact.engine import assess
    from gcip.models.propensity import model_table
    impact = assess(model_table(data, LIVE), PRODUCTS, LIVE)
    ctx.save_parquet(impact, "impact", "impact.parquet")
    return {"rows": len(impact), "status": impact["status"].value_counts().to_dict()}


def stage_recommend(ctx, data: Path) -> dict:
    from gcip.recommend.engine import recommend
    scores = ctx.read_parquet("scores", "live_scores.parquet")
    impact = ctx.read_parquet("impact", "impact.parquet")
    c360 = pd.read_parquet(data / f"customer_360_{LIVE}.parquet")
    recs, decisions = recommend(scores, impact, c360, ctx.cfg["recommend"], thresholds(ctx), memory=memory_signals(ctx, data))
    ctx.save_parquet(recs, "recommendations", "recommendations.parquet")
    ctx.save_parquet(decisions, "recommendations", "decisions.parquet")
    return {"recommendations": len(recs), "customers": int(recs["customer_id"].nunique()),
            "status": recs["status"].value_counts().to_dict(), "by_product": recs["product"].value_counts().to_dict()}


def stage_explain(ctx, data: Path) -> dict:
    from gcip.explain.run import build_items, explain
    recs = ctx.read_parquet("recommendations", "recommendations.parquet")
    out = explain(build_items(recs))
    ctx.save_parquet(out, "explanations", "explanations.parquet")
    summary = {"explanations": len(out), "template_guard_pass": int((out["template_problems"] == "").sum()),
               "reading_ease_mean": round(float(out["reading_ease"].mean()), 1), "words_mean": round(float(out["words"].mean()), 1)}
    ctx.save_json(summary, "metrics", "explanations.json")
    return summary


def stage_explain_llm(ctx, data: Path) -> dict:
    from gcip.explain.llm import make_explainer
    from gcip.explain.run import build_items, explain, reading_ease
    cfg = ctx.cfg["llm"]
    summary = json.loads((ctx.run_dir / "metrics" / "explanations.json").read_text())
    explainer = make_explainer(cfg)
    if explainer is None:
        summary["llm_eval"] = {"skipped": f"LLM provider {cfg.get('provider')!r} is not available here; templates only"}
        ctx.save_json(summary, "metrics", "explanations.json")
        return summary["llm_eval"]
    recs = ctx.read_parquet("recommendations", "recommendations.parquet")
    per = cfg["eval_set_size"] // len(PRODUCTS)
    sample = recs.loc[[i for _, g in recs.groupby("product") for i in g.sample(min(per, len(g)), random_state=7).index]]
    items = build_items(sample)
    t0 = time.time()
    ev = explain(items, generated=explainer.generate(items, cfg.get("concurrency", 4) if cfg.get("provider") == "gemini" else cfg["batch_size"]))
    ctx.save_parquet(ev, "explanations", "eval_set.parquet")
    passed = ev["source"] == "llm"
    problems = ev.loc[~passed, "guard_problems"].str.split(";").explode().str.strip().str.replace(r":.*", "", regex=True)
    summary["llm_eval"] = {
        "model": cfg["model"] if cfg.get("provider") == "gemini" else cfg.get("local_model"),
        "provider": cfg.get("provider"), "items": len(ev), "seconds": round(time.time() - t0, 1),
        "guard_pass_rate": round(float(passed.mean()), 3),
        "failure_reasons": {k: int(v) for k, v in problems.value_counts().items()},
        "pass_rate_by_product": ev.groupby("product")["source"].apply(lambda x: round(float((x == "llm").mean()), 3)).to_dict(),
        "words_mean": {"llm": round(float(ev.loc[passed, "words"].mean()), 1) if passed.any() else None,
                       "template": round(float(ev["template"].map(lambda t: len(t.split())).mean()), 1)},
        "reading_ease_mean": {"llm": round(float(ev.loc[passed, "reading_ease"].mean()), 1) if passed.any() else None,
                              "template": round(float(ev["template"].map(reading_ease).mean()), 1)},
    }
    # Rephrasings that passed every guard replace the template for those recommendations.
    out = ctx.read_parquet("explanations", "explanations.parquet")
    keyed = ev[passed].set_index(["customer_id", "product"])
    idx = out.set_index(["customer_id", "product"]).index
    hit = idx.isin(keyed.index)
    out.loc[hit, ["explanation", "source", "llm_text"]] = keyed.loc[idx[hit], ["explanation", "source", "llm_text"]].to_numpy()
    ctx.save_parquet(out, "explanations", "explanations.parquet")
    summary["source"] = {k: int(v) for k, v in out["source"].value_counts().items()}
    ctx.save_json(summary, "metrics", "explanations.json")
    return summary["llm_eval"]


def stage_impact_checks(ctx, data: Path) -> dict:
    from gcip.impact.checks import sensitivity, truth_recovery
    from gcip.models.propensity import model_table
    state_cols = ["customer_id", "has_car", "car_fuel", "annual_km_true", "km_per_litre_true", "extra_km_true",
                  "extra_km_per_litre_true", "extra_fuel", "gas_kwh_year_true", "elec_kwh_year_true"]
    state = pd.read_parquet(data / "truth_vault" / "state.parquet", columns=state_cols)
    recovery = truth_recovery(model_table(data, TRAIN), state)
    live = model_table(data, LIVE)
    tornado = sensitivity(live.sample(min(8000, len(live)), random_state=7), LIVE)
    ctx.save_json(recovery, "metrics", "impact_truth_recovery.json")
    tornado.to_csv(ctx.path("metrics", "impact_sensitivity.csv"), index=False)
    return {"truth_recovery": {k: v for k, v in recovery.items() if k != "note"},
            "largest_swings": tornado.sort_values("co2e_swing_pct", ascending=False).head(5)[
                ["assumption", "product", "co2e_swing_pct", "money_swing_pct"]].to_dict("records")}


def stage_memory(ctx, data: Path) -> dict:
    from gcip.memory.replay import replay
    scores = ctx.read_parquet("scores", "live_scores.parquet")
    impact = ctx.read_parquet("impact", "impact.parquet")
    c360 = pd.read_parquet(data / f"customer_360_{LIVE}.parquet")
    truth = pd.read_parquet(data / "truth_vault" / "labels.parquet", columns=["customer_id", "product", "p_true"])
    mcfg = ctx.cfg["memory"]
    metrics, events = replay(scores, impact, c360, ctx.cfg["recommend"], truth, mcfg["replay_customers"], mcfg["rounds"],
                             thresholds=thresholds(ctx), events=memory_events(data), memory_cfg=mcfg)
    ctx.save_parquet(events, "memory", "replay_events.parquet")
    ctx.save_json(metrics, "metrics", "memory_replay.json")
    return metrics


def stage_insights(ctx, data: Path) -> dict:
    from gcip.insights.bank import build, customer_monthly
    from gcip.models.propensity import model_table
    recs = ctx.read_parquet("recommendations", "recommendations.parquet")
    decisions = ctx.read_parquet("recommendations", "decisions.parquet")
    impact = ctx.read_parquet("impact", "impact.parquet")
    holdings = data / "warehouse" / "product_holdings.parquet"
    insights, trends = build(model_table(data, LIVE), recs, decisions, impact, data / "categorised", ctx.cfg["insights"]["segments_k"],
                             pd.read_parquet(holdings) if holdings.exists() else None)
    ctx.save_json(insights, "insights", "bank_insights.json")
    ctx.save_parquet(trends, "insights", "monthly_trends.parquet")
    ctx.save_parquet(customer_monthly(data / "categorised", pd.Timestamp(LIVE)), "insights", "customer_monthly_spend.parquet")
    return insights["overview"]


def stage_value(ctx, data: Path) -> dict:
    from gcip.config import load_config
    from gcip.impact.engine import assess
    from gcip.insights.value import backtest, measure_uplift, suggestion_value, summary
    from gcip.models.propensity import model_table
    names = {k: v["name"] for k, v in load_config("products")["products"].items()}
    recs = ctx.read_parquet("recommendations", "recommendations.parquet")
    values = suggestion_value(recs, pd.read_parquet(data / f"customer_360_{LIVE}.parquet"))
    ctx.save_parquet(values, "recommendations", "value.parquet")
    out = summary(values, names)
    # Backtest on what really happened: the training snapshot's test customers and their take-up in the six months after.
    held = pd.read_csv(ctx.run_dir / "predictions" / "heldout_predictions.csv")
    test = held[held["split"] == "test"][["customer_id", "product", "propensity"]]
    impact = assess(model_table(data, TRAIN), PRODUCTS, TRAIN).rename(columns={"status": "impact_status"})
    rows = test.merge(impact, on=["customer_id", "product"], how="left").assign(status="recommended")
    test_values = suggestion_value(rows, pd.read_parquet(data / f"customer_360_{TRAIN}.parquet"))
    out["backtest"] = backtest(held, test_values, thresholds(ctx), train_date=TRAIN, live_date=LIVE)
    labels = pd.read_parquet(data / "labels.parquet")
    out["measured_uplift"] = measure_uplift(held, test_values, labels, thresholds(ctx), train_date=TRAIN, live_date=LIVE)
    ctx.save_json(out, "metrics", "business_value.json")
    return {"portfolio": out["portfolio"], "backtest_total": out["backtest"]["total"]}


def stage_evaluate(ctx, data: Path) -> dict:
    from gcip.config import load_config
    from gcip.models.evaluate import build
    opportunity = {k: v["opportunity"] for k, v in load_config("products")["products"].items()}
    # The categoriser's report is written by the data build (<data>_run); keep a copy with this run's metrics.
    built, kept = data.parent / f"{data.name}_run" / "metrics" / "categoriser_report.json", ctx.path("metrics", "categoriser_report.json")
    if built.exists():
        kept.write_text(built.read_text())
    report = build(ctx.run_dir, kept if kept.exists() else None, opportunity, data / f"customer_360_{TRAIN}.parquet")
    return {"validation_mean": report["propensity"]["mean_over_products"]["validation"]["model"],
            "test_mean": report["propensity"]["mean_over_products"]["test"]["model"]}


def stage_report(ctx, data: Path) -> dict:
    metrics = ctx.run_dir / "metrics"
    summary = {"data": json.loads((data / "meta" / "data_validation.json").read_text()).get("summary")
               if (data / "meta" / "data_validation.json").exists() else None}
    for name in ("propensity_report", "evaluation_report", "explanations", "impact_truth_recovery", "memory_replay", "business_value"):
        path = metrics / f"{name}.json"
        if path.exists():
            summary[name] = json.loads(path.read_text())
    summary["insights_overview"] = json.loads((ctx.run_dir / "insights" / "bank_insights.json").read_text())["overview"]
    summary["stage_seconds"] = ctx.timings
    ctx.save_json(summary, "summary.json")
    return {"summary": str((ctx.run_dir / "summary.json").relative_to(ROOT)) if ctx.run_dir.is_relative_to(ROOT) else "summary.json"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="runs/full")
    parser.add_argument("--run-dir", default="runs/pipeline")
    parser.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES)
    parser.add_argument("--force", nargs="*", default=[], choices=STAGES, help="rerun these stages even if done")
    parser.add_argument("--llm-provider", choices=["gemini", "local", "none"], help="override llm.provider for this run")
    args = parser.parse_args()
    data = Path(args.data).resolve()
    ctx = make_context(Path(args.run_dir).resolve(), {"llm": {"provider": args.llm_provider}} if args.llm_provider else None)
    link = ctx.run_dir / "data"
    if not link.exists():
        link.symlink_to(data)
    for name in args.stages:
        if ctx.done(name) and name not in args.force:
            ctx.log.info("STAGE %s: already done, skipped", name)
            continue
        with ctx.stage(name):
            info = globals()[f"stage_{name}"](ctx, data)
        ctx.mark(name, json.loads(json.dumps(info, default=str)))
        ctx.log.info("STAGE %s: %s", name, json.dumps(info, default=str)[:600])


if __name__ == "__main__":
    main()
