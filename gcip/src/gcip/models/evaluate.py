"""Evaluation report: every model's held-out metrics on the validation set, next to the test set and a simple baseline.

Nothing is retrained; the report reads what the stages wrote:
  FR2 categoriser   categoriser_report.json from the data build: transactions of validation and test customers, split
                    into merchants seen in training and merchants never seen (generalisation); baseline: rules only
  FR3 propensity    predictions/heldout_predictions.csv: calibrated propensity of every eligible validation and test
                    customer with the observed take-up (1 Apr - 30 Sep 2026); baseline: the rules score; the suggestion
                    threshold (min_lift x the product's take-up rate on train) from models/thresholds.json
  FR4 impact        metrics/impact_truth_recovery.json: litres and kWh read from transactions vs the simulator's truth
  FR5 explanations  metrics/explanations.json (and explanations_gemini_sample.json): guard pass rate of the LLM
                    rewrites on the evaluation set
  targeting         metrics/business_value.json: take-ups when the model, the rules or chance picks whom to contact
Writes metrics/evaluation_report.json, metrics/evaluation_report.md (the tables in docs/EVALUATION.md) and
predictions/test_predictions.csv (test customers: propensity, confidence, whether it clears the product's suggestion
threshold, and the observed take-up).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gcip.models.propensity import score_metrics

SPLITS = ("validation", "test")
METRICS = ("roc_auc", "pr_auc", "brier", "ece", "lift_top10")


def _at_threshold(y: np.ndarray, p: np.ndarray, cut: float) -> dict:
    """What suggesting to everyone above the threshold would have done on this split."""
    chosen = p >= cut
    rate = float(y[chosen].mean()) if chosen.any() else None
    return {"selected_share": round(float(chosen.mean()), 4), "precision": None if rate is None else round(rate, 4),
            "recall": round(float(y[chosen].sum() / max(1, y.sum())), 4),
            "lift": None if rate is None or y.mean() == 0 else round(rate / float(y.mean()), 2)}


def propensity(held: pd.DataFrame, thresholds: dict) -> dict:
    """Per product and split: the model (calibrated LightGBM), the rules baseline, and the suggestion threshold."""
    out: dict = {"products": {}, "mean_over_products": {}}
    for product, part in held.groupby("product"):
        cut = thresholds[product]["min_propensity"]
        out["products"][product] = {"min_propensity": cut}
        for s in SPLITS:
            d = part[part["split"] == s]
            y, p = d["took_up"].to_numpy().astype(int), d["propensity"].to_numpy()
            out["products"][product][s] = {"model": score_metrics(y, p),
                                           "rules_baseline": score_metrics(y, d["rules_score"].to_numpy()),
                                           "at_threshold": _at_threshold(y, p, cut)}
    for s in SPLITS:
        rows = [v[s] for v in out["products"].values()]
        out["mean_over_products"][s] = {
            who: {m: round(float(np.mean([r[who][m] for r in rows if m in r[who]])), 4) for m in METRICS}
            for who in ("model", "rules_baseline")}
    return out


def fairness(held: pd.DataFrame, c360: pd.DataFrame, thresholds: dict, min_n: int = 100) -> dict:
    """Per product, split and attribute: each group's selection rate at the suggestion threshold, take-up rate and mean
    propensity (calibration within the group); the summary is the lowest / highest selection rate across groups of at
    least `min_n` customers and the largest gap between a group's take-up rate and its mean propensity."""
    who = c360[["customer_id", "age", "declared_income_band", "region", "vulnerability_flag"]].copy()
    who["age_band"] = pd.cut(who["age"], [17, 34, 54, 120], labels=["18-34", "35-54", "55+"]).astype(str)
    who["vulnerable"] = who["vulnerability_flag"].fillna(False).astype(bool).map({True: "yes", False: "no"})
    d = held.merge(who, on="customer_id", how="left")
    d["selected"] = d["propensity"] >= d["product"].map({k: v["min_propensity"] for k, v in thresholds.items()})
    out: dict = {}
    for (product, split), part in d.groupby(["product", "split"]):
        for attr in ("age_band", "declared_income_band", "region", "vulnerable"):
            g = part.groupby(attr).agg(n=("took_up", "size"), take_up_rate=("took_up", "mean"),
                                       mean_propensity=("propensity", "mean"), selected_rate=("selected", "mean"))
            big = g[g["n"] >= min_n]
            out.setdefault(product, {}).setdefault(split, {})[attr] = {
                "groups": g.round(4).to_dict("index"),
                "selected_rate_min": round(float(big["selected_rate"].min()), 4),
                "selected_rate_max": round(float(big["selected_rate"].max()), 4),
                "max_calibration_gap": round(float((big["take_up_rate"] - big["mean_propensity"]).abs().max()), 4)}
    return out


def categoriser(report: dict) -> dict:
    """Category accuracy and macro F1 of the full categoriser and of the rules alone, per split and merchant group."""
    out = {}
    for s in SPLITS:
        for group in ("seen", "unseen"):
            sl = report["slices"][f"{s}_{group}_merchants"]
            out[f"{s}_{group}_merchants"] = {
                who: {k: sl[layer][k] for k in ("transactions", "category_accuracy", "category_macro_f1", "sub_category_accuracy")}
                for who, layer in (("model", "full"), ("rules_baseline", "rules_only"))}
    out["out_of_distribution_us_descriptions"] = {k: report["ood_dodatathings"][k] for k in ("rows", "coarse_accuracy")}
    return out


def test_predictions(held: pd.DataFrame, thresholds: dict, opportunity: dict) -> pd.DataFrame:
    t = held[held["split"] == "test"].copy()
    t["opportunity"] = t["product"].map(opportunity)
    t["suggest"] = t["propensity"] >= t["product"].map({p: v["min_propensity"] for p, v in thresholds.items()})
    return t[["customer_id", "product", "opportunity", "propensity", "confidence", "suggest", "rules_score", "took_up"]]


def build(run_dir: Path, categoriser_report: Path | None, opportunity: dict, c360_train: Path | None = None) -> dict:
    metrics = run_dir / "metrics"
    held = pd.read_csv(run_dir / "predictions" / "heldout_predictions.csv")
    thresholds = json.loads((run_dir / "models" / "thresholds.json").read_text())
    report: dict = {
        "splits": {"customers": "70% train / 15% validation / 15% test, by customer (seeded), the same split for every model",
                   "propensity_rows": {s: int((held["split"] == s).sum()) for s in SPLITS},
                   "merchants": "20% of merchants never appear in training (categoriser generalisation)"},
        "propensity": propensity(held, thresholds)}
    if c360_train is not None and c360_train.exists():
        report["fairness"] = fairness(held, pd.read_parquet(c360_train), thresholds)
    if categoriser_report is not None and categoriser_report.exists():
        report["categoriser"] = categoriser(json.loads(categoriser_report.read_text()))
    for key, name in (("impact_truth_recovery", "impact_truth_recovery"), ("explanations", "explanations"),
                      ("business_value", "business_value")):
        path = metrics / f"{name}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if key == "explanations":
            data = {"template_guard_pass": data["template_guard_pass"], "explanations": data["explanations"],
                    "llm_eval": data.get("llm_eval")}
        elif key == "business_value":
            key, data = "targeting_backtest_test", data["backtest"]["total"]
        else:
            data = {k: v for k, v in data.items() if k != "note"}
        report[key] = data
    gemini = metrics / "explanations_gemini_sample.json"
    if gemini.exists() and "explanations" in report:
        g = json.loads(gemini.read_text())
        report["explanations"]["gemini_sample"] = {k: g[k] for k in ("models_used", "requests", "passed_guards", "measured")}
    test_predictions(held, thresholds, opportunity).to_csv(run_dir / "predictions" / "test_predictions.csv", index=False)
    (metrics / "evaluation_report.json").write_text(json.dumps(report, indent=2, default=str))
    (metrics / "evaluation_report.md").write_text(to_markdown(report))
    return report


def _f(x, pct: bool = False) -> str:
    if x is None:
        return "-"
    return f"{100 * x:.1f}%" if pct else f"{x:.3f}"


def to_markdown(report: dict) -> str:
    """The report's tables, generated so the numbers in docs/EVALUATION.md are never typed by hand."""
    lines = ["### FR3 propensity: validation and test (calibrated LightGBM vs rules baseline)", "",
             "| Product | Split | Customers | Take-ups | ROC-AUC model | ROC-AUC rules | PR-AUC model | PR-AUC rules | Brier | ECE "
             "| Lift top 10% | Above threshold | Precision | Recall |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for product, v in report["propensity"]["products"].items():
        for s in SPLITS:
            m, r, t = v[s]["model"], v[s]["rules_baseline"], v[s]["at_threshold"]
            lines.append(f"| {product} | {s} | {m['n']} | {m['positives']} | {_f(m.get('roc_auc'))} | {_f(r.get('roc_auc'))} "
                         f"| {_f(m.get('pr_auc'))} | {_f(r.get('pr_auc'))} | {_f(m.get('brier'))} | {_f(m.get('ece'))} "
                         f"| {m.get('lift_top10', 0):.2f} | {_f(t['selected_share'], True)} | {_f(t['precision'], True)} "
                         f"| {_f(t['recall'], True)} |")
    lines += ["", "Mean over the six products:", "", "| Split | Scorer | ROC-AUC | PR-AUC | Brier | ECE | Lift top 10% |",
              "|---|---|---|---|---|---|---|"]
    for s, by in report["propensity"]["mean_over_products"].items():
        for who, m in by.items():
            lines.append(f"| {s} | {who.replace('_', ' ')} | {_f(m['roc_auc'])} | {_f(m['pr_auc'])} | {_f(m['brier'])} "
                         f"| {_f(m['ece'])} | {m['lift_top10']:.2f} |")
    if "fairness" in report:
        lines += ["", "### Fairness at the suggestion threshold (groups of 100+ customers)", "",
                  "Selection rate = share of the group above the product's threshold; calibration gap = largest difference "
                  "between a group's take-up rate and its mean propensity.", "",
                  "| Product | Attribute | Validation: selection rate (lowest-highest group) | Validation: calibration gap "
                  "| Test: selection rate | Test: calibration gap |", "|---|---|---|---|---|---|"]
        for product, by in report["fairness"].items():
            for attr in by["validation"]:
                v, t = by["validation"][attr], by["test"][attr]
                lines.append(f"| {product} | {attr.replace('_', ' ')} | {_f(v['selected_rate_min'], True)}-{_f(v['selected_rate_max'], True)} "
                             f"| {_f(v['max_calibration_gap'], True)} | {_f(t['selected_rate_min'], True)}-{_f(t['selected_rate_max'], True)} "
                             f"| {_f(t['max_calibration_gap'], True)} |")
    if "categoriser" in report:
        lines += ["", "### FR2 categoriser: validation and test transactions (full categoriser vs rules only)", "",
                  "| Slice | Transactions | Category accuracy | Macro F1 | Sub-category accuracy | Rules only: accuracy | Rules only: macro F1 |",
                  "|---|---|---|---|---|---|---|"]
        for name, v in report["categoriser"].items():
            if name.startswith("out_of"):
                continue
            m, r = v["model"], v["rules_baseline"]
            lines.append(f"| {name.replace('_', ' ')} | {m['transactions']:,} | {_f(m['category_accuracy'], True)} "
                         f"| {_f(m['category_macro_f1'])} | {_f(m['sub_category_accuracy'], True)} "
                         f"| {_f(r['category_accuracy'], True)} | {_f(r['category_macro_f1'])} |")
        ood = report["categoriser"]["out_of_distribution_us_descriptions"]
        lines += ["", f"Out of distribution: {ood['rows']:,} real US bank descriptions, coarse agreement "
                      f"{_f(ood['coarse_accuracy'], True)}."]
    if "impact_truth_recovery" in report:
        lines += ["", "### FR4 impact inputs vs the simulator's truth", "",
                  "| Input | Customers | Median ratio | Median abs. error | Within 20% | Correlation |", "|---|---|---|---|---|---|"]
        for name, v in report["impact_truth_recovery"].items():
            if isinstance(v, dict) and "median_ratio" in v:
                lines.append(f"| {name.replace('_', ' ')} | {v['customers']:,} | {v['median_ratio']:.3f} "
                             f"| {v['median_abs_pct_error']:.1f}% | {_f(v['within_20pct'], True)} | {v['correlation']:.3f} |")
    ex = report.get("explanations", {})
    if ex.get("llm_eval"):
        e = ex["llm_eval"]
        lines += ["", "### FR5 explanations", "",
                  f"Templates: {ex['template_guard_pass']:,} of {ex['explanations']:,} pass every guard. "
                  f"LLM rewrites ({e['model']}, {e['provider']}): {_f(e['guard_pass_rate'], True)} of {e['items']} pass; "
                  "by product " + ", ".join(f"{k} {_f(v, True)}" for k, v in e["pass_rate_by_product"].items()) + "."]
        g = ex.get("gemini_sample")
        if g:
            lines += ["", f"Gemini API ({', '.join(g['models_used'])}), the provider at inference: {g['passed_guards']} of "
                          f"{g['requests']} rewrites pass every guard (sample, {g['measured']})."]
    bt = report.get("targeting_backtest_test")
    if bt:
        lines += ["", "### Targeting backtest (test customers, same number contacted by each strategy)", "",
                  "| Strategy | Contacted | Take-ups | Lifetime tCO2e | Bank value (£) |", "|---|---|---|---|---|"]
        for who in ("model", "rules", "random"):
            v = bt[who]
            lines.append(f"| {who} | {v['contacted']:,.0f} | {v['takeups']:,.0f} | {v['co2e_t_lifetime']:,.1f} | {v['bank_value_gbp']:,.0f} |")
    return "\n".join(lines) + "\n"
