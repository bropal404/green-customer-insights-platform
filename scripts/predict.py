#!/usr/bin/env python3
"""Inference with the trained models, without retraining (needs requirements.txt).

    # FR3: propensity, confidence and reason codes for every customer x product of a snapshot
    python scripts/predict.py propensity --data demo/data --models demo/pipeline/models --as-of 2026-09-30 --out scores.csv

    # FR2: category of transaction descriptions
    python scripts/predict.py categorise "SHELL PETROL 0423 LEEDS" "TFL TRAVEL CH"
    python scripts/predict.py categorise "BRITISH GAS" --ptype DD      # payment type and MCC, when known, sharpen it

The propensity models are the six LightGBM ensembles with their isotonic calibrators (models/propensity_GP0*.joblib,
written by the pipeline's propensity stage); the categoriser is the rules + merchant lookup + TF-IDF model written by
the data build (models/categoriser.joblib). Eligibility rules run first: ineligible customers get no score.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PRODUCTS = ["GP01", "GP02", "GP03", "GP04", "GP05", "GP06"]


def propensity(args: argparse.Namespace) -> None:
    from gcip.models.propensity import model_table, score_live
    from gcip.rules.eligibility import evaluate_eligibility
    table = model_table(Path(args.data), args.as_of)
    eligibility = evaluate_eligibility(table)
    rows = []
    for product in args.products:
        bundle = joblib.load(Path(args.models) / f"propensity_{product}.joblib")
        rows.append(score_live(bundle, table, eligibility, product))
    out = pd.concat(rows, ignore_index=True)
    out["reason_codes"] = out["reason_codes"].map(lambda c: ";".join(c))
    out["evidence"] = out["evidence"].map(lambda e: " | ".join(e))
    out.to_csv(args.out, index=False)
    scored = out[out["eligible"]]
    print(f"wrote {args.out}: {len(out):,} customer x product rows, {len(scored):,} eligible and scored")
    print(scored.groupby("product")["propensity"].describe()[["count", "mean", "50%", "max"]].round(4).to_string())


def categorise(args: argparse.Namespace) -> None:
    from gcip.categorise.model import Categoriser
    from gcip.categorise.normalise import normalise_text
    cat = Categoriser.load(Path(args.model))
    text = pd.Series(args.text)
    parts = pd.DataFrame({"ptype": args.ptype, "mcc": args.mcc, "amount_band": 3, "round_amount": 0, "direction": "D",
                          "norm_text": text.map(normalise_text)})
    pred = cat.predict(parts, cat.combine(cat.model_proba(parts)))
    for t, (_, p) in zip(text, pred.iterrows()):
        print(f"{t:<40} {p['pred_category']:<18} {p['pred_sub_category']:<24} confidence {p['cat_confidence']:.2f} ({p['layer']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("propensity", help="score a snapshot with the saved propensity models")
    p.add_argument("--data", default="demo/data", help="folder with customer_360_<as-of>.parquet and features_<as-of>.parquet")
    p.add_argument("--models", default="demo/pipeline/models")
    p.add_argument("--as-of", default="2026-09-30")
    p.add_argument("--products", nargs="*", default=PRODUCTS, choices=PRODUCTS)
    p.add_argument("--out", default="scores.csv")
    c = sub.add_parser("categorise", help="categorise transaction descriptions with the saved categoriser")
    c.add_argument("text", nargs="+")
    c.add_argument("--ptype", default="DEB", help="payment type, as in the bank's extract (DEB card, DD direct debit, SO, FPO ...)")
    c.add_argument("--mcc", default="", help="merchant category code when known (card payments); the rules use it first")
    c.add_argument("--model", default="demo/pipeline/models/categoriser.joblib")
    args = parser.parse_args()
    np.seterr(all="ignore")
    {"propensity": propensity, "categorise": categorise}[args.command](args)


if __name__ == "__main__":
    main()
