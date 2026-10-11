"""FR3: propensity, confidence and reason codes for every customer and product.

Per product, on customers the eligibility rules allow:
  1. a rules baseline (share of the product's reason signals that are on) and a logistic regression;
  2. LightGBM tuned by random search on train (scored on validation by average precision);
  3. out-of-fold isotonic calibration on train, so a propensity of 0.2 means about 1 in 5;
  4. a bootstrap ensemble of five models; their disagreement feeds confidence;
  5. confidence = sqrt(data coverage) x 1 / (1 + coefficient of variation across the ensemble);
  6. reason codes from LightGBM's per-feature contributions (TreeSHAP), summed per reason code, with evidence.
Metrics on the test split: ROC-AUC, PR-AUC, Brier, ECE, lift in the top 10%, the share of the gap to the oracle
(the simulator's Bayes-optimal probability, used for evaluation only) that the model closes, fairness slices and
ablations. Inputs per product are curated (curated_columns), not the whole table.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gcip.features.store import COVERAGE_FEATURES, PHYSICAL_FEATURES
from gcip.fr1.labels import takeup_labels
from gcip.models.design import REASON_FEATURES, coverage_score, design_matrix
from gcip.models.reasons import reason_contributions, top_reasons
from gcip.rules.eligibility import evaluate_eligibility

log = logging.getLogger("gcip.propensity")
PRODUCTS = ["GP01", "GP02", "GP03", "GP04", "GP05", "GP06"]
# Shared context for every product's model: it may inform the score but never becomes a reason code.
CONTEXT = ["age", "declared_income_band", "customer_tenure_years", "n_products", "balance_current", "balance_savings",
           "income_monthly", "debt_to_income_est", "logins_per_month_6m", "months_of_history"]
EXTRA = {
    "GP01": ["area_car_share_bev", "motor_loan_months_to_end", "asset_vehicle_age_years", "area_share_cars_0"],
    "GP02": ["area_gas_dom_mean_kwh", "mortgage_current_ltv", "asset_construction_period__pre-1930",
             "asset_construction_period__1930-1972", "asset_construction_period__1973-1999", "asset_floor_area_m2"],
    "GP03": ["area_pv_rate_per_household", "area_elec_dom_median_kwh", "asset_construction_period__2000-or-later"],
    "GP04": ["app_login_share_6m", "estatement_views_6m", "months_with_logins_6m"],
    "GP05": ["balance_investment", "n_investment_accounts", "investment_transfer_total_12m"],
    "GP06": ["n_personal_loan_accounts", "balance_credit_card", "owner_no_mortgage", "heat_kwh_12m"],
}
MEMORY_PREFIX = "own_campaign_"
LABEL_WINDOW = ("2026-04-01", "2026-09-30")


# --------------------------------------------------------------------------- data


def model_table(data_dir: Path, as_of: str) -> pd.DataFrame:
    c360 = pd.read_parquet(data_dir / f"customer_360_{as_of}.parquet")
    features = pd.read_parquet(data_dir / f"features_{as_of}.parquet")
    return c360.merge(features, on=["customer_id", "as_of"], how="left")


def curated_columns(product: str, available: list[str], drop: set[str] | None = None) -> list[str]:
    """The product's reason-code features, shared context, product extras and its own campaign memory."""
    wanted = [f for features in REASON_FEATURES[product].values() for f in features] + CONTEXT + EXTRA.get(product, [])
    wanted += [c for c in available if c.startswith(MEMORY_PREFIX)]
    seen, out = set(), []
    for c in wanted:
        if c in available and c not in seen and not (drop and c in drop):
            seen.add(c)
            out.append(c)
    return out


# --------------------------------------------------------------------------- metrics


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def lift_at(y: np.ndarray, p: np.ndarray, share: float = 0.10) -> float:
    k = max(1, int(round(len(p) * share)))
    top = np.argsort(-p)[:k]
    base = y.mean()
    return float(y[top].mean() / base) if base > 0 else float("nan")


def score_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if y.sum() == 0 or y.sum() == len(y):
        return {"n": int(len(y)), "positives": int(y.sum())}
    return {"n": int(len(y)), "positives": int(y.sum()), "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4), "brier": round(float(brier_score_loss(y, np.clip(p, 0, 1))), 5),
            "ece": round(expected_calibration_error(y, np.clip(p, 0, 1)), 5), "lift_top10": round(lift_at(y, p), 3)}


# --------------------------------------------------------------------------- models


def rules_baseline(table: pd.DataFrame, product: str, train_mask: np.ndarray) -> np.ndarray:
    """Share of the product's reason signals that are 'on' (primary feature above its training 75th percentile, or a flag)."""
    signals = []
    for features in REASON_FEATURES[product].values():
        column = next((f for f in features if f in table and pd.api.types.is_numeric_dtype(table[f])), None)
        if column is None:
            continue
        values = pd.to_numeric(table[column], errors="coerce").astype(float).fillna(0.0).to_numpy()
        threshold = np.quantile(values[train_mask], 0.75) if train_mask.any() else 0.0
        signals.append(values > max(threshold, 0.0))
    return np.mean(signals, axis=0) if signals else np.zeros(len(table))


def _lgb_params(rng: np.random.Generator) -> dict:
    return {"objective": "binary", "learning_rate": 0.03, "verbose": -1, "num_threads": 0,
            "num_leaves": int(rng.choice([7, 15, 31, 63])), "min_child_samples": int(rng.choice([20, 40, 80, 160])),
            "feature_fraction": float(rng.uniform(0.6, 1.0)), "bagging_fraction": float(rng.uniform(0.6, 1.0)),
            "bagging_freq": 1, "lambda_l2": float(rng.choice([0.0, 1.0, 5.0, 20.0])), "min_gain_to_split": 0.0}


def search_lightgbm(X: pd.DataFrame, y: np.ndarray, train: np.ndarray, valid: np.ndarray, trials: int, seed: int,
                    early_stopping: int) -> tuple[dict, int, float]:
    rng = np.random.default_rng(seed)
    best = ({}, 100, -1.0)
    dtrain = lgb.Dataset(X[train], y[train], free_raw_data=False, params={"feature_pre_filter": False})
    dvalid = lgb.Dataset(X[valid], y[valid], reference=dtrain, free_raw_data=False)
    for _ in range(trials):
        params = _lgb_params(rng) | {"seed": int(rng.integers(1 << 30))}
        booster = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dvalid],
                            callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        rounds = booster.best_iteration or booster.current_iteration()
        score = average_precision_score(y[valid], booster.predict(X[valid], num_iteration=rounds)) if y[valid].any() else 0.0
        if score > best[2]:
            best = (params, rounds, float(score))
    return best


def fit_product(X: pd.DataFrame, y: np.ndarray, split: np.ndarray, cfg: dict, seed: int) -> dict:
    """Tuned, calibrated bootstrap ensemble for one product."""
    train, valid = split == "train", split == "validation"
    params, rounds, valid_ap = search_lightgbm(X, y, train, valid, cfg["optuna_trials"], seed, cfg["early_stopping_rounds"])
    rounds = max(rounds, 50)
    # Out-of-fold raw scores on train -> isotonic calibration.
    oof = np.zeros(train.sum())
    Xt, yt = X[train], y[train]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for k, (a, b) in enumerate(folds.split(Xt, yt)):
        booster = lgb.train(params | {"seed": seed + k}, lgb.Dataset(Xt.iloc[a], yt[a]), num_boost_round=rounds)
        oof[b] = booster.predict(Xt.iloc[b])
    # Bounded: no customer is certain either way, and the top isotonic step rests on a handful of take-ups.
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=PROPENSITY_BOUNDS[0], y_max=PROPENSITY_BOUNDS[1]).fit(oof, yt)
    # Bootstrap ensemble on all training rows.
    rng = np.random.default_rng(seed)
    models = []
    for k in range(cfg["bootstrap_models"]):
        sample = rng.integers(0, len(yt), len(yt))
        models.append(lgb.train(params | {"seed": seed + 100 + k}, lgb.Dataset(Xt.iloc[sample], yt[sample]), num_boost_round=rounds))
    return {"params": params, "rounds": rounds, "validation_pr_auc": round(valid_ap, 4), "calibrator": calibrator,
            "models": models, "columns": list(X.columns)}


PROPENSITY_BOUNDS = (0.001, 0.95)


def predict(bundle: dict, X: pd.DataFrame, calibrated: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Propensity (calibrated mean of the ensemble) and the ensemble's coefficient of variation."""
    raw = np.column_stack([m.predict(X[bundle["columns"]]) for m in bundle["models"]])
    mean = raw.mean(axis=1)
    cv = raw.std(axis=1) / np.maximum(mean, 1e-6)
    return (np.clip(bundle["calibrator"].predict(mean), *PROPENSITY_BOUNDS) if calibrated else mean), cv


def contributions(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    """Mean TreeSHAP contributions (log-odds) across the ensemble, one column per feature (bias dropped)."""
    total = sum(m.predict(X[bundle["columns"]], pred_contrib=True) for m in bundle["models"])
    return (total / len(bundle["models"]))[:, :-1]


# --------------------------------------------------------------------------- stage


def _fairness(frame: pd.DataFrame, threshold: float) -> dict:
    out = {}
    age = pd.cut(frame["age"], [17, 34, 54, 120], labels=["18-34", "35-54", "55+"]).astype(str)
    for name, groups in (("age_band", age), ("income_band", frame["declared_income_band"]), ("region", frame["region"]),
                         ("vulnerable", frame["vulnerability_flag"].astype(str))):
        rows = {}
        for g, part in frame.groupby(groups.values):
            y, p = part["took_up"].to_numpy().astype(int), part["propensity"].to_numpy()
            rows[str(g)] = {"n": int(len(part)), "take_up_rate": round(float(y.mean()), 4), "mean_propensity": round(float(p.mean()), 4),
                            "selected_rate": round(float((p >= threshold).mean()), 4),
                            "roc_auc": round(float(roc_auc_score(y, p)), 4) if 0 < y.sum() < len(y) else None}
        out[name] = rows
    return out


def run(data_dir: Path, out_dir: Path, cfg: dict, truth_dir: Path | None = None, seed: int = 7,
        products: list[str] | None = None, ablations: bool = True) -> dict:
    """Train, evaluate and score every product. Writes models, reports, test predictions and live scores."""
    t0 = time.time()
    products = products or PRODUCTS
    for sub in ("models", "metrics", "predictions", "scores"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    train_table = model_table(data_dir, "2026-03-31")
    live_table = model_table(data_dir, "2026-09-30")
    warehouse = data_dir / "warehouse"
    labels = takeup_labels(pd.read_parquet(warehouse / "product_holdings.parquet"), train_table[["customer_id"]], *LABEL_WINDOW)
    eligibility = evaluate_eligibility(train_table)
    live_eligibility = evaluate_eligibility(live_table)
    oracle = (pd.read_parquet(truth_dir / "labels.parquet", columns=["customer_id", "product", "p_true"])
              if truth_dir is not None and (truth_dir / "labels.parquet").exists() else None)
    report: dict = {"products": {}, "ablations": {}, "config": {k: cfg[k] for k in ("optuna_trials", "bootstrap_models")}}
    test_rows, live_rows = [], []
    thresholds: dict[str, dict] = {}

    for i, product in enumerate(products):
        p0 = time.time()
        el = eligibility[eligibility["product"] == product].set_index("customer_id")
        frame = train_table.join(el[["eligible", "human_review"]], on="customer_id")
        frame = frame.merge(labels[labels["product"] == product][["customer_id", "took_up", "split"]], on="customer_id")
        frame = frame[frame["eligible"]].reset_index(drop=True)
        X_all = design_matrix(frame.drop(columns=["took_up", "split", "eligible", "human_review"]), product)
        columns = curated_columns(product, list(X_all.columns))
        X = X_all[columns]
        y = frame["took_up"].to_numpy().astype(int)
        split = frame["split"].to_numpy()
        bundle = fit_product(X, y, split, cfg, seed + i)
        test = split == "test"
        p_cal, cv = predict(bundle, X)
        p_raw, _ = predict(bundle, X, calibrated=False)
        coverage = coverage_score(frame, product)
        confidence = np.sqrt(coverage) * (1.0 / (1.0 + cv))
        logistic = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
        logistic.fit(X[split == "train"].fillna(0.0), y[split == "train"])
        p_logit = logistic.predict_proba(X.fillna(0.0))[:, 1]
        p_rules = rules_baseline(frame, product, split == "train")

        # Other gradient-boosting libraries as benchmarks (uncalibrated), when installed.
        benchmarks = {}
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=6, random_state=seed, eval_metric="logloss")
            xgb_model.fit(X[split == "train"], y[split == "train"])
            benchmarks["xgboost_raw"] = xgb_model.predict_proba(X)[:, 1]
        except ImportError:
            log.info("xgboost not installed: benchmark skipped")
        try:
            from catboost import CatBoostClassifier
            cb_model = CatBoostClassifier(iterations=200, learning_rate=0.03, depth=6, random_seed=seed, verbose=False,
                                          allow_writing_files=False)
            cb_model.fit(X[split == "train"].fillna(0.0), y[split == "train"])
            benchmarks["catboost_raw"] = cb_model.predict_proba(X.fillna(0.0))[:, 1]
        except ImportError:
            log.info("catboost not installed: benchmark skipped")

        scores = {"rules_baseline": p_rules, "logistic_regression": p_logit, "lightgbm_raw": p_raw, "lightgbm_calibrated": p_cal,
                  **benchmarks}

        metrics = {name: score_metrics(y[test], s[test]) for name, s in scores.items()}
        if oracle is not None:
            o = frame[["customer_id"]].merge(oracle[oracle["product"] == product], on="customer_id", how="left")["p_true"].fillna(0).to_numpy()
            metrics["oracle_p_true"] = score_metrics(y[test], o[test])
            base, model, best = (metrics["rules_baseline"].get("roc_auc"), metrics["lightgbm_calibrated"].get("roc_auc"),
                                 metrics["oracle_p_true"].get("roc_auc"))
            if None not in (base, model, best) and best > base:
                metrics["share_of_oracle_gap_closed"] = round((model - base) / (best - base), 3)
        frame["propensity"] = p_cal
        threshold = float(np.quantile(p_cal[split == "train"], 0.90))
        # Suggestion threshold: calibrated propensity at least `min_lift` x the product's take-up rate among eligible
        # training customers, i.e. clearly more likely than an average eligible customer. Checked on validation and test.
        base_rate = float(y[split == "train"].mean())
        cut = cfg.get("min_lift", 2.0) * base_rate
        thresholds[product] = {"base_rate": round(base_rate, 5), "min_propensity": round(cut, 5)}
        at_threshold = {}
        for part in ("validation", "test"):
            m = split == part
            chosen = m & (p_cal >= cut)
            rate = float(y[chosen].mean()) if chosen.any() else None
            at_threshold[part] = {"selected_share": round(float(chosen.sum() / max(1, m.sum())), 3),
                                  "takeup_rate_selected": None if rate is None else round(rate, 4),
                                  "lift": None if rate is None or y[m].mean() == 0 else round(rate / float(y[m].mean()), 2),
                                  "recall": round(float(y[chosen].sum() / max(1, y[m].sum())), 3)}
        report["products"][product] = {
            "eligible_customers": int(len(frame)), "positives": {s: int(y[split == s].sum()) for s in ("train", "validation", "test")},
            "features_used": len(columns), "lightgbm_rounds": bundle["rounds"], "validation_pr_auc": bundle["validation_pr_auc"],
            "test": metrics, "mean_confidence_test": round(float(confidence[test].mean()), 3),
            "threshold": thresholds[product] | {"min_lift": cfg.get("min_lift", 2.0), "at_threshold": at_threshold},
            "fairness_test": _fairness(frame[test], threshold),
            "top_features": _top_features(bundle, X),
        }
        joblib.dump(bundle, out_dir / "models" / f"propensity_{product}.joblib")
        held_out = split != "train"
        test_rows.append(pd.DataFrame({"customer_id": frame["customer_id"][held_out].values, "product": product,
                                       "propensity": p_cal[held_out].round(5), "confidence": confidence[held_out].round(4),
                                       "rules_score": np.asarray(p_rules)[held_out].round(5),
                                       "took_up": y[held_out], "split": split[held_out]}))
        live_rows.append(score_live(bundle, live_table, live_eligibility, product))
        if ablations:
            report["ablations"][product] = run_ablations(frame, X_all, y, split, product, cfg, seed + i)
        log.info("propensity %s: %s eligible, test AUC %s (oracle %s) in %.0fs", product, len(frame),
                 metrics["lightgbm_calibrated"].get("roc_auc"), metrics.get("oracle_p_true", {}).get("roc_auc"), time.time() - p0)

    pd.concat(test_rows, ignore_index=True).to_csv(out_dir / "predictions" / "heldout_predictions.csv", index=False)
    (out_dir / "models" / "thresholds.json").write_text(json.dumps(thresholds, indent=1))
    pd.concat(live_rows, ignore_index=True).to_parquet(out_dir / "scores" / "live_scores.parquet", index=False)
    report["elapsed_seconds"] = round(time.time() - t0, 1)
    (out_dir / "metrics" / "propensity_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def _top_features(bundle: dict, X: pd.DataFrame, k: int = 10) -> list:
    gains = sum(m.feature_importance(importance_type="gain") for m in bundle["models"])
    order = np.argsort(-gains)[:k]
    total = gains.sum() or 1.0
    return [{"feature": bundle["columns"][j], "gain_share": round(float(gains[j] / total), 4)} for j in order]


def score_live(bundle: dict, table: pd.DataFrame, eligibility: pd.DataFrame, product: str) -> pd.DataFrame:
    """Live snapshot: every customer gets a row; eligible ones get propensity, confidence, reason codes and evidence."""
    el = eligibility[eligibility["product"] == product].set_index("customer_id")
    frame = table.join(el[["eligible", "blocks", "human_review"]], on="customer_id")
    out = pd.DataFrame({"customer_id": frame["customer_id"], "product": product, "as_of": frame["as_of"],
                        "eligible": frame["eligible"], "blocks": frame["blocks"], "human_review": frame["human_review"],
                        "propensity": np.nan, "confidence": np.nan})
    reason_codes = np.empty(len(out), dtype=object)
    evidence_text = np.empty(len(out), dtype=object)
    for j in range(len(out)):
        reason_codes[j], evidence_text[j] = [], []
    ok = frame["eligible"].to_numpy()
    if ok.any():
        rows = frame[ok].reset_index(drop=True)
        X = design_matrix(rows, product).reindex(columns=bundle["columns"])
        p, cv = predict(bundle, X)
        confidence = np.sqrt(coverage_score(rows, product)) * (1.0 / (1.0 + cv))
        reasons = reason_contributions(contributions(bundle, X), bundle["columns"], product)
        codes, evidence = top_reasons(reasons, rows)
        idx = np.flatnonzero(ok)
        out.loc[idx, "propensity"] = p.round(5)
        out.loc[idx, "confidence"] = confidence.round(4)
        for j, row in enumerate(idx):
            reason_codes[row], evidence_text[row] = codes[j], evidence[j]
    out["reason_codes"] = reason_codes
    out["evidence"] = evidence_text
    return out


def run_ablations(frame: pd.DataFrame, X_all: pd.DataFrame, y: np.ndarray, split: np.ndarray, product: str, cfg: dict,
                  seed: int) -> dict:
    """Retrain with one ingredient removed (fewer search trials) and compare test metrics."""
    fast = cfg | {"optuna_trials": max(4, cfg["optuna_trials"] // 4), "bootstrap_models": 3}
    test = split == "test"
    variants = {
        "full": curated_columns(product, list(X_all.columns)),
        "price_normalisation_off": curated_columns(product, list(X_all.columns), drop=set(PHYSICAL_FEATURES)),
        "coverage_flags_off": curated_columns(product, list(X_all.columns), drop=set(COVERAGE_FEATURES)),
        "memory_off": [c for c in curated_columns(product, list(X_all.columns)) if not c.startswith(MEMORY_PREFIX)],
        "all_features": list(X_all.columns),
    }
    out = {}
    for name, columns in variants.items():
        bundle = fit_product(X_all[columns], y, split, fast, seed)
        p, _ = predict(bundle, X_all[columns])
        out[name] = score_metrics(y[test], p[test]) | {"features": len(columns)}
        if name == "full":
            raw, _ = predict(bundle, X_all[columns], calibrated=False)
            out["calibration_off"] = score_metrics(y[test], raw[test]) | {"features": len(columns)}
    return out
