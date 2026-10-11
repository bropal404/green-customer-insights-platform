"""Pipeline stage for FR2: train, evaluate and apply the categoriser."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from gcip.categorise.model import (CATEGORIES, Categoriser, boundary_errors, choose_unseen_merchants, collect_signatures,
                                   confusion, evaluate)
from gcip.categorise.normalise import add_signature, normalise_text, split_signature
from gcip.config import BENCHMARK_DIR
from gcip.pipeline.splits import customer_split


def make_finbert(model_name: str, batch_size: int, max_length: int, log):
    """Return a cached FinBERT text embedder using masked mean pooling."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as error:
        raise RuntimeError("FinBERT requires torch and transformers; install requirements-llm.txt") from error
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    cache: dict[str, np.ndarray] = {}

    def embed(texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            cache.update(zip(batch, vectors.cpu().numpy().astype(np.float32)))
        return np.stack([cache[t] for t in texts])

    log.info("FinBERT: %s on %s", model_name, device)
    return embed


def ood_benchmark(cat: Categoriser, finbert_fn) -> dict:
    """Coarse accuracy on DoDataThings US transactions (MIT): strings our generator never wrote."""
    path = BENCHMARK_DIR / "dodatathings_us_bank_categories.csv"
    if not path.exists():
        return {}
    data = pd.read_csv(path)
    direction = np.where(data["description"].str.startswith("[credit]"), "C", "D")
    text = data["description"].str.replace(r"^\[(debit|credit)\]\s*", "", regex=True)
    parts = pd.DataFrame({"ptype": np.where(direction == "C", "BGC", "DEB"), "mcc": "", "amount_band": 3, "round_amount": 0,
                          "direction": direction, "norm_text": text.map(normalise_text)})
    probas = cat.model_proba(parts, finbert_fn)
    pred = cat.predict(parts, probas["finbert"])
    coarse_ours = {"Fuel": "transport", "Public Transport": "transport", "Automotive": "transport",
                   "Electricity": "home_energy_utilities", "Gas": "home_energy_utilities", "Utilities": "home_energy_utilities",
                   "Home Improvement": "other", "Other": "other"}
    coarse_theirs = {"Transportation": "transport", "Utilities": "home_energy_utilities"}
    truth = data["category"].map(coarse_theirs).fillna("other").values
    ours = pred["pred_category"].map(coarse_ours).values
    table = pd.crosstab(pd.Series(truth, name="their_coarse"), pd.Series(ours, name="our_coarse"))
    return {"rows": int(len(data)), "coarse_accuracy": round(float((truth == ours).mean()), 4),
            "transport_recall": round(float((ours[truth == "transport"] == "transport").mean()), 4),
            "home_energy_utilities_recall": round(float((ours[truth == "home_energy_utilities"] == "home_energy_utilities").mean()), 4),
            "confusion": table.to_dict(),
            "note": "US merchants with no MCC or amount; their Transportation includes ride-hailing, which our taxonomy puts in Other (rule B7)."}


def run(ctx) -> dict:
    cfg = ctx.cfg["categoriser"]
    log = ctx.log
    customers = pd.read_parquet(ctx.warehouse / "customers.parquet", columns=["customer_id"])
    split_of = customer_split(customers["customer_id"])

    sig_path = ctx.path("categoriser", "signature_table.parquet")
    if sig_path.exists():
        table = pd.read_parquet(sig_path)
    else:
        table = collect_signatures(ctx.warehouse, ctx.truth, split_of)
        ctx.save_parquet(table, "categoriser", "signature_table.parquet")
    unseen = choose_unseen_merchants(table, cfg["unseen_merchant_share"], seed=11)
    table["unseen_merchant"] = table["merchant_key"].isin(unseen)
    log.info("categoriser: %s unique signature rows, %s merchants held out", len(table), len(unseen))

    train = table[(table["n_train"] > 0) & ~table["unseen_merchant"]][["signature", "true_sub_category", "n_train"]]
    finbert_fn = make_finbert(cfg["finbert"]["model"], cfg["finbert"]["batch_size"], cfg["finbert"]["max_length"], log)
    model = Categoriser(cfg)
    model.fit(train, finbert_fn)

    # Every unique signature once: shared by evaluation and by the final application.
    signatures = pd.Series(table["signature"].unique())
    parts = split_signature(signatures)
    probas = model.model_proba(parts, finbert_fn)
    sig_index = pd.Index(signatures)

    def slice_frame(split: str, unseen_flag: bool) -> pd.DataFrame:
        col = {"validation": "n_validation", "test": "n_test"}[split]
        return table[(table[col] > 0) & (table["unseen_merchant"] == unseen_flag)][["signature", "true_sub_category", col]].rename(columns={col: "n"})

    sub2cat = model.sub2cat
    combined = probas["finbert"]

    # Evaluate configurations (the categoriser-layer ablation) on every slice.
    configs = {
        "rules_only": dict(use_rules=True, use_lookup=False, use_model=False),
        "rules_lookup": dict(use_rules=True, use_lookup=True, use_model=False),
        "model_only": dict(use_rules=False, use_lookup=False, use_model=True),
        "full": dict(use_rules=True, use_lookup=True, use_model=True),
    }
    report: dict = {"model": "FinBERT embeddings + logistic regression head", "held_out_merchants": len(unseen), "slices": {}}
    for split in ("validation", "test"):
        for unseen_flag in (False, True):
            frame = slice_frame(split, unseen_flag)
            idx = sig_index.get_indexer(frame["signature"])
            sub_parts = parts.iloc[idx].reset_index(drop=True)
            name = f"{split}_{'unseen' if unseen_flag else 'seen'}_merchants"
            report["slices"][name] = {}
            for label, flags in configs.items():
                pred = model.predict(sub_parts, combined[idx], **flags)
                report["slices"][name][label] = evaluate(pred, frame["true_sub_category"].values, frame["n"].values, sub2cat)
                if label == "full":
                    true_cat = np.array([sub2cat.get(s, "Other") for s in frame["true_sub_category"].values])
                    report["slices"][name]["boundary_errors"] = boundary_errors(pred["pred_category"].values, true_cat, frame["n"].values.astype(float))
                    if split == "test":
                        ctx.save_json(confusion(pred["pred_category"].values, true_cat, frame["n"].values).to_dict(),
                                      "categoriser", f"confusion_{name}.json")
    report["ood_dodatathings"] = ood_benchmark(model, finbert_fn)

    # Apply to every transaction (all customers) and write categorised parts.
    all_pred = model.predict(parts, combined)
    all_pred["signature"] = signatures.values
    lookup = all_pred.set_index("signature")
    out_dir = ctx.data_dir / "categorised"
    out_dir.mkdir(parents=True, exist_ok=True)
    for tx_path in sorted((ctx.warehouse / "transactions").glob("part-*.parquet")):
        target = out_dir / tx_path.name
        tx = pd.read_parquet(tx_path)
        tx = add_signature(tx)
        joined = lookup.reindex(tx["signature"].values)
        out = pd.DataFrame({
            "transaction_id": tx["transaction_id"].values,
            "customer_id": tx["account_id"].str.rsplit("-", n=1).str[0].values,
            "account_id": tx["account_id"].values,
            "booking_datetime": tx["booking_datetime"].values,
            "amount": tx["amount"].values,
            "credit_debit_indicator": tx["credit_debit_indicator"].values,
            "transaction_information": tx["transaction_information"].values,
            "merchant_category_code": tx["merchant_category_code"].values,
            "payment_type": tx["proprietary_bank_transaction_code"].values,
            "pred_category": joined["pred_category"].values,
            "pred_sub_category": joined["pred_sub_category"].values,
            "cat_confidence": joined["cat_confidence"].values,
            "layer": joined["layer"].values,
        })
        out.to_parquet(target, index=False, compression="zstd")
    model.save(ctx.path("models", "categoriser.joblib"))
    ctx.save_json(report, "metrics", "categoriser_report.json")
    return report
