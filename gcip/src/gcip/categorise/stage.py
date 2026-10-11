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


def make_embedder(model_name: str, log):
    """Returns a cached text -> embedding function, or None if sentence-transformers is unavailable."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except Exception as error:  # optional dependency
        log.warning("embeddings disabled: %s", error)
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    cache: dict[str, np.ndarray] = {}

    def embed(texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        if missing:
            vectors = model.encode(missing, batch_size=512, show_progress_bar=len(missing) > 20000,
                                   convert_to_numpy=True, normalize_embeddings=True)
            cache.update(zip(missing, vectors.astype(np.float32)))
        return np.stack([cache[t] for t in texts])

    log.info("embeddings: %s on %s", model_name, device)
    return embed


def make_finbert(model_name: str, batch_size: int, max_length: int, log):
    """A cached FinBERT text embedder (masked mean pooling of the last hidden layer), or None if torch is unavailable."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as error:  # optional dependency (requirements-llm.txt)
        log.warning("FinBERT disabled: %s", error)
        return None
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
            cache.update(zip(batch, vectors.float().cpu().numpy()))
        return np.stack([cache[t] for t in texts]) if texts else np.zeros((0, model.config.hidden_size), dtype=np.float32)

    log.info("FinBERT: %s on %s", model_name, device)
    return embed


def ood_benchmark(cat: Categoriser, embed_fn, transformer_proba_fn=None, finbert_fn=None) -> dict:
    """Coarse accuracy on DoDataThings US transactions (MIT): strings our generator never wrote."""
    path = BENCHMARK_DIR / "dodatathings_us_bank_categories.csv"
    if not path.exists():
        return {}
    data = pd.read_csv(path)
    direction = np.where(data["description"].str.startswith("[credit]"), "C", "D")
    text = data["description"].str.replace(r"^\[(debit|credit)\]\s*", "", regex=True)
    parts = pd.DataFrame({"ptype": np.where(direction == "C", "BGC", "DEB"), "mcc": "", "amount_band": 3, "round_amount": 0,
                          "direction": direction, "norm_text": text.map(normalise_text)})
    probas = cat.model_proba(parts, embed_fn, finbert_fn=finbert_fn)
    pred = cat.predict(parts, cat.combine(probas))
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
    embed_fn = make_embedder(cfg["embedding_model"], log) if cfg.get("use_embeddings", True) else None
    fb = cfg.get("finbert", {})
    finbert_fn = make_finbert(fb["model"], fb.get("batch_size", 128), fb.get("max_length", 48), log) if cfg.get("use_finbert", False) else None
    model = Categoriser(cfg)
    model.fit(train, embed_fn, finbert_fn)

    # Every unique signature once: shared by evaluation and by the final application.
    signatures = pd.Series(table["signature"].unique())
    parts = split_signature(signatures)
    transformer_path = ctx.run_dir / "categoriser" / "transformer_proba.npz"
    transformer_proba = None
    if transformer_path.exists():
        saved = np.load(transformer_path, allow_pickle=True)
        order = pd.Index(saved["signatures"]).get_indexer(signatures)
        if (order >= 0).all() and list(saved["classes"]) == list(model.classes):
            transformer_proba = saved["proba"][order]
            log.info("categoriser: using fine-tuned transformer probabilities")
    probas = model.model_proba(parts, embed_fn, transformer_proba, finbert_fn=finbert_fn)
    sig_index = pd.Index(signatures)

    def slice_frame(split: str, unseen_flag: bool) -> pd.DataFrame:
        col = {"validation": "n_validation", "test": "n_test"}[split]
        return table[(table[col] > 0) & (table["unseen_merchant"] == unseen_flag)][["signature", "true_sub_category", col]].rename(columns={col: "n"})

    # Pick ensemble weights on validation, unseen merchants.
    val_unseen = slice_frame("validation", True)
    rows = sig_index.get_indexer(val_unseen["signature"])
    candidates = [{"tfidf": 1.0, "embedding": 0.0, "transformer": 0.0}]
    if "embedding" in probas:
        candidates += [{"tfidf": a, "embedding": 1 - a, "transformer": 0.0} for a in (0.3, 0.5, 0.7)]
    if "transformer" in probas:
        candidates += [{"tfidf": a, "embedding": b, "transformer": 1 - a - b} for a, b in ((0.2, 0.2), (0.3, 0.2), (0.4, 0.0), (0.0, 0.0))]
    if "finbert" in probas:
        candidates += [{"tfidf": a, "embedding": 0.0, "transformer": 0.0, "finbert": 1 - a} for a in (0.0, 0.3, 0.5, 0.7)]
    best, best_acc = candidates[0], -1.0
    sub2cat = model.sub2cat
    for weights in candidates:
        pred = model.predict(parts.iloc[rows].reset_index(drop=True), model.combine({k: v[rows] for k, v in probas.items()}, weights))
        acc = evaluate(pred, val_unseen["true_sub_category"].values, val_unseen["n"].values, sub2cat).get("category_accuracy", 0)
        if acc > best_acc:
            best, best_acc = weights, acc
    model.weights = best
    log.info("categoriser: ensemble weights %s (validation unseen-merchant accuracy %.4f)", best, best_acc)
    combined = model.combine(probas)

    # Evaluate configurations (the categoriser-layer ablation) on every slice.
    configs = {
        "rules_only": dict(use_rules=True, use_lookup=False, use_model=False),
        "rules_lookup": dict(use_rules=True, use_lookup=True, use_model=False),
        "model_only": dict(use_rules=False, use_lookup=False, use_model=True),
        "full": dict(use_rules=True, use_lookup=True, use_model=True),
    }
    report: dict = {"ensemble_weights": best, "held_out_merchants": len(unseen), "slices": {}}
    # Each text model on its own (model layer only, no rules or lookup), for a fair head-to-head.
    report["text_models"] = {}
    for name in probas:
        report["text_models"][name] = {}
        for split in ("validation", "test"):
            for unseen_flag, tag in ((False, "seen"), (True, "unseen")):
                sl = slice_frame(split, unseen_flag)
                r = sig_index.get_indexer(sl["signature"])
                pred = model.predict(parts.iloc[r].reset_index(drop=True), probas[name][r], use_rules=False, use_lookup=False)
                report["text_models"][name][f"{split}_{tag}"] = evaluate(pred, sl["true_sub_category"].values, sl["n"].values, sub2cat).get("category_accuracy")
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
    report["ood_dodatathings"] = ood_benchmark(model, embed_fn, finbert_fn=finbert_fn)

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
