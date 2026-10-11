"""Train, evaluate and apply the layered transaction categoriser (FR2).

Layers, in order: rules (rules.py) -> merchant lookup (exact normalised text seen
in training with >=95% purity) -> ensemble of text models: TF-IDF + linear model (default), and optionally sentence
embeddings + logistic regression, FinBERT embeddings + logistic regression (finance-domain language model), or a
fine-tuned DistilBERT; the ensemble weights are chosen on validation (unseen merchants).
The model predicts a sub-category; the category is its roll-up, so every
transaction gets exactly one of the eight categories.

Evaluation is weighted by transaction counts and split two ways: by customer
(train/validation/test, as everywhere) and by merchant (20% of merchants never
appear in training) so accuracy on unseen merchants measures generalisation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

from gcip.config import CONFIG_DIR
from gcip.categorise.normalise import add_signature, model_text, split_signature
from gcip.categorise.rules import apply_rules

log = logging.getLogger("gcip.categorise")
CATEGORIES = ["Fuel", "Electricity", "Gas", "Public Transport", "Home Improvement", "Automotive", "Utilities", "Other"]


def sub_to_category() -> dict[str, str]:
    taxonomy = yaml.safe_load((CONFIG_DIR / "taxonomy.yaml").read_text())
    return {sub: cat for cat, spec in taxonomy["categories"].items() for sub in spec["sub_categories"]}


# --------------------------------------------------------------------------- data


def collect_signatures(warehouse: Path, truth_dir: Path, split_of_customer: pd.Series) -> pd.DataFrame:
    """Unique (signature, merchant, true sub-category) rows with transaction counts per split."""
    parts = []
    tx_parts = sorted((warehouse / "transactions").glob("part-*.parquet"))
    for tx_path in tx_parts:
        truth_path = truth_dir / "transaction_truth" / tx_path.name
        tx = pd.read_parquet(tx_path, columns=["transaction_id", "account_id", "amount", "credit_debit_indicator",
                                               "transaction_information", "merchant_category_code",
                                               "proprietary_bank_transaction_code"])
        truth = pd.read_parquet(truth_path, columns=["transaction_id", "customer_id", "true_sub_category", "merchant_key"])
        frame = add_signature(tx).merge(truth, on="transaction_id")
        frame["split"] = frame["customer_id"].map(split_of_customer)
        grouped = frame.groupby(["signature", "merchant_key", "true_sub_category", "split"]).size().unstack(fill_value=0)
        parts.append(grouped)
    table = pd.concat(parts).groupby(level=[0, 1, 2]).sum().reset_index()
    for split in ("train", "validation", "test"):
        if split not in table:
            table[split] = 0
    return table.rename(columns={"train": "n_train", "validation": "n_validation", "test": "n_test"})


def choose_unseen_merchants(table: pd.DataFrame, share: float, seed: int) -> set[str]:
    """Hold out a share of merchants per sub-category, keeping at least one merchant in training."""
    rng = np.random.default_rng(seed)
    merchants = table.groupby(["true_sub_category", "merchant_key"]).size().reset_index()[["true_sub_category", "merchant_key"]]
    unseen: set[str] = set()
    for _, grp in merchants.groupby("true_sub_category"):
        keys = grp["merchant_key"].values
        if len(keys) < 2:
            continue
        k = min(int(round(len(keys) * share)), len(keys) - 1)
        unseen.update(rng.choice(keys, k, replace=False))
    return unseen


# --------------------------------------------------------------------------- models


class Categoriser:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.sub2cat = sub_to_category()
        self.classes: np.ndarray | None = None
        self.lookup: dict[str, tuple[str, float]] = {}
        self.char_vec: TfidfVectorizer | None = None
        self.word_vec: TfidfVectorizer | None = None
        self.linear: SGDClassifier | None = None
        self.onehot: OneHotEncoder | None = None
        self.emb_lr: LogisticRegression | None = None
        self.embedder = None
        self.finbert_lr: LogisticRegression | None = None
        self.weights = {"tfidf": 1.0, "embedding": 0.0, "transformer": 0.0, "finbert": 0.0}

    # ---------------------------------------------------------- training
    def fit(self, train: pd.DataFrame, embed_fn=None, finbert_fn=None) -> None:
        """``train`` has signature, true_sub_category, n_train (one row per unique signature and label)."""
        cfg = self.cfg
        cap = cfg["max_train_signatures"]
        if len(train) > cap:
            prob = np.sqrt(train["n_train"].values)
            prob = prob / prob.sum()
            keep = np.random.default_rng(0).choice(len(train), cap, replace=False, p=prob)
            train = train.iloc[np.sort(keep)]
        parts = split_signature(train["signature"])
        text = model_text(parts)
        y = train["true_sub_category"].values
        weight = np.minimum(np.sqrt(train["n_train"].values), 30.0)
        self.classes = np.unique(y)

        # Merchant lookup on (payment type, MCC, normalised text): majority label with its purity.
        key = parts["ptype"] + "|" + parts["mcc"] + "|" + parts["norm_text"]
        counts = (pd.DataFrame({"key": key.values, "sub": y, "n": train["n_train"].values})
                  .groupby(["key", "sub"], as_index=False)["n"].sum())
        totals = counts.groupby("key")["n"].sum()
        top = counts.sort_values("n", ascending=False).drop_duplicates("key").set_index("key")
        top["total"] = totals.reindex(top.index).values
        top["purity"] = top["n"] / top["total"]
        good = top[(top["total"] >= cfg["lookup_min_count"]) & (top["purity"] >= cfg["lookup_min_purity"])]
        self.lookup = dict(zip(good.index, zip(good["sub"], good["purity"].astype(float))))

        log.info("categoriser: %s training signatures, %s classes, %s lookup keys", len(train), len(self.classes), len(self.lookup))
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(cfg["tfidf_char"]["ngram_range"]),
                                        min_df=cfg["tfidf_char"]["min_df"], max_features=cfg["tfidf_char"]["max_features"],
                                        sublinear_tf=True, dtype=np.float32)
        self.word_vec = TfidfVectorizer(analyzer="word", ngram_range=tuple(cfg["tfidf_word"]["ngram_range"]),
                                        min_df=cfg["tfidf_word"]["min_df"], max_features=cfg["tfidf_word"]["max_features"],
                                        token_pattern=r"[^ ]+", sublinear_tf=True, dtype=np.float32)
        x = sparse.hstack([self.char_vec.fit_transform(parts["norm_text"]), self.word_vec.fit_transform(text)]).tocsr()
        self.linear = SGDClassifier(loss="log_loss", alpha=2e-6, max_iter=25, tol=1e-4, n_jobs=-1, random_state=0)
        self.linear.fit(x, y, sample_weight=weight)

        if embed_fn is not None:
            self.onehot = OneHotEncoder(handle_unknown="ignore")
            context = parts[["ptype", "mcc", "amount_band", "round_amount", "direction"]].astype(str)
            emb = embed_fn(parts["norm_text"].tolist())
            features = np.hstack([emb, self.onehot.fit_transform(context).toarray().astype(np.float32)])
            self.emb_lr = LogisticRegression(max_iter=400, C=8.0, n_jobs=-1)
            self.emb_lr.fit(features, y, sample_weight=weight)

        if finbert_fn is not None:
            # FinBERT gives finance-domain sentence representations; a logistic head maps them to our sub-categories.
            # Embedding is the slow part, so the head is fitted on a weighted sample of the training signatures.
            fcfg = cfg.get("finbert", {})
            rows = np.arange(len(parts))
            cap = fcfg.get("max_train_rows", 100_000)
            if len(rows) > cap:
                w = np.asarray(weight, dtype=float)
                rows = np.random.default_rng(0).choice(rows, size=cap, replace=False, p=w / w.sum())
            emb = finbert_fn(np.asarray(text)[rows].tolist())
            self.finbert_lr = LogisticRegression(max_iter=400, C=fcfg.get("logistic_C", 6.0), n_jobs=-1)
            self.finbert_lr.fit(emb, np.asarray(y)[rows], sample_weight=np.asarray(weight)[rows])

    # ---------------------------------------------------------- prediction
    def _aligned(self, model_classes: np.ndarray, proba: np.ndarray) -> np.ndarray:
        out = np.zeros((proba.shape[0], len(self.classes)), dtype=np.float32)
        index = {c: i for i, c in enumerate(self.classes)}
        for j, c in enumerate(model_classes):
            out[:, index[c]] = proba[:, j]
        return out

    @staticmethod
    def clean(parts: pd.DataFrame) -> pd.DataFrame:
        """Missing fields from a feed (null payment type, merchant code or description) become empty values, so no
        model ever reads 'nan'; amount bands default to 0."""
        if not parts[["ptype", "mcc", "direction", "norm_text", "amount_band", "round_amount"]].isna().any().any():
            return parts
        parts = parts.copy()
        for c in ("ptype", "mcc", "direction", "norm_text"):
            parts[c] = parts[c].fillna("").astype(str)
        for c in ("amount_band", "round_amount"):
            parts[c] = parts[c].fillna(0).astype(int)
        return parts

    def model_proba(self, parts: pd.DataFrame, embed_fn=None, transformer_proba: np.ndarray | None = None, finbert_fn=None,
                    batch: int = 200_000) -> dict[str, np.ndarray]:
        """Class probabilities per text model, scored in batches so memory stays flat however many signatures there are."""
        parts = self.clean(parts)
        tfidf, embedding, finbert = [], [], []
        for start in range(0, len(parts), batch):
            chunk = parts.iloc[start:start + batch]
            x = sparse.hstack([self.char_vec.transform(chunk["norm_text"]), self.word_vec.transform(model_text(chunk))]).tocsr()
            tfidf.append(self._aligned(self.linear.classes_, self.linear.predict_proba(x)).astype(np.float32))
            del x
            if self.emb_lr is not None and embed_fn is not None:
                context = chunk[["ptype", "mcc", "amount_band", "round_amount", "direction"]].astype(str)
                features = np.hstack([embed_fn(chunk["norm_text"].tolist()), self.onehot.transform(context).toarray().astype(np.float32)])
                embedding.append(self._aligned(self.emb_lr.classes_, self.emb_lr.predict_proba(features)).astype(np.float32))
            if self.finbert_lr is not None and finbert_fn is not None:
                finbert.append(self._aligned(self.finbert_lr.classes_, self.finbert_lr.predict_proba(
                    finbert_fn(model_text(chunk).tolist()))).astype(np.float32))
        probas = {"tfidf": np.vstack(tfidf) if tfidf else np.zeros((0, len(self.classes)), dtype=np.float32)}
        if embedding:
            probas["embedding"] = np.vstack(embedding)
        if finbert:
            probas["finbert"] = np.vstack(finbert)
        if transformer_proba is not None:
            probas["transformer"] = transformer_proba
        return probas

    def combine(self, probas: dict[str, np.ndarray], weights: dict[str, float] | None = None) -> np.ndarray:
        weights = weights or self.weights
        total = sum(w for k, w in weights.items() if k in probas and w > 0)
        out = sum(probas[k] * w for k, w in weights.items() if k in probas and w > 0)
        return out / max(total, 1e-9)

    def predict(self, parts: pd.DataFrame, proba: np.ndarray, use_rules: bool = True, use_lookup: bool = True,
                use_model: bool = True) -> pd.DataFrame:
        parts = self.clean(parts)
        n = len(parts)
        sub = np.full(n, "", dtype=object)
        conf = np.zeros(n)
        layer = np.full(n, "", dtype=object)
        if use_rules:
            rules = apply_rules(parts)
            fired = rules["rule_confidence"].values > 0
            sub[fired] = rules["rule_sub_category"].values[fired]
            conf[fired] = rules["rule_confidence"].values[fired]
            layer[fired] = rules["rule_id"].values[fired]
        if use_lookup and self.lookup:
            open_rows = np.where(layer == "")[0]
            key = (parts["ptype"] + "|" + parts["mcc"] + "|" + parts["norm_text"]).values[open_rows]
            hits = pd.Series(key).map(self.lookup)
            found = hits.notna().values
            rows = open_rows[found]
            sub[rows] = [h[0] for h in hits[found]]
            conf[rows] = [h[1] for h in hits[found]]
            layer[rows] = "lookup"
        # No description and no merchant code: nothing to read, so the payment is Other with low confidence. It is
        # never guessed into Fuel, Electricity or Gas, so it cannot inflate a fuel or energy estimate.
        blank = (layer == "") & (parts["norm_text"].values == "") & (parts["mcc"].values == "")
        sub[blank], conf[blank], layer[blank] = "retail", 0.3, "no_description"
        rest = layer == ""
        if use_model and rest.any():
            pr = proba[rest]
            best = pr.argmax(axis=1)
            sub[rest] = self.classes[best]
            # Category confidence: probability mass of the predicted sub-category's whole category.
            cats = np.array([self.sub2cat.get(c, "Other") for c in self.classes])
            cat_index = np.array([CATEGORIES.index(c) for c in cats])
            onehot = np.zeros((len(self.classes), len(CATEGORIES)), dtype=np.float32)
            onehot[np.arange(len(self.classes)), cat_index] = 1.0
            mass = pr @ onehot
            conf[rest] = mass[np.arange(len(best)), cat_index[best]]
            layer[rest] = "model"
        elif rest.any():
            sub[rest], conf[rest], layer[rest] = "retail", 0.3, "default"
        category = np.array([self.sub2cat.get(s, "Other") for s in sub])
        return pd.DataFrame({"pred_sub_category": sub, "pred_category": category, "cat_confidence": conf.round(4),
                             "layer": layer}, index=parts.index)

    def save(self, path: Path) -> None:
        state = {k: v for k, v in self.__dict__.items() if k != "embedder"}
        joblib.dump(state, path, compress=3)

    @classmethod
    def load(cls, path: Path) -> "Categoriser":
        obj = cls.__new__(cls)
        obj.__dict__.update(joblib.load(path))
        obj.embedder = None
        obj.__dict__.setdefault("finbert_lr", None)  # models saved before the FinBERT head existed
        return obj


# --------------------------------------------------------------------------- evaluation


def evaluate(pred: pd.DataFrame, truth_sub: np.ndarray, weight: np.ndarray, sub2cat: dict[str, str]) -> dict:
    true_cat = np.array([sub2cat.get(s, "Other") for s in truth_sub])
    w = weight.astype(float)
    if w.sum() == 0:
        return {}
    acc = accuracy_score(true_cat, pred["pred_category"].values, sample_weight=w)
    sub_acc = accuracy_score(truth_sub, pred["pred_sub_category"].values, sample_weight=w)
    macro = f1_score(true_cat, pred["pred_category"].values, average="macro", labels=CATEGORIES, sample_weight=w, zero_division=0)
    per_cat = f1_score(true_cat, pred["pred_category"].values, average=None, labels=CATEGORIES, sample_weight=w, zero_division=0)
    layers = pd.Series(w, index=pred["layer"].str.split(":").str[0].values).groupby(level=0).sum() / w.sum()
    return {"transactions": int(w.sum()), "category_accuracy": round(acc, 4), "category_macro_f1": round(macro, 4),
            "sub_category_accuracy": round(sub_acc, 4),
            "per_category_f1": {c: round(float(v), 4) for c, v in zip(CATEGORIES, per_cat)},
            "layer_share": {k: round(float(v), 4) for k, v in layers.items()}}


def confusion(pred_cat: np.ndarray, true_cat: np.ndarray, weight: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"true": true_cat, "pred": pred_cat, "w": weight})
    return frame.pivot_table(index="true", columns="pred", values="w", aggfunc="sum", fill_value=0).reindex(
        index=CATEGORIES, columns=CATEGORIES, fill_value=0)


BOUNDARY_PAIRS = {
    "B1 Fuel vs Automotive": ({"Fuel"}, {"Automotive"}),
    "B3 Fuel vs Other (forecourt shop)": ({"Fuel"}, {"Other"}),
    "B4 Electricity/Gas vs Utilities": ({"Electricity", "Gas"}, {"Utilities"}),
    "B7 Public Transport vs Other (taxis)": ({"Public Transport"}, {"Other"}),
    "B8 Home Improvement vs Other": ({"Home Improvement"}, {"Other"}),
}


def boundary_errors(pred_cat: np.ndarray, true_cat: np.ndarray, weight: np.ndarray) -> dict:
    out = {}
    for name, (a, b) in BOUNDARY_PAIRS.items():
        involved = np.isin(true_cat, list(a | b))
        crossed = (np.isin(true_cat, list(a)) & np.isin(pred_cat, list(b))) | (np.isin(true_cat, list(b)) & np.isin(pred_cat, list(a)))
        relevant = involved & (np.isin(true_cat, list(a)) | np.isin(pred_cat, list(a)))
        total = weight[relevant].sum()
        out[name] = {"transactions": int(total), "cross_boundary_error_rate": round(float(weight[crossed].sum() / total), 5) if total else None}
    return out
