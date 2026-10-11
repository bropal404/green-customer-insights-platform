"""Statement-text normalisation and transaction signatures.

A signature is what the categoriser sees for one transaction: normalised text,
payment type, MCC, amount band, round-amount flag and direction. Many
transactions share a signature, so training and prediction run on unique
signatures and are mapped back.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_CARD_SUFFIX = re.compile(r"\s+CD\s+\d{4}$")
_TOKEN = re.compile(r"[A-Z0-9&'./*#-]+")
AMOUNT_EDGES = np.array([5.0, 10.0, 20.0, 50.0, 100.0, 250.0, 1000.0])


def _mask_token(token: str) -> str:
    digits = sum(c.isdigit() for c in token)
    if digits >= 3 or (len(token) >= 6 and digits >= 2):
        return "#"
    return token


def normalise_text(text: str) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""  # a missing description stays missing (not the word "NONE" or "NAN")
    t = _CARD_SUFFIX.sub("", str(text).upper())
    tokens = [_mask_token(tok) for tok in _TOKEN.findall(t.replace("*", " * "))]
    out = " ".join(tokens)
    return re.sub(r"(# ?)+", "# ", out).strip()


def amount_band(amount: np.ndarray) -> np.ndarray:
    return np.digitize(amount, AMOUNT_EDGES)


def add_signature(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds norm_text, amount_band, round_amount and signature columns."""
    out = frame.copy()
    unique_text = pd.Series(out["transaction_information"].unique())
    mapping = dict(zip(unique_text, unique_text.map(normalise_text)))
    out["norm_text"] = out["transaction_information"].map(mapping)
    amount = out["amount"].values
    out["amount_band"] = amount_band(amount)
    out["round_amount"] = (np.abs(amount - np.round(amount / 5.0) * 5.0) < 0.005).astype(int)
    mcc = out["merchant_category_code"].fillna("").astype(str)
    direction = np.where(out["credit_debit_indicator"].values == "Credit", "C", "D")
    out["signature"] = (out["proprietary_bank_transaction_code"].astype(str) + "|" + mcc + "|" + out["amount_band"].astype(str)
                        + "|" + out["round_amount"].astype(str) + "|" + direction + "|" + out["norm_text"])
    return out


def split_signature(signatures: pd.Series) -> pd.DataFrame:
    parts = signatures.str.split("|", n=5, expand=True)
    parts.columns = ["ptype", "mcc", "amount_band", "round_amount", "direction", "norm_text"]
    parts["amount_band"] = parts["amount_band"].astype(int)
    parts["round_amount"] = parts["round_amount"].astype(int)
    return parts


def model_text(parts: pd.DataFrame) -> pd.Series:
    """The text the models read: context tokens plus the normalised narrative."""
    mcc = parts["mcc"].where(parts["mcc"] != "", "none")
    return ("__pt_" + parts["ptype"] + " __mcc_" + mcc + " __amt_" + parts["amount_band"].astype(str)
            + " __rnd_" + parts["round_amount"].astype(str) + " __dir_" + parts["direction"] + " " + parts["norm_text"])
