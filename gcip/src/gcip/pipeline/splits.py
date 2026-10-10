"""Customer-level train/validation/test split from a salted hash of the customer id.

Every row of a customer (transactions, snapshots, labels) lands in the same split, so nothing leaks across.
The salt and shares live in configs/simulation.yaml (`splits`).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from gcip.config import load_config


def split_of(customer_ids: pd.Series, salt: str, train: float, validation: float) -> np.ndarray:
    values = np.array([int(hashlib.sha256(f"{salt}:{c}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for c in customer_ids])
    return np.where(values < train, "train", np.where(values < train + validation, "validation", "test"))


def customer_split(customer_ids: pd.Series) -> pd.Series:
    cfg = load_config("simulation")["splits"]
    ids = pd.Series(customer_ids).reset_index(drop=True)
    return pd.Series(split_of(ids, cfg["hash_salt"], cfg["train"], cfg["validation"]), index=ids.values, name="split")
