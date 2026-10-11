"""Eligibility rules: decide which products may be SHOWN, using observed data only.

Rules never say a customer qualifies; real eligibility is the bank's application
process. Each block has a fixed id from configs/products.yaml so every decision
can be explained and audited. Credit products for customers with a
vulnerability flag are not blocked but routed to human review (FCA Consumer Duty).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from gcip.config import CONFIG_DIR

PRODUCTS = ["GP01", "GP02", "GP03", "GP04", "GP05", "GP06"]


def catalogue() -> dict:
    return yaml.safe_load((CONFIG_DIR / "products.yaml").read_text())["products"]


def _flag(frame: pd.DataFrame, column: str) -> np.ndarray:
    return frame[column].fillna(0).astype(float).values > 0


def block_masks(f: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Per product: block id -> boolean mask (True = blocked for that reason)."""
    arrears = _flag(f, "in_arrears")
    difficulty = _flag(f, "financial_difficulty")
    vulnerable = f["vulnerability_flag"].fillna(False).astype(bool).values
    vehicle_age = f["asset_vehicle_age_years"].astype(float).values
    ev_owner = _flag(f, "ev_owner")
    mortgage = _flag(f, "has_mortgage_evidence")
    renter = _flag(f, "renter_evidence")
    owner = _flag(f, "homeowner_evidence")
    return {
        "GP01": {
            "NO_VEHICLE": ~_flag(f, "vehicle_evidence"),
            "EV_OWNER_NO_REPLACEMENT": ev_owner & ~(np.nan_to_num(vehicle_age, nan=0.0) >= 8),
            "IN_ARREARS": arrears,
            "FINANCIAL_DIFFICULTY": difficulty,
            "ALREADY_HOLDS": f["holds_GP01"].fillna(False).astype(bool).values,
        },
        "GP02": {
            "NOT_MORTGAGED_OWNER": ~mortgage,
            "RENTER": renter & ~mortgage,
            "TOP_EPC_RATING": _flag(f, "epc_high_known"),
            "IN_ARREARS": arrears,
            "FINANCIAL_DIFFICULTY": difficulty,
            "ALREADY_HOLDS": f["holds_GP02"].fillna(False).astype(bool).values,
        },
        "GP03": {
            "NOT_HOMEOWNER": ~owner,
            "RENTER": renter & ~owner,
            "FLAT": _flag(f, "flat_known"),
            "SOLAR_FITTED": _flag(f, "solar_evidence"),
            "IN_ARREARS": arrears,
            "FINANCIAL_DIFFICULTY": difficulty,
            "ALREADY_HOLDS": f["holds_GP03"].fillna(False).astype(bool).values,
        },
        "GP04": {
            "ALREADY_PAPERLESS": ~_flag(f, "paper_statements"),
            "DIGITALLY_INACTIVE": _flag(f, "digitally_inactive"),
            "VULNERABLE": vulnerable,
        },
        "GP05": {
            "NO_INVESTMENTS": ~_flag(f, "investor_evidence"),
            "FINANCIAL_DIFFICULTY": difficulty | arrears,
            "VULNERABLE": vulnerable,
            "ALREADY_HOLDS": f["holds_GP05"].fillna(False).astype(bool).values,
        },
        "GP06": {
            "IN_ARREARS": arrears,
            "FINANCIAL_DIFFICULTY": difficulty,
            "HIGH_DEBT": f["debt_to_income_est"].fillna(0).values > 0.25,
            "ALREADY_HOLDS": f["holds_GP06"].fillna(False).astype(bool).values,
        },
    }


def evaluate_eligibility(features: pd.DataFrame) -> pd.DataFrame:
    """Long table: customer_id, product, eligible, blocks ('|'-joined ids), human_review."""
    cat = catalogue()
    masks = block_masks(features)
    vulnerable = features["vulnerability_flag"].fillna(False).astype(bool).values
    rows = []
    for product in PRODUCTS:
        declared = {b["id"] for b in cat[product]["blocks"]}
        product_masks = masks[product]
        unknown = set(product_masks) - declared
        if unknown:
            raise KeyError(f"{product}: rules {unknown} are not declared in configs/products.yaml")
        ids = list(product_masks)
        matrix = np.column_stack([product_masks[i] for i in ids])
        blocks = np.array(["|".join(np.array(ids)[row]) for row in matrix])
        human = vulnerable & ("VULNERABLE" in cat[product].get("human_review", []))
        rows.append(pd.DataFrame({
            "customer_id": features["customer_id"].values,
            "product": product,
            "eligible": ~matrix.any(axis=1),
            "blocks": blocks,
            "human_review": human & ~matrix.any(axis=1),
        }))
    return pd.concat(rows, ignore_index=True)
