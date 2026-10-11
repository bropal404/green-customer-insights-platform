"""Per-customer document: the Customer 360 and features by section, the recommendations with their explanations, and
the audited decision for every product. This is what the dashboard's customer view and the API return."""

from __future__ import annotations

import math
from datetime import date, datetime

import numpy as np
import pandas as pd

from gcip.features.store import FEATURE_SECTIONS
from gcip.fr1.customer360 import C360_SECTIONS


def clean(value):
    """JSON-safe value: NaN/NA to None, numpy scalars to Python, dates to ISO strings, arrays to lists."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sections(row: dict, spec: dict) -> list[dict]:
    return [{"section": name, "question": s["question"], "used_by": s["used_by"],
             "fields": {c: clean(row.get(c)) for c in s["columns"] if c in row}} for name, s in spec.items()]


def document(customer_id: str, c360: pd.DataFrame, features: pd.DataFrame, recs: pd.DataFrame, explanations: pd.DataFrame,
             decisions: pd.DataFrame, events: pd.DataFrame | None = None) -> dict | None:
    base = c360[c360["customer_id"] == customer_id]
    if base.empty:
        return None
    row = base.iloc[0].to_dict()
    feat = features[features["customer_id"] == customer_id]
    frow = feat.iloc[0].to_dict() if len(feat) else {}
    r = recs[recs["customer_id"] == customer_id].sort_values("rank")
    if len(explanations):
        r = r.merge(explanations[["customer_id", "product", "explanation", "source"]], on=["customer_id", "product"], how="left")
    d = decisions[decisions["customer_id"] == customer_id]
    mem = events[events["customer_id"] == customer_id] if events is not None and len(events) else pd.DataFrame()
    return clean({
        "customer_id": customer_id,
        "as_of": row.get("as_of"),
        "customer_360": sections(row, C360_SECTIONS),
        "features": sections(frow, FEATURE_SECTIONS),
        "recommendations": r.to_dict("records"),
        "decisions": d.to_dict("records"),
        "memory_events": mem.to_dict("records"),
    })
