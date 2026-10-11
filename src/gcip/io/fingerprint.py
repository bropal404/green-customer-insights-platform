"""Order-independent fingerprints of tables, for auditing ingestion.

Each cell is written in a canonical text form (dates to the second, numbers to
4 decimals, nulls empty) so the same record gives the same hash whatever dtype or
file format carried it. A table's fingerprint is its row count plus the sum of
its row hashes, so row order does not matter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEP = "\x1f"
# Every booked-transaction field except the id, which is the join key.
TRANSACTION_AUDIT_COLUMNS = ["account_id", "booking_datetime", "amount", "currency", "credit_debit_indicator",
                             "transaction_information", "merchant_name", "merchant_category_code",
                             "proprietary_bank_transaction_code", "bank_transaction_code", "status"]
# Natural key of each warehouse table, used to localise a fingerprint mismatch to a column.
TABLE_KEYS = {
    "customers": ["customer_id"],
    "accounts": ["account_id"],
    "balances": ["account_id", "month_end"],
    "product_holdings": ["account_id", "product_code", "start_date"],
    "preference_changes": ["account_id", "change_date"],
    "assets": ["asset_id"],
    "digital_activity": ["customer_id", "month"],
    "campaign_responses": ["campaign_id", "customer_id", "product"],
}


def canonical(series: pd.Series) -> pd.Series:
    """Canonical text form of one column."""
    if pd.api.types.is_datetime64_any_dtype(series):
        text = np.datetime_as_string(series.values.astype("datetime64[s]"), unit="s")
        return pd.Series(np.where(text == "NaT", "", text), index=series.index, dtype=object)
    if pd.api.types.is_bool_dtype(series):
        return series.astype(str).astype(object)
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").astype(float).round(4) + 0.0  # -0.0 and 0.0 are the same amount
        text = values.astype(str)
        return text.where(values.notna(), "").astype(object)
    values = series.astype(object)
    return values.where(values.notna(), "").map(str)


def row_hashes(frame: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
    """One uint64 per row over the given columns (all columns, sorted by name, by default)."""
    columns = sorted(frame.columns) if columns is None else list(columns)
    if not len(frame):
        return np.zeros(0, dtype=np.uint64)
    text = canonical(frame[columns[0]])
    for column in columns[1:]:
        text = text + SEP + canonical(frame[column])
    return pd.util.hash_array(text.values.astype(object))


def _digest(hashes: np.ndarray) -> str:
    return f"{int(hashes.sum(dtype=np.uint64)):016x}"


def table_fingerprint(frame: pd.DataFrame, key: list[str] | None = None) -> dict:
    """Row count and order-independent hash; with a key, also one hash per column (key + that column)."""
    out = {"rows": int(len(frame)), "columns": sorted(frame.columns), "hash": _digest(row_hashes(frame))}
    if key:
        out["column_hashes"] = {c: _digest(row_hashes(frame, key + [c])) for c in sorted(frame.columns) if c not in key}
    return out
