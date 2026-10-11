"""Render generated rows into Open Banking-shaped statement lines plus truth records.

Observed columns follow the UK Open Banking OBTransaction model (TransactionInformation,
MerchantDetails, BankTransactionCode, ProprietaryBankTransactionCode, CreditDebitIndicator).
Merchant category codes exist only on card payments, as in real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gcip.io.fingerprint import TRANSACTION_AUDIT_COLUMNS, row_hashes
from gcip.simulate.lexicon import FIRST_NAMES, SURNAMES, Merchants
from gcip.simulate.transactions import CHANNEL_PTYPE, PTYPE_ISO

_B36 = np.array(list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"))


def _base36(values: np.ndarray, width: int = 8) -> np.ndarray:
    values = values.astype(np.uint64)
    chars = np.empty((len(values), width), dtype="<U1")
    for i in range(width - 1, -1, -1):
        chars[:, i] = _B36[(values % 36).astype(int)]
        values = values // 36
    return np.array(["".join(r) for r in chars])


def render(frame: pd.DataFrame, state: pd.DataFrame, merchants: Merchants, chunk_id: int,
           rng: np.random.Generator, mcc_present: float = 0.97) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    table = merchants.table
    cust = frame["cust"].values
    mid = frame["merchant_id"].values.astype(np.int64)
    n = len(frame)

    channel = table["channel"].values[mid]
    ptype = frame["ptype"].values.astype(object)
    blank = (ptype == "") | pd.isna(ptype)
    ptype[blank] = np.array([CHANNEL_PTYPE.get(c, "DEB") for c in channel[blank]], dtype=object)

    # Variant: mostly stable per (customer, merchant), sometimes another form of the same name.
    n_var = np.array([len(v) for v in table["variants"].values])[mid]
    cust_num = state["customer_num"].values[cust].astype(np.uint64)
    stable = ((cust_num * np.uint64(31) + mid.astype(np.uint64) * np.uint64(7)) % n_var.astype(np.uint64)).astype(np.int64)
    var = np.where(rng.random(n) < 0.82, stable, rng.integers(0, 1 << 30, n) % n_var)

    # Placeholder values.
    store = ((cust_num * np.uint64(2654435761) + mid.astype(np.uint64) * np.uint64(40503)) % np.uint64(9000) + np.uint64(1000)).astype(np.int64)
    pair = cust_num * np.uint64(1000003) + mid.astype(np.uint64) * np.uint64(7919)
    unique_pairs, inverse = np.unique(pair, return_inverse=True)
    ref = _base36(unique_pairs % np.uint64(36 ** 8))[inverse]
    town = state["town"].values[cust]
    reg = state["car_reg"].values[cust]
    num = state["account_suffix"].values[cust]
    employer = state["employer_name"].values[cust]
    pension = state["pension_provider"].values[cust]
    surname = np.array(SURNAMES)[rng.integers(0, len(SURNAMES), n)]
    first = np.array(FIRST_NAMES)[rng.integers(0, len(FIRST_NAMES), n)]

    desc = np.empty(n, dtype=object)
    variants = table["variants"].values
    order = np.lexsort((var, mid))
    keys = mid[order] * 64 + var[order]
    boundaries = np.flatnonzero(np.diff(keys)) + 1
    for group in np.split(order, boundaries):
        template = variants[mid[group[0]]][var[group[0]]]
        if "{" not in template:
            desc[group] = template
            continue
        out = []
        for i in group:
            text = template
            if "{loc}" in text:
                text = text.replace("{loc}", town[i])
            if "{la}" in text:
                text = text.replace("{la}", town[i])
            if "{store}" in text:
                text = text.replace("{store}", str(store[i]))
            if "{ref}" in text:
                text = text.replace("{ref}", ref[i])
            if "{reg}" in text:
                text = text.replace("{reg}", reg[i])
            if "{num}" in text:
                text = text.replace("{num}", num[i])
            if "{employer}" in text:
                text = text.replace("{employer}", employer[i])
            if "{pension}" in text:
                text = text.replace("{pension}", pension[i])
            if "{surname}" in text:
                text = text.replace("{surname}", surname[i])
            if "{first}" in text:
                text = text.replace("{first}", first[i])
            out.append(text)
        desc[group] = out

    card4 = np.where(frame["on_card"].values, state["credit_card_last4"].values[cust], state["debit_card_last4"].values[cust])
    is_card = np.isin(ptype, ["DEB", "CPT"])
    info = np.where(
        is_card, pd.Series(desc).str.slice(0, 25).str.cat(card4, sep=" CD ").values,
        np.where(np.isin(ptype, ["DD", "SO"]), pd.Series(desc).str.slice(0, 26).str.rstrip().values,
                 pd.Series(desc).str.slice(0, 34).str.rstrip().values),
    )
    mcc = table["mcc"].values[mid].astype(object)
    has_mcc = (ptype == "DEB") & (mcc != "") & (rng.random(n) < mcc_present)
    seconds = np.where(is_card, rng.integers(7 * 3600, 22 * 3600 + 1800, n), 0).astype("timedelta64[s]")
    booked = frame["date"].values.astype("datetime64[s]") + seconds
    customer_id = state["customer_id"].values[cust]
    account_id = np.where(frame["on_card"].values, pd.Series(customer_id) + "-CC", pd.Series(customer_id) + "-CA")
    tx_id = np.array([f"T{chunk_id:04d}{i:07d}" for i in range(n)])

    observed = pd.DataFrame({
        "transaction_id": tx_id,
        "account_id": account_id,
        "booking_datetime": booked,
        "amount": frame["amount"].values,
        "currency": "GBP",
        "credit_debit_indicator": np.where(frame["credit"].values, "Credit", "Debit"),
        "transaction_information": info,
        "merchant_name": np.where(ptype == "DEB", pd.Series(desc).str.slice(0, 30).values, None),
        "merchant_category_code": np.where(has_mcc, mcc, None),
        "proprietary_bank_transaction_code": ptype,
        "bank_transaction_code": np.array([PTYPE_ISO.get(p, "Other") for p in ptype], dtype=object),
        "status": "Booked",
    })
    truth = pd.DataFrame({
        "transaction_id": tx_id,
        "customer_id": customer_id,
        "true_category": table["category"].values[mid],
        "true_sub_category": table["sub_category"].values[mid],
        "merchant_key": table["merchant_key"].values[mid],
        "is_local_merchant": table["local_la"].values[mid] != "",
        "quantity": frame["qty"].values,
        "quantity_unit": frame["qty_unit"].values,
        "fuel_type": frame["fuel"].values,
        "stream": frame["stream"].values,
        # Ingestion audit: the booked record as the bank's systems hold it before extraction.
        "audit_account_id": account_id,
        "audit_fingerprint": row_hashes(observed, TRANSACTION_AUDIT_COLUMNS),
    })
    return observed, truth
