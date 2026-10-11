"""Training labels from the bank's own product system: did the customer open a green product in the window?

This is how a bank would label take-up (no simulator truth needed). On the synthetic bank it reproduces the
truth-vault labels exactly (checked on a 2,000-customer build: 247 of 247 take-ups, same dates).
"""

from __future__ import annotations

import pandas as pd

from gcip.pipeline.splits import customer_split

PRODUCTS = ["GP01", "GP02", "GP03", "GP04", "GP05", "GP06"]


def takeup_labels(holdings: pd.DataFrame, customers: pd.DataFrame, window_start: str, window_end: str) -> pd.DataFrame:
    """One row per customer and product: took_up, takeup_date (first green holding that starts in the window), split."""
    start, end = pd.Timestamp(window_start), pd.Timestamp(window_end)
    green = holdings[holdings["green_product"] != ""]
    began = pd.to_datetime(green["start_date"])
    opened = green[(began >= start) & (began <= end)]
    first = (opened.assign(start_date=pd.to_datetime(opened["start_date"]))
             .groupby(["customer_id", "green_product"])["start_date"].min()
             .rename("takeup_date").reset_index().rename(columns={"green_product": "product"}))
    grid = customers[["customer_id"]].merge(pd.DataFrame({"product": PRODUCTS}), how="cross")
    labels = grid.merge(first, on=["customer_id", "product"], how="left")
    labels["took_up"] = labels["takeup_date"].notna()
    labels["split"] = labels["customer_id"].map(customer_split(customers["customer_id"]))
    return labels[["customer_id", "product", "took_up", "takeup_date", "split"]]
