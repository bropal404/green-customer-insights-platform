"""Rules layer of the categoriser. Every rule cites the taxonomy boundary rule it applies.

Rules fire first and only when they are near-certain; anything ambiguous (single
vs dual-fuel bills with no fuel tag, insurers selling motor and home cover, local
traders) is left to the models.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# MCC -> (category, sub_category, confidence, boundary rule)
MCC_RULES = {
    "5552": ("Fuel", "ev_public_charging", 0.97, "B2"),
    "5983": ("Utilities", "heating_oil_lpg", 0.95, "B4"),
    "4112": ("Public Transport", "rail", 0.96, ""),
    "4111": ("Public Transport", "metro_tram", 0.95, ""),
    "4131": ("Public Transport", "bus_coach", 0.95, ""),
    "4789": ("Public Transport", "micromobility", 0.93, "B7"),
    "4121": ("Other", "taxi_ridehail", 0.96, "B7"),
    "5200": ("Home Improvement", "diy_materials", 0.95, ""),
    "5211": ("Home Improvement", "diy_materials", 0.95, ""),
    "5231": ("Home Improvement", "diy_materials", 0.95, ""),
    "5251": ("Home Improvement", "diy_materials", 0.95, ""),
    "5074": ("Home Improvement", "diy_materials", 0.94, ""),
    "5713": ("Home Improvement", "diy_materials", 0.94, ""),
    "5722": ("Home Improvement", "appliances", 0.94, "B8"),
    "7538": ("Automotive", "maintenance", 0.96, "B1"),
    "5533": ("Automotive", "maintenance", 0.95, "B1"),
    "7523": ("Automotive", "parking_charges", 0.95, "B1"),
    "7549": ("Automotive", "breakdown_cover", 0.95, "B9"),
    "5511": ("Automotive", "vehicle_purchase", 0.92, "B1"),
    "5411": ("Other", "groceries", 0.96, ""),
    "5812": ("Other", "eating_out", 0.95, ""),
    "5311": ("Other", "retail", 0.90, ""),
    "5912": ("Other", "health_beauty", 0.95, ""),
    "5931": ("Other", "charity_second_hand", 0.93, ""),
    "4899": ("Other", "subscriptions", 0.94, ""),
    "7997": ("Other", "subscriptions", 0.93, ""),
    "4511": ("Other", "travel", 0.95, ""),
    "7011": ("Other", "travel", 0.95, ""),
    "6011": ("Other", "cash", 0.97, ""),
    "5940": ("Other", "retail", 0.92, ""),
}

# (pattern, payment types or None, direction or None, category, sub_category, confidence, rule)
TEXT_RULES = [
    (r"\bDVLA\b", None, "D", "Automotive", "vehicle_tax", 0.98, "B1"),
    (r"\bULEZ\b|TFL CC|CONGESTION CHARGE|CLEAN AIR ZONE", None, "D", "Automotive", "parking_charges", 0.98, "B6"),
    (r"\bCOUNCIL PARKING\b|\bCAR PARK\b", None, "D", "Automotive", "parking_charges", 0.95, "B1"),
    (r"^MORTGAGE PAYMENT\b|^MTG\b|^MORTGAGE FURTHER ADV", ("TFR",), "D", "Other", "housing", 0.97, "B11"),
    (r"^MOTOR FINANCE\b|^EV FINANCE\b|^GREEN CAR LOAN\b|^BLACK HORSE #", ("TFR",), "D", "Automotive", "vehicle_finance", 0.97, "B1"),
    (r"^LOAN REPAYMENT\b|^PERSONAL LOAN\b|^GREEN LOAN\b|^SOLAR FINANCE\b", ("TFR",), "D", "Other", "finance", 0.96, ""),
    (r"^TO SAVINGS\b|^TFR SAVER\b|^CREDIT CARD\b|^CARD PAYMENT\b", ("TFR",), "D", "Other", "finance", 0.96, ""),
    (r"^ISA INVESTMENT\b|^INVESTMENT ACCOUNT\b", ("TFR",), "D", "Other", "investments", 0.96, ""),
    (r"OVERDRAFT|UNPAID DD|RETURNED DIRECT DEBIT|ARRANGED OD", ("CHG",), "D", "Other", "finance", 0.97, ""),
    (r"^LNK\b|^CASH\b", ("CPT",), "D", "Other", "cash", 0.97, ""),
    (r"\bDWP\b|\bHMRC\b|PENSION|\bSLC\b|STUDENT LOANS|STUDENT FINANCE", ("BGC",), "C", "Other", "income", 0.95, ""),
    (r"\bSEG PAYMENT\b|SMART EXPORT|\bEXPORT\b", ("BGC",), "C", "Electricity", "electricity_bill", 0.92, "B10"),
    (r"\bWATER\b|\bWATR\b", ("DD",), "D", "Utilities", "water", 0.94, "B4"),
    (r"COUNCIL RENT|HOUSING RENT", None, "D", "Other", "housing", 0.95, "B11"),
    (r"\bCOUNCIL\b|\bCTAX\b|\bBC CT\b", ("DD",), "D", "Other", "council_tax", 0.94, ""),
    (r"\bPAYG ELEC\b|\bTOP UP ELEC\b", None, "D", "Electricity", "electricity_prepayment", 0.95, "B5"),
    (r"\bPAYG GAS\b|\bTOP UP GAS\b", None, "D", "Gas", "gas_prepayment", 0.95, "B5"),
    (r"\bELEC\b|\bELECTRICITY\b", ("DD",), "D", "Electricity", "electricity_bill", 0.92, "B4"),
    (r"\bGAS ACC\b|\bGAS SUPPLY\b|ENERGY GAS #|\b(?<!BRITISH )GAS #", ("DD",), "D", "Gas", "gas_bill", 0.90, "B4"),
]
_COMPILED = [(re.compile(p), pt, d, c, s, conf, rule) for p, pt, d, c, s, conf, rule in TEXT_RULES]


def apply_rules(sig: pd.DataFrame) -> pd.DataFrame:
    """Rule predictions for unique signatures.

    Expects columns: ptype, mcc, amount_band, round_amount, direction, norm_text.
    Returns rule_category, rule_sub_category, rule_confidence, rule_id (empty when no rule fired).
    """
    n = len(sig)
    category = np.full(n, "", dtype=object)
    sub = np.full(n, "", dtype=object)
    conf = np.zeros(n)
    rule_id = np.full(n, "", dtype=object)
    text = sig["norm_text"].values
    ptype = sig["ptype"].values
    direction = sig["direction"].values

    for pattern, ptypes, d, cat, s, c, rid in _COMPILED:
        free = conf == 0
        if not free.any():
            break
        mask = free & np.array([bool(pattern.search(t)) for t in text])
        if ptypes is not None:
            mask &= np.isin(ptype, ptypes)
        if d is not None:
            mask &= direction == d
        category[mask], sub[mask], conf[mask], rule_id[mask] = cat, s, c, f"text:{rid or 'domain'}"

    mcc = sig["mcc"].values
    free = conf == 0
    for code, (cat, s, c, rid) in MCC_RULES.items():
        mask = free & (mcc == code)
        category[mask], sub[mask], conf[mask], rule_id[mask] = cat, s, c, f"mcc:{code}{'/' + rid if rid else ''}"
    # Forecourt terminals: small, non-round purchases are shop items (B3); the rest is fuel.
    forecourt = (conf == 0) & np.isin(mcc, ["5541", "5542"])
    small = sig["amount_band"].values <= 1
    shop = forecourt & small & (sig["round_amount"].values == 0)
    category[shop], sub[shop], conf[shop], rule_id[shop] = "Other", "forecourt_shop", 0.85, "mcc:5541/B3"
    fuel = forecourt & ~shop
    category[fuel], sub[fuel], conf[fuel], rule_id[fuel] = "Fuel", "road_fuel", 0.92, "mcc:5541/B1"
    return pd.DataFrame({"rule_category": category, "rule_sub_category": sub, "rule_confidence": conf, "rule_id": rule_id},
                        index=sig.index)
