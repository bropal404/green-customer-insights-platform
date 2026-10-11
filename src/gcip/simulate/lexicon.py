"""Merchant lexicon: one table of every payee the synthetic bank can show, plus
descriptor rendering.

National merchants come from data/lexicon/merchants.yaml. Energy suppliers are
expanded into electricity, gas, dual-fuel, prepayment and export-payment
descriptors (taxonomy rule B4). Each local authority gets a pool of small local
businesses built from name patterns, so most local names are unique and the
categoriser has to generalise rather than memorise.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gcip.config import LEXICON_DIR

SURNAMES = [
    "SMITH", "JONES", "TAYLOR", "BROWN", "WILLIAMS", "WILSON", "JOHNSON", "DAVIES", "ROBINSON", "WRIGHT",
    "THOMPSON", "EVANS", "WALKER", "WHITE", "ROBERTS", "GREEN", "HALL", "WOOD", "JACKSON", "CLARKE",
    "PATEL", "KHAN", "LEWIS", "JAMES", "PHILLIPS", "MASON", "MITCHELL", "ROSE", "DAVIS", "MORRIS",
    "HUGHES", "WARD", "TURNER", "COOPER", "HILL", "MOORE", "KING", "LEE", "HARRIS", "MARTIN",
    "ALLEN", "YOUNG", "SCOTT", "BAKER", "ADAMS", "CAMPBELL", "ANDERSON", "PARKER", "SHAH", "ALI",
    "MURPHY", "OWEN", "PRICE", "BENNETT", "GRIFFITHS", "KELLY", "MARSHALL", "COLLINS", "CARTER", "BELL",
    "FOSTER", "SINGH", "HUSSAIN", "BAILEY", "COX", "RICHARDSON", "HOWARD", "CHAPMAN", "WEBB", "ELLIS",
]
FIRST_NAMES = [
    "OLIVER", "GEORGE", "HARRY", "JACK", "NOAH", "LEO", "ARTHUR", "MUHAMMAD", "OSCAR", "CHARLIE",
    "OLIVIA", "AMELIA", "ISLA", "AVA", "IVY", "FREYA", "LILY", "FLORENCE", "MIA", "WILLOW",
    "JAMES", "DAVID", "SARAH", "EMMA", "SOPHIE", "DANIEL", "THOMAS", "LAURA", "HANNAH", "PRIYA",
]


def stable_int(*parts: object, modulo: int) -> int:
    digest = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8).hexdigest()
    return int(digest, 16) % modulo


def stable_ref(*parts: object, length: int = 8) -> str:
    digest = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=12).hexdigest().upper()
    return digest[:length]


@dataclass
class Merchants:
    """All payees as one table, with fast per-row samplers.

    ``local_matrix[sub]`` is an (n_local_authorities, k) array of local-business ids
    so local picks are a single vectorised gather.
    """

    table: pd.DataFrame
    by_sub: dict[str, np.ndarray]
    la_codes: pd.Index
    local_matrix: dict[str, np.ndarray]
    _national_cache: dict | None = None

    def ids(self, sub_category: str) -> np.ndarray:
        return self.by_sub.get(sub_category, np.array([], dtype=int))

    def national_pool(self, sub: str, region: str, keys_filter: str | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._national_cache is None:
            self._national_cache = {}
        cache_key = (sub, region, keys_filter)
        if cache_key not in self._national_cache:
            rows = self.table.loc[self.ids(sub)]
            rows = rows[rows["local_la"] == ""]
            if keys_filter is not None:
                rows = rows[rows["merchant_key"].str.contains(keys_filter, regex=True)]
            ok = rows[[(not r) or (region in r) for r in rows["regions"]]]
            if ok.empty:
                ok = rows
            self._national_cache[cache_key] = (ok["merchant_id"].values, ok["weight"].values.astype(float),
                                               ok["green"].values.astype(float))
        return self._national_cache[cache_key]


def _row(key: str, category: str, sub: str, mcc: str, channel: str, weight: float, variants: list[str],
         regions=None, green: float = 0.0, intent: str = "", lpg: bool = False, local_la: str = "",
         supplier: str = "", fuel_tag: str = "", pool: str = "") -> dict:
    return {
        "pool": pool or sub,
        "merchant_key": key, "category": category, "sub_category": sub, "mcc": mcc, "channel": channel,
        "weight": float(weight), "variants": list(variants), "regions": tuple(regions) if regions else (),
        "green": float(green), "intent": intent, "lpg": bool(lpg), "local_la": local_la,
        "supplier": supplier, "fuel_tag": fuel_tag,
    }


def build_merchants(la_frame: pd.DataFrame, towns: dict[str, str], rng: np.random.Generator,
                    path: Path = LEXICON_DIR / "merchants.yaml", locals_per_la: int = 5) -> Merchants:
    spec = yaml.safe_load(path.read_text())["sub_categories"]
    rows: list[dict] = []
    for sub, block in spec.items():
        category = block["category"]
        sub_name = block.get("sub_category_override", sub)
        pool_name = sub if sub == "clean_air_charges" else sub_name
        default_mcc = str(block.get("mcc", ""))
        default_channel = block.get("channel", "card")
        block_green = float(block.get("green", 0.0))
        if sub == "energy_supplier":
            for m in block["merchants"]:
                name, short, key = m["name"], m["short"], m["key"]
                g = float(m.get("green", 0.0))
                w = m["weight"]
                rows.append(_row(f"{key}::dual", "Utilities", "dual_fuel_energy", "4900", "dd", w,
                                 [name, f"{name} {{ref}}", f"{short} ENERGY {{ref}}"], green=g, supplier=key))
                if m.get("dual_only"):
                    continue
                rows.append(_row(f"{key}::elec_tag", "Electricity", "electricity_bill", "4900", "dd", w,
                                 [f"{name} ELEC {{ref}}", f"{short} ELECTRICITY", f"{short} ELEC ACC {{ref}}"], green=g,
                                 supplier=key, fuel_tag="elec"))
                rows.append(_row(f"{key}::gas_tag", "Gas", "gas_bill", "4900", "dd", w,
                                 [f"{name} GAS {{ref}}", f"{short} GAS ACC {{ref}}", f"{short} GAS SUPPLY"], green=g,
                                 supplier=key, fuel_tag="gas"))
                # Untagged single-fuel bills read exactly like dual-fuel ones (rule B4 ambiguity).
                rows.append(_row(f"{key}::elec_plain", "Electricity", "electricity_bill", "4900", "dd", w,
                                 [name, f"{name} {{ref}}"], green=g, supplier=key, fuel_tag="elec"))
                rows.append(_row(f"{key}::gas_plain", "Gas", "gas_bill", "4900", "dd", w,
                                 [name, f"{name} {{ref}}"], green=g, supplier=key, fuel_tag="gas"))
                rows.append(_row(f"{key}::prepay_elec", "Electricity", "electricity_prepayment", "4900", "card", w,
                                 [f"{short} PAYG ELEC", f"{short} TOP UP ELEC", f"{name} PREPAY"], green=g,
                                 supplier=key, fuel_tag="elec"))
                rows.append(_row(f"{key}::prepay_gas", "Gas", "gas_prepayment", "4900", "card", w,
                                 [f"{short} PAYG GAS", f"{short} TOP UP GAS", f"{name} PREPAY"], green=g,
                                 supplier=key, fuel_tag="gas"))
                rows.append(_row(f"{key}::seg", "Electricity", "electricity_bill", "4900", "bgc", w,
                                 [f"{short} SEG PAYMENT", f"{name} EXPORT", f"{short} SMART EXPORT"], green=g,
                                 supplier=key, fuel_tag="elec"))
            for i, variant in enumerate(block.get("prepayment_networks", [])):
                for fuel_tag, cat, subcat in (("elec", "Electricity", "electricity_prepayment"),
                                              ("gas", "Gas", "gas_prepayment")):
                    rows.append(_row(f"prepay_network_{i}::{fuel_tag}", cat, subcat, "4900", "card", 1.0, [variant],
                                     fuel_tag=fuel_tag))
            continue
        if sub in ("income", "transfers", "forecourt_shop"):
            continue
        for m in block.get("merchants", []):
            rows.append(_row(
                m["key"], category, sub_name, str(m.get("mcc", default_mcc)), m.get("channel", default_channel),
                m.get("weight", 1.0), m["variants"], m.get("regions"), m.get("green", block_green),
                m.get("intent", ""), m.get("lpg", False), pool=pool_name,
            ))
        patterns = block.get("local_patterns", [])
        if patterns:
            for la_code in la_frame["la_code"].values:
                town = towns[la_code]
                for j in range(locals_per_la):
                    pattern = patterns[stable_int(la_code, sub, j, modulo=len(patterns))]
                    surname = SURNAMES[stable_int(la_code, sub, j, "s", modulo=len(SURNAMES))]
                    first = FIRST_NAMES[stable_int(la_code, sub, j, "f", modulo=len(FIRST_NAMES))]
                    name = pattern.replace("{loc}", town).replace("{surname}", surname).replace("{first}", first)
                    name = name.replace("{la}", town)
                    rows.append(_row(f"local::{sub}::{la_code}::{j}", category, sub_name, default_mcc,
                                     default_channel, 1.0, [name], green=block_green, local_la=la_code))

    # Forecourt shop purchases reuse the forecourt names (taxonomy rule B3).
    for m in spec["road_fuel"]["merchants"]:
        rows.append(_row(f"{m['key']}::shop", "Other", "forecourt_shop", "5541", "card", m["weight"], m["variants"]))
    # Special payees rendered from customer context.
    specials = [
        ("dvla", "Automotive", "vehicle_tax", "9399", "dd", ["DVLA-{reg}", "DVLA VEHICLE TAX {reg}"]),
        ("council_tax", "Other", "council_tax", "9311", "dd", ["{la} COUNCIL", "{la} CTAX", "{la} BC CT"]),
        ("salary", "Other", "income", "", "bgc", ["{employer}"]),
        ("pension_state", "Other", "income", "", "bgc", ["DWP STATE PENSION", "DWP SP {ref}"]),
        ("pension_private", "Other", "income", "", "bgc", ["{pension} {ref}"]),
        ("benefits", "Other", "income", "", "bgc", ["DWP UC", "DWP UC {ref}"]),
        ("child_benefit", "Other", "income", "", "bgc", ["HMRC CHILD BENEFIT"]),
        ("student_loan", "Other", "income", "", "bgc", ["SLC STUDENT FINANCE", "STUDENT LOANS CO"]),
        ("client_payment", "Other", "income", "", "fpi", ["{surname} LTD", "{surname} & CO INVOICE {ref}"]),
        ("p2p_out", "Other", "transfers", "", "fpo", ["{first} {surname}", "{first} {surname} {ref}"]),
        ("p2p_in", "Other", "transfers", "", "fpi", ["{first} {surname}", "{surname} {first}"]),
        ("atm", "Other", "cash", "6011", "cpt", ["LNK {loc}", "CASH {loc} {store}", "LNK {store} {loc}"]),
        ("mortgage_internal", "Other", "housing", "", "tfr", ["MORTGAGE PAYMENT {num}", "MTG {num}"]),
        ("motor_loan_internal", "Automotive", "vehicle_finance", "", "tfr", ["MOTOR FINANCE {num}", "BLACK HORSE {num}"]),
        ("ev_loan_internal", "Automotive", "vehicle_finance", "", "tfr", ["EV FINANCE {num}", "GREEN CAR LOAN {num}"]),
        ("personal_loan_internal", "Other", "finance", "", "tfr", ["LOAN REPAYMENT {num}", "PERSONAL LOAN {num}"]),
        ("green_loan_internal", "Other", "finance", "", "tfr", ["GREEN LOAN {num}", "LOAN REPAYMENT {num}"]),
        ("solar_loan_internal", "Other", "finance", "", "tfr", ["SOLAR FINANCE {num}", "LOAN REPAYMENT {num}"]),
        ("green_home_internal", "Other", "housing", "", "tfr", ["MORTGAGE FURTHER ADV {num}", "MTG {num}"]),
        ("savings_internal", "Other", "finance", "", "tfr", ["TO SAVINGS {num}", "TFR SAVER {num}"]),
        ("card_repayment_internal", "Other", "finance", "", "tfr", ["CREDIT CARD {num}", "CARD PAYMENT {num}"]),
        ("investment_internal", "Other", "investments", "", "tfr", ["ISA INVESTMENT {num}", "INVESTMENT ACCOUNT {num}"]),
        ("overdraft_interest", "Other", "finance", "", "chg", ["OVERDRAFT INTEREST", "ARRANGED OD INTEREST"]),
        ("unpaid_dd", "Other", "finance", "", "chg", ["UNPAID DD {ref}", "RETURNED DIRECT DEBIT"]),
        ("removals", "Other", "retail", "4214", "fpo", ["{loc} REMOVALS", "{surname} REMOVALS & STORAGE"]),
    ]
    for key, cat, sub, mcc, channel, variants in specials:
        rows.append(_row(key, cat, sub, mcc, channel, 1.0, variants))

    table = pd.DataFrame(rows).reset_index(drop=True)
    table.insert(0, "merchant_id", np.arange(len(table)))
    by_sub = {sub: grp["merchant_id"].values for sub, grp in table.groupby("pool")}
    la_index = pd.Index(la_frame["la_code"].values)
    local_matrix: dict[str, np.ndarray] = {}
    local = table[table["local_la"] != ""]
    for sub, grp in local.groupby("pool"):
        matrix = np.full((len(la_index), locals_per_la), -1, dtype=np.int64)
        for la_code, rows_la in grp.groupby("local_la"):
            ids = rows_la["merchant_id"].values
            matrix[la_index.get_loc(la_code), : len(ids)] = ids[:locals_per_la]
        local_matrix[sub] = matrix
    return Merchants(table=table, by_sub=by_sub, la_codes=la_index, local_matrix=local_matrix)


def pick_merchants(m: Merchants, sub_category: str, regions: np.ndarray, la_codes: np.ndarray,
                   rng: np.random.Generator, local_share: float = 0.0, green_affinity: np.ndarray | None = None,
                   keys_filter: str | None = None) -> np.ndarray:
    """Choose a merchant id per row for one sub-category, honouring regional coverage.

    ``local_share`` of rows go to the row's own local-authority pool when the
    sub-category has local businesses. ``green_affinity`` (per row, > 0) tilts
    the choice toward merchants with a positive ``green`` score.
    """
    n = len(regions)
    out = np.empty(n, dtype=np.int64)
    use_local = np.zeros(n, dtype=bool)
    matrix = m.local_matrix.get(sub_category)
    if matrix is not None and local_share > 0 and keys_filter is None:
        use_local = rng.random(n) < local_share
        if use_local.any():
            la_idx = m.la_codes.get_indexer(np.asarray(la_codes)[use_local])
            picks = matrix[np.maximum(la_idx, 0), rng.integers(0, matrix.shape[1], use_local.sum())]
            bad = (la_idx < 0) | (picks < 0)
            use_local_idx = np.where(use_local)[0]
            use_local[use_local_idx[bad]] = False
            out[use_local_idx[~bad]] = picks[~bad]
    remaining = ~use_local
    regions = np.asarray(regions)
    for region in np.unique(regions[remaining]):
        mask = remaining & (regions == region)
        ids, weights, greens = m.national_pool(sub_category, region, keys_filter)
        if len(ids) == 0:
            raise KeyError(f"no merchants for {sub_category} ({keys_filter})")
        if green_affinity is None or greens.max() <= 0:
            out[mask] = rng.choice(ids, mask.sum(), p=weights / weights.sum())
        else:
            aff = np.asarray(green_affinity)[mask][:, None]
            w = weights[None, :] * np.power(aff, greens[None, :])
            cumulative = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
            draws = rng.random(mask.sum())[:, None]
            out[mask] = ids[np.minimum((draws > cumulative).sum(axis=1), len(ids) - 1)]
    return out
