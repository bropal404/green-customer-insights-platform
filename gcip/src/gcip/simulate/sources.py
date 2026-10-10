"""Raw extracts from the bank's separate source systems.

A real bank's data does not arrive as one tidy warehouse. Each system has its own
format, keys, units, dates, codes and quality problems, all documented in
configs/source_systems.yaml; FR1 ingestion (gcip.fr1.ingest) has to harmonise them.

    core_banking/        customers.csv, accounts.csv, balances.csv
    card_processor/      card_registry.csv, part-XXXX.parquet
    payments_hub/        part-XXXX.csv.gz
    crm_marketing/       contacts.csv, campaign_responses.csv
    digital_analytics/   monthly_engagement.jsonl
    lending_assets/      property_valuations.xml, vehicle_records.csv
    product_system/      holdings.csv, statement_preference_events.csv
    _control.json        row counts and SHA-256 of every file, as a bank's file-transfer control totals

Identifiers are bijections of the customer number, so they never collide.
Transaction extracts are sharded by customer range: shard XXXX of the card and
payment feeds covers the same customers.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lxml import etree

from gcip.config import load_config
from gcip.io.files import sha256_file, write_json

ACCOUNT_SUFFIXES = ["CA", "SV", "CC", "MG", "ML", "PL", "IV", "EV", "GH", "SF", "SI", "GL", "EVN", "GHN", "SFN", "SIN", "GLN"]
SORT_CODES = [f"0010{i:02d}" for i in range(1, 13)]  # twelve fictitious branch sort codes


def dictionary() -> dict:
    return load_config("source_systems")


def _invert(table: dict) -> dict:
    return {v: k for k, v in table.items()}


def _number(customer_id: str) -> int:
    return int(re.sub(r"\D", "", customer_id))


def account_number(account_id: str) -> str:
    """Sort code and 8-digit account number; unique because it is a bijection of (customer, account type)."""
    customer, suffix = account_id.rsplit("-", 1)
    num = _number(customer)
    seq = num * 32 + ACCOUNT_SUFFIXES.index(suffix)
    return f"{SORT_CODES[num % len(SORT_CODES)]}-{(seq * 48271 + 104729) % 100_000_000:08d}"


def card_token(account_id: str) -> str:
    return hashlib.sha256(f"card:{account_id}".encode()).hexdigest()[:24].upper()


def contact_id(customer_id: str, secondary: bool = False) -> str:
    num = _number(customer_id)
    if secondary:
        return f"CRM-W{(num * 40503 + 7) % 10**9:09d}"
    return f"CRM-{(num * 2654435761 + 12345) % 10**10:010d}"


def analytics_uid(customer_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{customer_id}".encode()).hexdigest()[:16]


def _map_unique(values: pd.Series, fn) -> np.ndarray:
    uniques = pd.unique(values.dropna())
    return values.map(dict(zip(uniques, map(fn, uniques)))).to_numpy(dtype=object, copy=True)


def _date_text(values, fmt: str) -> np.ndarray:
    dates = pd.to_datetime(pd.Series(np.asarray(values)))
    return dates.dt.strftime(fmt).fillna("").to_numpy(dtype=object)


def _minor(values, scale: int = 100):
    """Money (or a rate) in integer minor units; blank when missing."""
    x = pd.to_numeric(pd.Series(np.asarray(values, dtype=float)), errors="coerce")
    return pd.array(np.round(x.to_numpy() * scale), dtype="Int64")


def _column(frame: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    return frame[name] if name in frame else pd.Series(default, index=frame.index)


# --------------------------------------------------------------------------- core banking


def write_core(customers: pd.DataFrame, accounts: pd.DataFrame, balances: pd.DataFrame, out: Path,
               rng: np.random.Generator) -> None:
    d = dictionary()
    codes, rates = d["codes"], d["quality_rates"]
    folder = out / "core_banking"
    folder.mkdir(parents=True, exist_ok=True)

    region = customers["region"].to_numpy(dtype=object).copy()
    upper = rng.random(len(region)) < rates["region_upper_case"]
    lower = ~upper & (rng.random(len(region)) < rates["region_lower_case"])
    region[upper] = [r.upper() for r in region[upper]]
    region[lower] = [r.lower() for r in region[lower]]
    pref = customers["stated_sustainability_preference"].to_numpy(dtype=float)
    pd.DataFrame({
        "CUST_ID": customers["customer_id"].to_numpy(),
        "BIRTH_YR": customers["birth_year"].to_numpy(),
        "LA_CD": customers["la_code"].to_numpy(),
        "LA_NM": customers["la_name"].to_numpy(),
        "REGION_NM": region,
        "RES_STATUS": customers["declared_residential_status"].map(_invert(codes["residential_status"])).to_numpy(),
        "EMP_STATUS": customers["declared_employment_status"].map(_invert(codes["employment_status"])).to_numpy(),
        "INC_BAND": customers["declared_income_band"].map(_invert(codes["income_band"])).to_numpy(),
        "CUST_SINCE": _date_text(customers["customer_since"], "%Y%m%d"),
        "MKT_CONSENT": np.where(customers["marketing_consent"].to_numpy(), "Y", "N"),
        "VULN_FLAG": np.where(customers["vulnerability_flag"].to_numpy(), "Y", "N"),
        "PREF_CHANNEL": customers["preferred_contact_channel"].map(_invert(codes["contact_channel"])).to_numpy(),
        "SUST_PREF": np.where(np.isnan(pref), "", np.where(pref == 1.0, "Y", "N")),
        "OCCUPATION_CD": customers["declared_occupation"].map(_invert(codes["occupation"])).to_numpy(),
        "DEPENDANTS": customers["declared_dependants"].to_numpy(),
        "YRS_AT_ADDR": customers["declared_years_at_address"].to_numpy(),
    }).to_csv(folder / "customers.csv", index=False)

    pd.DataFrame({
        "ACCT_NO": _map_unique(accounts["account_id"], account_number),
        "CUST_ID": accounts["customer_id"].to_numpy(),
        "ACCT_TYPE": accounts["account_id"].str.rsplit("-", n=1).str[1].to_numpy(),
        "ACCT_FAMILY": accounts["account_type"].str.upper().to_numpy(),
        "PRODUCT_CD": accounts["product_code"].to_numpy(),
        "OPEN_DT": _date_text(accounts["opened_date"], "%Y%m%d"),
        "CLOSE_DT": _date_text(_column(accounts, "closed_date", pd.NaT), "%Y%m%d"),
        "STMT_PREF": _column(accounts, "statement_preference", None).map(_invert(codes["statement_preference"])).fillna("").to_numpy(),
        "CREDIT_LIMIT_PENCE": _minor(_column(accounts, "credit_limit")),
        "ORIG_AMT_PENCE": _minor(_column(accounts, "original_amount")),
        "RATE_BPS": _minor(_column(accounts, "interest_rate"), 10000),
        "TERM_MTHS": _minor(_column(accounts, "term_months"), 1),
        "COLLATERAL_VALUE_PENCE": _minor(_column(accounts, "collateral_value_at_origination")),
        "LOAN_PURPOSE": _column(accounts, "loan_purpose", None).fillna("").str.upper().to_numpy(),
        "PROVIDER": accounts["provider"].to_numpy(),
    }).to_csv(folder / "accounts.csv", index=False)

    pd.DataFrame({
        "ACCT_NO": _map_unique(balances["account_id"], account_number),
        "MONTH_END": _date_text(balances["month_end"], "%Y%m%d"),
        "BAL_PENCE": _minor(balances["balance"]),
        "ARREARS_MTHS": balances["arrears_months"].to_numpy(),
    }).to_csv(folder / "balances.csv", index=False)


# --------------------------------------------------------------------------- product system


def write_product_system(holdings: pd.DataFrame, preference_changes: pd.DataFrame, out: Path) -> None:
    folder = out / "product_system"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "CUSTOMER_REF": holdings["customer_id"].to_numpy(),
        "ACCT_NO": _map_unique(holdings["account_id"], account_number),
        "PRODUCT_CD": holdings["product_code"].to_numpy(),
        "PRODUCT_FAMILY": holdings["product_family"].to_numpy(),
        "START_DATE": _date_text(holdings["start_date"], "%Y-%m-%d"),
        "END_DATE": _date_text(holdings["end_date"], "%Y-%m-%d"),
    }).to_csv(folder / "holdings.csv", index=False)
    pd.DataFrame({
        "ACCT_NO": _map_unique(preference_changes["account_id"], account_number),
        "EVENT_TS": _date_text(preference_changes["change_date"], "%Y-%m-%dT09:00:00"),
        "OLD_PREF": preference_changes["old_preference"].str.upper().to_numpy(),
        "NEW_PREF": preference_changes["new_preference"].str.upper().to_numpy(),
    }).to_csv(folder / "statement_preference_events.csv", index=False)


# --------------------------------------------------------------------------- lending assets


def write_lending_assets(assets: pd.DataFrame, accounts: pd.DataFrame, out: Path, extract_date: str) -> None:
    """Property valuations as XML from the mortgage origination system; vehicle records as CSV."""
    codes = dictionary()["codes"]
    folder = out / "lending_assets"
    folder.mkdir(parents=True, exist_ok=True)
    ids = accounts["account_id"]
    loan_of = {suffix: pd.Series(ids[ids.str.endswith(f"-{suffix}")].to_numpy(),
                                 index=accounts.loc[ids.str.endswith(f"-{suffix}"), "customer_id"].to_numpy())
               for suffix in ("MG", "ML")}

    epc_fuel = _invert(codes["epc_main_fuel"])
    prop = assets[assets["asset_type"] == "property"].reset_index(drop=True)
    mortgage = prop["customer_id"].map(loan_of["MG"])
    root = etree.Element("valuations", system="mortgage_origination", extract_date=extract_date)
    for i, row in prop.iterrows():
        node = etree.SubElement(root, "valuation", mortgage_account=account_number(mortgage[i]))
        attrs = {"type": str(row["property_type"]).title(), "detail": str(row["property_detail"]),
                 "built": str(row["construction_period"]), "bedrooms": str(int(row["bedrooms"]))}
        if pd.notna(row["floor_area_m2"]):
            attrs["floor_area_m2"] = f"{row['floor_area_m2']:g}"
        etree.SubElement(node, "property", **attrs)
        if pd.notna(row["epc_band"]):
            etree.SubElement(node, "epc", rating=str(row["epc_band"]), main_fuel=epc_fuel[row["epc_main_heating"]],
                             solar_pv="Y" if bool(row["solar_pv_on_record"]) else "N")
        etree.SubElement(node, "value", currency="GBP", pence=str(int(round(row["valuation_at_origination"] * 100))))
    etree.ElementTree(root).write(str(folder / "property_valuations.xml"), xml_declaration=True, encoding="UTF-8",
                                  pretty_print=True)

    veh = assets[assets["asset_type"] == "vehicle"].reset_index(drop=True)
    pd.DataFrame({
        "MOTOR_ACCT_NO": [account_number(a) for a in veh["customer_id"].map(loan_of["ML"])],
        "VRM": veh["vehicle_registration"].to_numpy(),
        "FUEL_TYPE": veh["vehicle_fuel_type"].map(_invert(codes["dvla_fuel"])).to_numpy(),
        "YEAR_OF_FIRST_REG": _minor(veh["vehicle_first_registered_year"], 1),
        "VALUATION_PENCE": _minor(veh["valuation_at_origination"]),
    }).to_csv(folder / "vehicle_records.csv", index=False)


# --------------------------------------------------------------------------- CRM


def write_crm(campaigns: pd.DataFrame, customers: pd.DataFrame, out: Path, rng: np.random.Generator) -> None:
    d = dictionary()
    rates = d["quality_rates"]
    folder = out / "crm_marketing"
    folder.mkdir(parents=True, exist_ok=True)
    ids = customers["customer_id"].to_numpy()
    second = ids[rng.random(len(ids)) < rates["crm_duplicate_contact"]]
    primary_of = {c: contact_id(c) for c in ids}
    second_of = {c: contact_id(c, secondary=True) for c in second}

    responses = campaigns.reset_index(drop=True)
    use_second = responses["customer_id"].isin(second_of).to_numpy() & (rng.random(len(responses)) < 0.5)
    contact = np.where(use_second, responses["customer_id"].map(second_of), responses["customer_id"].map(primary_of))

    # Prospects: contacted by campaigns but not customers of the bank.
    n_prospects = int(round(len(responses) * rates["crm_prospect"]))
    prospects = responses.sample(n_prospects, random_state=11).reset_index(drop=True)
    prospect_ids = np.array([f"CRM-P{i:08d}" for i in range(n_prospects)], dtype=object)

    contacts = pd.concat([
        pd.DataFrame({"CONTACT_ID": [primary_of[c] for c in ids], "CUST_REF": ids, "SOURCE": "core_sync"}),
        pd.DataFrame({"CONTACT_ID": [second_of[c] for c in second], "CUST_REF": second, "SOURCE": "web_signup"}),
        pd.DataFrame({"CONTACT_ID": prospect_ids, "CUST_REF": "", "SOURCE": "prospect_list"}),
    ], ignore_index=True)
    contacts.to_csv(folder / "contacts.csv", index=False)

    rows = pd.concat([responses.assign(contact=contact), prospects.assign(contact=prospect_ids)], ignore_index=True)
    outcome = np.empty(len(rows), dtype=object)
    for value, forms in d["codes"]["campaign_outcome"].items():
        mask = rows["response"].to_numpy() == value
        outcome[mask] = np.array(forms, dtype=object)[rng.integers(0, len(forms), mask.sum())]
    if pd.isna(outcome).any():
        raise ValueError(f"campaign responses without a CRM spelling: {set(rows['response'][pd.isna(outcome)])}")
    names = {code: spec["name"] for code, spec in load_config("products")["products"].items()}
    pd.DataFrame({
        "CONTACT_ID": rows["contact"].to_numpy(),
        "CAMPAIGN": rows["campaign_id"].to_numpy(),
        "SENT": _date_text(rows["sent_date"], "%d/%m/%Y"),
        "PRODUCT_NAME": rows["product"].map(names).to_numpy(),
        "CHANNEL": rows["channel"].str.upper().to_numpy(),
        "OUTCOME": outcome,
        "OUTCOME_DATE": _date_text(rows["response_date"], "%d/%m/%Y"),
    }).to_csv(folder / "campaign_responses.csv", index=False)


# --------------------------------------------------------------------------- digital analytics


def write_digital(digital: pd.DataFrame, out: Path) -> None:
    d = dictionary()
    salt = d["pseudonymisation"]["digital_analytics_salt"]
    folder = out / "digital_analytics"
    folder.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "uid": _map_unique(digital["customer_id"], lambda c: analytics_uid(c, salt)),
        "month": _date_text(digital["month"], "%Y-%m"),
        "app_sessions": digital["app_logins"].astype(int).to_numpy(),
        "web_sessions": digital["web_logins"].astype(int).to_numpy(),
        "branch_visits": digital["branch_visits"].astype(int).to_numpy(),
        "estatement_views": digital["estatement_views"].astype(int).to_numpy(),
        "paperless": digital["paperless_all_accounts"].astype(bool).to_numpy(),
    })
    n_orphans = int(round(len(frame) * d["quality_rates"]["digital_orphan"]))
    orphans = frame.sample(n_orphans, random_state=7).copy()
    orphans["uid"] = [hashlib.sha256(f"prospect:{i}".encode()).hexdigest()[:16] for i in range(n_orphans)]
    frame = pd.concat([frame, orphans], ignore_index=True).sample(frac=1.0, random_state=3)
    frame.to_json(folder / "monthly_engagement.jsonl", orient="records", lines=True)


# --------------------------------------------------------------------------- card processor and payments hub


def write_card_registry(accounts: pd.DataFrame, state: pd.DataFrame, out: Path) -> None:
    folder = out / "card_processor"
    folder.mkdir(parents=True, exist_ok=True)
    last4 = state.set_index("customer_id")
    rows = []
    for kind, suffix, col in (("DEBIT", "CA", "debit_card_last4"), ("CREDIT", "CC", "credit_card_last4")):
        accts = accounts[accounts["account_id"].str.endswith(f"-{suffix}")]
        rows.append(pd.DataFrame({
            "CARD_TOKEN": [card_token(a) for a in accts["account_id"]],
            "ACCT_NO": [account_number(a) for a in accts["account_id"]],
            "CARD_LAST4": accts["customer_id"].map(last4[col]).to_numpy(),
            "CARD_TYPE": kind,
        }))
    pd.concat(rows, ignore_index=True).to_csv(folder / "card_registry.csv", index=False)


def write_transaction_extracts(observed: pd.DataFrame, chunk_id: int, out: Path, rng: np.random.Generator) -> dict:
    """Split one shard of booked transactions into the card-processor and payments-hub feeds."""
    d = dictionary()
    codes, rates = d["codes"], d["quality_rates"]
    ptype = observed["proprietary_bank_transaction_code"].to_numpy()
    on_card = np.isin(ptype, list(codes["card_entry_type"].values()))
    cards = observed[on_card]
    card_frame = pd.DataFrame({
        "CARD_TOKEN": _map_unique(cards["account_id"], card_token),
        "TXN_REF": cards["transaction_id"].to_numpy(),
        "AUTH_TS": pd.to_datetime(cards["booking_datetime"]).dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy(),
        "AMOUNT_MINOR": np.round(cards["amount"].to_numpy() * 100).astype(np.int64),
        "CURRENCY": cards["currency"].to_numpy(),
        "DIRECTION": np.where(cards["credit_debit_indicator"].to_numpy() == "Credit", "C", "D"),
        "MERCHANT_DESCRIPTOR": cards["transaction_information"].str.replace(r" CD \d{4}$", "", regex=True).to_numpy(),
        "MERCHANT_NAME": cards["merchant_name"].to_numpy(),
        "MCC": cards["merchant_category_code"].to_numpy(),
        "ENTRY_TYPE": cards["proprietary_bank_transaction_code"].map(_invert(codes["card_entry_type"])).to_numpy(),
    })
    replays = card_frame.sample(frac=rates["card_replay"], random_state=chunk_id)
    card_frame = pd.concat([card_frame, replays], ignore_index=True)
    folder = out / "card_processor"
    folder.mkdir(parents=True, exist_ok=True)
    card_frame.to_parquet(folder / f"part-{chunk_id:04d}.parquet", index=False, compression="zstd")

    pays = observed[~on_card]
    amount = pays["amount"].to_numpy()
    signed = np.where(pays["credit_debit_indicator"].to_numpy() == "Credit", amount, -amount)
    acct_no = _map_unique(pays["account_id"], account_number)
    legacy = rng.random(len(pays)) < rates["legacy_account_format"]
    acct_no[legacy] = [a.replace("-", "") for a in acct_no[legacy]]
    pay_frame = pd.DataFrame({
        "PAYMENT_ID": pays["transaction_id"].to_numpy(),
        "ACCT_NO": acct_no,
        "VALUE_DATE": pd.to_datetime(pays["booking_datetime"]).dt.strftime("%d/%m/%Y").to_numpy(),
        "SCHEME": pays["proprietary_bank_transaction_code"].map(_invert(codes["payment_scheme"])).to_numpy(),
        "AMOUNT": [f"{v:.2f}" for v in signed],
        "CURRENCY": pays["currency"].to_numpy(),
        "NARRATIVE": pays["transaction_information"].str.ljust(35).to_numpy(),
    })
    folder = out / "payments_hub"
    folder.mkdir(parents=True, exist_ok=True)
    pay_frame.to_csv(folder / f"part-{chunk_id:04d}.csv.gz", index=False, compression="gzip")
    return {"card_rows": int(len(card_frame)), "card_replays": int(len(replays)), "payment_rows": int(len(pay_frame)),
            "legacy_account_numbers": int(legacy.sum())}


# --------------------------------------------------------------------------- control file


def _count_rows(path: Path) -> int:
    name = path.name
    if name.endswith(".parquet"):
        return pq.ParquetFile(path).metadata.num_rows
    if name.endswith(".csv.gz"):
        with gzip.open(path, "rb") as handle:
            return sum(1 for _ in handle) - 1
    if name.endswith(".csv"):
        with open(path, "rb") as handle:
            return sum(1 for _ in handle) - 1
    if name.endswith(".jsonl"):
        with open(path, "rb") as handle:
            return sum(1 for _ in handle)
    if name.endswith(".xml"):
        return sum(1 for _ in etree.iterparse(str(path), events=("end",), tag="valuation", resolve_entities=False))
    raise ValueError(f"unknown extract type: {path}")


def write_controls(sources: Path, extract_date: str) -> dict:
    """_control.json: what the receiving side checks before loading anything."""
    files = {}
    for path in sorted(p for p in sources.rglob("*") if p.is_file() and not p.name.startswith("_")):
        files[str(path.relative_to(sources))] = {"rows": _count_rows(path), "bytes": path.stat().st_size,
                                                 "sha256": sha256_file(path)}
    control = {"extract_date": extract_date, "files": files}
    write_json(sources / "_control.json", control)
    return control


def load_controls(sources: Path) -> dict:
    return json.loads((sources / "_control.json").read_text())
