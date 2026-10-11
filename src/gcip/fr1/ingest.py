"""FR1 ingestion: seven source-system extracts -> one harmonised warehouse.

Real bank data arrives from separate systems, each with its own format, keys,
units, dates, codes and quality problems (configs/source_systems.yaml). This
module, in order:

1. checks every extract against the control file (size and SHA-256 before
   parsing, row counts after) and stops the load if anything is off;
2. parses each system's dates, units and codes into one convention: ISO dates,
   GBP, rates as fractions, readable category values;
3. resolves keys through crosswalks: account numbers and card tokens to accounts,
   CRM contact ids and pseudonymised analytics ids to customers;
4. removes duplicate records and quarantines records that match no customer,
   each with its reason;
5. writes the harmonised warehouse that Customer 360 joins (gcip.fr1.customer360),
   plus meta/ingestion_report.json, meta/lineage.csv and quarantine/*.parquet;
6. when the simulator's truth vault is present, audits the result: every table's
   fingerprint and every transaction's record hash must match what the bank's
   systems held before extraction.

The warehouse is built in a staging folder and swapped in only when complete.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree
from tqdm import tqdm

from gcip.config import load_config
from gcip.io.files import sha256_file, write_json
from gcip.io.fingerprint import TABLE_KEYS, TRANSACTION_AUDIT_COLUMNS, row_hashes, table_fingerprint

log = logging.getLogger("gcip.fr1.ingest")
_G: dict = {}  # shared with forked shard workers

REGIONS = ["North East", "North West", "Yorkshire and The Humber", "East Midlands", "West Midlands", "East", "London",
           "South East", "South West", "Wales"]
WAREHOUSE_TABLES = ["customers", "accounts", "balances", "product_holdings", "preference_changes", "assets",
                    "digital_activity", "campaign_responses"]

# Where every warehouse column comes from: (table, column, source file, source field, transformation).
LINEAGE = [
    ("customers", "customer_id", "core_banking/customers.csv", "CUST_ID", "as delivered; duplicates quarantined"),
    ("customers", "birth_year", "core_banking/customers.csv", "BIRTH_YR", "integer"),
    ("customers", "la_code, la_name", "core_banking/customers.csv", "LA_CD, LA_NM", "as delivered"),
    ("customers", "region", "core_banking/customers.csv", "REGION_NM", "trimmed and re-cased to the ONS region name"),
    ("customers", "declared_residential_status", "core_banking/customers.csv", "RES_STATUS", "code table residential_status"),
    ("customers", "declared_employment_status", "core_banking/customers.csv", "EMP_STATUS", "code table employment_status"),
    ("customers", "declared_income_band", "core_banking/customers.csv", "INC_BAND", "code table income_band"),
    ("customers", "customer_since", "core_banking/customers.csv", "CUST_SINCE", "YYYYMMDD to date"),
    ("customers", "marketing_consent, vulnerability_flag", "core_banking/customers.csv", "MKT_CONSENT, VULN_FLAG", "Y/N to boolean"),
    ("customers", "preferred_contact_channel", "core_banking/customers.csv", "PREF_CHANNEL", "code table contact_channel"),
    ("customers", "stated_sustainability_preference", "core_banking/customers.csv", "SUST_PREF", "Y/N/blank to 1/0/null"),
    ("customers", "declared_occupation", "core_banking/customers.csv", "OCCUPATION_CD", "code table occupation (SOC 2020 major group or status)"),
    ("customers", "declared_dependants, declared_years_at_address", "core_banking/customers.csv", "DEPENDANTS, YRS_AT_ADDR", "number"),
    ("accounts", "account_id", "core_banking/accounts.csv", "CUST_ID, ACCT_TYPE", "harmonised key '<customer>-<account type>'"),
    ("accounts", "customer_id", "core_banking/accounts.csv", "CUST_ID", "must exist in customers, else quarantined"),
    ("accounts", "account_type", "core_banking/accounts.csv", "ACCT_FAMILY", "lower case"),
    ("accounts", "product_code", "core_banking/accounts.csv", "PRODUCT_CD", "as delivered"),
    ("accounts", "opened_date, closed_date", "core_banking/accounts.csv", "OPEN_DT, CLOSE_DT", "YYYYMMDD to date"),
    ("accounts", "status", "core_banking/accounts.csv", "CLOSE_DT", "closed if the close date is on or before the extract date"),
    ("accounts", "statement_preference", "core_banking/accounts.csv", "STMT_PREF", "code table statement_preference"),
    ("accounts", "credit_limit, original_amount, collateral_value_at_origination", "core_banking/accounts.csv",
     "CREDIT_LIMIT_PENCE, ORIG_AMT_PENCE, COLLATERAL_VALUE_PENCE", "pence to GBP"),
    ("accounts", "interest_rate", "core_banking/accounts.csv", "RATE_BPS", "basis points to a fraction"),
    ("accounts", "term_months, loan_purpose, provider", "core_banking/accounts.csv", "TERM_MTHS, LOAN_PURPOSE, PROVIDER", "number; lower case; as delivered"),
    ("balances", "account_id", "core_banking/balances.csv", "ACCT_NO", "account-number crosswalk from accounts.csv"),
    ("balances", "month_end", "core_banking/balances.csv", "MONTH_END", "YYYYMMDD to date"),
    ("balances", "balance", "core_banking/balances.csv", "BAL_PENCE", "pence to GBP"),
    ("balances", "arrears_months", "core_banking/balances.csv", "ARREARS_MTHS", "integer"),
    ("transactions", "transaction_id", "card_processor/part-*.parquet | payments_hub/part-*.csv.gz", "TXN_REF | PAYMENT_ID", "replayed duplicates removed"),
    ("transactions", "account_id", "card_processor/part-*.parquet | payments_hub/part-*.csv.gz", "CARD_TOKEN | ACCT_NO",
     "card token -> card_registry.csv -> account number -> crosswalk | account number normalised (legacy 14-digit form) -> crosswalk"),
    ("transactions", "booking_datetime", "card_processor | payments_hub", "AUTH_TS | VALUE_DATE", "ISO timestamp | DD/MM/YYYY to datetime"),
    ("transactions", "amount, credit_debit_indicator", "card_processor | payments_hub", "AMOUNT_MINOR, DIRECTION | AMOUNT",
     "pence to GBP, C/D to Credit/Debit | absolute value of signed pounds, sign to Credit/Debit"),
    ("transactions", "currency", "card_processor | payments_hub", "CURRENCY", "as delivered"),
    ("transactions", "transaction_information", "card_processor | payments_hub", "MERCHANT_DESCRIPTOR + CARD_LAST4 | NARRATIVE",
     "statement form '<descriptor> CD <last 4 digits>' | right-trimmed"),
    ("transactions", "merchant_name, merchant_category_code", "card_processor", "MERCHANT_NAME, MCC", "card payments only; null for payments"),
    ("transactions", "proprietary_bank_transaction_code", "card_processor | payments_hub", "ENTRY_TYPE | SCHEME", "code tables card_entry_type | payment_scheme"),
    ("transactions", "bank_transaction_code", "derived", "proprietary_bank_transaction_code", "code table open_banking_code"),
    ("transactions", "status", "derived", "-", "'Booked' (both feeds carry settled items only)"),
    ("product_holdings", "customer_id, account_id", "product_system/holdings.csv", "ACCT_NO, CUSTOMER_REF",
     "account-number crosswalk; owner checked against CUSTOMER_REF"),
    ("product_holdings", "product_code, product_family", "product_system/holdings.csv", "PRODUCT_CD, PRODUCT_FAMILY", "as delivered"),
    ("product_holdings", "green_product", "product_system/holdings.csv", "PRODUCT_CD", "product codes in configs/products.yaml"),
    ("product_holdings", "start_date, end_date", "product_system/holdings.csv", "START_DATE, END_DATE", "ISO date"),
    ("preference_changes", "account_id", "product_system/statement_preference_events.csv", "ACCT_NO", "account-number crosswalk"),
    ("preference_changes", "change_date", "product_system/statement_preference_events.csv", "EVENT_TS", "timestamp to date"),
    ("preference_changes", "old_preference, new_preference", "product_system/statement_preference_events.csv", "OLD_PREF, NEW_PREF", "lower case"),
    ("assets", "asset_id, customer_id", "lending_assets/property_valuations.xml | vehicle_records.csv", "mortgage_account | MOTOR_ACCT_NO",
     "loan account crosswalk to the owner; asset id '<customer>-PROP' or '<customer>-VEH'"),
    ("assets", "property_type, floor_area_m2", "lending_assets/property_valuations.xml", "property/@type, property/@floor_area_m2", "lower case; number"),
    ("assets", "property_detail, construction_period, bedrooms", "lending_assets/property_valuations.xml",
     "property/@detail, @built, @bedrooms", "as delivered; number (from the valuation report)"),
    ("assets", "epc_band, epc_main_heating, solar_pv_on_record", "lending_assets/property_valuations.xml", "epc/@rating, @main_fuel, @solar_pv",
     "as delivered; code table epc_main_fuel; Y/N to boolean (null when no EPC match)"),
    ("assets", "vehicle_fuel_type, vehicle_first_registered_year, vehicle_registration", "lending_assets/vehicle_records.csv",
     "FUEL_TYPE, YEAR_OF_FIRST_REG, VRM", "code table dvla_fuel; number; as delivered"),
    ("assets", "valuation_at_origination", "lending_assets/property_valuations.xml | vehicle_records.csv", "value/@pence | VALUATION_PENCE", "pence to GBP"),
    ("digital_activity", "customer_id", "digital_analytics/monthly_engagement.jsonl", "uid",
     "re-derive each customer's uid with the shared salt; unmatched uids quarantined"),
    ("digital_activity", "month", "digital_analytics/monthly_engagement.jsonl", "month", "YYYY-MM to first day of month"),
    ("digital_activity", "app_logins, web_logins, branch_visits, estatement_views", "digital_analytics/monthly_engagement.jsonl",
     "app_sessions, web_sessions, branch_visits, estatement_views", "integer"),
    ("digital_activity", "paperless_all_accounts", "digital_analytics/monthly_engagement.jsonl", "paperless", "boolean"),
    ("campaign_responses", "customer_id", "crm_marketing/campaign_responses.csv", "CONTACT_ID",
     "contacts.csv crosswalk (several contact ids per customer allowed); prospects quarantined"),
    ("campaign_responses", "campaign_id, channel", "crm_marketing/campaign_responses.csv", "CAMPAIGN, CHANNEL", "as delivered; lower case"),
    ("campaign_responses", "product", "crm_marketing/campaign_responses.csv", "PRODUCT_NAME", "product names in configs/products.yaml to codes"),
    ("campaign_responses", "response", "crm_marketing/campaign_responses.csv", "OUTCOME", "spellings in code table campaign_outcome"),
    ("campaign_responses", "sent_date, response_date", "crm_marketing/campaign_responses.csv", "SENT, OUTCOME_DATE", "DD/MM/YYYY to date"),
]


class IngestionError(RuntimeError):
    """A control-total or integrity failure that stops the load."""


# --------------------------------------------------------------------------- helpers


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _blank(series: pd.Series) -> pd.Series:
    return series.where(series != "")


def _dates(series: pd.Series, fmt: str) -> pd.Series:
    return pd.to_datetime(_blank(series), format=fmt)


def _scaled(series: pd.Series, scale: float, issues: dict, field: str) -> pd.Series:
    raw = _blank(series)
    values = pd.to_numeric(raw, errors="coerce")
    bad = int((raw.notna() & values.isna()).sum())
    if bad:
        issues[f"{field}: unparseable numbers set to null"] = bad
    return values.astype(float) / scale


def _decode(raw: pd.Series, table: dict, field: str, issues: dict) -> pd.Series:
    values = raw.map(table)
    unknown = (raw != "") & values.isna()
    if unknown.any():
        issues[f"{field}: unknown codes set to null ({', '.join(sorted(raw[unknown].unique())[:5])})"] = int(unknown.sum())
    return values


def normalise_account_number(raw: pd.Series) -> pd.Series:
    """'SSSSSS-NNNNNNNN' from either the current or the legacy 14-digit form; null when malformed."""
    digits = raw.astype(str).str.replace(r"\D", "", regex=True)
    return (digits.str[:6] + "-" + digits.str[6:]).where(digits.str.len() == 14)


def _owner(account_ids: pd.Series) -> pd.Series:
    return account_ids.str.rsplit("-", n=1).str[0]


class Quarantine:
    """Rows that could not be attributed, kept with the reason (never silently dropped)."""

    def __init__(self) -> None:
        self.frames: dict[str, list[pd.DataFrame]] = {}

    def add(self, table: str, rows: pd.DataFrame, reason: str) -> int:
        if len(rows):
            self.frames.setdefault(table, []).append(rows.astype(str).assign(reason=reason))
        return int(len(rows))

    def write(self, folder: Path) -> dict:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        counts = {}
        for table, frames in self.frames.items():
            frame = pd.concat(frames, ignore_index=True)
            frame.to_parquet(folder / f"{table}.parquet", index=False)
            counts[table] = {str(k): int(v) for k, v in frame["reason"].value_counts().items()}
        return counts


def _record(report: dict, name: str, rows_in: int, rows_out: int, issues: dict) -> None:
    report[name] = {"rows_in": int(rows_in), "rows_out": int(rows_out), "issues": {k: v for k, v in issues.items() if v}}


# --------------------------------------------------------------------------- control totals


def verify_controls(sources: Path, strict: bool = True) -> dict:
    """Sizes and checksums of every listed file, before anything is parsed."""
    path = sources / "_control.json"
    if not path.exists():
        raise IngestionError(f"no control file at {path}")
    control = json.loads(path.read_text())
    failures = []
    for rel, spec in tqdm(control["files"].items(), desc="control totals", unit="file", leave=False):
        target = sources / rel
        if not target.exists():
            failures.append({"file": rel, "problem": "missing"})
        elif target.stat().st_size != spec["bytes"]:
            failures.append({"file": rel, "problem": f"size {target.stat().st_size} != {spec['bytes']}"})
        elif sha256_file(target) != spec["sha256"]:
            failures.append({"file": rel, "problem": "SHA-256 differs"})
    listed = set(control["files"])
    unlisted = sorted(str(p.relative_to(sources)) for p in sources.rglob("*")
                      if p.is_file() and not p.name.startswith("_") and str(p.relative_to(sources)) not in listed)
    if failures and strict:
        raise IngestionError(f"control check failed for {len(failures)} files: {failures[:5]}")
    return {"extract_date": control["extract_date"], "files_listed": len(listed), "files_verified": len(listed) - len(failures),
            "failures": failures, "unlisted_files_ignored": unlisted, "expected_rows": {k: v["rows"] for k, v in control["files"].items()}}


# --------------------------------------------------------------------------- core banking


def ingest_customers(sources: Path, codes: dict, report: dict, quarantine: Quarantine) -> pd.DataFrame:
    name = "core_banking/customers.csv"
    raw = _csv(sources / name)
    issues: dict = {}
    dup = raw["CUST_ID"].duplicated(keep="first")
    issues["duplicate CUST_ID quarantined"] = quarantine.add("customers", raw[dup], "duplicate CUST_ID")
    rows_in = len(raw)
    raw = raw[~dup].reset_index(drop=True)

    region_raw = raw["REGION_NM"].str.strip()
    region = region_raw.str.lower().map({r.lower(): r for r in REGIONS})
    issues["region names re-cased to the ONS spelling"] = int((region.notna() & (region != region_raw)).sum())
    issues["region names not recognised (set to null)"] = int(region.isna().sum())
    for field in ("MKT_CONSENT", "VULN_FLAG"):
        issues[f"{field}: values other than Y/N read as N"] = int((~raw[field].isin(["Y", "N"])).sum())
    frame = pd.DataFrame({
        "customer_id": raw["CUST_ID"],
        "birth_year": pd.to_numeric(raw["BIRTH_YR"]).astype(int),
        "la_code": raw["LA_CD"],
        "la_name": raw["LA_NM"],
        "region": region,
        "declared_residential_status": _decode(raw["RES_STATUS"], codes["residential_status"], "RES_STATUS", issues),
        "declared_employment_status": _decode(raw["EMP_STATUS"], codes["employment_status"], "EMP_STATUS", issues),
        "declared_income_band": _decode(raw["INC_BAND"], codes["income_band"], "INC_BAND", issues),
        "customer_since": _dates(raw["CUST_SINCE"], "%Y%m%d"),
        "marketing_consent": raw["MKT_CONSENT"].eq("Y"),
        "vulnerability_flag": raw["VULN_FLAG"].eq("Y"),
        "preferred_contact_channel": _decode(raw["PREF_CHANNEL"], codes["contact_channel"], "PREF_CHANNEL", issues),
        "stated_sustainability_preference": raw["SUST_PREF"].map({"Y": 1.0, "N": 0.0}).astype(float),
        "declared_occupation": _decode(raw["OCCUPATION_CD"], codes["occupation"], "OCCUPATION_CD", issues),
        "declared_dependants": _scaled(raw["DEPENDANTS"], 1, issues, "DEPENDANTS"),
        "declared_years_at_address": _scaled(raw["YRS_AT_ADDR"], 1, issues, "YRS_AT_ADDR"),
    })
    _record(report, name, rows_in, len(frame), issues)
    return frame


def ingest_accounts(sources: Path, codes: dict, customer_ids: set, extract_date: pd.Timestamp, report: dict,
                    quarantine: Quarantine) -> tuple[pd.DataFrame, pd.Series]:
    """Accounts plus the crosswalk from account number to harmonised account id."""
    name = "core_banking/accounts.csv"
    raw = _csv(sources / name)
    issues: dict = {}
    acct_no = normalise_account_number(raw["ACCT_NO"])
    bad = acct_no.isna()
    orphan = ~bad & ~raw["CUST_ID"].isin(customer_ids)
    dup = ~bad & ~orphan & acct_no.duplicated(keep="first")
    issues["malformed account numbers quarantined"] = quarantine.add("accounts", raw[bad], "malformed ACCT_NO")
    issues["accounts of unknown customers quarantined"] = quarantine.add("accounts", raw[orphan], "CUST_ID not in customers")
    issues["duplicate account numbers quarantined"] = quarantine.add("accounts", raw[dup], "duplicate ACCT_NO")
    keep = ~(bad | orphan | dup)
    rows_in = len(raw)
    raw, acct_no = raw[keep].reset_index(drop=True), acct_no[keep].reset_index(drop=True)
    frame = pd.DataFrame({
        "account_id": raw["CUST_ID"] + "-" + raw["ACCT_TYPE"],
        "customer_id": raw["CUST_ID"],
        "account_type": raw["ACCT_FAMILY"].str.lower(),
        "product_code": raw["PRODUCT_CD"],
        "opened_date": _dates(raw["OPEN_DT"], "%Y%m%d"),
        "statement_preference": _decode(raw["STMT_PREF"], codes["statement_preference"], "STMT_PREF", issues),
        "credit_limit": _scaled(raw["CREDIT_LIMIT_PENCE"], 100, issues, "CREDIT_LIMIT_PENCE"),
        "original_amount": _scaled(raw["ORIG_AMT_PENCE"], 100, issues, "ORIG_AMT_PENCE"),
        "interest_rate": _scaled(raw["RATE_BPS"], 10000, issues, "RATE_BPS"),
        "term_months": _scaled(raw["TERM_MTHS"], 1, issues, "TERM_MTHS"),
        "collateral_value_at_origination": _scaled(raw["COLLATERAL_VALUE_PENCE"], 100, issues, "COLLATERAL_VALUE_PENCE"),
        "closed_date": _dates(raw["CLOSE_DT"], "%Y%m%d"),
        "loan_purpose": _blank(raw["LOAN_PURPOSE"]).str.lower(),
        "provider": raw["PROVIDER"],
    })
    frame["status"] = np.where(frame["closed_date"].notna() & (frame["closed_date"] <= extract_date), "closed", "open")
    issues["duplicate harmonised account ids"] = int(frame["account_id"].duplicated().sum())
    _record(report, name, rows_in, len(frame), issues)
    return frame, pd.Series(frame["account_id"].to_numpy(), index=acct_no.to_numpy())


def ingest_balances(sources: Path, crosswalk: pd.Series, report: dict, quarantine: Quarantine) -> pd.DataFrame:
    name = "core_banking/balances.csv"
    raw = _csv(sources / name)
    issues: dict = {}
    account_id = normalise_account_number(raw["ACCT_NO"]).map(crosswalk)
    orphan = account_id.isna()
    issues["balances of unknown accounts quarantined"] = quarantine.add("balances", raw[orphan], "ACCT_NO not in accounts")
    rows_in = len(raw)
    raw, account_id = raw[~orphan].reset_index(drop=True), account_id[~orphan].reset_index(drop=True)
    frame = pd.DataFrame({
        "account_id": account_id,
        "month_end": _dates(raw["MONTH_END"], "%Y%m%d"),
        "balance": _scaled(raw["BAL_PENCE"], 100, issues, "BAL_PENCE"),
        "arrears_months": pd.to_numeric(raw["ARREARS_MTHS"]).astype(int),
    })
    dup = frame.duplicated(["account_id", "month_end"], keep="first")
    issues["duplicate month-end rows quarantined"] = quarantine.add("balances", frame[dup], "duplicate (account, month end)")
    frame = frame[~dup].reset_index(drop=True)
    _record(report, name, rows_in, len(frame), issues)
    return frame


# --------------------------------------------------------------------------- product system


def ingest_product_system(sources: Path, crosswalk: pd.Series, report: dict,
                          quarantine: Quarantine) -> tuple[pd.DataFrame, pd.DataFrame]:
    products = load_config("products")["products"]
    green = {code: gp for gp, spec in products.items() for code in spec.get("product_codes", [])}

    name = "product_system/holdings.csv"
    raw = _csv(sources / name)
    issues: dict = {}
    account_id = normalise_account_number(raw["ACCT_NO"]).map(crosswalk)
    orphan = account_id.isna()
    issues["holdings of unknown accounts quarantined"] = quarantine.add("product_holdings", raw[orphan], "ACCT_NO not in accounts")
    rows_in = len(raw)
    raw, account_id = raw[~orphan].reset_index(drop=True), account_id[~orphan].reset_index(drop=True)
    owner = _owner(account_id)
    issues["CUSTOMER_REF differs from the account owner (owner kept)"] = int((owner != raw["CUSTOMER_REF"]).sum())
    holdings = pd.DataFrame({
        "customer_id": owner,
        "account_id": account_id,
        "product_code": raw["PRODUCT_CD"],
        "product_family": raw["PRODUCT_FAMILY"],
        "green_product": raw["PRODUCT_CD"].map(green).fillna(""),
        "start_date": _dates(raw["START_DATE"], "%Y-%m-%d"),
        "end_date": _dates(raw["END_DATE"], "%Y-%m-%d"),
    })
    _record(report, name, rows_in, len(holdings), issues)

    name = "product_system/statement_preference_events.csv"
    raw = _csv(sources / name)
    issues = {}
    account_id = normalise_account_number(raw["ACCT_NO"]).map(crosswalk)
    orphan = account_id.isna()
    issues["events of unknown accounts quarantined"] = quarantine.add("preference_changes", raw[orphan], "ACCT_NO not in accounts")
    rows_in = len(raw)
    raw, account_id = raw[~orphan].reset_index(drop=True), account_id[~orphan].reset_index(drop=True)
    changes = pd.DataFrame({
        "account_id": account_id,
        "change_date": pd.to_datetime(raw["EVENT_TS"], format="%Y-%m-%dT%H:%M:%S").dt.normalize(),
        "old_preference": raw["OLD_PREF"].str.lower(),
        "new_preference": raw["NEW_PREF"].str.lower(),
    })
    _record(report, name, rows_in, len(changes), issues)
    return holdings, changes


# --------------------------------------------------------------------------- lending assets


def _attr(element, key: str) -> str:
    return element.get(key, "") if element is not None else ""


def _property_rows(path: Path) -> pd.DataFrame:
    """Flatten the valuation XML; entity expansion and network access are disabled."""
    columns = ["MORTGAGE_ACCT_NO", "TYPE", "DETAIL", "BUILT", "BEDROOMS", "FLOOR_AREA", "EPC_RATING", "EPC_FUEL", "SOLAR_PV",
               "VALUE_PENCE"]
    rows = []
    for _, node in etree.iterparse(str(path), events=("end",), tag="valuation", resolve_entities=False, no_network=True):
        prop, epc, value = node.find("property"), node.find("epc"), node.find("value")
        rows.append([node.get("mortgage_account", ""), _attr(prop, "type"), _attr(prop, "detail"), _attr(prop, "built"),
                     _attr(prop, "bedrooms"), _attr(prop, "floor_area_m2"), _attr(epc, "rating"), _attr(epc, "main_fuel"),
                     _attr(epc, "solar_pv"), _attr(value, "pence")])
        node.clear()
    return pd.DataFrame(rows, columns=columns, dtype=object)


def ingest_assets(sources: Path, codes: dict, crosswalk: pd.Series, report: dict, quarantine: Quarantine) -> pd.DataFrame:
    name = "lending_assets/property_valuations.xml"
    raw = _property_rows(sources / name)
    issues: dict = {}
    account_id = normalise_account_number(raw["MORTGAGE_ACCT_NO"]).map(crosswalk)
    orphan = account_id.isna()
    issues["valuations of unknown mortgages quarantined"] = quarantine.add("assets", raw[orphan], "mortgage account not in accounts")
    rows_in = len(raw)
    raw, account_id = raw[~orphan].reset_index(drop=True), account_id[~orphan].reset_index(drop=True)
    customer = _owner(account_id)
    issues["properties without an EPC register match"] = int((raw["EPC_RATING"] == "").sum())
    properties = pd.DataFrame({
        "asset_id": customer + "-PROP",
        "customer_id": customer,
        "asset_type": "property",
        "property_type": raw["TYPE"].str.lower(),
        "property_detail": _blank(raw["DETAIL"]),
        "construction_period": _blank(raw["BUILT"]),
        "bedrooms": _scaled(raw["BEDROOMS"], 1, issues, "bedrooms"),
        "floor_area_m2": _scaled(raw["FLOOR_AREA"], 1, issues, "floor_area_m2"),
        "epc_band": _blank(raw["EPC_RATING"]),
        "epc_main_heating": _decode(raw["EPC_FUEL"], codes["epc_main_fuel"], "main_fuel", issues),
        "solar_pv_on_record": raw["SOLAR_PV"].map({"Y": True, "N": False}),
        "valuation_at_origination": _scaled(raw["VALUE_PENCE"], 100, issues, "pence"),
    })
    _record(report, name, rows_in, len(properties), issues)

    name = "lending_assets/vehicle_records.csv"
    raw = _csv(sources / name)
    issues = {}
    account_id = normalise_account_number(raw["MOTOR_ACCT_NO"]).map(crosswalk)
    orphan = account_id.isna()
    issues["vehicles of unknown motor loans quarantined"] = quarantine.add("assets", raw[orphan], "motor loan account not in accounts")
    rows_in = len(raw)
    raw, account_id = raw[~orphan].reset_index(drop=True), account_id[~orphan].reset_index(drop=True)
    customer = _owner(account_id)
    vehicles = pd.DataFrame({
        "asset_id": customer + "-VEH",
        "customer_id": customer,
        "asset_type": "vehicle",
        "vehicle_fuel_type": _decode(raw["FUEL_TYPE"], codes["dvla_fuel"], "FUEL_TYPE", issues),
        "vehicle_first_registered_year": _scaled(raw["YEAR_OF_FIRST_REG"], 1, issues, "YEAR_OF_FIRST_REG"),
        "vehicle_registration": raw["VRM"],
        "valuation_at_origination": _scaled(raw["VALUATION_PENCE"], 100, issues, "VALUATION_PENCE"),
    })
    _record(report, name, rows_in, len(vehicles), issues)
    return pd.concat([properties, vehicles], ignore_index=True)


# --------------------------------------------------------------------------- CRM


def ingest_crm(sources: Path, codes: dict, customer_ids: set, report: dict, quarantine: Quarantine) -> pd.DataFrame:
    products = load_config("products")["products"]
    product_of = {spec["name"].lower(): gp for gp, spec in products.items()}
    outcome_of = {form.strip().lower(): value for value, forms in codes["campaign_outcome"].items() for form in forms}

    name = "crm_marketing/contacts.csv"
    contacts = _csv(sources / name)
    issues: dict = {}
    dup = contacts["CONTACT_ID"].duplicated(keep="first")
    issues["duplicate contact ids quarantined"] = quarantine.add("crm_contacts", contacts[dup], "duplicate CONTACT_ID")
    contacts = contacts[~dup]
    known = contacts[contacts["CUST_REF"] != ""]
    per_customer = known.groupby("CUST_REF").size()
    issues["customers with more than one contact id (merged)"] = int((per_customer > 1).sum())
    issues["prospect contacts (no customer reference)"] = int((contacts["CUST_REF"] == "").sum())
    _record(report, name, len(contacts) + int(dup.sum()), len(known), issues)
    crosswalk = pd.Series(known["CUST_REF"].to_numpy(), index=known["CONTACT_ID"].to_numpy())
    secondary = set(known.loc[known["CUST_REF"].duplicated(keep="first"), "CONTACT_ID"])
    prospects = set(contacts.loc[contacts["CUST_REF"] == "", "CONTACT_ID"])

    name = "crm_marketing/campaign_responses.csv"
    raw = _csv(sources / name)
    issues = {}
    customer = raw["CONTACT_ID"].map(crosswalk)
    is_prospect = raw["CONTACT_ID"].isin(prospects)
    unknown = customer.isna() & ~is_prospect
    not_customer = customer.notna() & ~customer.isin(customer_ids)
    product = raw["PRODUCT_NAME"].str.strip().str.lower().map(product_of)
    no_product = customer.notna() & ~not_customer & product.isna()
    issues["responses from prospects quarantined"] = quarantine.add("campaign_responses", raw[is_prospect], "prospect: no customer reference")
    issues["responses from unknown contact ids quarantined"] = quarantine.add("campaign_responses", raw[unknown], "CONTACT_ID not in contacts")
    issues["responses for customers not in core banking quarantined"] = quarantine.add("campaign_responses", raw[not_customer], "CUST_REF not in customers")
    issues["responses for unknown products quarantined"] = quarantine.add("campaign_responses", raw[no_product], "PRODUCT_NAME not in catalogue")
    issues["responses resolved through a second contact id"] = int(raw["CONTACT_ID"].isin(secondary).sum())
    keep = ~(is_prospect | unknown | not_customer | no_product)
    rows_in = len(raw)
    raw, customer, product = raw[keep].reset_index(drop=True), customer[keep].reset_index(drop=True), product[keep].reset_index(drop=True)
    spelled = raw["OUTCOME"].str.strip().str.lower()
    response = spelled.map(outcome_of)
    issues["outcome spellings harmonised"] = int((spelled != response).sum())
    unknown_outcome = response.isna()
    if unknown_outcome.any():
        issues[f"outcomes not recognised, set to no_response ({', '.join(sorted(raw.loc[unknown_outcome, 'OUTCOME'].unique())[:5])})"] = int(unknown_outcome.sum())
    frame = pd.DataFrame({
        "campaign_id": raw["CAMPAIGN"],
        "customer_id": customer,
        "sent_date": _dates(raw["SENT"], "%d/%m/%Y"),
        "product": product,
        "channel": raw["CHANNEL"].str.lower(),
        "response": response.fillna("no_response"),
        "response_date": _dates(raw["OUTCOME_DATE"], "%d/%m/%Y"),
    })
    _record(report, name, rows_in, len(frame), issues)
    return frame


# --------------------------------------------------------------------------- digital analytics


def ingest_digital(sources: Path, salt: str, customer_ids: list, report: dict, quarantine: Quarantine) -> pd.DataFrame:
    name = "digital_analytics/monthly_engagement.jsonl"
    raw = pd.read_json(sources / name, lines=True, dtype={"uid": str, "month": str}, convert_dates=False)
    issues: dict = {}
    uid_of = {hashlib.sha256(f"{salt}:{c}".encode()).hexdigest()[:16]: c for c in customer_ids}
    customer = raw["uid"].map(uid_of)
    orphan = customer.isna()
    issues["records with no matching customer (prospects) quarantined"] = quarantine.add("digital_activity", raw[orphan], "uid matches no customer")
    rows_in = len(raw)
    raw, customer = raw[~orphan].reset_index(drop=True), customer[~orphan].reset_index(drop=True)
    frame = pd.DataFrame({
        "customer_id": customer,
        "month": pd.to_datetime(raw["month"], format="%Y-%m"),
        "app_logins": raw["app_sessions"].astype(int),
        "web_logins": raw["web_sessions"].astype(int),
        "branch_visits": raw["branch_visits"].astype(int),
        "estatement_views": raw["estatement_views"].astype(int),
        "paperless_all_accounts": raw["paperless"].astype(bool),
    })
    dup = frame.duplicated(["customer_id", "month"], keep="first")
    issues["duplicate customer-months quarantined"] = quarantine.add("digital_activity", frame[dup], "duplicate (customer, month)")
    frame = frame[~dup].sort_values(["customer_id", "month"]).reset_index(drop=True)
    _record(report, name, rows_in, len(frame), issues)
    return frame


# --------------------------------------------------------------------------- transactions


def ingest_card_registry(sources: Path, crosswalk: pd.Series, report: dict, quarantine: Quarantine) -> pd.DataFrame:
    name = "card_processor/card_registry.csv"
    raw = _csv(sources / name)
    issues: dict = {}
    account_id = normalise_account_number(raw["ACCT_NO"]).map(crosswalk)
    orphan = account_id.isna()
    dup = ~orphan & raw["CARD_TOKEN"].duplicated(keep="first")
    issues["cards of unknown accounts quarantined"] = quarantine.add("card_registry", raw[orphan], "ACCT_NO not in accounts")
    issues["duplicate card tokens quarantined"] = quarantine.add("card_registry", raw[dup], "duplicate CARD_TOKEN")
    keep = ~(orphan | dup)
    registry = pd.DataFrame({"account_id": account_id[keep].to_numpy(), "last4": raw.loc[keep, "CARD_LAST4"].to_numpy()},
                            index=raw.loc[keep, "CARD_TOKEN"].to_numpy())
    _record(report, name, len(raw), len(registry), issues)
    return registry


def _ingest_shard(shard: str) -> dict:
    """One shard: card feed + payments feed -> one harmonised transactions part (same customers)."""
    g = _G
    codes = g["codes"]
    card = pd.read_parquet(g["sources"] / "card_processor" / f"{shard}.parquet")
    pays = pd.read_csv(g["sources"] / "payments_hub" / f"{shard}.csv.gz", dtype=str, keep_default_na=False)
    stats: dict = {"shard": shard, "card_rows": int(len(card)), "payment_rows": int(len(pays))}
    quarantined = []

    replay = card.duplicated("TXN_REF", keep="first")
    stats["card_replays_removed"] = int(replay.sum())
    card = card[~replay].reset_index(drop=True)
    account_id = card["CARD_TOKEN"].map(g["registry"]["account_id"])
    unknown = account_id.isna()
    if unknown.any():
        quarantined.append(card[unknown].astype(str).assign(reason="card token not in registry"))
    card, account_id = card[~unknown].reset_index(drop=True), account_id[~unknown].reset_index(drop=True)
    last4 = card["CARD_TOKEN"].map(g["registry"]["last4"])
    card_tx = pd.DataFrame({
        "transaction_id": card["TXN_REF"].to_numpy(),
        "account_id": account_id.to_numpy(),
        "booking_datetime": pd.to_datetime(card["AUTH_TS"], format="%Y-%m-%dT%H:%M:%S").to_numpy(),
        "amount": card["AMOUNT_MINOR"].to_numpy() / 100.0,
        "currency": card["CURRENCY"].to_numpy(),
        "credit_debit_indicator": card["DIRECTION"].map({"C": "Credit", "D": "Debit"}).to_numpy(),
        "transaction_information": (card["MERCHANT_DESCRIPTOR"] + " CD " + last4).to_numpy(),
        "merchant_name": card["MERCHANT_NAME"].to_numpy(),
        "merchant_category_code": card["MCC"].to_numpy(),
        "proprietary_bank_transaction_code": card["ENTRY_TYPE"].map(codes["card_entry_type"]).to_numpy(),
    })

    acct_no = normalise_account_number(pays["ACCT_NO"])
    stats["legacy_account_numbers_normalised"] = int((acct_no.notna() & (acct_no != pays["ACCT_NO"])).sum())
    account_id = acct_no.map(g["crosswalk"])
    unknown = account_id.isna()
    if unknown.any():
        quarantined.append(pays[unknown].astype(str).assign(reason="ACCT_NO not in accounts"))
    pays, account_id = pays[~unknown].reset_index(drop=True), account_id[~unknown].reset_index(drop=True)
    dup = pays["PAYMENT_ID"].duplicated(keep="first")
    stats["payment_duplicates_removed"] = int(dup.sum())
    pays, account_id = pays[~dup].reset_index(drop=True), account_id[~dup].reset_index(drop=True)
    signed = pd.to_numeric(pays["AMOUNT"]).to_numpy()
    pay_tx = pd.DataFrame({
        "transaction_id": pays["PAYMENT_ID"].to_numpy(),
        "account_id": account_id.to_numpy(),
        "booking_datetime": pd.to_datetime(pays["VALUE_DATE"], format="%d/%m/%Y").to_numpy(),
        "amount": np.abs(signed),
        "currency": pays["CURRENCY"].to_numpy(),
        "credit_debit_indicator": np.where(signed > 0, "Credit", "Debit"),
        "transaction_information": pays["NARRATIVE"].str.rstrip().to_numpy(),
        "merchant_name": None,
        "merchant_category_code": None,
        "proprietary_bank_transaction_code": pays["SCHEME"].map(codes["payment_scheme"]).to_numpy(),
    })

    tx = pd.concat([card_tx, pay_tx], ignore_index=True)
    tx["bank_transaction_code"] = tx["proprietary_bank_transaction_code"].map(codes["open_banking_code"])
    tx["status"] = "Booked"
    stats["unknown_payment_codes"] = int(tx["proprietary_bank_transaction_code"].isna().sum())
    cross = tx["transaction_id"].duplicated(keep="first")
    stats["ids_in_both_feeds_removed"] = int(cross.sum())
    tx = tx[~cross].sort_values("transaction_id", kind="stable").reset_index(drop=True)
    tx.to_parquet(g["staging"] / "transactions" / f"{shard}.parquet", index=False, compression="zstd")
    stats["rows_out"] = int(len(tx))
    stats["quarantined"] = int(sum(len(q) for q in quarantined))
    if quarantined:
        pd.concat(quarantined, ignore_index=True).to_parquet(g["staging_quarantine"] / f"transactions-{shard}.parquet", index=False)
    stats["customers"] = sorted(_owner(pd.Series(tx["account_id"].unique())).unique().tolist())

    truth_path = g["truth"] / "transaction_truth" / f"{shard}.parquet" if g["truth"] else None
    if truth_path is not None and truth_path.exists():
        truth = pd.read_parquet(truth_path, columns=["transaction_id", "customer_id", "audit_account_id", "audit_fingerprint"])
        mine = pd.DataFrame({"transaction_id": tx["transaction_id"], "account_id": tx["account_id"],
                             "fingerprint": row_hashes(tx, TRANSACTION_AUDIT_COLUMNS)})
        both = truth.merge(mine, on="transaction_id", how="inner")
        stats["audit"] = {
            "expected": int(len(truth)),
            "recovered": int(len(both)),
            "missing": int(len(truth) - len(both)),
            "unexpected": int(len(mine) - len(both)),
            "wrong_account": int((both["account_id"] != both["audit_account_id"]).sum()),
            "wrong_customer": int((_owner(both["account_id"]) != both["customer_id"]).sum()),
            "field_mismatch": int((both["fingerprint"].to_numpy() != both["audit_fingerprint"].to_numpy()).sum()),
        }
    return stats


def ingest_transactions(sources: Path, staging: Path, truth: Path | None, codes: dict, crosswalk: pd.Series,
                        registry: pd.DataFrame, workers: int | None) -> list[dict]:
    shards = sorted(p.stem for p in (sources / "card_processor").glob("part-*.parquet"))
    missing = [s for s in shards if not (sources / "payments_hub" / f"{s}.csv.gz").exists()]
    if missing:
        raise IngestionError(f"payments-hub shards missing for {missing[:5]}")
    (staging / "transactions").mkdir(parents=True, exist_ok=True)
    staging_quarantine = staging / "_quarantine"
    staging_quarantine.mkdir(exist_ok=True)
    _G.update(sources=sources, staging=staging, staging_quarantine=staging_quarantine, truth=truth, codes=codes,
              crosswalk=crosswalk, registry=registry)
    workers = workers or max(1, min(len(shards), (os.cpu_count() or 2) - 1))
    if workers == 1:
        results = [_ingest_shard(s) for s in tqdm(shards, desc="transactions", unit="shard")]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            results = list(tqdm(pool.imap_unordered(_ingest_shard, shards), total=len(shards), desc="transactions", unit="shard"))
    return sorted(results, key=lambda r: r["shard"])


# --------------------------------------------------------------------------- audit and report


def audit_tables(tables: dict[str, pd.DataFrame], truth: Path) -> dict:
    """Compare each harmonised table with the fingerprint of the bank's system of record."""
    expected = json.loads((truth / "system_of_record.json").read_text())
    out = {}
    for name, frame in tables.items():
        exp = expected[name]
        got = table_fingerprint(frame, TABLE_KEYS[name])
        out[name] = {
            "rows_expected": exp["rows"], "rows_ingested": got["rows"],
            "exact_match": got["hash"] == exp["hash"] and got["columns"] == exp["columns"],
            "columns_missing": sorted(set(exp["columns"]) - set(got["columns"])),
            "columns_extra": sorted(set(got["columns"]) - set(exp["columns"])),
            "columns_differing": sorted(c for c, h in exp.get("column_hashes", {}).items() if got["column_hashes"].get(c) != h),
        }
    return out


def _summary(report: dict) -> list[str]:
    tx = report["transactions"]
    lines = [
        f"{len(report['systems'])} source systems, {report['control']['files_listed']} files; "
        f"{report['control']['files_verified']} passed size and SHA-256 checks; row counts "
        f"{'all match' if not report['control']['row_count_mismatches'] else 'DIFFER for ' + str(len(report['control']['row_count_mismatches'])) + ' files'}.",
        f"Transactions: {tx['card_rows']:,} card-feed rows ({tx['card_replays_removed']:,} replayed duplicates removed) and "
        f"{tx['payment_rows']:,} payments-hub rows ({tx['legacy_account_numbers_normalised']:,} legacy account numbers normalised) "
        f"-> {tx['rows_out']:,} transactions in {tx['shards']} parts.",
    ]
    for table, reasons in report["quarantine"].items():
        lines.append(f"Quarantined from {table}: " + "; ".join(f"{n:,} ({reason})" for reason, n in reasons.items()) + ".")
    if "audit" in report:
        audit = report["audit"]
        matched = [t for t, a in audit["tables"].items() if a["exact_match"]]
        lines.append(f"Audit: {len(matched)} of {len(audit['tables'])} tables identical to the bank's system of record.")
        t = audit["transactions"]
        lines.append(f"Audit: {t['recovered']:,} of {t['expected']:,} transactions recovered; {t['missing']} missing, "
                     f"{t['unexpected']} unexpected, {t['wrong_account']} on the wrong account, {t['wrong_customer']} on the wrong customer, "
                     f"{t['field_mismatch']} with any field different.")
    return lines


def ingest(data_dir: str | Path, workers: int | None = None, strict: bool = True) -> dict:
    """sources/ -> warehouse/, meta/ingestion_report.json, meta/lineage.csv, quarantine/."""
    t0 = time.time()
    data = Path(data_dir)
    sources = data / "sources"
    truth = data / "truth_vault" if (data / "truth_vault" / "system_of_record.json").exists() else None
    staging = data / "warehouse.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    dictionary = load_config("source_systems")
    codes = dictionary["codes"]
    report: dict = {"systems": {k: v["description"] for k, v in dictionary["systems"].items()}, "files": {}}
    quarantine = Quarantine()

    steps = tqdm(total=9, desc="FR1 ingestion", unit="step")
    control = verify_controls(sources, strict); steps.update()
    extract_date = pd.Timestamp(control["extract_date"])
    customers = ingest_customers(sources, codes, report["files"], quarantine); steps.update()
    customer_ids = set(customers["customer_id"])
    accounts, crosswalk = ingest_accounts(sources, codes, customer_ids, extract_date, report["files"], quarantine); steps.update()
    balances = ingest_balances(sources, crosswalk, report["files"], quarantine); steps.update()
    holdings, changes = ingest_product_system(sources, crosswalk, report["files"], quarantine); steps.update()
    assets = ingest_assets(sources, codes, crosswalk, report["files"], quarantine); steps.update()
    campaigns = ingest_crm(sources, codes, customer_ids, report["files"], quarantine); steps.update()
    digital = ingest_digital(sources, dictionary["pseudonymisation"]["digital_analytics_salt"], sorted(customer_ids),
                             report["files"], quarantine); steps.update()
    registry = ingest_card_registry(sources, crosswalk, report["files"], quarantine)
    tables = {"customers": customers, "accounts": accounts, "balances": balances, "product_holdings": holdings,
              "preference_changes": changes, "assets": assets, "digital_activity": digital, "campaign_responses": campaigns}
    for name, frame in tables.items():
        frame.to_parquet(staging / f"{name}.parquet", index=False, compression="zstd")
    shards = ingest_transactions(sources, staging, truth, codes, crosswalk, registry, workers); steps.update()
    steps.close()

    # Row counts against the control file, now that every file has been read.
    delivered = {name: info["rows_in"] for name, info in report["files"].items()}
    for s in shards:
        delivered[f"card_processor/{s['shard']}.parquet"] = s["card_rows"]
        delivered[f"payments_hub/{s['shard']}.csv.gz"] = s["payment_rows"]
    mismatches = {k: {"control": v, "read": delivered.get(k)} for k, v in control.pop("expected_rows").items() if delivered.get(k) != v}
    control["row_count_mismatches"] = mismatches
    if mismatches and strict:
        raise IngestionError(f"row counts differ from the control file: {dict(list(mismatches.items())[:5])}")
    report["control"] = control

    seen: dict[str, int] = {}
    for s in shards:
        for c in s.pop("customers"):
            seen[c] = seen.get(c, 0) + 1
    split_customers = sum(1 for v in seen.values() if v > 1)
    if split_customers:
        raise IngestionError(f"{split_customers} customers appear in more than one transaction shard")
    keys = ["card_rows", "card_replays_removed", "payment_rows", "legacy_account_numbers_normalised", "payment_duplicates_removed",
            "ids_in_both_feeds_removed", "unknown_payment_codes", "quarantined", "rows_out"]
    report["transactions"] = {k: int(sum(s[k] for s in shards)) for k in keys} | {"shards": len(shards), "customers": len(seen)}

    for path in sorted((staging / "_quarantine").glob("transactions-*.parquet")):
        quarantine.frames.setdefault("transactions", []).append(pd.read_parquet(path))
    shutil.rmtree(staging / "_quarantine")
    report["quarantine"] = quarantine.write(data / "quarantine")

    if truth is not None:
        report["audit"] = {"tables": audit_tables(tables, truth),
                           "transactions": {k: int(sum(s["audit"][k] for s in shards if "audit" in s))
                                            for k in ("expected", "recovered", "missing", "unexpected", "wrong_account",
                                                      "wrong_customer", "field_mismatch")}}
        t = report["audit"]["transactions"]
        report["audit"]["pass"] = (all(a["exact_match"] for a in report["audit"]["tables"].values())
                                   and t["missing"] == t["unexpected"] == t["wrong_account"] == t["wrong_customer"] == t["field_mismatch"] == 0)
    report["shards"] = shards
    report["elapsed_seconds"] = round(time.time() - t0, 1)
    report["summary"] = _summary(report)

    target, previous = data / "warehouse", data / "warehouse.previous"
    if previous.exists():
        shutil.rmtree(previous)
    if target.exists():
        target.rename(previous)
    staging.rename(target)
    if previous.exists():
        shutil.rmtree(previous)
    (data / "meta").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(LINEAGE, columns=["table", "column", "source", "source_field", "transformation"]).to_csv(
        data / "meta" / "lineage.csv", index=False)
    write_json(data / "meta" / "ingestion_report.json", report)
    for line in report["summary"]:
        log.info("ingestion: %s", line)
    return report
