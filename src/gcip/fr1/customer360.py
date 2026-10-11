"""FR1: Customer 360.

Real banks keep customers, accounts, balances, transactions, holdings, assets and
digital activity in separate systems. This module checks those tables (keys,
referential integrity, value ranges) and joins them into one point-in-time row per
customer as of a given date. Transactions are streamed part by part, so memory
stays flat however many customers there are; every part holds whole customers.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

TABLES = ["customers", "accounts", "balances", "product_holdings", "preference_changes", "assets",
          "digital_activity", "campaign_responses"]

# Public statistics for the customer's area (Census 2021, DESNZ, DfT, PVGIS), added as area_<column>. These describe
# the area, not the person. Only the ones a model or the carbon engine reads are kept.
AREA_COLUMNS = ["share_cars_0", "car_share_bev", "gas_dom_mean_kwh", "elec_dom_median_kwh", "pv_rate_per_household",
                "pv_kwh_per_kwp_year"]
GREEN = ["GP01", "GP02", "GP03", "GP04", "GP05", "GP06"]
# The curated Customer 360, in sections. Each section answers one question and names who reads it, so the view
# explains itself: the schema export, the per-customer JSON document and the dashboard all follow these sections.
C360_SECTIONS = {
    "keys": {
        "question": "Which customer, as of which date",
        "used_by": "every stage",
        "columns": ["customer_id", "as_of"],
    },
    "who_they_are": {
        "question": "Who the customer is and what they agreed to",
        "used_by": "models (context only, never a reason code), fairness audit, recommendation gating (consent, vulnerability, channel)",
        "columns": ["age", "declared_income_band", "declared_employment_status", "declared_residential_status",
                    "declared_years_at_address", "customer_tenure_years", "marketing_consent", "vulnerability_flag",
                    "preferred_contact_channel", "stated_sustainability_preference"],
    },
    "where_they_live": {
        "question": "Where they live and what their area is like (public statistics, not about the person)",
        "used_by": "models, bank insights by region",
        "columns": ["la_code", "la_name", "region", *[f"area_{c}" for c in AREA_COLUMNS]],
    },
    "what_they_hold": {
        "question": "Which accounts and loans they hold, and what the loans are secured on",
        "used_by": "eligibility rules, models, PCAF financed emissions",
        "columns": ["n_products", "n_mortgage_accounts", "n_motor_loan_accounts", "n_personal_loan_accounts",
                    "n_investment_accounts", "paper_accounts", "mortgage_property_value_at_origination",
                    "mortgage_current_ltv", "motor_loan_months_to_end", "personal_loan_purpose"],
    },
    "green_products_held": {
        "question": "Which green products they already have",
        "used_by": "already-holds blocks, conversion insights",
        "columns": [f"holds_{gp}" for gp in GREEN],
    },
    "money_health": {
        "question": "How their finances look",
        "used_by": "affordability and financial-difficulty blocks, confidence, models",
        "columns": ["balance_current", "balance_savings", "balance_credit_card", "balance_mortgage", "balance_investment",
                    "months_overdrawn_6m", "max_arrears_months_6m", "n_transactions_12m"],
    },
    "home_and_car_on_record": {
        "question": "What the bank knows about their home and car as a lender",
        "used_by": "eligibility rules, carbon engine, reason codes",
        "columns": ["asset_property_type", "asset_construction_period", "asset_floor_area_m2", "asset_epc_band",
                    "asset_epc_main_heating", "asset_solar_pv_on_record", "asset_vehicle_fuel_type", "asset_vehicle_age_years"],
    },
    "how_they_bank": {
        "question": "How they use digital banking",
        "used_by": "paperless product, delivery channel, models",
        "columns": ["logins_per_month_6m", "app_login_share_6m", "estatement_views_6m", "months_with_logins_6m"],
    },
    "what_we_said_before": {
        "question": "What we offered before and how they responded",
        "used_by": "customer memory (cool-downs), models",
        "columns": [*[f"last_response_{gp}" for gp in GREEN], *[f"last_response_date_{gp}" for gp in GREEN]],
    },
}
C360_COLUMNS = [c for section in C360_SECTIONS.values() for c in section["columns"]]
SECTION_OF = {c: name for name, section in C360_SECTIONS.items() for c in section["columns"]}

HPI_TYPE_COLUMN = {"detached": "price_detached", "semi": "price_semi", "terraced": "price_terraced", "flat": "price_flat"}


def load_area_context(reference_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Public area statistics a bank may legitimately use: authority profile and the UK House Price Index."""
    from gcip.config import REFERENCE_DIR
    directory = reference_dir or REFERENCE_DIR
    prices = pd.read_csv(directory / "house_prices_la_monthly.csv", parse_dates=["month"])
    return {"la": pd.read_csv(directory / "la_profile.csv"), "house_prices": prices}


def _hpi_lookup(prices: pd.DataFrame, la_codes: pd.Series, months: pd.Series, prop_type: pd.Series) -> np.ndarray:
    """HPI price by authority, month (clamped to the published range) and property type (average if unknown)."""
    month = pd.to_datetime(months).dt.to_period("M").dt.to_timestamp()
    month = month.clip(prices["month"].min(), prices["month"].max())
    table = prices.set_index(["la_code", "month"])
    keys = pd.MultiIndex.from_arrays([la_codes.to_numpy(), month.to_numpy()])
    out = table["avg_price"].reindex(keys).to_numpy(dtype=float)
    for kind, column in HPI_TYPE_COLUMN.items():
        mask = (prop_type == kind).to_numpy()
        if mask.any():
            values = table[column].reindex(keys[mask]).to_numpy(dtype=float)
            out[mask] = np.where(np.isfinite(values), values, out[mask])
    return out


def transaction_parts(warehouse: Path) -> list[Path]:
    return sorted((warehouse / "transactions").glob("part-*.parquet"))


def iter_transactions(warehouse: Path, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
    for part in transaction_parts(warehouse):
        yield pd.read_parquet(part, columns=columns)


def load_tables(warehouse: Path) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(warehouse / f"{name}.parquet") for name in TABLES}


def customer_of_account(account_ids: pd.Series) -> pd.Series:
    """Account ids are '<customer_id>-<suffix>'."""
    return account_ids.str.rsplit("-", n=1).str[0]


def data_quality(tables: dict[str, pd.DataFrame], warehouse: Path) -> dict:
    """Keys, referential integrity and value checks. Each check records pass/fail and a count."""
    checks = []

    def add(name: str, failures: int, detail: str = "") -> None:
        checks.append({"check": name, "failures": int(failures), "pass": failures == 0, "detail": detail})

    customers = set(tables["customers"]["customer_id"])
    accounts = tables["accounts"]
    account_ids = set(accounts["account_id"])
    add("customers.customer_id unique", tables["customers"]["customer_id"].duplicated().sum())
    add("accounts.account_id unique", accounts["account_id"].duplicated().sum())
    add("accounts.customer_id exists in customers", (~accounts["customer_id"].isin(customers)).sum())
    add("balances key (account_id, month_end) unique", tables["balances"].duplicated(["account_id", "month_end"]).sum())
    add("balances.account_id exists in accounts", (~tables["balances"]["account_id"].isin(account_ids)).sum())
    for name in ("product_holdings", "assets", "digital_activity", "campaign_responses"):
        add(f"{name}.customer_id exists in customers", (~tables[name]["customer_id"].isin(customers)).sum())
    add("preference_changes.account_id exists in accounts", (~tables["preference_changes"]["account_id"].isin(account_ids)).sum())
    add("digital_activity key (customer_id, month) unique", tables["digital_activity"].duplicated(["customer_id", "month"]).sum())

    rows = dup_ids = bad_account = bad_amount = bad_indicator = mcc_off_card = 0
    first, last = None, None
    seen_prefixes: set[str] = set()
    for part in iter_transactions(warehouse, ["transaction_id", "account_id", "booking_datetime", "amount",
                                              "credit_debit_indicator", "merchant_category_code",
                                              "proprietary_bank_transaction_code"]):
        rows += len(part)
        dup_ids += part["transaction_id"].duplicated().sum()
        prefix = part["transaction_id"].str[:5].iloc[0]
        dup_ids += int(prefix in seen_prefixes) * len(part)
        seen_prefixes.add(prefix)
        bad_account += (~part["account_id"].isin(account_ids)).sum()
        bad_amount += (~np.isfinite(part["amount"]) | (part["amount"] <= 0)).sum()
        bad_indicator += (~part["credit_debit_indicator"].isin(["Credit", "Debit"])).sum()
        mcc_off_card += (part["merchant_category_code"].notna() & (part["proprietary_bank_transaction_code"] != "DEB")).sum()
        lo, hi = part["booking_datetime"].min(), part["booking_datetime"].max()
        first = lo if first is None else min(first, lo)
        last = hi if last is None else max(last, hi)
    add("transactions.transaction_id unique", dup_ids)
    add("transactions.account_id exists in accounts", bad_account)
    add("transactions.amount positive and finite", bad_amount)
    add("transactions.credit_debit_indicator valid", bad_indicator)
    add("merchant category code only on card payments", mcc_off_card)

    nulls = {}
    for name, frame in tables.items():
        rates = frame.isna().mean()
        nulls[name] = {k: round(float(v), 4) for k, v in rates[rates > 0].items()}
    return {
        "row_counts": {name: int(len(frame)) for name, frame in tables.items()} | {"transactions": int(rows)},
        "transaction_date_range": [str(first), str(last)],
        "checks": checks,
        "passed": int(sum(c["pass"] for c in checks)),
        "total": len(checks),
        "null_rates": nulls,
    }


def _open_at(frame: pd.DataFrame, start: str, end: str | None, as_of: np.datetime64) -> pd.Series:
    opened = pd.to_datetime(frame[start]) <= as_of
    if end and end in frame:
        closed = pd.to_datetime(frame[end])
        return opened & (closed.isna() | (closed > as_of))
    return opened


def build_base_view(tables: dict[str, pd.DataFrame], warehouse: Path, as_of: str, history_months: int = 12,
                    area: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """One row per customer joining every warehouse table, using only data dated on or before ``as_of``,
    plus labelled public context for the customer's area and values a lender derives (indexed property value)."""
    area = area if area is not None else load_area_context()
    t = np.datetime64(as_of)
    t_ts = pd.Timestamp(as_of)
    window_start = (t_ts - pd.DateOffset(months=history_months)) + pd.Timedelta(days=1)
    six_months = (t_ts - pd.DateOffset(months=6)) + pd.Timedelta(days=1)

    cust = tables["customers"].copy()
    base = pd.DataFrame({
        "customer_id": cust["customer_id"],
        "age": t_ts.year - cust["birth_year"],
        "la_code": cust["la_code"], "la_name": cust["la_name"], "region": cust["region"],
        "declared_residential_status": cust["declared_residential_status"],
        "declared_employment_status": cust["declared_employment_status"],
        "declared_income_band": cust["declared_income_band"],
        "customer_tenure_years": ((t - cust["customer_since"].values.astype("datetime64[D]")).astype(int) / 365.25).round(1),
        "marketing_consent": cust["marketing_consent"],
        "vulnerability_flag": cust["vulnerability_flag"],
        "preferred_contact_channel": cust["preferred_contact_channel"],
        "stated_sustainability_preference": cust["stated_sustainability_preference"] if "stated_sustainability_preference" in cust else np.nan,
    })
    for column in ("declared_occupation", "declared_dependants", "declared_years_at_address"):
        base[column] = cust[column] if column in cust else np.nan

    # Accounts open on the as-of date and statement preference on that date.
    acc = tables["accounts"].copy()
    acc = acc[_open_at(acc, "opened_date", "closed_date", t)]
    pivot = acc.pivot_table(index="customer_id", columns="account_type", values="account_id", aggfunc="count").fillna(0)
    for kind in ["current", "savings", "credit_card", "mortgage", "motor_loan", "personal_loan", "investment",
                 "mortgage_further_advance"]:
        base[f"n_{kind}_accounts"] = base["customer_id"].map(pivot[kind]).fillna(0).astype(int) if kind in pivot else 0
    base["n_products"] = base["customer_id"].map(acc.groupby("customer_id").size()).fillna(0).astype(int)
    changes = tables["preference_changes"]
    switched = set(changes.loc[pd.to_datetime(changes["change_date"]) <= t_ts, "account_id"])
    pref = acc[acc["statement_preference"].notna()].copy()
    pref["paper_at_as_of"] = (pref["statement_preference"] == "paper") & ~pref["account_id"].isin(switched)
    # Accounts switched after the as-of date were still on paper then.
    later = set(changes.loc[pd.to_datetime(changes["change_date"]) > t_ts, "account_id"])
    pref.loc[pref["account_id"].isin(later), "paper_at_as_of"] = True
    base["paper_accounts"] = base["customer_id"].map(pref.groupby("customer_id")["paper_at_as_of"].sum()).fillna(0).astype(int)
    mort = acc[acc["account_type"] == "mortgage"].set_index("customer_id")
    base["mortgage_original_amount"] = base["customer_id"].map(mort["original_amount"])
    base["mortgage_property_value_at_origination"] = base["customer_id"].map(mort["collateral_value_at_origination"])
    # A customer can hold two motor loans (an existing one plus EV finance taken in the window): keep the latest.
    motor = acc[acc["account_type"] == "motor_loan"].sort_values("opened_date").groupby("customer_id").tail(1).set_index("customer_id")
    motor_end = pd.to_datetime(motor["closed_date"])
    base["motor_loan_months_to_end"] = base["customer_id"].map(((motor_end - t_ts).dt.days / 30.44).round(1))
    base["motor_loan_product_code"] = base["customer_id"].map(motor["product_code"])
    pl = acc[acc["account_type"] == "personal_loan"].groupby("customer_id")["loan_purpose"].first()
    base["personal_loan_purpose"] = base["customer_id"].map(pl)

    # Balances: latest month-end on or before the as-of date; arrears and overdraft in the last six months.
    bal = tables["balances"]
    bal = bal[pd.to_datetime(bal["month_end"]) <= t_ts].copy()
    bal["customer_id"] = customer_of_account(bal["account_id"])
    bal["suffix"] = bal["account_id"].str.rsplit("-", n=1).str[1]
    latest = bal.sort_values("month_end").groupby("account_id").tail(1)
    for suffix, name in (("CA", "current"), ("SV", "savings"), ("CC", "credit_card"), ("MG", "mortgage"),
                         ("ML", "motor_loan"), ("PL", "personal_loan"), ("IV", "investment")):
        series = latest[latest["suffix"] == suffix].groupby("customer_id")["balance"].sum()
        base[f"balance_{name}"] = base["customer_id"].map(series).fillna(0.0)
    recent = bal[pd.to_datetime(bal["month_end"]) >= six_months]
    base["months_overdrawn_6m"] = base["customer_id"].map(
        recent[(recent["suffix"] == "CA") & (recent["balance"] < 0)].groupby("customer_id").size()).fillna(0).astype(int)
    base["max_arrears_months_6m"] = base["customer_id"].map(recent.groupby("customer_id")["arrears_months"].max()).fillna(0).astype(int)

    # Green products held on the as-of date.
    hold = tables["product_holdings"]
    held = hold[(hold["green_product"] != "") & (pd.to_datetime(hold["start_date"]) <= t_ts)
                & (pd.to_datetime(hold["end_date"]).isna() | (pd.to_datetime(hold["end_date"]) > t_ts))]
    for product in ["GP01", "GP02", "GP03", "GP04", "GP05", "GP06"]:
        base[f"holds_{product}"] = base["customer_id"].isin(held.loc[held["green_product"] == product, "customer_id"])

    # Assets the bank holds as lender.
    assets = tables["assets"]
    prop = assets[assets["asset_type"] == "property"].set_index("customer_id")
    for column in ("property_type", "property_detail", "construction_period", "bedrooms", "floor_area_m2", "epc_band",
                   "epc_main_heating", "solar_pv_on_record"):
        base[f"asset_{column}"] = base["customer_id"].map(prop[column]) if column in prop else np.nan

    # What a lender derives for homes it lends on: value indexed by the UK HPI, and the current loan-to-value.
    mort_start = base["customer_id"].map(pd.to_datetime(mort["opened_date"])) if len(mort) else pd.Series(pd.NaT, index=base.index)
    has_value = base["mortgage_property_value_at_origination"].notna() & mort_start.notna()
    base["mortgage_indexed_property_value"] = np.nan
    base["mortgage_current_ltv"] = np.nan
    if has_value.any():
        rows = base[has_value]
        then = _hpi_lookup(area["house_prices"], rows["la_code"], mort_start[has_value], rows["asset_property_type"])
        now = _hpi_lookup(area["house_prices"], rows["la_code"], pd.Series(t_ts, index=rows.index), rows["asset_property_type"])
        indexed = rows["mortgage_property_value_at_origination"].to_numpy() * now / then
        base.loc[has_value, "mortgage_indexed_property_value"] = np.round(indexed, -3)
        base.loc[has_value, "mortgage_current_ltv"] = (rows["balance_mortgage"].abs().to_numpy() / indexed).round(3) \
            if "balance_mortgage" in rows else np.nan
    veh = assets[assets["asset_type"] == "vehicle"].set_index("customer_id")
    base["asset_vehicle_fuel_type"] = base["customer_id"].map(veh["vehicle_fuel_type"]) if len(veh) else np.nan
    base["asset_vehicle_age_years"] = base["customer_id"].map(t_ts.year - veh["vehicle_first_registered_year"]) if len(veh) else np.nan

    # Digital engagement over the last six months.
    dig = tables["digital_activity"]
    dig = dig[(pd.to_datetime(dig["month"]) <= t_ts) & (pd.to_datetime(dig["month"]) >= six_months.replace(day=1))]
    grouped = dig.groupby("customer_id")
    base["logins_per_month_6m"] = base["customer_id"].map((grouped["app_logins"].sum() + grouped["web_logins"].sum()) / 6.0).fillna(0.0)
    base["app_login_share_6m"] = base["customer_id"].map(grouped["app_logins"].sum() / (grouped["app_logins"].sum() + grouped["web_logins"].sum()).replace(0, np.nan)).fillna(0.0)
    base["branch_visits_6m"] = base["customer_id"].map(grouped["branch_visits"].sum()).fillna(0).astype(int)
    base["estatement_views_6m"] = base["customer_id"].map(grouped["estatement_views"].sum()).fillna(0).astype(int)
    base["months_with_logins_6m"] = base["customer_id"].map(dig[(dig["app_logins"] + dig["web_logins"]) > 0].groupby("customer_id").size()).fillna(0).astype(int)

    # Campaign history before the as-of date (memory).
    camp = tables["campaign_responses"]
    camp = camp[pd.to_datetime(camp["sent_date"]) <= t_ts].sort_values("sent_date")
    last = camp.groupby(["customer_id", "product"]).tail(1)
    for product in ["GP01", "GP02", "GP03", "GP04", "GP05", "GP06"]:
        rows = last[last["product"] == product].set_index("customer_id")
        base[f"last_response_{product}"] = base["customer_id"].map(rows["response"]).fillna("none")
        base[f"last_response_date_{product}"] = base["customer_id"].map(rows["response_date"])

    # Transaction summary over the history window.
    summaries = []
    for part in iter_transactions(warehouse, ["account_id", "booking_datetime", "amount", "credit_debit_indicator"]):
        part = part[(part["booking_datetime"] <= t_ts + pd.Timedelta(hours=23, minutes=59)) & (part["booking_datetime"] >= window_start)]
        part["customer_id"] = customer_of_account(part["account_id"])
        part["month"] = part["booking_datetime"].dt.to_period("M")
        debit = part["credit_debit_indicator"] == "Debit"
        summary = part.groupby("customer_id").agg(n_transactions_12m=("amount", "size"), months_active_12m=("month", "nunique"))
        summary["debit_total_12m"] = part[debit].groupby("customer_id")["amount"].sum()
        summary["credit_total_12m"] = part[~debit].groupby("customer_id")["amount"].sum()
        summaries.append(summary)
    tx = pd.concat(summaries)
    for column in tx.columns:
        base[column] = base["customer_id"].map(tx[column]).fillna(0)
    # Area context: public statistics for the customer's authority, labelled as area-level.
    la = area["la"].set_index("la_code")
    context = {f"area_{column}": base["la_code"].map(la[column]) for column in AREA_COLUMNS}
    context = {k: (v.round(0) if k.endswith(("_kwh", "kwp_year")) else v.round(4)) for k, v in context.items()}
    base = pd.concat([base, pd.DataFrame(context, index=base.index)], axis=1)
    base["declared_years_at_address"] = base["declared_years_at_address"].round().astype("Int64")
    base["asset_solar_pv_on_record"] = base["asset_solar_pv_on_record"].astype("boolean")
    base["as_of"] = pd.Timestamp(as_of)
    return base[C360_COLUMNS]


def write_sqlite(path: Path, frames: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as con:
        for name, frame in frames.items():
            out = frame.copy()
            for column in out.columns:
                if pd.api.types.is_datetime64_any_dtype(out[column]):
                    out[column] = out[column].astype(str)
            out.to_sql(name, con, index=False, chunksize=5000)
        con.execute("CREATE INDEX IF NOT EXISTS ix_c360 ON customer_360(customer_id)")
