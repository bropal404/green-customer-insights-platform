"""FR1 ingestion: exact recovery of the bank's records from seven messy source systems."""

import shutil

import pandas as pd
import pytest

from gcip.fr1.customer360 import data_quality, load_tables
from gcip.fr1.ingest import IngestionError, audit_tables, ingest, normalise_account_number
from gcip.io.fingerprint import TRANSACTION_AUDIT_COLUMNS, row_hashes
from gcip.simulate.sources import ACCOUNT_SUFFIXES, account_number, contact_id


def test_audit_passes(tiny_bank):
    _, report = tiny_bank
    assert report["audit"]["pass"], report["summary"]
    t = report["audit"]["transactions"]
    assert t["expected"] == t["recovered"] > 0


def test_injected_issues_are_handled(tiny_bank):
    _, report = tiny_bank
    assert report["transactions"]["card_replays_removed"] > 0
    assert report["transactions"]["legacy_account_numbers_normalised"] > 0
    assert report["quarantine"]["digital_activity"]["uid matches no customer"] > 0
    assert report["quarantine"]["campaign_responses"]["prospect: no customer reference"] > 0
    assert report["files"]["core_banking/customers.csv"]["issues"]["region names re-cased to the ONS spelling"] > 0


def test_warehouse_passes_customer360_checks(tiny_bank):
    out, _ = tiny_bank
    dq = data_quality(load_tables(out / "warehouse"), out / "warehouse")
    assert dq["passed"] == dq["total"], [c for c in dq["checks"] if not c["pass"]]


def test_tampered_extract_stops_the_load(tiny_bank, tmp_path):
    out, _ = tiny_bank
    shutil.copytree(out / "sources", tmp_path / "sources")
    path = tmp_path / "sources" / "core_banking" / "customers.csv"
    text = path.read_text()
    assert ",Y,N," in text
    path.write_text(text.replace(",Y,N,", ",N,N,", 1))  # same size: only the checksum can catch it
    with pytest.raises(IngestionError, match="SHA-256"):
        ingest(tmp_path, workers=1)


def test_audit_detects_a_one_penny_change(tiny_bank):
    out, _ = tiny_bank
    balances = pd.read_parquet(out / "warehouse" / "balances.parquet")
    balances.loc[0, "balance"] += 0.01
    result = audit_tables({"balances": balances}, out / "truth_vault")["balances"]
    assert not result["exact_match"]
    assert result["columns_differing"] == ["balance"]


def test_transaction_fingerprint_detects_a_changed_narrative(tiny_bank):
    out, _ = tiny_bank
    part = sorted((out / "warehouse" / "transactions").glob("part-*.parquet"))[0]
    tx = pd.read_parquet(part)
    truth = pd.read_parquet(out / "truth_vault" / "transaction_truth" / part.name, columns=["transaction_id", "audit_fingerprint"])
    tx.loc[0, "transaction_information"] += " X"
    merged = tx.assign(fp=row_hashes(tx, TRANSACTION_AUDIT_COLUMNS)).merge(truth, on="transaction_id")
    assert len(merged) == len(tx)
    assert int((merged["fp"] != merged["audit_fingerprint"]).sum()) == 1


def test_identifiers_never_collide():
    customers = [f"C{100000 + i}" for i in range(200_000)]
    numbers = {account_number(f"{c}-{s}") for c in customers[:20_000] for s in ACCOUNT_SUFFIXES}
    assert len(numbers) == 20_000 * len(ACCOUNT_SUFFIXES)
    assert len({contact_id(c) for c in customers}) == len(customers)
    assert len({contact_id(c, secondary=True) for c in customers}) == len(customers)


def test_normalise_account_number():
    out = normalise_account_number(pd.Series(["001004-12345678", "00100412345678", " 001004 12345678 ", "12345"]))
    assert out.tolist()[:3] == ["001004-12345678"] * 3
    assert pd.isna(out.iloc[3])
