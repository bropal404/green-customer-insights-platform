"""Shared fixtures: a small synthetic bank built once per test session."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def tiny_bank(tmp_path_factory):
    """300 customers: source extracts, truth vault and the ingested warehouse."""
    from gcip.fr1.ingest import ingest
    from gcip.simulate.generate import generate

    out = tmp_path_factory.mktemp("bank")
    generate(out, n_customers=300, workers=1, chunk_size=150)
    report = ingest(out, workers=1)
    return out, report
