#!/usr/bin/env python3
"""Build the four model inputs for a built data directory (after scripts/generate_data.py):

    customer_360_<as_of>.parquet   FR1 Customer 360 for each snapshot
    categorised/part-*.parquet     FR2 categorised transactions
    features_<as_of>.parquet       point-in-time features (keys + features)
    labels.parquet                 take-up labels from the bank's product system, with each customer's split

    python scripts/build_model_inputs.py --data runs/full [--embeddings] [--finbert]
Each step writes <data>/_done/<step>.done and is skipped on a rerun.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from gcip.categorise.stage import run as run_categoriser  # noqa: E402
from gcip.features.store import build_features, feature_file  # noqa: E402
from gcip.fr1.customer360 import build_base_view, load_tables  # noqa: E402
from gcip.fr1.labels import takeup_labels  # noqa: E402
from gcip.pipeline.context import make_context  # noqa: E402

SNAPSHOTS = ("2026-03-31", "2026-09-30")
log = logging.getLogger("gcip.inputs")


def done(data: Path, step: str) -> bool:
    return (data / "_done" / f"{step}.done").exists()


def mark(data: Path, step: str, seconds: float) -> None:
    (data / "_done").mkdir(exist_ok=True)
    (data / "_done" / f"{step}.done").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {seconds:.0f}s\n")
    log.info("%s done in %.0fs", step, seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--embeddings", action="store_true", help="add MiniLM sentence embeddings to the categoriser (slower)")
    parser.add_argument("--finbert", action="store_true", help="add FinBERT embeddings to the categoriser (needs requirements-llm.txt; a GPU)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    data = Path(args.data).resolve()
    warehouse = data / "warehouse"

    if not done(data, "customer_360"):
        t0 = time.time()
        tables = load_tables(warehouse)
        for as_of in SNAPSHOTS:
            build_base_view(tables, warehouse, as_of).to_parquet(data / f"customer_360_{as_of}.parquet", index=False)
        mark(data, "customer_360", time.time() - t0)

    if not done(data, "categorised"):
        t0 = time.time()
        run_dir = data.parent / f"{data.name}_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        link = run_dir / "data"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(data)
        overrides = {"categoriser": {"use_embeddings": bool(args.embeddings), "use_finbert": bool(args.finbert),
                                     "transformer": {"enabled": False}}}
        run_categoriser(make_context(run_dir, overrides))
        mark(data, "categorised", time.time() - t0)

    if not done(data, "features"):
        t0 = time.time()
        for as_of in SNAPSHOTS:
            base = pd.read_parquet(data / f"customer_360_{as_of}.parquet")
            feature_file(build_features(data / "categorised", base, as_of)).to_parquet(data / f"features_{as_of}.parquet", index=False)
        mark(data, "features", time.time() - t0)

    if not done(data, "labels"):
        t0 = time.time()
        labels = takeup_labels(pd.read_parquet(warehouse / "product_holdings.parquet"),
                               pd.read_parquet(warehouse / "customers.parquet", columns=["customer_id"]), "2026-04-01", "2026-09-30")
        labels.to_parquet(data / "labels.parquet", index=False)
        mark(data, "labels", time.time() - t0)
    log.info("model inputs ready in %s", data)


if __name__ == "__main__":
    main()
