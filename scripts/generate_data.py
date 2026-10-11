#!/usr/bin/env python3
"""Build the synthetic bank: source-system extracts + truth vault, then FR1 ingestion,
validation against public statistics and the data card.

    python scripts/generate_data.py --out runs/data --n 30000 --workers 32
Re-running resumes: finished generation stages and transaction chunks are skipped;
ingestion, validation and the data card are rebuilt from the extracts.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gcip.fr1.ingest import ingest  # noqa: E402
from gcip.simulate.datacard import write_data_card  # noqa: E402
from gcip.simulate.generate import generate  # noqa: E402
from gcip.simulate.validate import validate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "runs" / "data"))
    parser.add_argument("--n", type=int, default=None, help="number of customers (default: configs/simulation.yaml)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--sources-only", action="store_true", help="stop after writing the source-system extracts")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    manifest = generate(args.out, n_customers=args.n, workers=args.workers, chunk_size=args.chunk_size)
    print(f"generated in {manifest['elapsed_seconds']}s -> {args.out}")
    if args.sources_only:
        return
    report = ingest(args.out, workers=args.workers)
    print("\n".join(report["summary"]))
    validation = validate(args.out)
    print(f"validation: {validation['passed']} of {validation['total']} checks within tolerance")
    print(f"data card: {write_data_card(args.out)}")


if __name__ == "__main__":
    main()
