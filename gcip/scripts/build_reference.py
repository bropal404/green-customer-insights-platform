#!/usr/bin/env python3
"""Download and process the public UK reference data into data/reference/.

Run once on a machine with internet (laptop or Ada login node). The outputs are
small and are shipped with the code; raw downloads stay in raw_cache/.

    python scripts/build_reference.py [--offline] [--raw-dir raw_cache] [--out-dir data/reference]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gcip.reference.build import build_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=str(ROOT / "raw_cache"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "reference"))
    parser.add_argument("--offline", action="store_true", help="use only files already in the raw cache")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    manifest = build_all(Path(args.raw_dir), Path(args.out_dir), offline=args.offline)
    print(f"built {len(manifest['processed_files'])} files; join stats: {manifest['join_stats']}")
    for item in manifest["processed_files"]:
        print(f"  {item['file']:<52} rows={item['rows']:>7}  cols={item['columns']:>3}")


if __name__ == "__main__":
    main()
