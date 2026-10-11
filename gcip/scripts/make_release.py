#!/usr/bin/env python3
"""Pack a finished build into the bundles that `run.sh` downloads, and write release/manifest.json (sizes, SHA-256).

    python scripts/make_release.py --data runs/full --run-dir runs/pipeline --out dist

Bundles (upload everything in dist/ to the shared Drive folder named in release/manifest.json):
  gcip_outputs.tar.gz    pipeline outputs: models, live scores, impact, recommendations, explanations, insights, metrics
  gcip_bank_core.tar.gz  what the pipeline and the dashboards read: Customer 360 and features (both snapshots), labels,
                         categorised transactions, product holdings, truth-vault labels and state, data card and reports
  gcip_bank_full.tar.gz  the rest of the synthetic bank: the 7 source-system extracts, the full warehouse, the truth
                         vault's transaction truth and the ingestion quarantine (for audit and re-running FR1)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ["customer_360_2026-03-31.parquet", "customer_360_2026-09-30.parquet", "features_2026-03-31.parquet",
        "features_2026-09-30.parquet", "labels.parquet", "categorised", "meta", "_done",
        "warehouse/product_holdings.parquet", "warehouse/campaign_responses.parquet", "truth_vault/labels.parquet",
        "truth_vault/state.parquet"]
FULL = ["sources", "warehouse", "truth_vault", "quarantine", "_checkpoints"]
SKIP_SUFFIXES = (".zip",)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pack(target: Path, members: list[tuple[Path, str]]) -> dict:
    with tarfile.open(target, "w:gz", compresslevel=1) as tar:
        for path, arcname in members:
            tar.add(path, arcname=arcname)
    return {"file": target.name, "bytes": target.stat().st_size, "sha256": sha256(target)}


def files_under(base: Path, rel: str) -> list[Path]:
    p = base / rel
    if p.is_file():
        return [p]
    return sorted(q for q in p.rglob("*") if q.is_file() and not q.name.endswith(SKIP_SUFFIXES))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="runs/full")
    parser.add_argument("--run-dir", default="runs/pipeline")
    parser.add_argument("--out", default="dist")
    args = parser.parse_args()
    data, run, out = (ROOT / args.data).resolve(), ROOT / args.run_dir, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    bundles = []

    outputs = [(p, f"pipeline/{p.relative_to(run)}") for p in sorted(run.rglob("*"))
               if p.is_file() and p.relative_to(run).parts[0] != "data"]
    bundles.append(pack(out / "gcip_outputs.tar.gz", outputs) | {"extract_to": "runs", "contents": "pipeline outputs",
                                                                 "required": True})

    core_files = [f for rel in CORE for f in files_under(data, rel)]
    core_set = set(core_files)
    bundles.append(pack(out / "gcip_bank_core.tar.gz", [(f, f"full/{f.relative_to(data)}") for f in core_files])
                   | {"extract_to": "runs", "contents": "model inputs, categorised transactions, reports",
                      "required": True})

    full_files = [f for rel in FULL for f in files_under(data, rel) if f not in core_set]
    bundles.append(pack(out / "gcip_bank_full.tar.gz", [(f, f"full/{f.relative_to(data)}") for f in full_files])
                   | {"extract_to": "runs", "contents": "source-system extracts, warehouse, truth vault, quarantine",
                      "required": False})

    manifest_path = ROOT / "release" / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {
        "drive_folder": old.get("drive_folder", ""),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "customers": json.loads((data / "meta" / "generation_manifest.json").read_text()).get("customers"),
        "bundles": bundles,
        "model": old.get("model", {"repo": "Qwen/Qwen2.5-1.5B-Instruct", "source": "Hugging Face"}),
    }
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    for b in bundles:
        print(f"{b['file']}: {b['bytes'] / 1e6:,.0f} MB")
    print(f"wrote {manifest_path.relative_to(ROOT)}; upload {out.relative_to(ROOT)}/*.tar.gz to the Drive folder")


if __name__ == "__main__":
    main()
