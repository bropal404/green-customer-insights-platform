#!/usr/bin/env python3
"""Download the prebuilt data and pipeline outputs (release/manifest.json) and the local explanation model.

    python scripts/fetch_release.py            # data bundles from the Drive folder, checked by SHA-256, into runs/
    python scripts/fetch_release.py --all      # also the full bank: 7 source-system extracts, warehouse, truth vault
    python scripts/fetch_release.py --model    # also the local LLM from Hugging Face (about 3 GB)
    python scripts/fetch_release.py --model-only

The Drive folder must be shared as "anyone with the link can view". Downloads land in downloads/ and are verified
against the checksums committed in release/manifest.json before anything is extracted; archives are extracted with
path checks (no absolute paths, no "..", no links outside runs/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release" / "manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_bundles(manifest: dict, downloads: Path, everything: bool) -> list[Path]:
    import gdown

    downloads.mkdir(exist_ok=True)
    wanted = {b["file"]: b for b in manifest["bundles"] if everything or b.get("required", True)}
    missing = [name for name in wanted
               if not (downloads / name).exists() or sha256(downloads / name) != wanted[name]["sha256"]]
    if missing:
        print(f"listing {manifest['drive_folder']}")
        listing = gdown.download_folder(manifest["drive_folder"], skip_download=True, quiet=True) or []
        ids = {Path(f.path).name: f.id for f in listing}
        for name in missing:
            if name not in ids:
                raise SystemExit(f"{name} is not in the Drive folder (is it shared as 'anyone with the link can view'?)")
            gdown.download(id=ids[name], output=str(downloads / name), quiet=False)
    paths = []
    for name, spec in wanted.items():
        path = downloads / name
        if not path.exists() or sha256(path) != spec["sha256"]:
            raise SystemExit(f"{name}: missing or checksum does not match release/manifest.json")
        paths.append(path)
    return paths


def extract(path: Path, target: Path) -> None:
    target.mkdir(exist_ok=True)
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts or not (member.isfile() or member.isdir()):
                raise SystemExit(f"{path.name}: unsafe member {member.name!r}")
        tar.extractall(target, filter="data")
    print(f"extracted {path.name} into {target.relative_to(ROOT)}/")


def download_model(manifest: dict) -> None:
    from huggingface_hub import snapshot_download

    repo = manifest["model"]["repo"]
    print(f"downloading {repo} from Hugging Face")
    print(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="store_true", help="also download the local LLM")
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--all", action="store_true", help="also the full bank (source extracts, warehouse, truth vault)")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    if not args.model_only:
        if not manifest.get("bundles"):
            raise SystemExit("release/manifest.json lists no bundles yet (run scripts/make_release.py and upload dist/)")
        for path in download_bundles(manifest, ROOT / "downloads", args.all):
            extract(path, ROOT / "runs")
        link = ROOT / "runs" / "pipeline" / "data"
        if not link.exists():
            link.symlink_to(Path("..") / "full")
    if args.model or args.model_only:
        download_model(manifest)


if __name__ == "__main__":
    sys.exit(main())
