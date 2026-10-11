"""Project paths and YAML config loading."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = Path(os.environ.get("GCIP_DATA_DIR", ROOT / "data"))
REFERENCE_DIR = DATA_DIR / "reference"
LEXICON_DIR = DATA_DIR / "lexicon"
BENCHMARK_DIR = DATA_DIR / "benchmarks"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml."""
    return load_yaml(CONFIG_DIR / f"{name}.yaml")


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``base`` with nested ``overrides`` applied."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged
