"""Run context shared by every stage: paths, config, logging, checkpoints and timings."""

from __future__ import annotations

import contextlib
import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from gcip.config import deep_update, load_config
from gcip.io.files import write_json


@dataclass
class RunContext:
    run_dir: Path
    cfg: dict[str, Any]
    log: logging.Logger
    timings: dict[str, float] = field(default_factory=dict)

    # ---------------------------------------------------------------- paths
    @property
    def data_dir(self) -> Path:
        return self.run_dir / "data"

    @property
    def warehouse(self) -> Path:
        return self.data_dir / "warehouse"

    @property
    def truth(self) -> Path:
        return self.data_dir / "truth_vault"

    def path(self, *parts: str) -> Path:
        target = self.run_dir.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    # ---------------------------------------------------------------- checkpoints
    def done(self, stage: str) -> bool:
        return (self.run_dir / "_checkpoints" / f"{stage}.done").exists()

    def mark(self, stage: str, info: dict | None = None) -> None:
        write_json(self.run_dir / "_checkpoints" / f"{stage}.done",
                   {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), **(info or {})})

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.time()
        self.log.info("=" * 70)
        self.log.info("STAGE %s: start", name)
        yield
        elapsed = time.time() - start
        self.timings[name] = round(elapsed, 1)
        self.log.info("STAGE %s: done in %.1f s", name, elapsed)
        write_json(self.run_dir / "logs" / "stage_timings.json", self.timings)

    # ---------------------------------------------------------------- io helpers
    def save_json(self, payload: Any, *parts: str) -> Path:
        target = self.path(*parts)
        write_json(target, payload)
        return target

    def save_parquet(self, frame: pd.DataFrame, *parts: str) -> Path:
        target = self.path(*parts)
        tmp = target.with_suffix(".tmp.parquet")
        frame.to_parquet(tmp, index=False, compression="zstd")
        tmp.replace(target)
        return target

    def read_parquet(self, *parts: str, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(self.run_dir.joinpath(*parts), columns=columns)


def make_context(run_dir: str | Path, overrides: dict | None = None) -> RunContext:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    cfg = load_config("pipeline")
    if overrides:
        cfg = deep_update(cfg, overrides)
    log = logging.getLogger("gcip")
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(fmt)
        log.addHandler(stream)
        file_handler = logging.FileHandler(run_dir / "logs" / "pipeline.log")
        file_handler.setFormatter(fmt)
        log.addHandler(file_handler)
    ctx = RunContext(run_dir=run_dir, cfg=cfg, log=log)
    existing = run_dir / "logs" / "stage_timings.json"
    if existing.exists():
        ctx.timings.update(json.loads(existing.read_text()))
    ctx.save_json({"python": sys.version, "platform": platform.platform(), "config": cfg}, "logs", "run_environment.json")
    return ctx
