#!/usr/bin/env python3
"""Copy what the dashboards read from a finished run into demo/ (committed, about 25 MB), so `./run.sh` can serve the
30,000-customer results in a couple of minutes without downloading or rebuilding anything.

    python scripts/make_demo.py --data runs/full --run-dir runs/pipeline --out demo
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = "2026-09-30"
RUN_FILES = ["scores/live_scores.parquet", "impact/impact.parquet", "recommendations/recommendations.parquet",
             "recommendations/decisions.parquet", "explanations/explanations.parquet", "explanations/eval_set.parquet",
             "insights/bank_insights.json", "insights/monthly_trends.parquet", "insights/customer_monthly_spend.parquet",
             "metrics/propensity_report.json", "metrics/explanations.json", "metrics/impact_truth_recovery.json",
             "metrics/impact_sensitivity.csv", "metrics/memory_replay.json", "models/thresholds.json",
             "metrics/explanations_gemini_sample.json", "memory/replay_events.parquet", "logs/stage_timings.json",
             "summary.json"]
DATA_FILES = [f"customer_360_{LIVE}.parquet", f"features_{LIVE}.parquet", "meta/data_validation.json",
              "meta/ingestion_report.json", "meta/generation_manifest.json"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="runs/full")
    parser.add_argument("--run-dir", default="runs/pipeline")
    parser.add_argument("--out", default="demo")
    args = parser.parse_args()
    data, run, out = ROOT / args.data, ROOT / args.run_dir, ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    copies = [(run / f, out / "pipeline" / f) for f in RUN_FILES]
    copies += [(p, out / "pipeline" / "_checkpoints" / p.name) for p in sorted((run / "_checkpoints").glob("*.done"))]
    copies += [(data / f, out / "data" / f) for f in DATA_FILES]
    for src, dst in copies:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out.relative_to(ROOT)}/: {len(copies)} files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
