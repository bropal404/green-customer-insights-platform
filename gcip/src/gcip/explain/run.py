"""Write explanations for recommendations: template for all, guarded LLM rephrasing when enabled.

    python -m gcip.explain.run --run-dir RUN [--llm MODEL]              (one process)
    torchrun --nproc_per_node 4 -m gcip.explain.run --run-dir RUN --llm MODEL   (one shard per GPU)
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from gcip.explain.guards import check
from gcip.explain.templates import facts, template_parts


def build_items(recs: pd.DataFrame) -> list[dict]:
    items = []
    for rec in recs.to_dict("records"):
        f = facts(rec)
        body, footer = template_parts(f)
        items.append({"customer_id": rec["customer_id"], "product": rec["product"], "facts": f, "draft": f"{body} {footer}",
                      "body": body, "footer": footer})
    return items


def compose(item: dict, rewritten_body: str | None) -> str | None:
    """The LLM rewrites only the body; the fixed compliance sentence is appended unchanged."""
    return f"{rewritten_body.strip()} {item['footer']}" if rewritten_body else None


def reading_ease(text: str) -> float:
    """Flesch reading ease with a simple syllable count (higher = easier)."""
    words = re.findall(r"[A-Za-z]+", text)
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    syllables = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower())) - (1 if w.lower().endswith("e") and len(w) > 3 else 0)) for w in words)
    n = max(1, len(words))
    return round(206.835 - 1.015 * n / sentences - 84.6 * syllables / n, 1)


def explain(items: list[dict], llm: str | None = None, batch_size: int = 8, max_new_tokens: int = 220, dtype: str = "float16",
            generated: list[str | None] | None = None) -> pd.DataFrame:
    """Template for every item; an LLM rephrasing replaces it only when it passes every guard. `generated` takes
    rephrasings made elsewhere (the API explainer); `llm` runs a local Hugging Face model instead."""
    rows = []
    if llm and generated is None:
        from gcip.explain.llm import generate
        generated = generate(items, llm, batch_size, max_new_tokens, dtype)
    for k, item in enumerate(items):
        text, source, problems = item["draft"], "template", []
        candidate = compose(item, generated[k]) if generated is not None else None
        if generated is not None:
            problems = check(candidate, item["facts"], draft=item["draft"]) if generated[k] else ["no reply from the LLM"]
            if not problems:
                text, source = candidate, "llm"
        rows.append({"customer_id": item["customer_id"], "product": item["product"], "explanation": text, "source": source,
                     "template": item["draft"], "llm_text": candidate,
                     "guard_problems": "; ".join(problems), "template_problems": "; ".join(check(item["draft"], item["facts"])),
                     "words": len(text.split()), "reading_ease": reading_ease(text)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--llm", default=None)
    parser.add_argument("--limit", type=int, default=None, help="explain only the first N recommendations (evaluation)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    recs = pd.read_parquet(run_dir / "recommendations" / "recommendations.parquet")
    if args.limit:
        recs = recs.head(args.limit)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    items = build_items(recs)[rank::world]
    out = explain(items, args.llm, args.batch_size, args.max_new_tokens, args.dtype)
    (run_dir / "explanations").mkdir(parents=True, exist_ok=True)
    out.to_parquet(run_dir / "explanations" / f"part-{rank:02d}.parquet", index=False)
    print(f"rank {rank}: {len(out)} explanations, LLM used for {int((out['source'] == 'llm').sum())}")


if __name__ == "__main__":
    main()
