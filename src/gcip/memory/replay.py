"""Customer memory over time: responses are stored as events and change what is shown next.

The memory store is an event log (customer_id, product, response, response_date, source): the CRM's campaign history
plus the responses given in the apps. gcip.memory.signals turns it into the weight, streak and state the recommendation
engine uses. `apply_events` is the older, simpler view (each product's last response written into the Customer 360).

`replay` shows memory building up over several monthly rounds: customers respond to what they were shown (responses
drawn from the simulator's true take-up probability, so evaluation only), every response, including ignoring the
suggestion, is stored, the clock moves on a month, and suggestions are made again from the updated memory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gcip.recommend.engine import recommend

from gcip.memory.signals import EVENT_COLUMNS, empty_events, signals  # noqa: F401  (re-exported)

RESPONSES = ["clicked", "dismissed", "not_interested", "no_response"]


def apply_events(c360: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Customer 360 with each product's last response replaced by the latest stored event, if newer."""
    if events.empty:
        return c360
    out = c360.copy()
    latest = events.sort_values("response_date").groupby(["customer_id", "product"]).tail(1)
    for product, rows in latest.groupby("product"):
        rows = rows.set_index("customer_id")
        col, date_col = f"last_response_{product}", f"last_response_date_{product}"
        out[date_col] = pd.to_datetime(out[date_col])
        new_date = pd.to_datetime(out["customer_id"].map(rows["response_date"]))
        newer = new_date.notna() & (out[date_col].isna() | (new_date > out[date_col]))
        out.loc[newer, col] = out.loc[newer, "customer_id"].map(rows["response"])
        out.loc[newer, date_col] = new_date[newer]
    return out


def _respond(shown: pd.DataFrame, p_true: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Simulated response to a shown product: more likely to click when the true take-up probability is high."""
    p = shown.set_index(["customer_id", "product"]).index.map(p_true).to_numpy(dtype=float)
    p = np.nan_to_num(p, nan=0.02)
    click = np.clip(0.05 + 2.5 * p, 0, 0.8)
    rest = 1 - click
    probs = np.stack([click, rest * 0.35, rest * 0.20, rest * 0.45], axis=1)
    u = rng.random(len(p))[:, None]
    return np.array(RESPONSES, dtype=object)[(u > probs.cumsum(axis=1)).sum(axis=1)]


def replay(scores: pd.DataFrame, impact: pd.DataFrame, c360: pd.DataFrame, cfg: dict, truth: pd.DataFrame,
           customers: int = 300, rounds: int = 3, days_per_round: int = 30, seed: int = 11,
           thresholds: dict | None = None, events: pd.DataFrame | None = None, memory_cfg: dict | None = None
           ) -> tuple[dict, pd.DataFrame]:
    """Run `rounds` rounds for a sample of customers who get at least one suggestion; return metrics and the event log."""
    from gcip.config import load_config

    memory_cfg = memory_cfg or load_config("pipeline")["memory"]
    events = empty_events() if events is None else events
    rng = np.random.default_rng(seed)
    as_of = pd.Timestamp(c360["as_of"].iloc[0])

    def memory_at(when, ids, log):
        return signals(log[log["customer_id"].isin(ids)], when, memory_cfg, cfg["cooldown_days"])

    first, _ = recommend(scores, impact, c360, cfg, thresholds, memory=memory_at(as_of, set(c360["customer_id"]), events))
    pool = first["customer_id"].unique()
    chosen = set(rng.choice(pool, size=min(customers, len(pool)), replace=False))
    s, i, c = (scores[scores["customer_id"].isin(chosen)], impact[impact["customer_id"].isin(chosen)],
               c360[c360["customer_id"].isin(chosen)].copy())
    p_true = truth.set_index(["customer_id", "product"])["p_true"]
    log = events[events["customer_id"].isin(chosen)]
    new_events = empty_events()
    history, previous = [], None
    for r in range(1, rounds + 1):
        when = as_of + pd.Timedelta(days=days_per_round * (r - 1))
        shown, decisions = recommend(s, i, c.assign(as_of=when), cfg, thresholds, memory=memory_at(when, chosen, log))
        shown = shown[shown["status"] == "recommended"]
        pairs = set(zip(shown["customer_id"], shown["product"]))
        memory_decisions = decisions["decision"].str.startswith("memory")
        stats = {"round": r, "date": str(when.date()), "customers": len(chosen), "shown": len(shown),
                 "customers_shown_something": int(shown["customer_id"].nunique()),
                 "suppressed_by_memory": int(memory_decisions.sum()),
                 "paused_no_interest": int(decisions["decision"].str.contains("no interest").sum())}
        if previous is not None:
            cooled = {(a, b) for a, b, resp in previous if resp in ("dismissed", "not_interested")}
            stats["reshown_within_cooldown"] = len(pairs & cooled)
            stats["new_products_shown"] = len(pairs - {(a, b) for a, b, _ in previous})
        responses = _respond(shown, p_true, rng)
        stats["responses"] = {k: int(v) for k, v in pd.Series(responses).value_counts().items()}
        batch = pd.DataFrame({"customer_id": shown["customer_id"].to_numpy(), "product": shown["product"].to_numpy(),
                              "response": responses, "response_date": when, "source": f"replay round {r}"})
        log = pd.concat([log, batch], ignore_index=True)
        new_events = pd.concat([new_events, batch], ignore_index=True)
        previous = list(zip(shown["customer_id"], shown["product"], responses))
        history.append(stats)
    return {"customers": len(chosen), "rounds": history, "cooldown_days": cfg["cooldown_days"],
            "fatigue_contacts": memory_cfg.get("fatigue_contacts")}, new_events
