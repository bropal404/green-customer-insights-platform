"""Customer memory: responses are stored as events and change what is shown next.

The memory store is an event log (customer_id, product, response, response_date, source). Applying it to a Customer
360 snapshot overwrites the last response and its date per product, which is exactly what the recommendation engine
reads (cool-downs for 'dismissed' and 'not_interested', suppression for 'already_have', a priority boost for 'clicked').

`replay` shows memory working over several rounds: each round, customers respond to what they were shown (responses
drawn from the simulator's true take-up probability, so evaluation only), the events are stored, the clock moves on a
month, and recommendations are made again from the updated memory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gcip.recommend.engine import recommend

RESPONSES = ["clicked", "dismissed", "not_interested", "none"]
EVENT_COLUMNS = ["customer_id", "product", "response", "response_date", "source"]


def empty_events() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c == "response_date" else object) for c in EVENT_COLUMNS})


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
           thresholds: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Run `rounds` rounds for a sample of customers who get at least one recommendation; return metrics and the event log."""
    rng = np.random.default_rng(seed)
    first, _ = recommend(scores, impact, c360, cfg, thresholds)
    pool = first["customer_id"].unique()
    chosen = set(rng.choice(pool, size=min(customers, len(pool)), replace=False))
    s, i, c = (scores[scores["customer_id"].isin(chosen)], impact[impact["customer_id"].isin(chosen)],
               c360[c360["customer_id"].isin(chosen)].copy())
    p_true = truth.set_index(["customer_id", "product"])["p_true"]
    events = empty_events()
    as_of = pd.Timestamp(c360["as_of"].iloc[0])
    history, previous = [], None
    for r in range(1, rounds + 1):
        when = as_of + pd.Timedelta(days=days_per_round * (r - 1))
        view = apply_events(c, events).assign(as_of=when)
        shown, decisions = recommend(s, i, view, cfg, thresholds)
        shown = shown[shown["status"] == "recommended"]
        pairs = set(zip(shown["customer_id"], shown["product"]))
        stats = {"round": r, "date": str(when.date()), "customers": len(chosen), "shown": len(shown),
                 "customers_shown_something": int(shown["customer_id"].nunique()),
                 "suppressed_by_memory": int(decisions["decision"].str.startswith("memory").sum())}
        if previous is not None:
            cooled = {(a, b) for a, b, resp in previous if resp in ("dismissed", "not_interested")}
            clicked = {(a, b) for a, b, resp in previous if resp == "clicked"}
            stats["reshown_within_cooldown"] = len(pairs & cooled)
            stats["clicked_reshown"] = len(pairs & clicked)
            stats["clicked_previous_round"] = len(clicked)
            stats["new_products_shown"] = len(pairs - {(a, b) for a, b, _ in previous})
        responses = _respond(shown, p_true, rng)
        stats["responses"] = {k: int(v) for k, v in pd.Series(responses).value_counts().items()}
        new = pd.DataFrame({"customer_id": shown["customer_id"].to_numpy(), "product": shown["product"].to_numpy(),
                            "response": responses, "response_date": when, "source": f"replay round {r}"})
        events = pd.concat([events, new[new["response"] != "none"]], ignore_index=True)
        previous = list(zip(shown["customer_id"], shown["product"], responses))
        history.append(stats)
    return {"customers": len(chosen), "rounds": history, "cooldown_days": cfg["cooldown_days"]}, events
