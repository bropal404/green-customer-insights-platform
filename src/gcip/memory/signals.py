"""Customer memory as a dynamic signal: every contact and response, not only the last one.

The memory store is one event log (customer_id, product, response, response_date, source) built from the CRM's campaign
history and the responses given in the apps. `signals` turns it into, per customer x product:

  weight   product of a factor per event in the last `lookback_days` (interest raises it, being ignored, dismissed or
           turned down lowers it), clipped to `weight_bounds`; the suggestion's priority is propensity x confidence x
           weight, so memory changes the order of suggestions but never the model's propensity (brief: memory must not
           override the models, the eligibility rules or the carbon engine)
  streak   contacts in a row with no engagement (no response, or seen without acting)
  state    what memory does today: stopped (already has it), paused after "dismissed" or "not interested" (cool-downs),
           or paused after `fatigue_contacts` contacts in a row with no engagement, however strong the propensity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["customer_id", "product", "response", "response_date", "source"]
IGNORED = {"no_response", "viewed"}
PRODUCTS = ["GP01", "GP02", "GP03", "GP04", "GP05", "GP06"]


def empty_events() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c == "response_date" else object) for c in EVENT_COLUMNS})


def events_from_crm(campaigns: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """CRM campaign history as memory events (a contact with no response is dated by when it was sent)."""
    c = campaigns[pd.to_datetime(campaigns["sent_date"]) <= as_of]
    date = pd.to_datetime(c["response_date"]).fillna(pd.to_datetime(c["sent_date"]))
    return pd.DataFrame({"customer_id": c["customer_id"].to_numpy(), "product": c["product"].to_numpy(),
                         "response": c["response"].to_numpy(), "response_date": date.to_numpy(), "source": "CRM campaign"})


def events_from_c360(c360: pd.DataFrame) -> pd.DataFrame:
    """Fallback when no campaign log is at hand: the Customer 360's last response per product, as single events."""
    rows = []
    for p in PRODUCTS:
        col, date_col = f"last_response_{p}", f"last_response_date_{p}"
        if col not in c360:
            continue
        r = c360[["customer_id", col, date_col]].rename(columns={col: "response", date_col: "response_date"})
        r = r[r["response"].fillna("none") != "none"]
        rows.append(r.assign(product=p, source="Customer 360"))
    if not rows:
        return empty_events()
    out = pd.concat(rows, ignore_index=True)
    out["response_date"] = pd.to_datetime(out["response_date"])
    return out[EVENT_COLUMNS]


def signals(events: pd.DataFrame, as_of: pd.Timestamp, cfg: dict, cooldown_days: dict) -> pd.DataFrame:
    """Per customer x product: contacts, weight, streak, last response, and the state memory puts it in today."""
    cols = ["customer_id", "product", "contacts", "weight", "streak", "last_response", "last_date", "last_source",
            "last_clicked", "state", "until"]
    if events is None or events.empty:
        return pd.DataFrame(columns=cols).set_index(["customer_id", "product"])
    as_of = pd.Timestamp(as_of)
    e = events.copy()
    e["response_date"] = pd.to_datetime(e["response_date"])
    e = e[e["response_date"].notna() & (e["response_date"] <= as_of)]
    recent = e[e["response_date"] > as_of - pd.Timedelta(days=cfg.get("lookback_days", 365))]
    key = ["customer_id", "product"]

    log_w = np.log(recent["response"].map(cfg.get("weights", {})).fillna(1.0).clip(lower=1e-6))
    weight = np.exp(log_w.groupby([recent["customer_id"], recent["product"]]).sum())
    lo, hi = cfg.get("weight_bounds", [0.05, 2.0])
    out = pd.DataFrame({"contacts": recent.groupby(key).size(), "weight": weight.clip(lo, hi)})

    last = e.sort_values("response_date").groupby(key).tail(1).set_index(key)
    out = out.join(last[["response", "response_date", "source"]].rename(
        columns={"response": "last_response", "response_date": "last_date", "source": "last_source"}), how="outer")
    out["contacts"] = out["contacts"].fillna(0).astype(int)
    out["weight"] = out["weight"].fillna(1.0)

    out["last_clicked"] = recent[recent["response"] == "clicked"].groupby(key)["response_date"].max().reindex(out.index)

    # Streak: ignored contacts after the last time the customer engaged in any way.
    engaged = recent[~recent["response"].isin(IGNORED)].groupby(key)["response_date"].max().rename("engaged")
    ign = recent[recent["response"].isin(IGNORED)].join(engaged, on=key)
    ign = ign[ign["engaged"].isna() | (ign["response_date"] > ign["engaged"])]
    out["streak"] = ign.groupby(key).size().reindex(out.index).fillna(0).astype(int)

    ever_have = e[e["response"] == "already_have"].groupby(key).size()
    state = pd.Series("", index=out.index, dtype=object)
    until = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    for kind, days in cooldown_days.items():
        end = out["last_date"] + pd.Timedelta(days=days)
        hit = (out["last_response"] == kind) & (end > as_of) & (state == "")
        state[hit], until[hit] = f"{kind.replace('_', ' ')} {int(days)}-day cool-down", end[hit]
    n, pause = cfg.get("fatigue_contacts", 3), cfg.get("fatigue_pause_days", 180)
    end = out["last_date"] + pd.Timedelta(days=pause)
    tired = (out["streak"] >= n) & (end > as_of) & (state == "")
    state[tired], until[tired] = f"no interest after {n} contacts", end[tired]
    have = out.index.isin(ever_have.index)
    state[have], until[have] = "already has it", pd.NaT
    out["state"], out["until"] = state, until
    return out[cols[2:]]
