"""FR5 customer-friendly explanations from the structured recommendation only (never from raw data).

`facts` turns one recommendation record into the only facts an explanation may use. `template_text` writes the
deterministic explanation every recommendation gets; an LLM may rephrase it (gcip.explain.llm), and the guards in
gcip.explain.guards check that the rephrasing adds no number, promise or advice that the facts do not support.
"""

from __future__ import annotations

import math

PRODUCT_DISCLAIMER = {"GP05": "Capital at risk. This is information, not advice."}


def _num(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value


def _round_kg(kg: float) -> str:
    if kg >= 1000:
        return f"{kg / 1000:,.1f} tonnes"
    return f"{kg:,.0f} kg" if kg >= 10 else f"{kg:.1f} kg"


def facts(rec: dict) -> dict:
    """The facts an explanation may state, already formatted the way the text should show them."""
    out = {
        "product": rec["product_name"],
        "action": rec.get("customer_action", ""),
        "measure": rec.get("measure") or "",
        "reasons": [str(e) for e in (rec.get("evidence") if rec.get("evidence") is not None else [])],
        "data_date": rec["data_timestamp"],
        "confidence": "high" if _num(rec.get("confidence")) >= 0.7 else ("medium" if _num(rec.get("confidence")) >= 0.45 else "low"),
        "disclaimer": PRODUCT_DISCLAIMER.get(rec["product"], ""),
    }
    kg = _num(rec.get("co2e_avoided_kg_per_year"))
    if rec.get("impact_status") == "ok" and kg > 0:
        out["carbon_per_year"] = _round_kg(kg)
        horizon = _num(rec.get("horizon_years"))
        if horizon > 1:
            out["carbon_horizon"] = _round_kg(_num(rec.get("co2e_avoided_kg_horizon")))
            out["horizon_years"] = f"{horizon:.0f}"
    change = _num(rec.get("running_cost_change_gbp_per_year"))
    if math.isfinite(change) and abs(change) >= 5:
        out["money_per_year"] = f"£{abs(change):,.0f}"
        out["money_direction"] = "less" if change < 0 else "more"
    upfront = _num(rec.get("upfront_cost_gbp"))
    if math.isfinite(upfront) and upfront > 0:
        out["upfront"] = f"£{upfront:,.0f}"
    payback = _num(rec.get("payback_years"))
    if math.isfinite(payback) and payback <= 30:
        out["payback_years"] = f"{payback:.0f}"
    return out


def template_text(f: dict) -> str:
    """Deterministic explanation: why, what it changes, what it costs, and when the data is from."""
    body, footer = template_parts(f)
    return f"{body} {footer}"


def compliance_text(f: dict) -> str:
    """Fixed wording every explanation ends with, never rewritten: the data date, the caveat and any disclaimer."""
    text = f"These are estimates from your account data up to {f['data_date']}; prices and your situation can change."
    return f"{text} {f['disclaimer']}" if f["disclaimer"] else text


def template_parts(f: dict) -> tuple[str, str]:
    """(body an LLM may rephrase, fixed compliance sentence)."""
    parts = []
    reasons = f["reasons"]
    if reasons:
        joined = reasons[0] if len(reasons) == 1 else ", ".join(reasons[:-1]) + " and " + reasons[-1]
        parts.append(f"Why {f['product']} could suit you: {joined}.")
    else:
        parts.append(f"{f['product']} could suit you based on how you bank with us.")
    if "carbon_per_year" in f:
        measure = f["measure"][:1].upper() + f["measure"][1:] if f["measure"] else "This"  # keep "kWp" as written
        line = f"{measure} could cut about {f['carbon_per_year']} of CO2e a year"
        if "carbon_horizon" in f:
            line += f" (about {f['carbon_horizon']} over {f['horizon_years']} years)"
        parts.append(line + ".")
    if "money_per_year" in f:
        parts.append(f"Running costs would be about {f['money_per_year']} a year {f['money_direction']} at today's prices.")
    if "upfront" in f:
        line = f"The upfront cost is about {f['upfront']}"
        if "payback_years" in f:
            line += f", paying back in about {f['payback_years']} years"
        parts.append(line + ".")
    return " ".join(parts), compliance_text(f)
