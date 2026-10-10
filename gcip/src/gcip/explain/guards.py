"""Guardrails for generated explanations: the text may only state facts from the structured record."""

from __future__ import annotations

import re

BANNED = [r"\bguarantee", r"\bwill save\b", r"\bdefinitely\b", r"\brisk[- ]free\b", r"\bfree money\b", r"\bpre-?approved\b",
          r"\byou qualify\b", r"\bapproved\b", r"\bbest investment\b", r"\brecommend(ed)? (that )?you invest\b",
          r"\bperfect\b", r"\bideal\b", r"\bamazing\b", r"\bexcit", r"\bgreat news\b", r"\bhi there\b",
          r"\bwill (reduce|decrease|cut|lower|pay|save)\b", r"\byou'?ll (save|reduce|cut)\b", r"\bshould see you\b"]
MONTHLY = re.compile(r"\b(each|per|a|every) month\b|\bmonthly\b")
NUMBER = re.compile(r"£?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return {n.replace("£", "").replace(",", "") for n in NUMBER.findall(text)}


def check(text: str, facts: dict, max_words: int = 120, draft: str | None = None) -> list[str]:
    """Problems found (empty list = passes): unsupported numbers, banned claims, missing disclaimer or date, length.
    With the `draft` that was rewritten, also: every number of the draft must be kept, and yearly or 12-month figures
    must not turn into monthly ones (a rewrite can keep a number and still change what it means)."""
    problems = []
    if draft is not None:
        dropped = _numbers(draft) - _numbers(text)
        if dropped:
            problems.append(f"numbers dropped: {sorted(dropped)[:5]}")
        if MONTHLY.search(text.lower()) and not MONTHLY.search(draft.lower()):
            problems.append("changes the time period to monthly")
    allowed = _numbers(" ".join(str(v) for v in facts.values() if not isinstance(v, list)) + " " + " ".join(facts.get("reasons", [])))
    allowed |= {"2", "1"}  # "CO2", "CO2e"
    extra = _numbers(text) - allowed
    if extra:
        problems.append(f"numbers not in the facts: {sorted(extra)[:5]}")
    lowered = text.lower()
    for pattern in BANNED:
        if re.search(pattern, lowered):
            problems.append(f"banned claim: {pattern}")
    if facts.get("disclaimer") and facts["disclaimer"].lower() not in lowered:
        problems.append("missing disclaimer")
    if facts["data_date"] not in text:
        problems.append("missing data date")
    words = len(text.split())
    if words > max_words or words < 15:
        problems.append(f"length {words} words")
    return problems
