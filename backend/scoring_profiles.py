"""Horizon-specific score weights (long-term vs short-term growth)."""

from __future__ import annotations

from typing import Literal

Horizon = Literal["long", "short"]

# Weights must sum to 1.0 per profile.
LONG_TERM_WEIGHTS = {
    "tech": 0.15,
    "fund": 0.35,
    "sent": 0.12,
    "market": 0.28,
    "session": 0.10,
}

SHORT_TERM_WEIGHTS = {
    "tech": 0.42,
    "fund": 0.12,
    "sent": 0.22,
    "market": 0.14,
    "session": 0.10,
}

BUY_THRESHOLD = 0.18
SELL_THRESHOLD = -0.18


def combined_for_horizon(
    horizon: Horizon,
    tech_s: float,
    fund_s: float,
    sent_s: float,
    market_s: float,
    session_s: float,
) -> float:
    w = LONG_TERM_WEIGHTS if horizon == "long" else SHORT_TERM_WEIGHTS
    return (
        w["tech"] * tech_s
        + w["fund"] * fund_s
        + w["sent"] * sent_s
        + w["market"] * market_s
        + w["session"] * session_s
    )


def direction_from_score(score: float) -> str:
    if score >= BUY_THRESHOLD:
        return "BUY"
    if score <= SELL_THRESHOLD:
        return "SELL"
    return "HOLD"
