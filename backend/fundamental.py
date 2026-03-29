"""Fundamental metrics from yfinance `info` and optional financial statement trends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class FundamentalSnapshot:
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    eps_ttm: Optional[float]
    revenue: Optional[float]
    revenue_growth: Optional[float]  # often YoY from info if present
    earnings_growth: Optional[float]
    profit_margins: Optional[float]
    debt_to_equity: Optional[float]
    raw: dict[str, Any]


def _safe_float(val: Any) -> Optional[float]:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def extract_fundamentals(info: dict[str, Any]) -> FundamentalSnapshot:
    """
    Pull key ratios from ticker.info. Keys vary by listing; use fallbacks.
    """
    pe = _safe_float(info.get("trailingPE") or info.get("trailingPe"))
    fpe = _safe_float(info.get("forwardPE") or info.get("forwardPe"))
    eps = _safe_float(info.get("trailingEps") or info.get("epsTrailingTwelveMonths"))
    rev = _safe_float(info.get("totalRevenue"))
    rev_g = _safe_float(info.get("revenueGrowth"))
    earn_g = _safe_float(info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth"))
    margin = _safe_float(info.get("profitMargins"))
    de = _safe_float(info.get("debtToEquity"))

    raw = {
        "trailing_pe": pe,
        "forward_pe": fpe,
        "eps_ttm": eps,
        "total_revenue": rev,
        "revenue_growth": rev_g,
        "earnings_growth": earn_g,
        "profit_margins": margin,
        "debt_to_equity": de,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }

    return FundamentalSnapshot(
        pe_ratio=pe,
        forward_pe=fpe,
        eps_ttm=eps,
        revenue=rev,
        revenue_growth=rev_g,
        earnings_growth=earn_g,
        profit_margins=margin,
        debt_to_equity=de,
        raw=raw,
    )


def fundamental_score(f: FundamentalSnapshot) -> float:
    """
    Simple score [-1, 1]: growth and margins positive; high debt negative; extreme P/E slightly negative.
    """
    parts: list[float] = []
    weights: list[float] = []

    if f.revenue_growth is not None:
        rg = f.revenue_growth
        if rg > 0.1:
            parts.append(0.35)
        elif rg > 0:
            parts.append(0.15)
        elif rg < -0.05:
            parts.append(-0.35)
        else:
            parts.append(-0.1)
        weights.append(1.0)

    if f.earnings_growth is not None:
        eg = f.earnings_growth
        parts.append(0.3 if eg > 0.05 else (-0.3 if eg < -0.05 else 0.0))
        weights.append(0.8)

    if f.profit_margins is not None:
        pm = f.profit_margins
        parts.append(0.25 if pm > 0.1 else (-0.15 if pm < 0 else 0.0))
        weights.append(0.7)

    if f.debt_to_equity is not None:
        de = f.debt_to_equity
        parts.append(-0.25 if de > 200 else (-0.1 if de > 100 else 0.1))
        weights.append(0.5)

    if f.pe_ratio is not None and f.pe_ratio > 0:
        pe = f.pe_ratio
        parts.append(-0.2 if pe > 60 else (0.1 if pe < 25 else 0.0))
        weights.append(0.4)

    if not parts:
        return 0.0

    w = np.array(weights, dtype=float)
    p = np.array(parts, dtype=float)
    return float(np.clip(np.dot(p, w) / w.sum(), -1.0, 1.0))
