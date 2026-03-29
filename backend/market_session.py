"""Broad market (SPY, VIX) and extended-hours context from Yahoo Finance."""

from __future__ import annotations

import time
from typing import Any, Optional

import pandas as pd
import yfinance as yf

# Reuse one SPY/VIX fetch per process for a short window (batch analyzes many tickers).
_market_cache: tuple[float, dict[str, Any]] | None = None
_CACHE_TTL_SEC = 90.0


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def get_market_snapshot() -> dict[str, Any]:
    """
    SPY trend vs MAs + VIX level. Cached ~90s to avoid hammering Yahoo on multi-ticker runs.
    Returns dict with scores and display fields.
    """
    global _market_cache
    now = time.monotonic()
    if _market_cache is not None and (now - _market_cache[0]) < _CACHE_TTL_SEC:
        return _market_cache[1]

    out: dict[str, Any] = {
        "spy_score": 0.0,
        "vix_adjustment": 0.0,
        "market_score": 0.0,
        "spy_last": None,
        "spy_sma_50": None,
        "spy_sma_200": None,
        "spy_return_5d": None,
        "vix_last": None,
        "error": None,
    }

    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="400d", interval="1d", auto_adjust=True)
        if hist is not None and not hist.empty:
            hist.columns = [str(c).lower() for c in hist.columns]
            close = hist["close"]
            last = float(close.iloc[-1])
            sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
            sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
            ret5 = None
            if len(close) >= 6:
                ret5 = (last / float(close.iloc[-6]) - 1.0) * 100.0

            out["spy_last"] = last
            out["spy_sma_50"] = sma50
            out["spy_sma_200"] = sma200
            out["spy_return_5d"] = ret5

            ss = 0.0
            if sma50 and sma200:
                if last > sma50 > sma200:
                    ss += 0.35
                elif last < sma50 < sma200:
                    ss -= 0.35
                elif last > sma200:
                    ss += 0.1
                elif last < sma200:
                    ss -= 0.1
            if ret5 is not None:
                if ret5 > 2.0:
                    ss += 0.15
                elif ret5 < -2.0:
                    ss -= 0.15
            out["spy_score"] = float(max(-1.0, min(1.0, ss)))
    except Exception as e:
        out["error"] = str(e)

    vix_adj = 0.0
    try:
        vx = yf.Ticker("^VIX")
        vh = vx.history(period="30d", interval="1d", auto_adjust=True)
        if vh is not None and not vh.empty:
            vh.columns = [str(c).lower() for c in vh.columns]
            vix_last = float(vh["close"].iloc[-1])
            out["vix_last"] = vix_last
            # Elevated fear: slight headwind for risk-on bias; very low VIX: tiny caution
            if vix_last >= 28:
                vix_adj = -0.18
            elif vix_last >= 22:
                vix_adj = -0.08
            elif vix_last <= 13:
                vix_adj = -0.05
            out["vix_adjustment"] = vix_adj
    except Exception:
        pass

    spy_s = float(out["spy_score"])
    out["market_score"] = float(max(-1.0, min(1.0, spy_s + vix_adj)))
    _market_cache = (now, out)
    return out


def extended_session_from_info(info: dict[str, Any]) -> dict[str, Any]:
    """
    Pre-market / after-hours vs regular session from quote summary fields when present.
    Yahoo key names vary; we normalize to a score in [-1, 1] and raw percentages.
    """
    reg = _safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
    prev = _safe_float(info.get("regularMarketPreviousClose") or info.get("previousClose"))

    pre_pct = _safe_float(info.get("preMarketChangePercent"))
    post_pct = _safe_float(info.get("postMarketChangePercent"))

    pre_price = _safe_float(info.get("preMarketPrice"))
    post_price = _safe_float(info.get("postMarketPrice"))

    state = str(info.get("marketState") or info.get("market_state") or "")

    # If only prices, derive % vs previous close
    if prev and prev > 0:
        if pre_pct is None and pre_price is not None:
            pre_pct = (pre_price / prev - 1.0) * 100.0
        if post_pct is None and post_price is not None and reg is not None:
            post_pct = (post_price / reg - 1.0) * 100.0 if reg else None

    score = 0.0
    parts: list[str] = []
    st = state.upper()

    def _contrib(pct: Optional[float], label: str) -> None:
        nonlocal score
        if pct is None:
            return
        c = max(-0.22, min(0.22, pct / 25.0))
        score += c
        parts.append(f"{label} {pct:+.2f}%")

    # Prefer the session Yahoo marks active to avoid double-counting stale fields
    if "PRE" in st and pre_pct is not None:
        _contrib(pre_pct, "Pre-market")
    elif "POST" in st and post_pct is not None:
        _contrib(post_pct, "After-hours")
    else:
        _contrib(pre_pct, "Pre-market")
        _contrib(post_pct, "After-hours")

    score = float(max(-1.0, min(1.0, score)))

    return {
        "session_score": score,
        "regular_market_price": reg,
        "previous_close": prev,
        "pre_market_change_pct": pre_pct,
        "post_market_change_pct": post_pct,
        "pre_market_price": pre_price,
        "post_market_price": post_price,
        "market_state": state or None,
        "summary": "; ".join(parts) if parts else "No extended-hours % in quote (may be closed or data unavailable)",
    }
