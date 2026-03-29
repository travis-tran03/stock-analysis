"""Technical indicators: RSI, MACD, MAs, ATR, pivots, support/resistance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta


@dataclass
class TechnicalSnapshot:
    last_close: float
    rsi_14: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_hist: Optional[float]
    sma_50: Optional[float]
    sma_200: Optional[float]
    atr_14: Optional[float]
    pivot: Optional[float]
    r1: Optional[float]
    s1: Optional[float]
    support: Optional[float]
    resistance: Optional[float]
    raw_row: dict[str, Any]


def _last_valid(series: Optional[pd.Series]) -> Optional[float]:
    if series is None or series.empty:
        return None
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def _classic_pivots(prev_high: float, prev_low: float, prev_close: float) -> dict[str, float]:
    """Classic daily pivot from prior session H/L/C."""
    p = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * p - prev_low
    s1 = 2 * p - prev_high
    return {"pivot": p, "r1": r1, "s1": s1}


def _swing_support_resistance(
    high: pd.Series, low: pd.Series, lookback: int = 20
) -> tuple[Optional[float], Optional[float]]:
    """Recent swing low as support, swing high as resistance (last `lookback` bars)."""
    if len(high) < lookback:
        lookback = len(high)
    if lookback < 2:
        return None, None
    seg_h = high.iloc[-lookback:]
    seg_l = low.iloc[-lookback:]
    return float(seg_l.min()), float(seg_h.max())


def compute_technicals(df: pd.DataFrame) -> TechnicalSnapshot:
    """
    Expect columns: open, high, low, close, volume (lowercase).
    Computes RSI(14), MACD, SMA50/200, ATR(14), prior-day pivots, S/R from recent range.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    rsi_series = ta.rsi(close, length=14)
    macd_df = ta.macd(close)
    sma50 = ta.sma(close, length=50)
    sma200 = ta.sma(close, length=200)
    atr_series = ta.atr(high, low, close, length=14)

    rsi_14 = _last_valid(rsi_series)
    macd_val = macd_sig = macd_hist = None
    if macd_df is not None and not macd_df.empty:
        # Typical columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        cols = list(macd_df.columns)
        for c in cols:
            if c.startswith("MACD_") and "MACDh" not in c and "MACDs" not in c:
                macd_val = _last_valid(macd_df[c])
            elif c.startswith("MACDs_"):
                macd_sig = _last_valid(macd_df[c])
            elif c.startswith("MACDh_"):
                macd_hist = _last_valid(macd_df[c])

    sma_50 = _last_valid(sma50)
    sma_200 = _last_valid(sma200)
    atr_14 = _last_valid(atr_series)
    last_close = float(close.iloc[-1])

    pivot = r1 = s1 = None
    if len(df) >= 2:
        prev = df.iloc[-2]
        ph, pl, pc = float(prev["high"]), float(prev["low"]), float(prev["close"])
        piv = _classic_pivots(ph, pl, pc)
        pivot, r1, s1 = piv["pivot"], piv["r1"], piv["s1"]

    sup, res = _swing_support_resistance(high, low, lookback=20)

    raw_row = {
        "rsi_14": rsi_14,
        "macd": macd_val,
        "macd_signal": macd_sig,
        "macd_histogram": macd_hist,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "atr_14": atr_14,
        "pivot": pivot,
        "resistance_r1": r1,
        "support_s1": s1,
        "support_20d": sup,
        "resistance_20d": res,
        "last_close": last_close,
    }

    return TechnicalSnapshot(
        last_close=last_close,
        rsi_14=rsi_14,
        macd=macd_val,
        macd_signal=macd_sig,
        macd_hist=macd_hist,
        sma_50=sma_50,
        sma_200=sma_200,
        atr_14=atr_14,
        pivot=pivot,
        r1=r1,
        s1=s1,
        support=sup,
        resistance=res,
        raw_row=raw_row,
    )


def technical_score(tech: TechnicalSnapshot) -> float:
    """
    Map indicators to a score in [-1, 1]: bullish positive, bearish negative.
    """
    parts: list[float] = []
    weights: list[float] = []

    # RSI: <30 oversold (bullish mean reversion hint), >70 overbought
    if tech.rsi_14 is not None:
        r = tech.rsi_14
        if r < 30:
            parts.append(0.35)
        elif r > 70:
            parts.append(-0.35)
        elif r < 45:
            parts.append(0.15)
        elif r > 55:
            parts.append(-0.15)
        else:
            parts.append(0.0)
        weights.append(1.0)

    # MACD vs signal
    if tech.macd is not None and tech.macd_signal is not None:
        parts.append(0.4 if tech.macd > tech.macd_signal else -0.4)
        weights.append(1.0)

    # Price vs MAs (trend)
    if tech.sma_50 is not None and tech.sma_200 is not None:
        if tech.last_close > tech.sma_50 > tech.sma_200:
            parts.append(0.45)
        elif tech.last_close < tech.sma_50 < tech.sma_200:
            parts.append(-0.45)
        elif tech.last_close > tech.sma_200:
            parts.append(0.15)
        elif tech.last_close < tech.sma_200:
            parts.append(-0.15)
        else:
            parts.append(0.0)
        weights.append(1.0)

    if not parts:
        return 0.0

    w = np.array(weights, dtype=float)
    p = np.array(parts, dtype=float)
    return float(np.dot(p, w) / w.sum())
