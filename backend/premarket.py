"""Pre-market / after-hours fetch + adjustment logic.

This module augments the core technical/fundamental/sentiment pipeline with
extended-hours *price action* (pre-market / after-hours) to adjust entries when
there is a meaningful gap/move vs the previous close.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from backend.schemas import AfterhoursAnalysis, EntryRange, PremarketAnalysis, RiskReward, TradeDirection


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


@dataclass
class PremarketFetchResult:
    previous_close: Optional[float]
    premarket_price: Optional[float]
    premarket_high: Optional[float]
    premarket_low: Optional[float]
    premarket_volume: Optional[float]


@dataclass
class AfterhoursFetchResult:
    regular_close: Optional[float]
    afterhours_price: Optional[float]
    afterhours_high: Optional[float]
    afterhours_low: Optional[float]
    afterhours_volume: Optional[float]


@dataclass
class ExtendedHoursFetchResult:
    premarket: PremarketFetchResult
    afterhours: AfterhoursFetchResult


def _fetch_extended_hours_df(ticker: str) -> Optional[pd.DataFrame]:
    t = yf.Ticker(ticker)
    df = t.history(period="1d", interval="1m", prepost=True, auto_adjust=True)
    if df is None or df.empty:
        return None
    df = df.dropna(how="all")
    if df.empty:
        return None
    df.columns = [str(c).lower() for c in df.columns]
    try:
        if getattr(df.index, "tz", None) is not None:
            df = df.tz_convert("America/New_York")
    except Exception:
        pass
    return df


def fetch_extended_hours_data(ticker: str, info: dict[str, Any]) -> ExtendedHoursFetchResult:
    """Fetch and summarize both pre-market and after-hours windows."""
    prev_close = _safe_float(info.get("regularMarketPreviousClose") or info.get("previousClose"))
    reg_close = _safe_float(info.get("regularMarketPrice") or info.get("regularMarketClose") or info.get("currentPrice"))

    df = _fetch_extended_hours_df(ticker)
    if df is None:
        return ExtendedHoursFetchResult(
            premarket=PremarketFetchResult(prev_close, None, None, None, None),
            afterhours=AfterhoursFetchResult(reg_close, None, None, None, None),
        )

    # Pre-market 04:00–09:29
    try:
        pre = df.between_time(time(4, 0), time(9, 29))
    except Exception:
        pre = None
    if pre is None or pre.empty:
        premkt = PremarketFetchResult(prev_close, None, None, None, None)
    else:
        premkt = PremarketFetchResult(
            prev_close,
            _safe_float(pre["close"].iloc[-1]) if "close" in pre.columns else None,
            _safe_float(pre["high"].max()) if "high" in pre.columns else None,
            _safe_float(pre["low"].min()) if "low" in pre.columns else None,
            _safe_float(pre["volume"].sum()) if "volume" in pre.columns else None,
        )

    # After-hours 16:00–19:59
    try:
        post = df.between_time(time(16, 0), time(19, 59))
    except Exception:
        post = None
    if post is None or post.empty:
        aft = AfterhoursFetchResult(reg_close, None, None, None, None)
    else:
        aft = AfterhoursFetchResult(
            reg_close,
            _safe_float(post["close"].iloc[-1]) if "close" in post.columns else None,
            _safe_float(post["high"].max()) if "high" in post.columns else None,
            _safe_float(post["low"].min()) if "low" in post.columns else None,
            _safe_float(post["volume"].sum()) if "volume" in post.columns else None,
        )

    return ExtendedHoursFetchResult(premarket=premkt, afterhours=aft)


def fetch_premarket_data(ticker: str, info: dict[str, Any]) -> PremarketFetchResult:
    """Fetch pre-market intraday bars (when available) and summarize.

    Uses `prepost=True` minute data and then slices the pre-market window (US equities):
    04:00–09:29 ET.
    """
    prev_close = _safe_float(info.get("regularMarketPreviousClose") or info.get("previousClose"))

    t = yf.Ticker(ticker)
    df = t.history(period="1d", interval="1m", prepost=True, auto_adjust=True)
    if df is None or df.empty:
        return PremarketFetchResult(prev_close, None, None, None, None)

    df = df.dropna(how="all")
    if df.empty:
        return PremarketFetchResult(prev_close, None, None, None, None)

    # Normalize columns
    df.columns = [str(c).lower() for c in df.columns]

    idx = df.index
    # Ensure we can slice by local ET time-of-day
    try:
        if getattr(idx, "tz", None) is not None:
            df = df.tz_convert("America/New_York")
    except Exception:
        # If tz conversion fails, still try time slicing on naive index
        pass

    # Pre-market window: 04:00–09:29 (inclusive)
    try:
        pre = df.between_time(time(4, 0), time(9, 29))
    except Exception:
        pre = df

    if pre is None or pre.empty:
        return PremarketFetchResult(prev_close, None, None, None, None)

    price = _safe_float(pre["close"].iloc[-1]) if "close" in pre.columns else None
    high = _safe_float(pre["high"].max()) if "high" in pre.columns else None
    low = _safe_float(pre["low"].min()) if "low" in pre.columns else None
    vol = _safe_float(pre["volume"].sum()) if "volume" in pre.columns else None

    return PremarketFetchResult(prev_close, price, high, low, vol)


def analyze_premarket_conditions(
    premkt: PremarketFetchResult,
    avg_daily_volume: Optional[float] = None,
    significant_move_abs_pct: float = 3.0,
) -> PremarketAnalysis:
    """Convert raw premarket fetch result into structured analysis + signal label."""
    prev = premkt.previous_close
    pre_price = premkt.premarket_price
    chg_pct = None
    if prev is not None and prev > 0 and pre_price is not None:
        chg_pct = ((pre_price - prev) / prev) * 100.0

    # Volume heuristic: pre-market volume is usually a small fraction of daily volume.
    # We treat larger fractions as stronger confirmation for the move.
    signal = "NEUTRAL"
    note_parts: list[str] = []
    if chg_pct is not None and abs(chg_pct) > significant_move_abs_pct:
        note_parts.append(f"Significant pre-market move ({chg_pct:+.2f}%).")
    else:
        note_parts.append("Pre-market move not significant; wait for market open confirmation.")

    vol = premkt.premarket_volume
    if vol is not None and avg_daily_volume is not None and avg_daily_volume > 0:
        frac = vol / avg_daily_volume
        # These thresholds are intentionally conservative to avoid overfitting.
        if frac >= 0.02:
            signal = "STRONG"
            note_parts.append(f"Pre-market volume looks meaningful (~{frac:.2%} of avg daily).")
        elif frac <= 0.005:
            signal = "WEAK"
            note_parts.append(f"Pre-market volume is light (~{frac:.2%} of avg daily).")
        else:
            signal = "NEUTRAL"
            note_parts.append(f"Pre-market volume is moderate (~{frac:.2%} of avg daily).")
    elif vol is not None:
        note_parts.append("Pre-market volume present, but no avg daily volume for comparison.")

    return PremarketAnalysis(
        premarket_price=pre_price,
        premarket_change_percent=None if chg_pct is None else float(chg_pct),
        premarket_high=premkt.premarket_high,
        premarket_low=premkt.premarket_low,
        premarket_volume=vol,
        previous_close=prev,
        adjusted_entry_range=[],
        premarket_signal=signal,
        note=" ".join(note_parts),
    )


def analyze_afterhours_conditions(
    aft: AfterhoursFetchResult,
    avg_daily_volume: Optional[float] = None,
    significant_move_abs_pct: float = 3.0,
) -> AfterhoursAnalysis:
    """Convert raw after-hours fetch result into structured analysis + signal label."""
    reg = aft.regular_close
    post_price = aft.afterhours_price
    chg_pct = None
    if reg is not None and reg > 0 and post_price is not None:
        chg_pct = ((post_price - reg) / reg) * 100.0

    signal = "NEUTRAL"
    note_parts: list[str] = []
    if chg_pct is not None and abs(chg_pct) > significant_move_abs_pct:
        note_parts.append(f"Significant after-hours move ({chg_pct:+.2f}%).")
    else:
        note_parts.append("After-hours move not significant; wait for next session confirmation.")

    vol = aft.afterhours_volume
    if vol is not None and avg_daily_volume is not None and avg_daily_volume > 0:
        frac = vol / avg_daily_volume
        if frac >= 0.02:
            signal = "STRONG"
            note_parts.append(f"After-hours volume looks meaningful (~{frac:.2%} of avg daily).")
        elif frac <= 0.005:
            signal = "WEAK"
            note_parts.append(f"After-hours volume is light (~{frac:.2%} of avg daily).")
        else:
            signal = "NEUTRAL"
            note_parts.append(f"After-hours volume is moderate (~{frac:.2%} of avg daily).")
    elif vol is not None:
        note_parts.append("After-hours volume present, but no avg daily volume for comparison.")

    return AfterhoursAnalysis(
        afterhours_price=post_price,
        afterhours_change_percent=None if chg_pct is None else float(chg_pct),
        afterhours_high=aft.afterhours_high,
        afterhours_low=aft.afterhours_low,
        afterhours_volume=vol,
        regular_close=reg,
        adjusted_entry_range=[],
        afterhours_signal=signal,
        note=" ".join(note_parts),
    )


def _market_state(info: dict[str, Any]) -> str:
    return str(info.get("marketState") or info.get("market_state") or "").upper()


def pick_helpful_extended_hours(
    info: dict[str, Any],
    pre: PremarketAnalysis,
    post: AfterhoursAnalysis,
) -> tuple[str, Optional[PremarketAnalysis], Optional[AfterhoursAnalysis]]:
    """Pick which session is most relevant to adjust entries.

    Preference:
    - If Yahoo indicates PRE or POST, use that.
    - Otherwise choose the session with larger absolute % move.
    """
    st = _market_state(info)
    if "PRE" in st:
        return "PRE", pre, None
    if "POST" in st:
        return "POST", None, post

    pre_abs = abs(pre.premarket_change_percent) if pre.premarket_change_percent is not None else 0.0
    post_abs = abs(post.afterhours_change_percent) if post.afterhours_change_percent is not None else 0.0
    if post_abs > pre_abs:
        return "POST", None, post
    return "PRE", pre, None


def _rr_label(ratio: Optional[float]) -> str:
    return f"~{ratio:.2f}:1" if ratio is not None else "n/a"


def adjust_trade_levels(
    direction: TradeDirection,
    planned_range: EntryRange,
    planned_entry_price: Optional[float],
    atr: Optional[float],
    resistance: Optional[float],
    support: Optional[float],
    premkt: PremarketAnalysis,
    significant_move_abs_pct: float = 3.0,
) -> tuple[EntryRange, Optional[float], Optional[float], list[float], RiskReward, PremarketAnalysis, bool]:
    """Adjust entry/levels using breakout logic on significant pre-market moves.

    - If |pre-market % change| <= threshold: keep planned levels, but return a "wait" flag.
    - If significant: use breakout entry at pre-market high/low and use pre-market high/low
      as key levels for stop placement.
    """
    chg = premkt.premarket_change_percent
    pre_high = premkt.premarket_high
    pre_low = premkt.premarket_low
    pre_price = premkt.premarket_price

    # Default: unchanged
    wait_for_open = True
    if chg is None or abs(chg) <= significant_move_abs_pct:
        premkt.adjusted_entry_range = [planned_range.low, planned_range.high]
        premkt.premarket_signal = premkt.premarket_signal or "NEUTRAL"
        return planned_range, planned_entry_price, None, [], RiskReward(ratio=None, label="n/a"), premkt, wait_for_open

    wait_for_open = False
    a = (atr if atr and atr > 0 else (planned_entry_price or pre_price or 0.0) * 0.02) or 0.0

    # Breakout: entry trigger is pre-market high for longs, pre-market low for shorts.
    if direction == TradeDirection.BUY:
        entry = pre_high or pre_price or planned_entry_price
        if entry is None:
            premkt.adjusted_entry_range = [planned_range.low, planned_range.high]
            return planned_range, planned_entry_price, None, [], RiskReward(ratio=None, label="n/a"), premkt, wait_for_open

        # Use pre-market low as support when available (gap-and-go failure point).
        stop = (pre_low - 0.15 * a) if pre_low is not None else (entry - 1.5 * a)
        tp1 = (resistance - 0.05 * a) if (resistance is not None and resistance > entry) else (entry + 1.5 * a)
        tp2 = (resistance + 0.5 * a) if (resistance is not None and resistance > entry) else (entry + 2.5 * a)

        final = EntryRange(low=round(entry, 4), high=round(entry + 0.25 * a, 4))
        risk = entry - stop
        reward = max(tp1 - entry, a) if entry is not None else None
        rr = (reward / risk) if (risk and risk > 0 and reward is not None) else None
        premkt.adjusted_entry_range = [final.low, final.high]
        premkt.premarket_signal = "STRONG" if premkt.premarket_signal == "STRONG" else (premkt.premarket_signal or "NEUTRAL")
        premkt.note = (premkt.note + " Using breakout above pre-market high.").strip()
        return (
            final,
            round(float(entry), 4),
            round(float(stop), 4) if stop is not None else None,
            [round(float(tp1), 4), round(float(tp2), 4)],
            RiskReward(ratio=round(rr, 3) if rr is not None else None, label=_rr_label(rr)),
            premkt,
            wait_for_open,
        )

    if direction == TradeDirection.SELL:
        entry = pre_low or pre_price or planned_entry_price
        if entry is None:
            premkt.adjusted_entry_range = [planned_range.low, planned_range.high]
            return planned_range, planned_entry_price, None, [], RiskReward(ratio=None, label="n/a"), premkt, wait_for_open

        stop = (pre_high + 0.15 * a) if pre_high is not None else (entry + 1.5 * a)
        tp1 = (support + 0.05 * a) if (support is not None and support < entry) else (entry - 1.5 * a)
        tp2 = (entry - 2.5 * a)

        final = EntryRange(low=round(entry - 0.25 * a, 4), high=round(entry, 4))
        risk = stop - entry
        reward = max(entry - tp1, a)
        rr = (reward / risk) if risk and risk > 0 else None
        premkt.adjusted_entry_range = [final.low, final.high]
        premkt.premarket_signal = "STRONG" if premkt.premarket_signal == "STRONG" else (premkt.premarket_signal or "NEUTRAL")
        premkt.note = (premkt.note + " Using breakout below pre-market low.").strip()
        return (
            final,
            round(float(entry), 4),
            round(float(stop), 4) if stop is not None else None,
            [round(float(tp1), 4), round(float(tp2), 4)],
            RiskReward(ratio=round(rr, 3) if rr is not None else None, label=_rr_label(rr)),
            premkt,
            wait_for_open,
        )

    premkt.adjusted_entry_range = [planned_range.low, planned_range.high]
    return planned_range, planned_entry_price, None, [], RiskReward(ratio=None, label="n/a"), premkt, True


def adjust_trade_levels_afterhours(
    direction: TradeDirection,
    planned_range: EntryRange,
    planned_entry_price: Optional[float],
    atr: Optional[float],
    resistance: Optional[float],
    support: Optional[float],
    aft: AfterhoursAnalysis,
    significant_move_abs_pct: float = 3.0,
) -> tuple[EntryRange, Optional[float], Optional[float], list[float], RiskReward, AfterhoursAnalysis, bool]:
    """Adjust entry/levels using breakout logic on significant after-hours moves."""
    chg = aft.afterhours_change_percent
    post_high = aft.afterhours_high
    post_low = aft.afterhours_low
    post_price = aft.afterhours_price

    wait = True
    if chg is None or abs(chg) <= significant_move_abs_pct:
        aft.adjusted_entry_range = [planned_range.low, planned_range.high]
        aft.afterhours_signal = aft.afterhours_signal or "NEUTRAL"
        return planned_range, planned_entry_price, None, [], RiskReward(ratio=None, label="n/a"), aft, wait

    wait = False
    a = (atr if atr and atr > 0 else (planned_entry_price or post_price or 0.0) * 0.02) or 0.0

    if direction == TradeDirection.BUY:
        entry = post_high or post_price or planned_entry_price
        if entry is None:
            aft.adjusted_entry_range = [planned_range.low, planned_range.high]
            return planned_range, planned_entry_price, None, [], RiskReward(ratio=None, label="n/a"), aft, wait
        stop = (post_low - 0.15 * a) if post_low is not None else (entry - 1.5 * a)
        tp1 = (resistance - 0.05 * a) if (resistance is not None and resistance > entry) else (entry + 1.5 * a)
        tp2 = (resistance + 0.5 * a) if (resistance is not None and resistance > entry) else (entry + 2.5 * a)
        final = EntryRange(low=round(entry, 4), high=round(entry + 0.25 * a, 4))
        risk = entry - stop
        reward = max(tp1 - entry, a)
        rr = (reward / risk) if risk and risk > 0 else None
        aft.adjusted_entry_range = [final.low, final.high]
        aft.afterhours_signal = "STRONG" if aft.afterhours_signal == "STRONG" else (aft.afterhours_signal or "NEUTRAL")
        aft.note = (aft.note + " Using breakout above after-hours high.").strip()
        return (
            final,
            round(float(entry), 4),
            round(float(stop), 4) if stop is not None else None,
            [round(float(tp1), 4), round(float(tp2), 4)],
            RiskReward(ratio=round(rr, 3) if rr is not None else None, label=_rr_label(rr)),
            aft,
            wait,
        )

    if direction == TradeDirection.SELL:
        entry = post_low or post_price or planned_entry_price
        if entry is None:
            aft.adjusted_entry_range = [planned_range.low, planned_range.high]
            return planned_range, planned_entry_price, None, [], RiskReward(ratio=None, label="n/a"), aft, wait
        stop = (post_high + 0.15 * a) if post_high is not None else (entry + 1.5 * a)
        tp1 = (support + 0.05 * a) if (support is not None and support < entry) else (entry - 1.5 * a)
        tp2 = entry - 2.5 * a
        final = EntryRange(low=round(entry - 0.25 * a, 4), high=round(entry, 4))
        risk = stop - entry
        reward = max(entry - tp1, a)
        rr = (reward / risk) if risk and risk > 0 else None
        aft.adjusted_entry_range = [final.low, final.high]
        aft.afterhours_signal = "STRONG" if aft.afterhours_signal == "STRONG" else (aft.afterhours_signal or "NEUTRAL")
        aft.note = (aft.note + " Using breakout below after-hours low.").strip()
        return (
            final,
            round(float(entry), 4),
            round(float(stop), 4) if stop is not None else None,
            [round(float(tp1), 4), round(float(tp2), 4)],
            RiskReward(ratio=round(rr, 3) if rr is not None else None, label=_rr_label(rr)),
            aft,
            wait,
        )

    aft.adjusted_entry_range = [planned_range.low, planned_range.high]
    return planned_range, planned_entry_price, None, [], RiskReward(ratio=None, label="n/a"), aft, True

