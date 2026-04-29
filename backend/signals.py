"""Combine technical, fundamental, and sentiment inputs into trade signals and levels."""

from __future__ import annotations

from typing import Any, Optional

from backend.data_fetch import FetchedStockData, fetch_stock_data
from backend.fundamental import extract_fundamentals, fundamental_score
from backend.market_session import get_market_snapshot
from backend.premarket import (
    adjust_trade_levels,
    adjust_trade_levels_afterhours,
    analyze_afterhours_conditions,
    analyze_premarket_conditions,
    pick_helpful_extended_hours,
)
from backend.schemas import EntryRange, RiskReward, StockAnalysis, TradeDirection
from backend.sentiment import analyze_news_sentiment, sentiment_score
from backend.technical import TechnicalSnapshot, compute_technicals, technical_score

# Weights include broad market (SPY/VIX) and extended-hours session tilt
WT_TECH = 0.40
WT_FUND = 0.18
WT_SENT = 0.18
WT_MARKET = 0.14
WT_SESSION = 0.10

BUY_THRESHOLD = 0.18
SELL_THRESHOLD = -0.18
MAX_EXT_HOURS_ENTRY_SHIFT_PCT = 0.08
MAX_EXT_HOURS_TP1_SHIFT_PCT = 0.12
MAX_LEVEL_MOD_SHIFT = 0.18
EXT_HOURS_SIGNIFICANT_MOVE_PCT = 4.0
EXT_HOURS_STRONG_MOVE_PCT = 6.0
EXT_HOURS_MIN_BLEND = 0.10
EXT_HOURS_MAX_BLEND = 0.35


def _combined_score(
    tech_s: float,
    fund_s: float,
    sent_s: float,
    market_s: float,
    session_s: float,
) -> float:
    return (
        WT_TECH * tech_s
        + WT_FUND * fund_s
        + WT_SENT * sent_s
        + WT_MARKET * market_s
        + WT_SESSION * session_s
    )


def _direction_from_score(s: float) -> TradeDirection:
    if s >= BUY_THRESHOLD:
        return TradeDirection.BUY
    if s <= SELL_THRESHOLD:
        return TradeDirection.SELL
    return TradeDirection.HOLD


def _cap_ext_hours_levels(
    entry: Optional[float],
    stop: Optional[float],
    tps: list[float],
    reference_price: Optional[float],
) -> tuple[Optional[float], Optional[float], list[float]]:
    """Keep extended-hours-derived levels within sane bounds vs reference price."""
    if reference_price is None or reference_price <= 0:
        return entry, stop, tps

    lo_entry = reference_price * (1.0 - MAX_EXT_HOURS_ENTRY_SHIFT_PCT)
    hi_entry = reference_price * (1.0 + MAX_EXT_HOURS_ENTRY_SHIFT_PCT)
    lo_tp = reference_price * (1.0 - MAX_EXT_HOURS_TP1_SHIFT_PCT)
    hi_tp = reference_price * (1.0 + MAX_EXT_HOURS_TP1_SHIFT_PCT)

    capped_entry = entry
    if capped_entry is not None:
        capped_entry = max(lo_entry, min(hi_entry, float(capped_entry)))

    capped_stop = stop
    if capped_stop is not None:
        capped_stop = max(lo_tp, min(hi_tp, float(capped_stop)))

    capped_tps: list[float] = []
    for tp in tps:
        capped_tps.append(max(lo_tp, min(hi_tp, float(tp))))

    return capped_entry, capped_stop, capped_tps


def _blend_scalar(a: Optional[float], b: Optional[float], w: float) -> Optional[float]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return (1.0 - w) * float(a) + w * float(b)


def _ext_hours_blend_weight(
    change_pct: Optional[float],
    session_signal: Optional[str],
    combined_score: float,
) -> float:
    """Map extended-hours confidence to a bounded influence weight."""
    if change_pct is None:
        return 0.0
    mag = abs(float(change_pct))
    # Magnitude scaled around 4%-8% band.
    mag_norm = max(0.0, min(1.0, (mag - EXT_HOURS_SIGNIFICANT_MOVE_PCT) / 4.0))
    w = EXT_HOURS_MIN_BLEND + (EXT_HOURS_MAX_BLEND - EXT_HOURS_MIN_BLEND) * mag_norm

    sig = (session_signal or "").upper()
    if sig == "WEAK":
        w *= 0.55
    elif sig == "NEUTRAL":
        w *= 0.80
    elif sig == "STRONG":
        w *= 1.00

    # If regular blended score is already directional, reduce ext-hours dominance.
    if abs(combined_score) >= 0.18:
        w *= 0.70

    return float(max(0.0, min(EXT_HOURS_MAX_BLEND, w)))


def _should_override_direction(
    current: TradeDirection,
    implied: TradeDirection,
    change_pct: Optional[float],
    session_signal: Optional[str],
    combined_score: float,
) -> bool:
    """Conservative override: only when ext-hours signal is strong and core score is near-neutral."""
    if implied == current:
        return False
    if current != TradeDirection.HOLD:
        return False
    if change_pct is None or abs(float(change_pct)) < EXT_HOURS_STRONG_MOVE_PCT:
        return False
    if (session_signal or "").upper() != "STRONG":
        return False
    # Only override HOLD when core model is near neutral.
    return abs(combined_score) <= 0.12


def _conviction_strength(
    direction: TradeDirection,
    combined: float,
    tech_s: float,
    fund_s: float,
    sent_s: float,
    market_s: float,
    session_s: float,
    rsi_14: Optional[float],
) -> float:
    """Return 0..1 confidence that levels can be more aggressive for this direction."""
    sign = 1.0 if direction == TradeDirection.BUY else (-1.0 if direction == TradeDirection.SELL else 0.0)
    if sign == 0.0:
        return 0.5

    aligned = [
        combined * sign,
        tech_s * sign,
        fund_s * sign,
        sent_s * sign,
        market_s * sign,
        session_s * sign,
    ]
    # Map average aligned score to [0,1] with conservative bounds.
    avg_aligned = sum(aligned) / len(aligned)
    base = (max(-0.35, min(0.35, avg_aligned)) + 0.35) / 0.70

    # RSI tilt: supportive RSI adds conviction, adverse RSI reduces.
    rsi_adj = 0.0
    if rsi_14 is not None:
        if direction == TradeDirection.BUY:
            if rsi_14 < 35:
                rsi_adj += 0.08
            elif rsi_14 > 68:
                rsi_adj -= 0.10
        elif direction == TradeDirection.SELL:
            if rsi_14 > 65:
                rsi_adj += 0.08
            elif rsi_14 < 32:
                rsi_adj -= 0.10

    return float(max(0.0, min(1.0, base + rsi_adj)))


def _apply_level_modifiers(
    direction: TradeDirection,
    entry_range: EntryRange,
    entry_price: Optional[float],
    stop: Optional[float],
    take_profits: list[float],
    price: float,
    atr: Optional[float],
    conviction: float,
) -> tuple[EntryRange, Optional[float], Optional[float], list[float], RiskReward]:
    """
    Nudge levels using conviction while preserving structure:
    - stronger conviction => earlier entry + farther TP
    - weaker conviction => more conservative entry + nearer TP
    """
    if direction not in (TradeDirection.BUY, TradeDirection.SELL):
        rr = RiskReward(ratio=None, label="wait for clarity")
        return entry_range, entry_price, stop, take_profits, rr

    if entry_price is None or stop is None or not take_profits:
        rr = RiskReward(ratio=None, label="n/a")
        return entry_range, entry_price, stop, take_profits, rr

    base_a = atr if atr and atr > 0 else price * 0.02
    # conviction around 0.5 => neutral; below/above shifts up to MAX_LEVEL_MOD_SHIFT of ATR.
    tilt = (conviction - 0.5) * 2.0
    shift = max(-MAX_LEVEL_MOD_SHIFT, min(MAX_LEVEL_MOD_SHIFT, tilt)) * base_a

    e = float(entry_price)
    s = float(stop)
    tps = [float(x) for x in take_profits]
    low = float(entry_range.low) if entry_range.low is not None else e
    high = float(entry_range.high) if entry_range.high is not None else e
    width = max(0.0, high - low)
    half_w = width / 2.0

    if direction == TradeDirection.BUY:
        # Stronger bullish conviction: accept slightly higher/earlier entry and aim farther TP.
        e += shift
        s += 0.4 * shift
        tps = [tp + 0.8 * shift for tp in tps]
    else:
        # Stronger bearish conviction: accept slightly lower/earlier short entry and farther downside TP.
        e -= shift
        s -= 0.4 * shift
        tps = [tp - 0.8 * shift for tp in tps]

    # Keep entry range centered on adjusted entry, preserving prior width.
    adj_range = EntryRange(low=round(e - half_w, 4), high=round(e + half_w, 4))

    if direction == TradeDirection.BUY:
        risk = e - s
        reward = tps[0] - e
    else:
        risk = s - e
        reward = e - tps[0]
    rr_ratio = (reward / risk) if (risk > 0 and reward > 0) else None
    rr = RiskReward(ratio=round(rr_ratio, 3) if rr_ratio is not None else None, label=f"~{rr_ratio:.2f}:1" if rr_ratio else "n/a")

    return adj_range, round(e, 4), round(s, 4), [round(x, 4) for x in tps], rr


def _long_only_levels_for_sell_signal(
    price: float,
    atr: Optional[float],
    support: Optional[float],
    resistance: Optional[float],
    market_score: float,
    vix_last: Optional[float],
    conviction: float,
    rsi_14: Optional[float],
) -> tuple[EntryRange, Optional[float], Optional[float], list[float], RiskReward]:
    """
    Long-only interpretation for bearish signals:
    seek a deeper pullback entry, then sell into rebound targets.
    """
    base_a = atr if atr and atr > 0 else price * 0.02
    bm, _lx = _entry_market_adjustment(market_score, vix_last, atr, price)
    a = base_a * bm

    # Evidence-based guardrail:
    # Pullback entries in volatile tapes are commonly around ~1.0–2.0 ATR from anchor,
    # not extreme multi-ATR discounts.
    desired_discount_atr = 1.0 + 0.55 * conviction + 0.25 * max(0.0, -market_score)
    if rsi_14 is not None:
        # If already very oversold, avoid demanding an even deeper discount.
        if rsi_14 <= 30:
            desired_discount_atr -= 0.18
        elif rsi_14 >= 45:
            desired_discount_atr += 0.10
    desired_discount_atr = max(0.8, min(1.9, desired_discount_atr))

    # Hard percentage cap to prevent unrealistic buy lows in large caps.
    max_discount_pct = 0.07
    discount_amt = min(desired_discount_atr * a, price * max_discount_pct)

    center = price - discount_amt
    half_width = max(0.18 * a, price * 0.006)
    low = center - half_width
    high = center + half_width

    # Respect support structure but avoid setting entries too far below it.
    if support is not None and support < price:
        support_floor = support - 0.35 * a
        support_ceiling = support + 0.25 * a
        low = max(low, support_floor)
        high = min(high, support_ceiling if support_ceiling > low else high)
    if high <= low:
        high = low + 0.25 * a

    entry = (low + high) / 2.0
    stop = low - 0.8 * a
    tp1 = max(price, entry + 0.8 * a)
    if resistance is not None and resistance > entry:
        tp1 = min(tp1, resistance - 0.05 * a)
        tp1 = max(tp1, entry + 0.6 * a)
    tp2 = max(tp1 + 0.7 * a, min(price + 1.0 * a, entry + 2.0 * a))

    risk = entry - stop
    reward = tp1 - entry
    rr = (reward / risk) if (risk and risk > 0 and reward > 0) else None
    return (
        EntryRange(low=round(low, 4), high=round(high, 4)),
        round(entry, 4),
        round(stop, 4),
        [round(tp1, 4), round(tp2, 4)],
        RiskReward(ratio=round(rr, 3) if rr is not None else None, label=f"~{rr:.2f}:1" if rr else "n/a"),
    )


def _confidence_from_score(s: float, has_fundamentals: bool, news_count: int) -> float:
    base = min(1.0, abs(s) * 1.4 + 0.15)
    if not has_fundamentals:
        base *= 0.88
    if news_count == 0:
        base *= 0.9
    return float(max(0.1, min(0.95, base)))


def _entry_market_adjustment(
    market_score: float,
    vix_last: Optional[float],
    atr: Optional[float],
    price: float,
) -> tuple[float, float]:
    """
    Scale ATR-based bands with tape + vol; add extra room *below* last for long-leaning zones.

    Returns:
        band_mult: multiply effective ATR (widen ranges in risk-off / high VIX / volatile names).
        low_extra_frac: additional fraction of scaled ATR to extend the low side (dip / flush room).
    """
    base_a = atr if atr and atr > 0 else price * 0.02
    atr_pct = (base_a / price) if price > 0 else 0.02

    band_mult = 1.0
    if vix_last is not None and vix_last > 18:
        band_mult += min(0.3, (vix_last - 18) * 0.012)
    if market_score < 0:
        band_mult += min(0.28, abs(market_score) * 0.55)
    if atr_pct > 0.028:
        band_mult += min(0.22, (atr_pct - 0.028) * 5.0)
    band_mult = max(1.0, min(1.85, band_mult))

    low_extra_frac = 0.0
    if market_score < 0:
        low_extra_frac += min(0.38, abs(market_score) * 0.65)
    if vix_last is not None and vix_last > 20:
        low_extra_frac += min(0.22, (vix_last - 20) * 0.016)
    if atr_pct > 0.038:
        low_extra_frac += min(0.2, (atr_pct - 0.038) * 4.0)
    low_extra_frac = min(0.7, low_extra_frac)

    return band_mult, low_extra_frac


def _levels_buy(
    price: float,
    atr: Optional[float],
    support: Optional[float],
    resistance: Optional[float],
    market_score: float = 0.0,
    vix_last: Optional[float] = None,
) -> tuple[EntryRange, Optional[float], Optional[float], list[float], RiskReward]:
    """Long-oriented entry zone, stop under support/ATR, targets toward resistance/ATR multiples."""
    base_a = atr if atr and atr > 0 else price * 0.02
    bm, lx = _entry_market_adjustment(market_score, vix_last, atr, price)
    a = base_a * bm
    low = price - 0.35 * a - lx * base_a
    high = price + 0.25 * a
    if support and support < price:
        low = max(low, support - 0.1 * a)
    entry_price = (low + high) / 2.0
    stop = (support - 0.5 * a) if support else (price - 1.5 * a)
    stop = min(stop, price - 0.5 * a)

    tp1 = price + 1.5 * a
    tp2 = price + 2.5 * a
    if resistance and resistance > price:
        tp1 = min(tp1, resistance - 0.05 * a)
        tp2 = max(tp2, min(resistance + 0.5 * a, price + 3.5 * a))

    risk = entry_price - stop
    reward = max(tp1 - entry_price, a)
    rr = (reward / risk) if risk and risk > 0 else None
    label = f"~{rr:.2f}:1" if rr else "n/a"

    return (
        EntryRange(low=round(low, 4), high=round(high, 4)),
        round(entry_price, 4),
        round(stop, 4),
        [round(tp1, 4), round(tp2, 4)],
        RiskReward(ratio=round(rr, 3) if rr else None, label=label),
    )


def _levels_sell(
    price: float,
    atr: Optional[float],
    support: Optional[float],
    resistance: Optional[float],
    market_score: float = 0.0,
    vix_last: Optional[float] = None,
) -> tuple[EntryRange, Optional[float], Optional[float], list[float], RiskReward]:
    """Short-oriented: entry near resistance, stop above, targets lower."""
    base_a = atr if atr and atr > 0 else price * 0.02
    bm, _lx = _entry_market_adjustment(market_score, vix_last, atr, price)
    a = base_a * bm
    low = price - 0.25 * a
    high = price + 0.35 * a
    if resistance and resistance > price:
        high = min(high, resistance + 0.1 * a)
    entry_price = (low + high) / 2.0
    stop = (resistance + 0.5 * a) if resistance else (price + 1.5 * a)
    stop = max(stop, price + 0.5 * a)

    tp1 = price - 1.5 * a
    tp2 = price - 2.5 * a
    if support and support < price:
        tp1 = max(tp1, support + 0.05 * a)

    risk = stop - entry_price
    reward = max(entry_price - tp1, a)
    rr = (reward / risk) if risk and risk > 0 else None
    label = f"~{rr:.2f}:1" if rr else "n/a"

    return (
        EntryRange(low=round(low, 4), high=round(high, 4)),
        round(entry_price, 4),
        round(stop, 4),
        [round(tp1, 4), round(tp2, 4)],
        RiskReward(ratio=round(rr, 3) if rr else None, label=label),
    )


def _levels_hold(
    price: float,
    atr: Optional[float],
    market_score: float = 0.0,
    vix_last: Optional[float] = None,
) -> tuple[EntryRange, Optional[float], Optional[float], list[float], RiskReward]:
    base_a = atr if atr and atr > 0 else price * 0.02
    bm, lx = _entry_market_adjustment(market_score, vix_last, atr, price)
    a = base_a * bm
    low = price - 0.3 * a - lx * base_a
    high = price + 0.3 * a
    entry_price = (low + high) / 2.0
    return (
        EntryRange(low=round(low, 4), high=round(high, 4)),
        round(entry_price, 4),
        round(price - 1.2 * a, 4),
        [round(price + 1.2 * a, 4), round(price - 1.2 * a, 4)],
        RiskReward(ratio=None, label="wait for clarity"),
    )


def build_stock_analysis(data: FetchedStockData) -> StockAnalysis:
    """Run all analyses and produce one StockAnalysis object."""
    tech: TechnicalSnapshot = compute_technicals(data.history)
    fund = extract_fundamentals(data.info)
    sent = analyze_news_sentiment(data.news, ticker=data.ticker)
    market = get_market_snapshot()
    sess = data.extended_session
    ms = float(market.get("market_score") or 0.0)
    es = float(sess.get("session_score") or 0.0)

    ts = technical_score(tech)
    fs = fundamental_score(fund)
    ss = sentiment_score(sent)
    comb = _combined_score(ts, fs, ss, ms, es)
    direction = _direction_from_score(comb)
    final_direction = direction

    has_fund = fund.pe_ratio is not None or fund.eps_ttm is not None
    conf = _confidence_from_score(comb, has_fund, sent.headline_count)

    price = tech.last_close
    atr = tech.atr_14
    sup, res = tech.support, tech.resistance
    vix = market.get("vix_last")
    vix_f = float(vix) if vix is not None else None
    bm, lx = _entry_market_adjustment(ms, vix_f, atr, price)

    if direction == TradeDirection.BUY:
        planned_range, planned_entry, stop, tps, rr = _levels_buy(price, atr, sup, res, ms, vix_f)
    elif direction == TradeDirection.SELL:
        planned_range, planned_entry, stop, tps, rr = _levels_sell(price, atr, sup, res, ms, vix_f)
    else:
        planned_range, planned_entry, stop, tps, rr = _levels_hold(price, atr, ms, vix_f)

    final_range = planned_range
    final_entry = planned_entry
    premarket_analysis = None

    # Extended-hours adjustment:
    # - Always compute pre/after-hours diagnostics (for display + scoring context).
    # - Apply price-level adjustments:
    #   - For BUY/SELL: adjust using same direction.
    #   - For HOLD: only adjust if extended-hours move is significant; use move sign to pick a
    #     "conditional breakout" direction (gap up -> BUY breakout, gap down -> SELL breakdown).
    afterhours_analysis = None

    ext_hours_influence = 0.0
    if data.extended_hours is not None:
        avg_vol = None
        try:
            avg_vol = float(data.info.get("averageVolume") or data.info.get("averageVolume10days") or 0) or None
        except Exception:
            avg_vol = None
        premarket_analysis = analyze_premarket_conditions(data.extended_hours.premarket, avg_daily_volume=avg_vol)
        afterhours_analysis = analyze_afterhours_conditions(data.extended_hours.afterhours, avg_daily_volume=avg_vol)

        mode, use_pre, use_post = pick_helpful_extended_hours(data.info, premarket_analysis, afterhours_analysis)
        if mode == "PRE" and use_pre is not None:
            implied_dir = direction
            if direction == TradeDirection.HOLD:
                chg = premarket_analysis.premarket_change_percent
                if chg is not None and abs(chg) > 3.0:
                    implied_dir = TradeDirection.BUY if chg > 0 else TradeDirection.SELL
            adj_range, adj_entry, adj_stop, adj_tps, adj_rr, premarket_analysis, wait_open = adjust_trade_levels(
                direction=implied_dir,
                planned_range=planned_range,
                planned_entry_price=planned_entry,
                atr=atr,
                resistance=res,
                support=sup,
                premkt=premarket_analysis,
                significant_move_abs_pct=EXT_HOURS_SIGNIFICANT_MOVE_PCT,
            )
            if not wait_open and adj_entry is not None and adj_stop is not None and adj_tps:
                capped_entry, capped_stop, capped_tps = _cap_ext_hours_levels(adj_entry, adj_stop, adj_tps, price)
                if capped_entry is not None and capped_stop is not None and capped_tps:
                    w = _ext_hours_blend_weight(premarket_analysis.premarket_change_percent, premarket_analysis.premarket_signal, comb)
                    ext_hours_influence = max(ext_hours_influence, w)
                    final_range = EntryRange(
                        low=round(_blend_scalar(planned_range.low, adj_range.low, w) or planned_range.low or 0.0, 4),
                        high=round(_blend_scalar(planned_range.high, adj_range.high, w) or planned_range.high or 0.0, 4),
                    )
                    final_entry = _blend_scalar(planned_entry, capped_entry, w)
                    stop = _blend_scalar(stop, capped_stop, w)
                    tps = [
                        round(_blend_scalar(tps[i], capped_tps[i], w) or capped_tps[i], 4)
                        for i in range(min(len(tps), len(capped_tps)))
                    ]
                    if _should_override_direction(
                        current=direction,
                        implied=implied_dir,
                        change_pct=premarket_analysis.premarket_change_percent,
                        session_signal=premarket_analysis.premarket_signal,
                        combined_score=comb,
                    ):
                        final_direction = implied_dir
            if premarket_analysis.premarket_signal == "WEAK":
                conf = float(max(0.1, min(0.95, conf * 0.92)))
            elif premarket_analysis.premarket_signal == "STRONG":
                conf = float(max(0.1, min(0.95, conf * 1.03)))
        elif mode == "POST" and use_post is not None:
            implied_dir = direction
            if direction == TradeDirection.HOLD:
                chg = afterhours_analysis.afterhours_change_percent if afterhours_analysis else None
                if chg is not None and abs(chg) > 3.0:
                    implied_dir = TradeDirection.BUY if chg > 0 else TradeDirection.SELL
            adj_range, adj_entry, adj_stop, adj_tps, adj_rr, afterhours_analysis, wait_post = adjust_trade_levels_afterhours(
                direction=implied_dir,
                planned_range=planned_range,
                planned_entry_price=planned_entry,
                atr=atr,
                resistance=res,
                support=sup,
                aft=afterhours_analysis,
                significant_move_abs_pct=EXT_HOURS_SIGNIFICANT_MOVE_PCT,
            )
            if not wait_post and adj_entry is not None and adj_stop is not None and adj_tps:
                capped_entry, capped_stop, capped_tps = _cap_ext_hours_levels(adj_entry, adj_stop, adj_tps, price)
                if capped_entry is not None and capped_stop is not None and capped_tps:
                    w = _ext_hours_blend_weight(afterhours_analysis.afterhours_change_percent, afterhours_analysis.afterhours_signal, comb)
                    ext_hours_influence = max(ext_hours_influence, w)
                    final_range = EntryRange(
                        low=round(_blend_scalar(planned_range.low, adj_range.low, w) or planned_range.low or 0.0, 4),
                        high=round(_blend_scalar(planned_range.high, adj_range.high, w) or planned_range.high or 0.0, 4),
                    )
                    final_entry = _blend_scalar(planned_entry, capped_entry, w)
                    stop = _blend_scalar(stop, capped_stop, w)
                    tps = [
                        round(_blend_scalar(tps[i], capped_tps[i], w) or capped_tps[i], 4)
                        for i in range(min(len(tps), len(capped_tps)))
                    ]
                    if _should_override_direction(
                        current=direction,
                        implied=implied_dir,
                        change_pct=afterhours_analysis.afterhours_change_percent,
                        session_signal=afterhours_analysis.afterhours_signal,
                        combined_score=comb,
                    ):
                        final_direction = implied_dir
            if afterhours_analysis.afterhours_signal == "WEAK":
                conf = float(max(0.1, min(0.95, conf * 0.92)))
            elif afterhours_analysis.afterhours_signal == "STRONG":
                conf = float(max(0.1, min(0.95, conf * 1.03)))

    # Apply directional conviction + RSI to level aggressiveness.
    conviction = _conviction_strength(
        final_direction,
        comb,
        ts,
        fs,
        ss,
        ms,
        es,
        tech.rsi_14,
    )
    final_range, final_entry, stop, tps, rr = _apply_level_modifiers(
        final_direction,
        final_range,
        final_entry,
        stop,
        tps,
        price,
        atr,
        conviction,
    )
    if final_direction == TradeDirection.SELL:
        final_range, final_entry, stop, tps, rr = _long_only_levels_for_sell_signal(
            price=price,
            atr=atr,
            support=sup,
            resistance=res,
            market_score=ms,
            vix_last=vix_f,
            conviction=conviction,
            rsi_14=tech.rsi_14,
        )

    summary_points: list[str] = []
    if tech.rsi_14 is not None:
        summary_points.append(f"RSI(14) ≈ {tech.rsi_14:.1f}")
    if tech.sma_50 and tech.sma_200:
        summary_points.append(
            "Price vs 50/200 SMA: "
            f"{'above both' if price > tech.sma_50 > tech.sma_200 else 'mixed / below key MAs'}"
        )
    if fund.revenue_growth is not None:
        summary_points.append(f"Revenue growth (info): {fund.revenue_growth * 100:.1f}%")
    if sent.headline_count:
        summary_points.append(
            f"News sentiment polarity (avg): {sent.mean_polarity:.2f} over {sent.headline_count} snippets"
        )
    else:
        summary_points.append("No recent headlines for sentiment")

    if market.get("vix_last") is not None:
        summary_points.append(f"VIX last ~{float(market['vix_last']):.2f}")
    if market.get("spy_return_5d") is not None:
        summary_points.append(f"SPY 5d return (context): {float(market['spy_return_5d']):+.2f}%")
    summary_points.append(f"Market context score: {ms:+.2f}; session (pre/post) score: {es:+.2f}")
    summary_points.append(
        f"Entry vs market: ATR scale ×{bm:.2f}, extra dip room +{lx:.2f}×ATR (weak tape / VIX / vol)"
    )
    summary_points.append(f"Extended-hours influence weight: {ext_hours_influence:.2f} (capped; blended into planned levels).")
    summary_points.append(f"Level conviction modifier: {conviction:.2f} (includes score alignment + RSI tilt).")
    if final_direction == TradeDirection.SELL:
        summary_points.append("Long-only mode: SELL means wait for a deeper buy entry, then sell into rebound targets.")
    summary_points.append(sess.get("summary") or "Extended hours: n/a")
    if final_direction != direction:
        summary_points.append(f"Direction override from {direction.value} to {final_direction.value} due to significant extended-hours move.")
    if premarket_analysis and premarket_analysis.premarket_change_percent is not None:
        summary_points.append(f"Pre-market change vs prev close: {premarket_analysis.premarket_change_percent:+.2f}% ({premarket_analysis.premarket_signal})")
    if afterhours_analysis and afterhours_analysis.afterhours_change_percent is not None:
        summary_points.append(f"After-hours change vs regular close: {afterhours_analysis.afterhours_change_percent:+.2f}% ({afterhours_analysis.afterhours_signal})")

    rationale = (
        f"Blended score {comb:.2f} (tech {ts:.2f}, fund {fs:.2f}, sent {ss:.2f}, "
        f"market {ms:.2f}, session {es:.2f}). "
        f"Direction {final_direction.value} with confidence {conf:.0%}."
    )
    if final_direction == TradeDirection.SELL:
        rationale += " Long-only interpretation applied: no short plan; levels represent a lower buy zone and rebound exits."

    triggers: list[str] = []
    if tech.sma_200:
        triggers.append(f"Monitor daily close vs 200 SMA (~{tech.sma_200:.2f})")
    if tech.macd is not None and tech.macd_signal is not None:
        triggers.append("Watch MACD cross vs signal line")
    if market.get("vix_last") is not None and float(market["vix_last"]) >= 25:
        triggers.append("Elevated VIX: size down or wait for market stabilization")
    triggers.append(f"Invalidation: stop near {stop}")

    steps: list[str] = [
        f"1. Planned entry zone: {planned_range.low} – {planned_range.high} (mid ~{planned_entry}; last ~{price:.2f}).",
        f"2. Set stop at {stop}; size position so risk fits your rules.",
        f"3. Take profit scale-out at {tps}.",
        "4. Re-evaluate if macro or earnings change the thesis.",
    ]
    if premarket_analysis:
        steps.insert(
            1,
            "1b. Pre-market check: "
            + (premarket_analysis.note or "n/a")
            + (" (entry adjusted)." if final_range != planned_range else " (no entry adjustment)."),
        )
    if afterhours_analysis:
        steps.insert(
            1,
            "1c. After-hours check: "
            + (afterhours_analysis.note or "n/a")
            + (" (entry adjusted)." if final_range != planned_range else " (no entry adjustment)."),
        )

    details: dict[str, Any] = {
        "technical": tech.raw_row,
        "fundamental": fund.raw,
        "sentiment": {
            "mean_polarity": sent.mean_polarity,
            "headline_count": sent.headline_count,
            "sample_headlines": sent.sample_headlines,
        },
        "scores": {
            "technical": ts,
            "fundamental": fs,
            "sentiment": ss,
            "market": ms,
            "session": es,
            "combined": comb,
        },
        "entry_vs_market": {
            "atr_band_multiplier": round(float(bm), 4),
            "extra_low_atr_units": round(float(lx), 4),
            "note": "Wider bands + lower entry when market_score<0, VIX elevated, or stock ATR% high.",
        },
        "market": market,
        "extended_session": sess,
        "premarket": premarket_analysis.model_dump(mode="json") if premarket_analysis else {},
        "afterhours": afterhours_analysis.model_dump(mode="json") if afterhours_analysis else {},
    }

    # Keep planned vs adjusted zones in details for transparency/debugging.
    details["entry_ranges"] = {
        "planned": {"low": planned_range.low, "high": planned_range.high},
        "final": {"low": final_range.low, "high": final_range.high},
    }

    return StockAnalysis(
        ticker=data.ticker,
        direction=final_direction,
        confidence=conf,
        entry_range=final_range,
        entry_price=final_entry,
        stop_loss=stop,
        take_profits=tps,
        risk_reward=rr,
        premarket_analysis=premarket_analysis,
        afterhours_analysis=afterhours_analysis,
        rationale=rationale,
        summary_points=summary_points,
        monitoring_triggers=triggers,
        plan_steps=steps,
        details=details,
    )


def analyze_ticker(ticker: str) -> StockAnalysis:
    """Fetch data and return analysis for one symbol."""
    data = fetch_stock_data(ticker)
    return build_stock_analysis(data)


def analyze_tickers(tickers: list[str]) -> tuple[list[StockAnalysis], list[str]]:
    """Analyze multiple tickers; collect per-ticker errors without failing the whole batch."""
    results: list[StockAnalysis] = []
    errors: list[str] = []
    for raw in tickers:
        t = raw.strip().upper()
        if not t:
            continue
        try:
            results.append(analyze_ticker(t))
        except Exception as e:
            errors.append(f"{t}: {e}")
    return results, errors
