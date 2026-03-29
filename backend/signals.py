"""Combine technical, fundamental, and sentiment inputs into trade signals and levels."""

from __future__ import annotations

from typing import Any, Optional

from backend.data_fetch import FetchedStockData, fetch_stock_data
from backend.fundamental import extract_fundamentals, fundamental_score
from backend.market_session import get_market_snapshot
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


def _confidence_from_score(s: float, has_fundamentals: bool, news_count: int) -> float:
    base = min(1.0, abs(s) * 1.4 + 0.15)
    if not has_fundamentals:
        base *= 0.88
    if news_count == 0:
        base *= 0.9
    return float(max(0.1, min(0.95, base)))


def _levels_buy(
    price: float,
    atr: Optional[float],
    support: Optional[float],
    resistance: Optional[float],
) -> tuple[EntryRange, Optional[float], list[float], RiskReward]:
    """Long-oriented entry zone, stop under support/ATR, targets toward resistance/ATR multiples."""
    a = atr if atr and atr > 0 else price * 0.02
    low = price - 0.35 * a
    high = price + 0.25 * a
    if support and support < price:
        low = max(low, support - 0.1 * a)
    entry_mid = (low + high) / 2.0
    stop = (support - 0.5 * a) if support else (price - 1.5 * a)
    stop = min(stop, price - 0.5 * a)

    tp1 = price + 1.5 * a
    tp2 = price + 2.5 * a
    if resistance and resistance > price:
        tp1 = min(tp1, resistance - 0.05 * a)
        tp2 = max(tp2, min(resistance + 0.5 * a, price + 3.5 * a))

    risk = entry_mid - stop
    reward = max(tp1 - entry_mid, a)
    rr = (reward / risk) if risk and risk > 0 else None
    label = f"~{rr:.2f}:1" if rr else "n/a"

    return (
        EntryRange(low=round(low, 4), high=round(high, 4)),
        round(stop, 4),
        [round(tp1, 4), round(tp2, 4)],
        RiskReward(ratio=round(rr, 3) if rr else None, label=label),
    )


def _levels_sell(
    price: float,
    atr: Optional[float],
    support: Optional[float],
    resistance: Optional[float],
) -> tuple[EntryRange, Optional[float], list[float], RiskReward]:
    """Short-oriented: entry near resistance, stop above, targets lower."""
    a = atr if atr and atr > 0 else price * 0.02
    low = price - 0.25 * a
    high = price + 0.35 * a
    if resistance and resistance > price:
        high = min(high, resistance + 0.1 * a)
    entry_mid = (low + high) / 2.0
    stop = (resistance + 0.5 * a) if resistance else (price + 1.5 * a)
    stop = max(stop, price + 0.5 * a)

    tp1 = price - 1.5 * a
    tp2 = price - 2.5 * a
    if support and support < price:
        tp1 = max(tp1, support + 0.05 * a)

    risk = stop - entry_mid
    reward = max(entry_mid - tp1, a)
    rr = (reward / risk) if risk and risk > 0 else None
    label = f"~{rr:.2f}:1" if rr else "n/a"

    return (
        EntryRange(low=round(low, 4), high=round(high, 4)),
        round(stop, 4),
        [round(tp1, 4), round(tp2, 4)],
        RiskReward(ratio=round(rr, 3) if rr else None, label=label),
    )


def _levels_hold(
    price: float,
    atr: Optional[float],
) -> tuple[EntryRange, Optional[float], list[float], RiskReward]:
    a = atr if atr and atr > 0 else price * 0.02
    low = price - 0.3 * a
    high = price + 0.3 * a
    return (
        EntryRange(low=round(low, 4), high=round(high, 4)),
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

    has_fund = fund.pe_ratio is not None or fund.eps_ttm is not None
    conf = _confidence_from_score(comb, has_fund, sent.headline_count)

    price = tech.last_close
    atr = tech.atr_14
    sup, res = tech.support, tech.resistance

    if direction == TradeDirection.BUY:
        entry, stop, tps, rr = _levels_buy(price, atr, sup, res)
    elif direction == TradeDirection.SELL:
        entry, stop, tps, rr = _levels_sell(price, atr, sup, res)
    else:
        entry, stop, tps, rr = _levels_hold(price, atr)

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
    summary_points.append(sess.get("summary") or "Extended hours: n/a")

    rationale = (
        f"Blended score {comb:.2f} (tech {ts:.2f}, fund {fs:.2f}, sent {ss:.2f}, "
        f"market {ms:.2f}, session {es:.2f}). "
        f"Direction {direction.value} with confidence {conf:.0%}."
    )

    triggers: list[str] = []
    if tech.sma_200:
        triggers.append(f"Monitor daily close vs 200 SMA (~{tech.sma_200:.2f})")
    if tech.macd is not None and tech.macd_signal is not None:
        triggers.append("Watch MACD cross vs signal line")
    if market.get("vix_last") is not None and float(market["vix_last"]) >= 25:
        triggers.append("Elevated VIX: size down or wait for market stabilization")
    triggers.append(f"Invalidation: stop near {stop}")

    steps: list[str] = [
        f"1. Plan entry between {entry.low} and {entry.high} (last ~{price:.2f}).",
        f"2. Set stop at {stop}; size position so risk fits your rules.",
        f"3. Take profit scale-out at {tps}.",
        "4. Re-evaluate if macro or earnings change the thesis.",
    ]

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
        "market": market,
        "extended_session": sess,
    }

    return StockAnalysis(
        ticker=data.ticker,
        direction=direction,
        confidence=conf,
        entry_range=entry,
        stop_loss=stop,
        take_profits=tps,
        risk_reward=rr,
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
