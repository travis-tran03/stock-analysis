"""Scan universes and rank horizon-specific buy recommendations."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Optional

from backend.cache import get_cached_analysis, set_cached_analysis
from backend.data_fetch import fetch_stock_data
from backend.schemas import RecommendationPick, RecommendationsResponse, StockAnalysis
from backend.scoring_profiles import Horizon
from backend.signals import build_stock_analysis
from backend.universe import get_universe_with_info, passes_large_cap_liquidity_filter

THROTTLE_SECONDS = 0.4
SCORE_KEY: dict[Horizon, str] = {"long": "long_term", "short": "short_term"}
DIRECTION_KEY: dict[Horizon, str] = {"long": "long_term", "short": "short_term"}


def _horizon_score(analysis: StockAnalysis, horizon: Horizon) -> float:
    scores = (analysis.details or {}).get("scores") or {}
    return float(scores.get(SCORE_KEY[horizon]) or 0.0)


def _horizon_direction(analysis: StockAnalysis, horizon: Horizon) -> str:
    hd = (analysis.details or {}).get("horizon_direction") or {}
    return str(hd.get(DIRECTION_KEY[horizon]) or analysis.direction.value)


def _analyze_with_cache(ticker: str, *, apply_large_cap_filter: bool) -> StockAnalysis:
    cached = get_cached_analysis(ticker)
    if cached is not None:
        return cached
    data = fetch_stock_data(ticker)
    if apply_large_cap_filter and not passes_large_cap_liquidity_filter(data.info):
        raise ValueError("failed liquidity filter")
    analysis = build_stock_analysis(data)
    set_cached_analysis(analysis)
    return analysis


def process_ticker_for_recommendations(
    ticker: str,
    horizon: Horizon,
    min_confidence: float,
    min_rr: Optional[float] = None,
) -> tuple[Optional[tuple[float, StockAnalysis]], Optional[str], bool]:
    """
    Analyze one symbol for the recommendation scan.

    Returns:
        (rank_score, analysis) if it qualifies as a BUY candidate, else None;
        error message if any;
        scanned_increment (True if analysis ran).
    """
    try:
        analysis = _analyze_with_cache(
            ticker, apply_large_cap_filter=(horizon == "long")
        )
        direction = _horizon_direction(analysis, horizon)
        if direction != "BUY":
            return None, None, True
        if analysis.confidence < min_confidence:
            return None, None, True
        if min_rr is not None:
            rr = analysis.risk_reward.ratio
            if rr is None or rr < min_rr:
                return None, None, True
        score = _horizon_score(analysis, horizon)
        rank = score * analysis.confidence
        return (rank, analysis), None, True
    except ValueError as e:
        if "liquidity filter" in str(e).lower():
            return None, None, False
        return None, f"{ticker}: {e}", False
    except Exception as e:
        return None, f"{ticker}: {e}", False


def finalize_recommendations(
    horizon: Horizon,
    top_n: int,
    universe_size: int,
    candidates: list[tuple[float, StockAnalysis]],
    errors: list[str],
    scanned_count: int,
    *,
    cancelled: bool = False,
    universe_source: str = "",
    universe_source_message: str = "",
) -> RecommendationsResponse:
    candidates.sort(key=lambda x: x[0], reverse=True)
    picks: list[RecommendationPick] = []
    for rank_val, analysis in candidates[:top_n]:
        score = _horizon_score(analysis, horizon)
        picks.append(
            RecommendationPick(
                ticker=analysis.ticker,
                direction=_horizon_direction(analysis, horizon),
                horizon_score=score,
                confidence=analysis.confidence,
                rank_score=rank_val,
                analysis=analysis,
            )
        )
    return RecommendationsResponse(
        horizon=horizon,
        top_n=top_n,
        scanned_count=scanned_count,
        universe_size=universe_size,
        picks=picks,
        errors=errors,
        as_of=datetime.now(timezone.utc).isoformat(),
        cancelled=cancelled,
        universe_source=universe_source,
        universe_source_message=universe_source_message,
    )


def run_recommendations(
    horizon: Horizon,
    top_n: int = 10,
    min_confidence: float = 0.5,
    min_rr: Optional[float] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> RecommendationsResponse:
    """
    Scan universe, score each ticker, return top BUY picks for the horizon.
  """
    universe = get_universe_with_info(horizon)
    tickers = universe.tickers
    total = len(tickers)
    u_src, u_msg = universe.source, universe.message
    scanned = 0
    errors: list[str] = []
    candidates: list[tuple[float, StockAnalysis]] = []

    for i, ticker in enumerate(tickers, start=1):
        if should_stop and should_stop():
            return finalize_recommendations(
                horizon,
                top_n,
                total,
                candidates,
                errors + ["Scan stopped by user."],
                scanned,
                cancelled=True,
                universe_source=u_src,
                universe_source_message=u_msg,
            )
        if progress_callback:
            progress_callback(i, total, ticker)
        candidate, err, counted = process_ticker_for_recommendations(
            ticker, horizon, min_confidence, min_rr
        )
        if counted:
            scanned += 1
        if err:
            errors.append(err)
        if candidate is not None:
            candidates.append(candidate)
        time.sleep(THROTTLE_SECONDS)

    return finalize_recommendations(
        horizon,
        top_n,
        total,
        candidates,
        errors,
        scanned,
        cancelled=False,
        universe_source=u_src,
        universe_source_message=u_msg,
    )
