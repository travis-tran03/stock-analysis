"""Fetch OHLCV, fundamentals metadata, and news from Yahoo Finance via yfinance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
from backend.market_session import extended_session_from_info
from backend.premarket import ExtendedHoursFetchResult, fetch_extended_hours_data


def normalize_ticker(raw: str) -> str:
    """Uppercase, strip, and remove common separators."""
    s = raw.strip().upper()
    s = re.sub(r"\s+", "", s)
    return s


def validate_ticker_symbol(symbol: str) -> bool:
    """Basic validation: non-empty alphanumeric (allow dots for BRK.B)."""
    return bool(symbol) and bool(re.match(r"^[A-Z0-9.\-^]+$", symbol))


def _flatten_yfinance_news_item(raw: dict[str, Any]) -> dict[str, Any]:
    """
    yfinance often returns nested `content: {title, summary, ...}` instead of top-level keys.
    Copy title/summary up so sentiment and display logic can use a single shape.
    """
    out = dict(raw)
    c = raw.get("content")
    if isinstance(c, dict):
        if not out.get("title"):
            out["title"] = c.get("title") or ""
        if not out.get("summary"):
            out["summary"] = c.get("summary") or ""
    return out


@dataclass
class FetchedStockData:
    ticker: str
    history: pd.DataFrame
    info: dict[str, Any]
    news: list[dict[str, Any]]
    extended_session: dict[str, Any]
    extended_hours: Optional[ExtendedHoursFetchResult] = None


def fetch_stock_data(ticker: str, history_period: str = "2y") -> FetchedStockData:
    """
    Download daily OHLCV (long enough for 200-day MA), company info, and recent news.
    Raises ValueError if data is unusable (empty history).
    """
    t = yf.Ticker(ticker)
    # Daily bars for technical indicators
    hist = t.history(period=history_period, interval="1d", auto_adjust=True)
    if hist is None or hist.empty:
        raise ValueError(f"No price history for {ticker}")

    hist = hist.dropna(how="all")
    if hist.empty:
        raise ValueError(f"No price history for {ticker}")

    # Normalize column names to lowercase for technical analysis
    hist.columns = [str(c).lower() for c in hist.columns]

    info: dict[str, Any] = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    news: list[dict[str, Any]] = []
    try:
        raw_news = getattr(t, "news", None)
        if raw_news is None:
            raw_news = []
        if isinstance(raw_news, list):
            news = [_flatten_yfinance_news_item(x) for x in raw_news if isinstance(x, dict)]
    except Exception:
        news = []

    ext_sess = extended_session_from_info(info)

    ext_hours: Optional[ExtendedHoursFetchResult] = None
    try:
        ext_hours = fetch_extended_hours_data(ticker, info)
    except Exception:
        ext_hours = None

    return FetchedStockData(
        ticker=ticker,
        history=hist,
        info=info,
        news=news,
        extended_session=ext_sess,
        extended_hours=ext_hours,
    )
