"""Fetch OHLCV, fundamentals metadata, and news from Yahoo Finance via yfinance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import yfinance as yf


def normalize_ticker(raw: str) -> str:
    """Uppercase, strip, and remove common separators."""
    s = raw.strip().upper()
    s = re.sub(r"\s+", "", s)
    return s


def validate_ticker_symbol(symbol: str) -> bool:
    """Basic validation: non-empty alphanumeric (allow dots for BRK.B)."""
    return bool(symbol) and bool(re.match(r"^[A-Z0-9.\-^]+$", symbol))


@dataclass
class FetchedStockData:
    ticker: str
    history: pd.DataFrame
    info: dict[str, Any]
    news: list[dict[str, Any]]


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

    # Normalize column names to lowercase for consistency with pandas_ta
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
            news = raw_news
    except Exception:
        news = []

    return FetchedStockData(ticker=ticker, history=hist, info=info, news=news)
