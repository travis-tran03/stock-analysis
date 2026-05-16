"""Stock universes for long-term and short-term scans (both use major US indexes)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

from backend.data_fetch import normalize_ticker, validate_ticker_symbol
from backend.index_constituents import (
    build_major_index_universe,
    major_index_universe_summary,
)

logger = logging.getLogger(__name__)

Horizon = Literal["long", "short"]

_UNIVERSES_DIR = Path(__file__).resolve().parent.parent / "data" / "universes"
LARGE_CAP_FILE = _UNIVERSES_DIR / "large_cap_100.txt"
SHORT_TERM_CANDIDATES_FILE = _UNIVERSES_DIR / "short_term_index_candidates.txt"
SHORT_TERM_META_FILE = _UNIVERSES_DIR / "short_term_index_candidates.meta.json"

# Includes S&P 400 mid-caps; index members below this are skipped at scan time.
MIN_LARGE_CAP_MARKET_CAP = 2_000_000_000
MIN_LARGE_CAP_AVG_VOLUME = 500_000
MIN_SHORT_AVG_VOLUME = 300_000

SHORT_TERM_CANDIDATE_CACHE_TTL = 24 * 3600
DEFAULT_SHORT_SCAN_SIZE = 200
_DOWNLOAD_CHUNK = 60


def _load_ticker_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    seen: set[str] = set()
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        t = normalize_ticker(line.split("#")[0].strip())
        if not t or not validate_ticker_symbol(t) or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def load_large_cap_universe(*, force_refresh: bool = False) -> list[str]:
    """
    US major index constituents (merged, deduped):
    S&P 500, S&P 400, S&P 600, NASDAQ-100, Dow Jones Industrial Average.
    """
    tickers = build_major_index_universe(force_refresh=force_refresh)
    if tickers:
        return tickers
    return _load_ticker_file(LARGE_CAP_FILE)


def _composite_momentum_score(close: pd.Series) -> Optional[float]:
    """Higher = stronger recent price trend (for ranking index members)."""
    s = close.dropna()
    if len(s) < 22:
        return None
    last = float(s.iloc[-1])
    if last <= 0:
        return None

    ret_1m: Optional[float] = None
    if len(s) >= 22:
        base = float(s.iloc[-22])
        if base > 0:
            ret_1m = (last / base - 1.0) * 100.0

    ret_3m: Optional[float] = None
    if len(s) >= 64:
        base = float(s.iloc[-64])
        if base > 0:
            ret_3m = (last / base - 1.0) * 100.0

    if ret_3m is None and ret_1m is None:
        return None
    if ret_3m is None:
        return ret_1m
    if ret_1m is None:
        return ret_3m
    return 0.65 * ret_3m + 0.35 * ret_1m


def _close_series_for_ticker(raw: pd.DataFrame, ticker: str) -> Optional[pd.Series]:
    if raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(0):
            sub = raw[ticker]
            if isinstance(sub, pd.DataFrame) and "Close" in sub.columns:
                return sub["Close"]
        if "Close" in raw.columns.get_level_values(0) and ticker in raw["Close"].columns:
            return raw["Close"][ticker]
        return None
    if "Close" in raw.columns:
        return raw["Close"]
    return None


def _momentum_scores_for_chunk(tickers: list[str]) -> dict[str, float]:
    import yfinance as yf

    scores: dict[str, float] = {}
    if not tickers:
        return scores

    try:
        raw = yf.download(
            tickers,
            period="6mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )
    except Exception as e:
        logger.debug("Chunk download failed: %s", e)
        raw = pd.DataFrame()

    if len(tickers) == 1:
        close = _close_series_for_ticker(raw, tickers[0]) if not raw.empty else None
        if close is None and not raw.empty and "Close" in raw.columns:
            close = raw["Close"]
        if close is not None:
            m = _composite_momentum_score(close)
            if m is not None:
                scores[tickers[0]] = m
        return scores

    for sym in tickers:
        close = _close_series_for_ticker(raw, sym)
        if close is None:
            continue
        m = _composite_momentum_score(close)
        if m is not None:
            scores[sym] = m
    return scores


def _rank_index_by_short_term_momentum(
    index_tickers: list[str],
    max_symbols: int,
) -> list[str]:
    """Pick index members with the strongest 1–3 month price momentum."""
    ranked: list[tuple[float, str]] = []
    for start in range(0, len(index_tickers), _DOWNLOAD_CHUNK):
        chunk = index_tickers[start : start + _DOWNLOAD_CHUNK]
        batch = _momentum_scores_for_chunk(chunk)
        for sym, score in batch.items():
            ranked.append((score, sym))
        time.sleep(0.15)

    if not ranked:
        return index_tickers[:max_symbols]

    ranked.sort(key=lambda x: x[0], reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for _, sym in ranked:
        if sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
        if len(out) >= max_symbols:
            break
    return out


def _short_term_cache_fresh() -> bool:
    if not SHORT_TERM_META_FILE.exists() or not SHORT_TERM_CANDIDATES_FILE.exists():
        return False
    try:
        meta = json.loads(SHORT_TERM_META_FILE.read_text(encoding="utf-8"))
        age = time.time() - float(meta.get("built_at", 0))
        return age < SHORT_TERM_CANDIDATE_CACHE_TTL
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return False


def _write_short_term_cache(tickers: list[str], index_count: int) -> None:
    SHORT_TERM_CANDIDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SHORT_TERM_CANDIDATES_FILE.write_text("\n".join(tickers) + "\n", encoding="utf-8")
    SHORT_TERM_META_FILE.write_text(
        json.dumps(
            {
                "built_at": time.time(),
                "count": len(tickers),
                "index_pool": index_count,
                "source": major_index_universe_summary(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_short_term_index_universe(
    max_symbols: int = DEFAULT_SHORT_SCAN_SIZE,
    *,
    force_refresh: bool = False,
) -> list[str]:
    """
    Same major indexes as long-term, narrowed to names with the strongest
    short-term price momentum (1–3 month) before full scoring.
    """
    if not force_refresh and _short_term_cache_fresh():
        cached = _load_ticker_file(SHORT_TERM_CANDIDATES_FILE)
        if cached:
            return cached[:max_symbols]

    pool = load_large_cap_universe()
    if not pool:
        return []

    candidates = _rank_index_by_short_term_momentum(pool, max_symbols=max_symbols)
    if candidates:
        _write_short_term_cache(candidates, index_count=len(pool))
        return candidates

    logger.warning("Short-term momentum rank failed; using index pool slice")
    return pool[:max_symbols]


def short_term_universe_summary() -> str:
    meta: Optional[dict] = None
    if SHORT_TERM_META_FILE.exists():
        try:
            meta = json.loads(SHORT_TERM_META_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            meta = None
    if meta:
        return (
            f"Top {meta.get('count', '?')} short-term momentum picks from "
            f"{meta.get('index_pool', '?')} index symbols ({meta.get('source', 'major indexes')})"
        )
    return (
        f"Top {DEFAULT_SHORT_SCAN_SIZE} by 1–3 month momentum from major US indexes "
        f"({major_index_universe_summary()})"
    )


def get_universe(horizon: Horizon, max_short: int = DEFAULT_SHORT_SCAN_SIZE) -> list[str]:
    if horizon == "long":
        return load_large_cap_universe()
    return load_short_term_index_universe(max_symbols=max_short)


def passes_large_cap_liquidity_filter(info: dict) -> bool:
    """Post-fetch filter for long-term universe."""
    try:
        cap = info.get("marketCap")
        if cap is not None and float(cap) < MIN_LARGE_CAP_MARKET_CAP:
            return False
        vol = info.get("averageVolume") or info.get("averageVolume10days")
        if vol is not None and float(vol) < MIN_LARGE_CAP_AVG_VOLUME:
            return False
    except (TypeError, ValueError):
        pass
    return True
