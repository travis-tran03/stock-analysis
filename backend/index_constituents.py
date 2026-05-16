"""Merge US major index constituents (Wikipedia) with disk cache."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from backend.data_fetch import normalize_ticker, validate_ticker_symbol

logger = logging.getLogger(__name__)

_UNIVERSES_DIR = Path(__file__).resolve().parent.parent / "data" / "universes"
MERGED_CACHE_FILE = _UNIVERSES_DIR / "major_indexes_merged.txt"
MERGED_META_FILE = _UNIVERSES_DIR / "major_indexes_merged.meta.json"
FALLBACK_FILE = _UNIVERSES_DIR / "large_cap_100.txt"

CACHE_TTL_SECONDS = 7 * 24 * 3600

_WIKI_HEADERS = {
    "User-Agent": "StockApp/1.0 (index-constituent research; +https://github.com/)",
}

# label, Wikipedia URL
MAJOR_INDEX_SOURCES: list[tuple[str, str]] = [
    ("sp500", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
    ("sp400", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"),
    ("sp600", "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"),
    ("nasdaq100", "https://en.wikipedia.org/wiki/Nasdaq-100"),
    ("dow", "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"),
]

_SYMBOL_COLUMNS = ("Symbol", "Ticker", "Company Ticker")


def _normalize_wiki_symbol(raw: str) -> str:
    # Wikipedia uses BRK.B; Yahoo uses BRK-B
    cleaned = str(raw).strip().upper().replace(".", "-")
    return normalize_ticker(cleaned)


def _extract_symbols_from_frame(df: pd.DataFrame) -> list[str]:
    col: Optional[str] = None
    for name in _SYMBOL_COLUMNS:
        if name in df.columns:
            col = name
            break
    if col is None:
        return []

    out: list[str] = []
    for raw in df[col].astype(str):
        sym = _normalize_wiki_symbol(raw)
        if sym and validate_ticker_symbol(sym):
            out.append(sym)
    return out


def _fetch_index_symbols(label: str, url: str) -> list[str]:
    """Pick the Wikipedia table that looks most like an index constituent list."""
    try:
        tables = pd.read_html(url, storage_options=_WIKI_HEADERS)
    except Exception as e:
        logger.warning("Wikipedia read failed for %s: %s", label, e)
        return []

    best: list[str] = []
    for df in tables:
        syms = _extract_symbols_from_frame(df)
        # Dow ~30, Nasdaq-100 ~100, S&P 500 ~500 — prefer the largest plausible list
        if len(syms) > len(best):
            best = syms
    return best


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


def _read_cache_meta() -> Optional[dict[str, Any]]:
    if not MERGED_META_FILE.exists():
        return None
    try:
        return json.loads(MERGED_META_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _cache_is_fresh() -> bool:
    meta = _read_cache_meta()
    if not meta or not MERGED_CACHE_FILE.exists():
        return False
    try:
        age = time.time() - float(meta.get("built_at", 0))
        return age < CACHE_TTL_SECONDS
    except (TypeError, ValueError):
        return False


def _write_merged_cache(tickers: list[str], sources: dict[str, int]) -> None:
    MERGED_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    MERGED_CACHE_FILE.write_text("\n".join(tickers) + "\n", encoding="utf-8")
    MERGED_META_FILE.write_text(
        json.dumps(
            {
                "built_at": time.time(),
                "count": len(tickers),
                "sources": sources,
                "indexes": [label for label, _ in MAJOR_INDEX_SOURCES],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_major_index_universe(*, force_refresh: bool = False) -> list[str]:
    """
    Merge constituents from major US equity indexes:
    S&P 500, S&P MidCap 400, S&P SmallCap 600, NASDAQ-100, Dow Jones Industrial Average.
    """
    if not force_refresh:
        cached = _load_ticker_file(MERGED_CACHE_FILE)
        if cached and _cache_is_fresh():
            return cached

    seen: set[str] = set()
    merged: list[str] = []
    sources: dict[str, int] = {}

    for label, url in MAJOR_INDEX_SOURCES:
        batch = _fetch_index_symbols(label, url)
        sources[label] = len(batch)
        for sym in batch:
            if sym not in seen:
                seen.add(sym)
                merged.append(sym)

    if merged:
        _write_merged_cache(merged, sources)
        logger.info(
            "Built major-index universe: %d symbols (%s)",
            len(merged),
            ", ".join(f"{k}={v}" for k, v in sources.items()),
        )
        return merged

    stale = _load_ticker_file(MERGED_CACHE_FILE)
    if stale:
        logger.warning("Major-index fetch failed; using existing merged cache (%d symbols)", len(stale))
        return stale

    logger.warning("Major-index fetch produced no symbols; using bundled fallback list")
    return _load_ticker_file(FALLBACK_FILE)


def major_index_universe_summary() -> str:
    """Human-readable description of which indexes are included."""
    meta = _read_cache_meta()
    if meta and meta.get("indexes"):
        n = meta.get("count", "?")
        names = ", ".join(str(i).upper() for i in meta["indexes"])
        return f"{n} symbols from {names}"
    labels = ", ".join(label.upper() for label, _ in MAJOR_INDEX_SOURCES)
    return f"Major US indexes: {labels}"
