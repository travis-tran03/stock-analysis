"""Merge US major index constituents (Wikipedia) with disk cache."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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

# wikipedia_fresh | cache_fresh | cache_stale | bundled_fallback
SOURCE_WIKIPEDIA_FRESH = "wikipedia_fresh"
SOURCE_CACHE_FRESH = "cache_fresh"
SOURCE_CACHE_STALE = "cache_stale"
SOURCE_BUNDLED_FALLBACK = "bundled_fallback"


@dataclass
class IndexUniverseBuildResult:
    tickers: list[str]
    source: str
    message: str
    built_at: Optional[float] = None
    symbol_count: int = 0


def _format_built_at(ts: Optional[float]) -> str:
    if ts is None:
        return "unknown date"
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except (OSError, OverflowError, ValueError):
        return "unknown date"


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


def build_major_index_universe_with_info(
    *, force_refresh: bool = False
) -> IndexUniverseBuildResult:
    """
    Merge constituents from major US equity indexes:
    S&P 500, S&P MidCap 400, S&P SmallCap 600, NASDAQ-100, Dow Jones Industrial Average.
    """
    meta = _read_cache_meta()

    if not force_refresh:
        cached = _load_ticker_file(MERGED_CACHE_FILE)
        if cached and _cache_is_fresh():
            built = float(meta["built_at"]) if meta and meta.get("built_at") else None
            return IndexUniverseBuildResult(
                tickers=cached,
                source=SOURCE_CACHE_FRESH,
                built_at=built,
                symbol_count=len(cached),
                message=(
                    f"Using on-disk index list ({len(cached)} symbols, last built "
                    f"{_format_built_at(built)}). Wikipedia was not called (cache still fresh)."
                ),
            )

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
        built = time.time()
        logger.info(
            "Built major-index universe: %d symbols (%s)",
            len(merged),
            ", ".join(f"{k}={v}" for k, v in sources.items()),
        )
        return IndexUniverseBuildResult(
            tickers=merged,
            source=SOURCE_WIKIPEDIA_FRESH,
            built_at=built,
            symbol_count=len(merged),
            message=(
                f"Fetched fresh index constituents from Wikipedia ({len(merged)} symbols, "
                f"{_format_built_at(built)})."
            ),
        )

    stale = _load_ticker_file(MERGED_CACHE_FILE)
    if stale:
        built = float(meta["built_at"]) if meta and meta.get("built_at") else None
        logger.warning("Major-index fetch failed; using existing merged cache (%d symbols)", len(stale))
        return IndexUniverseBuildResult(
            tickers=stale,
            source=SOURCE_CACHE_STALE,
            built_at=built,
            symbol_count=len(stale),
            message=(
                f"Wikipedia fetch failed; using bundled/cached merged index list ({len(stale)} symbols, "
                f"file dated {_format_built_at(built)})."
            ),
        )

    fallback = _load_ticker_file(FALLBACK_FILE)
    logger.warning("Major-index fetch produced no symbols; using bundled fallback list")
    return IndexUniverseBuildResult(
        tickers=fallback,
        source=SOURCE_BUNDLED_FALLBACK,
        symbol_count=len(fallback),
        message=(
            f"Index fetch failed; using small bundled fallback list ({len(fallback)} symbols from "
            f"{FALLBACK_FILE.name})."
        ),
    )


def build_major_index_universe(*, force_refresh: bool = False) -> list[str]:
    return build_major_index_universe_with_info(force_refresh=force_refresh).tickers


def peek_index_universe_provenance() -> str:
    """Fast hint (no Wikipedia) for what list the next long scan would likely use."""
    meta = _read_cache_meta()
    merged = _load_ticker_file(MERGED_CACHE_FILE)
    if merged and _cache_is_fresh():
        built = float(meta["built_at"]) if meta and meta.get("built_at") else None
        return (
            f"Likely source: fresh on-disk cache ({len(merged)} symbols, "
            f"{_format_built_at(built)})."
        )
    if merged:
        built = float(meta["built_at"]) if meta and meta.get("built_at") else None
        return (
            f"Likely source: merged list in app ({len(merged)} symbols, "
            f"{_format_built_at(built)}). Scan will try Wikipedia first."
        )
    fallback = _load_ticker_file(FALLBACK_FILE)
    if fallback:
        return (
            f"No merged list on disk. Scan will try Wikipedia, or fall back to "
            f"{len(fallback)} symbols in {FALLBACK_FILE.name}."
        )
    return "Scan will try Wikipedia for index constituents."


def major_index_universe_summary() -> str:
    """Human-readable description of which indexes are included."""
    meta = _read_cache_meta()
    if meta and meta.get("indexes"):
        n = meta.get("count", "?")
        names = ", ".join(str(i).upper() for i in meta["indexes"])
        return f"{n} symbols from {names}"
    labels = ", ".join(label.upper() for label, _ in MAJOR_INDEX_SOURCES)
    return f"Major US indexes: {labels}"
