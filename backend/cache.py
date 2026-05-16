"""Per-ticker analysis file cache with TTL."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from backend.schemas import StockAnalysis

_CACHE_ROOT = Path(__file__).resolve().parent.parent / ".cache" / "stock_analysis"
DEFAULT_TTL_SECONDS = 24 * 3600


def _cache_path(ticker: str) -> Path:
    safe = ticker.upper().replace("/", "_")
    return _CACHE_ROOT / f"{safe}.json"


def get_cached_analysis(ticker: str, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> Optional[StockAnalysis]:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    try:
        age = time.time() - path.stat().st_mtime
        if age > ttl_seconds:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return StockAnalysis.model_validate(data)
    except Exception:
        return None


def set_cached_analysis(analysis: StockAnalysis) -> None:
    path = _cache_path(analysis.ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(analysis.model_dump_json(), encoding="utf-8")
