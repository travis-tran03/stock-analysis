"""Shared batch analysis: normalize tickers, run signals pipeline, return AnalyzeResponse.

Used by the FastAPI app and by the Streamlit UI (single process, no HTTP).
"""

from __future__ import annotations

from backend.data_fetch import normalize_ticker, validate_ticker_symbol
from backend.schemas import AnalyzeResponse
from backend.signals import analyze_tickers


def run_analyze(raw_tickers: list[str]) -> AnalyzeResponse:
    """
    Validate and dedupe tickers, run analysis, return results + per-ticker errors.

    Raises:
        ValueError: No valid tickers after normalization (maps to HTTP 422 in API).
        RuntimeError: Every ticker failed to analyze (maps to HTTP 502 in API).
    """
    normalized: list[str] = []
    parse_errors: list[str] = []
    for raw in raw_tickers:
        t = normalize_ticker(raw)
        if not t:
            continue
        if not validate_ticker_symbol(t):
            parse_errors.append(f"{t}: invalid symbol format")
            continue
        normalized.append(t)

    seen: set[str] = set()
    tickers: list[str] = []
    for t in normalized:
        if t not in seen:
            seen.add(t)
            tickers.append(t)

    if not tickers:
        msg = "No valid tickers supplied."
        if parse_errors:
            msg += " " + "; ".join(parse_errors)
        raise ValueError(msg)

    results, fetch_errors = analyze_tickers(tickers)
    all_errors = parse_errors + fetch_errors

    if not results and all_errors:
        raise RuntimeError("Could not analyze any ticker. " + " | ".join(all_errors))

    return AnalyzeResponse(results=results, errors=all_errors)
