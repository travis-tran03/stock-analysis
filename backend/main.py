"""FastAPI application: CORS for local Streamlit, POST /analyze for batch ticker analysis."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.data_fetch import normalize_ticker, validate_ticker_symbol
from backend.schemas import AnalyzeRequest, AnalyzeResponse
from backend.signals import analyze_tickers

app = FastAPI(title="Stock Trade Analysis API", version="1.0.0")

# Allow Streamlit (default 8501) and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """
    Accept a list of tickers, run technical / fundamental / sentiment pipeline,
    return structured trade ideas per symbol plus any per-ticker error messages.
    """
    normalized: list[str] = []
    parse_errors: list[str] = []
    for raw in req.tickers:
        t = normalize_ticker(raw)
        if not t:
            continue
        if not validate_ticker_symbol(t):
            parse_errors.append(f"{t}: invalid symbol format")
            continue
        normalized.append(t)

    # Deduplicate preserving order
    seen: set[str] = set()
    tickers: list[str] = []
    for t in normalized:
        if t not in seen:
            seen.add(t)
            tickers.append(t)

    if not tickers:
        raise HTTPException(
            status_code=422,
            detail="No valid tickers supplied. " + ("; ".join(parse_errors) if parse_errors else ""),
        )

    results, fetch_errors = analyze_tickers(tickers)
    all_errors = parse_errors + fetch_errors

    if not results and all_errors:
        raise HTTPException(
            status_code=502,
            detail="Could not analyze any ticker. " + " | ".join(all_errors),
        )

    return AnalyzeResponse(results=results, errors=all_errors)
