"""FastAPI application: optional REST API wrapping the same logic as the Streamlit app."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.analysis_runner import run_analyze
from backend.recommendations import run_recommendations
from backend.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    RecommendationsRequest,
    RecommendationsResponse,
)

app = FastAPI(title="Stock Trade Analysis API", version="1.0.0")

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
    """Optional: same pipeline as Streamlit (`run_analyze`)."""
    try:
        return run_analyze(req.tickers)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.post("/recommendations", response_model=RecommendationsResponse)
def recommendations(req: RecommendationsRequest) -> RecommendationsResponse:
    horizon = req.horizon.strip().lower()
    if horizon not in ("long", "short"):
        raise HTTPException(status_code=422, detail="horizon must be 'long' or 'short'")
    return run_recommendations(
        horizon=horizon,  # type: ignore[arg-type]
        top_n=req.top_n,
        min_confidence=req.min_confidence,
        min_rr=req.min_rr,
    )
