"""News headline sentiment using TextBlob; optional RSS headlines via feedparser."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import feedparser
from textblob import TextBlob


def _yahoo_rss_headlines(ticker: str, limit: int = 12) -> list[dict[str, Any]]:
    """Fallback headlines when yfinance `news` is empty."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(ticker)}&region=US&lang=en-US"
    parsed = feedparser.parse(url)
    out: list[dict[str, Any]] = []
    for entry in getattr(parsed, "entries", [])[:limit]:
        title = getattr(entry, "title", "") or ""
        if title:
            out.append({"title": title})
    return out


@dataclass
class SentimentSnapshot:
    mean_polarity: float
    headline_count: int
    sample_headlines: list[str]
    per_headline_polarity: list[float]


def _extract_title(item: dict[str, Any]) -> str:
    t = item.get("title") or item.get("Title")
    if not t:
        c = item.get("content")
        if isinstance(c, dict):
            t = c.get("title")
    return str(t or "")


def analyze_news_sentiment(
    news_items: list[dict[str, Any]],
    ticker: str = "",
    max_headlines: int = 15,
) -> SentimentSnapshot:
    """
    Run TextBlob polarity on Yahoo Finance news titles (and optional summary).
    If `news_items` is empty, try Yahoo RSS for `ticker`.
    Polarity in [-1, 1].
    """
    items = list(news_items)
    if not items and ticker:
        items = _yahoo_rss_headlines(ticker, limit=max_headlines)

    texts: list[str] = []
    for n in items[:max_headlines]:
        title = _extract_title(n)
        if title:
            texts.append(title)
        summary = n.get("summary") or n.get("Summary")
        if not summary:
            c = n.get("content")
            if isinstance(c, dict):
                summary = c.get("summary")
        if summary and isinstance(summary, str) and len(summary) < 400:
            texts.append(summary[:300])

    if not texts:
        return SentimentSnapshot(
            mean_polarity=0.0,
            headline_count=0,
            sample_headlines=[],
            per_headline_polarity=[],
        )

    polarities: list[float] = []
    for t in texts:
        blob = TextBlob(t)
        polarities.append(float(blob.sentiment.polarity))

    mean_p = sum(polarities) / len(polarities) if polarities else 0.0
    samples = [_extract_title(n) for n in items[:5] if _extract_title(n)]

    return SentimentSnapshot(
        mean_polarity=mean_p,
        headline_count=len(texts),
        sample_headlines=samples,
        per_headline_polarity=polarities[:10],
    )


def sentiment_score(snap: SentimentSnapshot) -> float:
    """Map mean polarity to [-1, 1] (already in range for TextBlob)."""
    if snap.headline_count == 0:
        return 0.0
    return float(max(-1.0, min(1.0, snap.mean_polarity * 1.5)))
