"""Streamlit UI: ticker input, call FastAPI /analyze, display cards and detail expanders."""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_API = os.environ.get("API_BASE", "http://127.0.0.1:8000")


def parse_tickers(text: str) -> list[str]:
    """Split on commas and newlines, strip, drop empties."""
    parts: list[str] = []
    for line in text.replace(",", "\n").split("\n"):
        s = line.strip().upper()
        if s:
            parts.append(s)
    # Dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def direction_badge(direction: str) -> str:
    """HTML span with color for BUY / SELL / HOLD."""
    colors = {
        "BUY": "#0d9488",
        "SELL": "#dc2626",
        "HOLD": "#ca8a04",
    }
    c = colors.get(direction, "#64748b")
    return f'<span style="background:{c};color:white;padding:4px 10px;border-radius:6px;font-weight:600;">{direction}</span>'


def main() -> None:
    st.set_page_config(page_title="Stock Trade Analysis", layout="wide")
    st.title("Stock trade analysis")
    st.caption("Technical, fundamental, and news sentiment — powered by your local FastAPI backend.")

    api_base = st.sidebar.text_input("API base URL", value=DEFAULT_API, help="Where uvicorn is running")

    raw = st.text_area(
        "Tickers (comma or line separated)",
        value="TSLA, AAPL, NVDA",
        height=100,
        placeholder="e.g. TSLA, AAPL, NVDA",
    )

    if st.button("Analyze", type="primary"):
        tickers = parse_tickers(raw)
        if not tickers:
            st.warning("Enter at least one ticker.")
            return

        url = api_base.rstrip("/") + "/analyze"
        try:
            with st.spinner("Analyzing…"):
                r = httpx.post(url, json={"tickers": tickers}, timeout=120.0)
        except httpx.RequestError as e:
            st.error(f"Could not reach API at {url}: {e}")
            st.info("Start the backend: `python -m uvicorn backend.main:app --reload --port 8000` from the project root.")
            return

        if r.status_code == 422:
            st.error(r.json().get("detail", r.text))
            return
        if r.status_code == 502:
            st.error(r.json().get("detail", r.text))
            return
        if r.status_code != 200:
            st.error(f"HTTP {r.status_code}: {r.text}")
            return

        data = r.json()
        errors = data.get("errors") or []
        results = data.get("results") or []

        for e in errors:
            st.warning(e)

        if not results:
            st.info("No results returned.")
            return

        # Summary table
        rows: list[dict[str, Any]] = []
        for item in results:
            er = item.get("entry_range") or {}
            tps = item.get("take_profits") or []
            rr = item.get("risk_reward") or {}
            rows.append(
                {
                    "Ticker": item.get("ticker"),
                    "Direction": item.get("direction"),
                    "Confidence": f"{float(item.get('confidence') or 0):.0%}",
                    "Entry low": er.get("low"),
                    "Entry high": er.get("high"),
                    "Stop": item.get("stop_loss"),
                    "TPs": ", ".join(str(x) for x in tps),
                    "R:R": rr.get("label") or "",
                    "Rationale (short)": (item.get("rationale") or "")[:120] + "…"
                    if len(item.get("rationale") or "") > 120
                    else (item.get("rationale") or ""),
                }
            )

        st.subheader("Summary")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.subheader("Details")
        for item in results:
            direction = item.get("direction") or "HOLD"
            st.markdown(direction_badge(str(direction)), unsafe_allow_html=True)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"**{item.get('ticker')}**  \nConfidence: **{float(item.get('confidence') or 0):.0%}**")
                er = item.get("entry_range") or {}
                st.write("**Entry range:**", f"{er.get('low')} – {er.get('high')}")
                st.write("**Stop:**", item.get("stop_loss"))
                st.write("**Take profits:**", item.get("take_profits"))
                rr = item.get("risk_reward") or {}
                st.write("**Risk/Reward:**", rr.get("label"), rr.get("ratio"))
            with c2:
                st.write("**Rationale**")
                st.write(item.get("rationale") or "")
                pts = item.get("summary_points") or []
                if pts:
                    st.write("**Summary points**")
                    for p in pts:
                        st.markdown(f"- {p}")

            with st.expander(f"Technical / fundamental / sentiment — {item.get('ticker')}"):
                det = item.get("details") or {}
                ttab, ftab, stab = st.tabs(["Technical", "Fundamental", "Sentiment"])
                with ttab:
                    st.json(det.get("technical") or {})
                with ftab:
                    st.json(det.get("fundamental") or {})
                with stab:
                    st.json(det.get("sentiment") or {})
                st.write("**Monitoring triggers**")
                for tr in item.get("monitoring_triggers") or []:
                    st.markdown(f"- {tr}")
                st.write("**Plan steps**")
                for s in item.get("plan_steps") or []:
                    st.markdown(f"- {s}")
            st.divider()


if __name__ == "__main__":
    main()
