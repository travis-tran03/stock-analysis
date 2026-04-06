"""Streamlit UI: imports backend directly (no HTTP). Suitable for Streamlit Community Cloud."""

from __future__ import annotations

from typing import Any

import streamlit as st

from backend.analysis_runner import run_analyze


def parse_tickers(text: str) -> list[str]:
    """Split on commas and newlines, strip, drop empties, dedupe."""
    parts: list[str] = []
    for line in text.replace(",", "\n").split("\n"):
        s = line.strip().upper()
        if s:
            parts.append(s)
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
    return (
        f'<span style="background:{c};color:white;padding:4px 10px;'
        f'border-radius:6px;font-weight:600;">{direction}</span>'
    )


def pct_badge(pct: float) -> str:
    c = "#0d9488" if pct >= 0 else "#dc2626"
    return (
        f'<span style="background:{c};color:white;padding:3px 8px;'
        f'border-radius:6px;font-weight:600;">{pct:+.2f}%</span>'
    )


def main() -> None:
    st.set_page_config(page_title="Stock Trade Analysis", layout="wide")
    st.title("Stock trade analysis")
    st.caption(
        "Technical, fundamental, and news sentiment — analysis runs in-process "
        "(no separate API server required)."
    )

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

        try:
            with st.spinner("Analyzing…"):
                response = run_analyze(tickers)
        except ValueError as e:
            st.error(str(e))
            return
        except RuntimeError as e:
            st.error(str(e))
            return

        errors = response.errors or []
        results = response.results or []

        for e in errors:
            st.warning(e)

        if not results:
            st.info("No results returned.")
            return

        # Pydantic models → plain dicts for display
        results_dicts: list[dict[str, Any]] = [r.model_dump(mode="json") for r in results]

        rows: list[dict[str, Any]] = []
        for item in results_dicts:
            tps = item.get("take_profits") or []
            rr = item.get("risk_reward") or {}
            rat = item.get("rationale") or ""
            planned = item.get("planned_entry_range") or {}
            final = item.get("final_entry_range") or {}
            pm = item.get("premarket_analysis") or {}
            ah = item.get("afterhours_analysis") or {}
            rows.append(
                {
                    "Ticker": item.get("ticker"),
                    "Direction": item.get("direction"),
                    "Confidence": f"{float(item.get('confidence') or 0):.0%}",
                    "Planned entry": f"{planned.get('low')} – {planned.get('high')}",
                    "Final entry": f"{final.get('low')} – {final.get('high')}",
                    "Entry price": item.get("entry_price"),
                    "Pre %": pm.get("premarket_change_percent"),
                    "Pre signal": pm.get("premarket_signal"),
                    "AH %": ah.get("afterhours_change_percent"),
                    "AH signal": ah.get("afterhours_signal"),
                    "Stop": item.get("stop_loss"),
                    "TPs": ", ".join(str(x) for x in tps),
                    "R:R": rr.get("label") or "",
                    "Rationale (short)": rat[:120] + "…" if len(rat) > 120 else rat,
                }
            )

        st.subheader("Summary")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.subheader("Details")
        for item in results_dicts:
            direction = item.get("direction") or "HOLD"
            st.markdown(direction_badge(str(direction)), unsafe_allow_html=True)
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(
                    f"**{item.get('ticker')}**  \n"
                    f"Confidence: **{float(item.get('confidence') or 0):.0%}**"
                )
                planned = item.get("planned_entry_range") or {}
                final = item.get("final_entry_range") or {}
                st.write("**Planned entry:**", f"{planned.get('low')} – {planned.get('high')}")
                st.write("**Final entry:**", f"{final.get('low')} – {final.get('high')}")
                st.write("**Entry price (single):**", item.get("entry_price"))
                st.write("**Stop:**", item.get("stop_loss"))
                st.write("**Take profits:**", item.get("take_profits"))
                rr = item.get("risk_reward") or {}
                st.write("**Risk/Reward:**", rr.get("label"), rr.get("ratio"))

                pm = item.get("premarket_analysis") or {}
                if pm:
                    st.write("**Pre-market**")
                    pct = pm.get("premarket_change_percent")
                    if pct is not None:
                        st.markdown(pct_badge(float(pct)), unsafe_allow_html=True)
                    st.write("Pre price:", pm.get("premarket_price"))
                    st.write("Pre high / low:", pm.get("premarket_high"), "/", pm.get("premarket_low"))
                    st.write("Pre volume:", pm.get("premarket_volume"))
                    st.write("Signal:", pm.get("premarket_signal"))
                    st.caption(pm.get("note") or "")

                ah = item.get("afterhours_analysis") or {}
                if ah:
                    st.write("**After-hours**")
                    pct = ah.get("afterhours_change_percent")
                    if pct is not None:
                        st.markdown(pct_badge(float(pct)), unsafe_allow_html=True)
                    st.write("AH price:", ah.get("afterhours_price"))
                    st.write("AH high / low:", ah.get("afterhours_high"), "/", ah.get("afterhours_low"))
                    st.write("AH volume:", ah.get("afterhours_volume"))
                    st.write("Signal:", ah.get("afterhours_signal"))
                    st.caption(ah.get("note") or "")
            with c2:
                st.write("**Rationale**")
                st.write(item.get("rationale") or "")
                pts = item.get("summary_points") or []
                if pts:
                    st.write("**Summary points**")
                    for p in pts:
                        st.markdown(f"- {p}")

            with st.expander(f"Details — {item.get('ticker')}"):
                det = item.get("details") or {}
                ttab, ftab, stab, mtab, xtab = st.tabs(
                    ["Technical", "Fundamental", "Sentiment", "Market (SPY/VIX)", "Pre / after hours"]
                )
                with ttab:
                    st.json(det.get("technical") or {})
                with ftab:
                    st.json(det.get("fundamental") or {})
                with stab:
                    st.json(det.get("sentiment") or {})
                with mtab:
                    st.json(det.get("market") or {})
                with xtab:
                    st.json(det.get("extended_session") or {})
                st.write("**Monitoring triggers**")
                for tr in item.get("monitoring_triggers") or []:
                    st.markdown(f"- {tr}")
                st.write("**Plan steps**")
                for step in item.get("plan_steps") or []:
                    st.markdown(f"- {step}")
            st.divider()


if __name__ == "__main__":
    main()
