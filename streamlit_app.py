"""Streamlit UI: imports backend directly (no HTTP). Suitable for Streamlit Community Cloud."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import streamlit as st

from backend.analysis_runner import run_analyze


def _num(v: Any) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _optional_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_compare_dataframe(results_dicts: list[dict[str, Any]]) -> pd.DataFrame:
    """Wide comparison table; default sort by blended score desc applied by caller."""
    rows: list[dict[str, Any]] = []
    for item in results_dicts:
        det = item.get("details") or {}
        tech = det.get("technical") or {}
        scores = det.get("scores") or {}
        er = item.get("entry_range") or {}
        pm = item.get("premarket_analysis") or {}
        ah = item.get("afterhours_analysis") or {}
        rr = item.get("risk_reward") or {}

        last = _optional_float(tech.get("last_close"))
        low = _optional_float(er.get("low"))
        high = _optional_float(er.get("high"))

        delta_to_low: Optional[float] = None
        if last is not None and low is not None and last > 0:
            delta_to_low = (last - low) / last * 100.0

        comb = _optional_float(scores.get("combined"))
        conf = _optional_float(item.get("confidence"))

        direction = str(item.get("direction") or "HOLD")
        buy_lean: Optional[float] = None
        if direction == "BUY" and comb is not None and conf is not None:
            buy_lean = comb * conf

        # ProgressColumn uses printf formatting; values 0–1 with "%.0f%%" print as 0%.
        conf_pct = conf * 100.0 if conf is not None else float("nan")

        rows.append(
            {
                "Ticker": item.get("ticker"),
                "Direction": direction,
                "Confidence": conf_pct,
                "Last": last if last is not None else float("nan"),
                "Buy low": low if low is not None else float("nan"),
                "Buy high": high if high is not None else float("nan"),
                "Day Δ %": _num(tech.get("change_1d_pct")),
                "Δ to buy low %": delta_to_low if delta_to_low is not None else float("nan"),
                "Blended": comb if comb is not None else float("nan"),
                "Tech": _num(scores.get("technical")),
                "Fund": _num(scores.get("fundamental")),
                "Sent": _num(scores.get("sentiment")),
                "Market": _num(scores.get("market")),
                "Sess": _num(scores.get("session")),
                "R:R": _num(rr.get("ratio")),
                "R:R label": rr.get("label") or "",
                "RSI": _num(tech.get("rsi_14")),
                "Pre %": _num(pm.get("premarket_change_percent")),
                "AH %": _num(ah.get("afterhours_change_percent")),
                "Buy lean": buy_lean if buy_lean is not None else float("nan"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty and "Blended" in df.columns:
        df = df.sort_values("Blended", ascending=False, na_position="last")
    return df


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

        compare_df = build_compare_dataframe(results_dicts)
        st.subheader("Compare tickers")
        st.caption(
            "Sorted by blended score (highest first). "
            "“Δ to buy low %” is how far last price sits above the suggested entry low."
        )
        st.dataframe(
            compare_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Direction": st.column_config.TextColumn("Direction"),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.0f%%",
                    min_value=0.0,
                    max_value=100.0,
                ),
                "Last": st.column_config.NumberColumn("Last", format="%.2f"),
                "Buy low": st.column_config.NumberColumn("Buy low", format="%.2f"),
                "Buy high": st.column_config.NumberColumn("Buy high", format="%.2f"),
                "Day Δ %": st.column_config.NumberColumn("Day Δ %", format="%.2f"),
                "Δ to buy low %": st.column_config.NumberColumn("Δ to buy low %", format="%.2f"),
                "Blended": st.column_config.NumberColumn("Blended", format="%.3f"),
                "Tech": st.column_config.NumberColumn("Tech", format="%.2f"),
                "Fund": st.column_config.NumberColumn("Fund", format="%.2f"),
                "Sent": st.column_config.NumberColumn("Sent", format="%.2f"),
                "Market": st.column_config.NumberColumn("Market", format="%.2f"),
                "Sess": st.column_config.NumberColumn("Sess", format="%.2f"),
                "R:R": st.column_config.NumberColumn("R:R", format="%.2f"),
                "R:R label": st.column_config.TextColumn("R:R label"),
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                "Pre %": st.column_config.NumberColumn("Pre %", format="%.2f"),
                "AH %": st.column_config.NumberColumn("AH %", format="%.2f"),
                "Buy lean": st.column_config.NumberColumn("Buy lean", format="%.3f"),
            },
        )

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
                er = item.get("entry_range") or {}
                st.write("**Entry range:**", f"{er.get('low')} – {er.get('high')}")
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
