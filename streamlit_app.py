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
            # Percent move from last down to the suggested entry low.
            # Typically negative (buy-low is below last).
            delta_to_low = (low - last) / last * 100.0

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
                "RSI": _num(tech.get("rsi_14")),
                "Blended": comb if comb is not None else float("nan"),
                "Tech": _num(scores.get("technical")),
                "Fund": _num(scores.get("fundamental")),
                "Sent": _num(scores.get("sentiment")),
                "Market": _num(scores.get("market")),
                "Sess": _num(scores.get("session")),
                "R:R": _num(rr.get("ratio")),
                "R:R label": rr.get("label") or "",
                "Pre %": _num(pm.get("premarket_change_percent")),
                "AH %": _num(ah.get("afterhours_change_percent")),
                "Buy lean": buy_lean if buy_lean is not None else float("nan"),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty and "Blended" in df.columns:
        df = df.sort_values("Blended", ascending=False, na_position="last")
    return df


def _compare_column_guide() -> dict[str, dict[str, str]]:
    """
    Human-friendly guide shown in the UI. Scores are generally in [-1, 1].
    Keep this lightweight: it's a heuristic guide, not financial advice.
    """
    return {
        "Ticker": {"meaning": "Stock symbol."},
        "Direction": {"meaning": "Model trade lean from blended score.", "good": "BUY / SELL when supported by other columns."},
        "Confidence": {"meaning": "How confident the model is (0–100%).", "good": "70%+ strong, 50–70% moderate, <40% weak."},
        "Last": {"meaning": "Last regular-session close.", "good": "Contextual (compare vs entry range)."},
        "Buy low": {"meaning": "Suggested lower bound of entry zone.", "good": "For BUY: last near/under this is favorable."},
        "Buy high": {"meaning": "Suggested upper bound of entry zone.", "good": "For BUY: last below this is generally better than above."},
        "Day Δ %": {
            "meaning": "1-day % change (close vs prior close).",
            "good": "For BUY: ≤ 0% good, 0% to +1.5% ok, > +1.5% stretched. For SELL: ≥ 0% good, 0% to −1.5% ok, < −1.5% stretched.",
        },
        "Δ to buy low %": {
            "meaning": "Percent move from last price down to the suggested entry low (typically negative).",
            "good": "For BUY: ~0% to −1% great, −1% to −3% ok, <−5% far away.",
        },
        "RSI": {"meaning": "RSI(14).", "good": "For BUY: <45 favorable (esp <30). For SELL: >55 favorable (esp >70)."},
        "Blended": {
            "meaning": "Weighted score combining Tech/Fund/Sent/Market/Sess (range ~[-1, 1]).",
            "good": "BUY if ≥ +0.18, SELL if ≤ −0.18 (stronger magnitude is better).",
        },
        "Tech": {
            "meaning": "Technical sub-score (~[-1, 1]).",
            "good": "For BUY: ≥ +0.22 strong, +0.08 to +0.22 ok, < +0.08 weak. For SELL: ≤ −0.22 strong, −0.22 to −0.08 ok, > −0.08 weak.",
        },
        "Fund": {
            "meaning": "Fundamental sub-score (~[-1, 1]).",
            "good": "For BUY: ≥ +0.22 strong, +0.08 to +0.22 ok, < +0.08 weak. For SELL: ≤ −0.22 strong, −0.22 to −0.08 ok, > −0.08 weak.",
        },
        "Sent": {
            "meaning": "News sentiment sub-score (~[-1, 1]).",
            "good": "For BUY: ≥ +0.22 strong, +0.08 to +0.22 ok, < +0.08 weak. For SELL: ≤ −0.22 strong, −0.22 to −0.08 ok, > −0.08 weak.",
        },
        "Market": {
            "meaning": "Broad market context score (SPY/VIX) (~[-1, 1]).",
            "good": "For BUY: ≥ +0.22 supportive, +0.08 to +0.22 mildly supportive, < +0.08 neutral/against. For SELL: ≤ −0.22 supportive, −0.22 to −0.08 mildly supportive, > −0.08 neutral/against.",
        },
        "Sess": {
            "meaning": "Extended-hours session score (~[-1, 1]).",
            "good": "For BUY: ≥ +0.22 supportive, +0.08 to +0.22 mildly supportive, < +0.08 neutral/against. For SELL: ≤ −0.22 supportive, −0.22 to −0.08 mildly supportive, > −0.08 neutral/against.",
        },
        "R:R": {"meaning": "Estimated risk/reward ratio for the plan.", "good": "≥2.0 great, 1.3–2.0 ok, <1.3 weak."},
        "R:R label": {"meaning": "Text label for risk/reward.", "good": "Higher is better (or 'wait for clarity' on HOLD)."},
        "Pre %": {"meaning": "Pre-market % change vs prior regular close.", "good": "Aligned with direction is supportive; big moves (>|3%|) deserve caution."},
        "AH %": {"meaning": "After-hours % change vs regular close.", "good": "Aligned with direction is supportive; big moves (>|3%|) deserve caution."},
        "Buy lean": {
            "meaning": "BUY-only: blended × confidence (higher = stronger BUY conviction).",
            "good": "For BUY: ≥ 0.16 strong, 0.12–0.16 ok, < 0.12 weak; blank/NaN for non-BUY rows.",
        },
    }


def _style_compare_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def css(bg: str) -> str:
        return f"background-color: {bg};"

    GREEN = "#d1fae5"   # emerald-100
    YELLOW = "#fef3c7"  # amber-100
    RED = "#fee2e2"     # red-100

    def row_style(row: pd.Series) -> list[str]:
        direction = str(row.get("Direction") or "HOLD").upper()
        out = [""] * len(row)
        idx = {c: i for i, c in enumerate(row.index)}

        def set_cell(col: str, which: str) -> None:
            if col not in idx:
                return
            out[idx[col]] = css({"g": GREEN, "y": YELLOW, "r": RED}.get(which, ""))

        def f(col: str) -> float:
            try:
                return float(row.get(col))
            except Exception:
                return float("nan")

        # Directional interpretation helpers (scores in [-1, 1])
        sign = 1.0 if direction == "BUY" else (-1.0 if direction == "SELL" else 0.0)

        # Confidence (0-100)
        conf = f("Confidence")
        if pd.notna(conf):
            set_cell("Confidence", "g" if conf >= 70 else ("y" if conf >= 50 else "r"))

        # Blended score
        blended = f("Blended")
        if pd.notna(blended):
            if direction == "BUY":
                set_cell("Blended", "g" if blended >= 0.18 else ("y" if blended >= 0.05 else "r"))
            elif direction == "SELL":
                set_cell("Blended", "g" if blended <= -0.18 else ("y" if blended <= -0.05 else "r"))
            else:
                set_cell("Blended", "y" if abs(blended) >= 0.12 else "")

        # Δ to buy low % (closer to 0 is better for BUY; typically negative)
        dlow = f("Δ to buy low %")
        if pd.notna(dlow):
            if direction == "BUY":
                # Example: -0.5% (great), -2% (ok), -6% (far).
                set_cell("Δ to buy low %", "g" if dlow >= -1.0 else ("y" if dlow >= -3.0 else "r"))
            else:
                set_cell("Δ to buy low %", "y" if dlow >= -1.0 else "")

        # RSI (14)
        rsi = f("RSI")
        if pd.notna(rsi):
            if direction == "BUY":
                set_cell("RSI", "g" if rsi < 45 else ("y" if rsi < 55 else "r"))
                if rsi < 30:
                    set_cell("RSI", "g")
            elif direction == "SELL":
                set_cell("RSI", "g" if rsi > 55 else ("y" if rsi > 45 else "r"))
                if rsi > 70:
                    set_cell("RSI", "g")
            else:
                set_cell("RSI", "y" if (rsi < 35 or rsi > 65) else "")

        # Subscores Tech/Fund/Sent/Market/Sess: prefer aligned sign with decent magnitude
        for col in ("Tech", "Fund", "Sent", "Market", "Sess"):
            v = f(col)
            if pd.isna(v) or sign == 0.0:
                continue
            aligned = v * sign
            set_cell(col, "g" if aligned >= 0.22 else ("y" if aligned >= 0.08 else "r"))

        # Risk/Reward
        rr = f("R:R")
        if pd.notna(rr):
            set_cell("R:R", "g" if rr >= 2.0 else ("y" if rr >= 1.3 else "r"))

        # Day Δ %: mild directional tint
        day = f("Day Δ %")
        if pd.notna(day):
            if direction == "BUY":
                set_cell("Day Δ %", "g" if day <= 0.0 else ("y" if day <= 1.5 else "r"))
            elif direction == "SELL":
                set_cell("Day Δ %", "g" if day >= 0.0 else ("y" if day >= -1.5 else "r"))

        # Extended session moves: big adverse moves red, aligned moves green
        for col in ("Pre %", "AH %"):
            v = f(col)
            if pd.isna(v) or sign == 0.0:
                continue
            if abs(v) >= 3.0 and (v * sign) < 0:
                set_cell(col, "r")
            elif abs(v) >= 1.5 and (v * sign) > 0:
                set_cell(col, "g")

        # Buy lean (only meaningful for BUY)
        bl = f("Buy lean")
        if pd.notna(bl) and direction == "BUY":
            # Example: blended 0.18 with 70% confidence => 0.126 (solid "ok").
            set_cell("Buy lean", "g" if bl >= 0.16 else ("y" if bl >= 0.12 else "r"))

        return out

    return df.style.apply(row_style, axis=1)


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
            "“Δ to buy low %” is the % move from last down to the suggested entry low."
        )
        guide = _compare_column_guide()
        with st.expander("Column guide (what it means + what looks good)"):
            st.markdown(
                "\n".join(
                    [
                        f"- **{col}**: {meta.get('meaning','')}"
                        + (f"  \n  _Looks good_: {meta.get('good','')}" if meta.get("good") else "")
                        for col, meta in guide.items()
                        if col in compare_df.columns
                    ]
                )
            )
        st.dataframe(
            _style_compare_table(compare_df),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", help=guide["Ticker"]["meaning"]),
                "Direction": st.column_config.TextColumn("Direction", help=guide["Direction"]["meaning"]),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.0f%%",
                    min_value=0.0,
                    max_value=100.0,
                    help=guide["Confidence"]["meaning"],
                ),
                "Last": st.column_config.NumberColumn("Last", format="%.2f", help=guide["Last"]["meaning"]),
                "Buy low": st.column_config.NumberColumn("Buy low", format="%.2f", help=guide["Buy low"]["meaning"]),
                "Buy high": st.column_config.NumberColumn("Buy high", format="%.2f", help=guide["Buy high"]["meaning"]),
                "Day Δ %": st.column_config.NumberColumn("Day Δ %", format="%.2f", help=guide["Day Δ %"]["meaning"]),
                "Δ to buy low %": st.column_config.NumberColumn(
                    "Δ to buy low %", format="%.2f", help=guide["Δ to buy low %"]["meaning"]
                ),
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f", help=guide["RSI"]["meaning"]),
                "Blended": st.column_config.NumberColumn("Blended", format="%.3f", help=guide["Blended"]["meaning"]),
                "Tech": st.column_config.NumberColumn("Tech", format="%.2f", help=guide["Tech"]["meaning"]),
                "Fund": st.column_config.NumberColumn("Fund", format="%.2f", help=guide["Fund"]["meaning"]),
                "Sent": st.column_config.NumberColumn("Sent", format="%.2f", help=guide["Sent"]["meaning"]),
                "Market": st.column_config.NumberColumn("Market", format="%.2f", help=guide["Market"]["meaning"]),
                "Sess": st.column_config.NumberColumn("Sess", format="%.2f", help=guide["Sess"]["meaning"]),
                "R:R": st.column_config.NumberColumn("R:R", format="%.2f", help=guide["R:R"]["meaning"]),
                "R:R label": st.column_config.TextColumn("R:R label", help=guide["R:R label"]["meaning"]),
                "Pre %": st.column_config.NumberColumn("Pre %", format="%.2f", help=guide["Pre %"]["meaning"]),
                "AH %": st.column_config.NumberColumn("AH %", format="%.2f", help=guide["AH %"]["meaning"]),
                "Buy lean": st.column_config.NumberColumn("Buy lean", format="%.3f", help=guide["Buy lean"]["meaning"]),
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
