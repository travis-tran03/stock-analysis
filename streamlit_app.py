"""Streamlit UI: imports backend directly (no HTTP). Suitable for Streamlit Community Cloud."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
from supabase import Client, create_client

from backend.analysis_runner import run_analyze
from backend.index_constituents import major_index_universe_summary
from backend.recommendations import (
    finalize_recommendations,
    process_ticker_for_recommendations,
)
from backend.schemas import RecommendationsResponse, StockAnalysis
from backend.universe import get_universe, load_large_cap_universe, short_term_universe_summary

REC_SCAN_STATE_KEY = "recommendation_scan"
REC_SCAN_STOP_KEY = "recommendation_scan_stop"

WATCHLISTS_PATH = Path(__file__).with_name("watchlists.json")
SUPABASE_TABLE = "watchlist_groups"
RECOMMENDATION_RUNS_TABLE = "recommendation_runs"


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
        take_profits = item.get("take_profits") or []
        tp1: Optional[float] = None
        if isinstance(take_profits, list) and take_profits:
            tp1 = _optional_float(take_profits[0])

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
                "Sell": tp1 if tp1 is not None else float("nan"),
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
        "Direction": {
            "meaning": "Model trade lean from blended score.",
            "good": "BUY = strongest long setup; HOLD = wait/neutral; SELL = wait for deeper discount (long-only mode).",
        },
        "Confidence": {
            "meaning": "How confident the model is (0–100%).",
            "good": "Strong: >=70 | Moderate: 50-69 | Weak: <50.",
        },
        "Last": {
            "meaning": "Last regular-session close.",
            "good": "No fixed universal range; compare against Buy low/Buy high/Sell for context.",
        },
        "Buy low": {
            "meaning": "Suggested lower bound of entry zone.",
            "good": "No fixed universal range; closer to current Last means sooner fill, farther below means more patience needed.",
        },
        "Buy high": {
            "meaning": "Suggested upper bound of entry zone.",
            "good": "No fixed universal range; if Last <= Buy high, entry is currently in-range.",
        },
        "Sell": {
            "meaning": "Suggested first take-profit level (TP1). In long-only mode this is a sell target for both BUY and SELL signals (SELL means buy lower, then sell rebound).",
            "good": "No fixed universal range; should typically be above entry zone for a valid long setup.",
        },
        "Day Δ %": {
            "meaning": "1-day % change (close vs prior close).",
            "good": "Strong: >=+1.5 | Neutral: 0 to +1.49 | Weak: <0.",
        },
        "Δ to buy low %": {
            "meaning": "Percent move from last price down to the suggested entry low (typically negative).",
            "good": "Strong: >=-1% | OK: -3% to -1% | Weak: <-3%. (Higher/closer to 0 is better.)",
        },
        "RSI": {
            "meaning": "RSI(14).",
            "good": "Strong: >=60 | OK: 45-59 | Weak: <45.",
        },
        "Blended": {
            "meaning": "Weighted score combining Tech/Fund/Sent/Market/Sess (range ~[-1, 1]).",
            "good": "Strong: >=+0.18 | OK: +0.08 to +0.179 | Weak: <+0.08.",
        },
        "Tech": {
            "meaning": "Technical sub-score (~[-1, 1]).",
            "good": "Strong: >=+0.22 | OK: +0.08 to +0.219 | Weak: <+0.08.",
        },
        "Fund": {
            "meaning": "Fundamental sub-score (~[-1, 1]).",
            "good": "Strong: >=+0.22 | OK: +0.08 to +0.219 | Weak: <+0.08.",
        },
        "Sent": {
            "meaning": "News sentiment sub-score (~[-1, 1]).",
            "good": "Strong: >=+0.22 | OK: +0.08 to +0.219 | Weak: <+0.08.",
        },
        "Market": {
            "meaning": "Broad market context score (SPY/VIX) (~[-1, 1]).",
            "good": "Strong: >=+0.22 | OK: +0.08 to +0.219 | Weak: <+0.08.",
        },
        "Sess": {
            "meaning": "Extended-hours session score (~[-1, 1]).",
            "good": "Strong: >=+0.22 | OK: +0.08 to +0.219 | Weak: <+0.08.",
        },
        "R:R": {
            "meaning": "Estimated risk/reward ratio for the plan.",
            "good": "Strong: >=2.0 | OK: 1.3-1.99 | Weak: <1.3.",
        },
        "R:R label": {
            "meaning": "Text label for risk/reward.",
            "good": "Higher ratio is better; 'wait for clarity' means no strong setup yet.",
        },
        "Pre %": {
            "meaning": "Pre-market % change vs prior regular close.",
            "good": "Strong: >=+1.5% | Neutral: 0% to +1.49% | Weak: <0%.",
        },
        "AH %": {
            "meaning": "After-hours % change vs regular close.",
            "good": "Strong: >=+1.5% | Neutral: 0% to +1.49% | Weak: <0%.",
        },
        "Buy lean": {
            "meaning": "BUY-only: blended × confidence (higher = stronger BUY conviction).",
            "good": "Strong: >=0.16 | OK: 0.12-0.159 | Weak: <0.12. Blank for non-BUY rows.",
        },
    }


def _help_text(guide: dict[str, dict[str, str]], col: str) -> str:
    meta = guide.get(col) or {}
    meaning = (meta.get("meaning") or "").strip()
    good = (meta.get("good") or "").strip()
    if meaning and good:
        return f"{meaning}\n\nLooks good: {good}"
    return meaning or good or ""


def _style_compare_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def css(bg: str) -> str:
        return f"background-color: {bg};"

    GREEN = "#d1fae5"   # emerald-100
    YELLOW = "#fef3c7"  # amber-100
    RED = "#fee2e2"     # red-100

    def row_style(row: pd.Series) -> list[str]:
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

        # Confidence (0-100)
        conf = f("Confidence")
        if pd.notna(conf):
            set_cell("Confidence", "g" if conf >= 70 else ("y" if conf >= 50 else "r"))

        # Blended score (higher is better)
        blended = f("Blended")
        if pd.notna(blended):
            set_cell("Blended", "g" if blended >= 0.18 else ("y" if blended >= 0.08 else "r"))

        # Δ to buy low %: closer to 0 is better
        dlow = f("Δ to buy low %")
        if pd.notna(dlow):
            ad = abs(dlow)
            set_cell("Δ to buy low %", "g" if ad <= 1.0 else ("y" if ad <= 3.0 else "r"))

        # RSI (14): higher is better for long-only momentum view
        rsi = f("RSI")
        if pd.notna(rsi):
            set_cell("RSI", "g" if rsi >= 60 else ("y" if rsi >= 45 else "r"))

        # Subscores Tech/Fund/Sent/Market/Sess: higher is better
        for col in ("Tech", "Fund", "Sent", "Market", "Sess"):
            v = f(col)
            if pd.isna(v):
                continue
            set_cell(col, "g" if v >= 0.22 else ("y" if v >= 0.08 else "r"))

        # Risk/Reward
        rr = f("R:R")
        if pd.notna(rr):
            set_cell("R:R", "g" if rr >= 2.0 else ("y" if rr >= 1.3 else "r"))

        # Day Δ %: higher is better
        day = f("Day Δ %")
        if pd.notna(day):
            set_cell("Day Δ %", "g" if day >= 1.5 else ("y" if day >= 0.0 else "r"))

        # Extended session moves: higher is better
        for col in ("Pre %", "AH %"):
            v = f(col)
            if pd.isna(v):
                continue
            set_cell(col, "g" if v >= 1.5 else ("y" if v >= 0.0 else "r"))

        # Buy lean score (only present for BUY rows)
        bl = f("Buy lean")
        if pd.notna(bl):
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


def load_watchlists() -> dict[str, list[str]]:
    """Load saved watchlist groups from disk."""
    if not WATCHLISTS_PATH.exists():
        return {}
    try:
        raw = json.loads(WATCHLISTS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}

    cleaned: dict[str, list[str]] = {}
    for name, values in raw.items():
        if not isinstance(name, str) or not isinstance(values, list):
            continue
        tickers = parse_tickers(",".join(str(v) for v in values))
        if tickers:
            cleaned[name.strip()] = tickers
    return cleaned


def _get_supabase_client() -> Optional[Client]:
    """Create Supabase client from Streamlit secrets, if configured."""
    try:
        url = st.secrets.get("SUPABASE_URL")
        key = st.secrets.get("SUPABASE_KEY")
    except Exception:
        return None
    if not url or not key:
        return None
    try:
        return create_client(str(url), str(key))
    except Exception:
        return None


def load_watchlists_from_supabase(client: Client) -> dict[str, list[str]]:
    """Load watchlist groups from Supabase table."""
    data = client.table(SUPABASE_TABLE).select("group_name,symbols").execute().data or []
    out: dict[str, list[str]] = {}
    for row in data:
        name = str(row.get("group_name") or "").strip()
        symbols = row.get("symbols") or []
        if not name or not isinstance(symbols, list):
            continue
        tickers = parse_tickers(",".join(str(v) for v in symbols))
        if tickers:
            out[name] = tickers
    return out


def save_watchlists(watchlists: dict[str, list[str]]) -> None:
    """Persist watchlist groups to disk."""
    payload = {k: v for k, v in sorted(watchlists.items()) if k and v}
    WATCHLISTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_watchlists_to_supabase(client: Client, watchlists: dict[str, list[str]]) -> None:
    """Persist all watchlist groups to Supabase table."""
    payload = [
        {"group_name": name, "symbols": symbols}
        for name, symbols in sorted(watchlists.items())
        if name and symbols
    ]
    client.table(SUPABASE_TABLE).delete().neq("group_name", "").execute()
    if payload:
        client.table(SUPABASE_TABLE).insert(payload).execute()


def load_recommendations_from_supabase(client: Client, horizon: str) -> Optional[dict[str, Any]]:
    """Load most recent recommendation run for a horizon."""
    try:
        rows = (
            client.table(RECOMMENDATION_RUNS_TABLE)
            .select("results_json,created_at")
            .eq("horizon", horizon)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        if not rows:
            return None
        row = rows[0]
        payload = row.get("results_json")
        if isinstance(payload, dict):
            payload = dict(payload)
            if row.get("created_at"):
                payload["_supabase_created_at"] = row["created_at"]
            return payload
    except Exception:
        return None
    return None


def save_recommendations_to_supabase(
    client: Client, response: RecommendationsResponse
) -> None:
    client.table(RECOMMENDATION_RUNS_TABLE).insert(
        {
            "horizon": response.horizon,
            "results_json": response.model_dump(mode="json"),
        }
    ).execute()


def build_recommendations_dataframe(
    picks: list[dict[str, Any]], score_label: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pick in picks:
        analysis = pick.get("analysis") or {}
        det = analysis.get("details") or {}
        tech = det.get("technical") or {}
        scores = det.get("scores") or {}
        rows.append(
            {
                "Ticker": pick.get("ticker"),
                "Direction": pick.get("direction"),
                "Confidence": float(pick.get("confidence") or 0) * 100.0,
                score_label: _num(pick.get("horizon_score")),
                "Rank": _num(pick.get("rank_score")),
                "Last": _num(tech.get("last_close")),
                "3mo %": _num(tech.get("momentum_3m_pct")),
                "RSI": _num(tech.get("rsi_14")),
                "Blended": _num(scores.get("combined")),
            }
        )
    return pd.DataFrame(rows)


def _complete_recommendation_scan(
    state: dict[str, Any],
    supabase_client: Optional[Client],
    *,
    cancelled: bool,
) -> None:
    horizon = state["horizon"]
    candidates: list[tuple[float, StockAnalysis]] = []
    for item in state.get("candidates") or []:
        candidates.append((float(item["rank"]), StockAnalysis.model_validate(item["analysis"])))

    response = finalize_recommendations(
        horizon=horizon,  # type: ignore[arg-type]
        top_n=int(state["top_n"]),
        universe_size=int(state.get("universe_size") or len(state.get("tickers") or [])),
        candidates=candidates,
        errors=list(state.get("errors") or []),
        scanned_count=int(state.get("scanned") or 0),
        cancelled=cancelled,
    )
    if cancelled:
        response.errors = list(response.errors) + ["Scan stopped by user."]

    session_key = state["session_key"]
    st.session_state[session_key] = response.model_dump(mode="json")
    if supabase_client is not None:
        try:
            save_recommendations_to_supabase(supabase_client, response)
        except Exception as e:
            st.warning(f"Could not save scan to Supabase: {e}")

    st.session_state.pop(REC_SCAN_STATE_KEY, None)
    st.session_state.pop(REC_SCAN_STOP_KEY, None)


def _start_recommendation_scan(
    horizon: str,
    title: str,
    session_key: str,
    top_n: int,
    min_confidence: float,
) -> None:
    if st.session_state.get(REC_SCAN_STATE_KEY):
        st.warning("A scan is already in progress. Stop it or wait for it to finish.")
        return
    st.session_state[REC_SCAN_STATE_KEY] = {
        "horizon": horizon,
        "session_key": session_key,
        "title": title,
        "top_n": top_n,
        "min_confidence": min_confidence,
        "phase": "universe",
        "tickers": [],
        "i": 0,
        "candidates": [],
        "errors": [],
        "scanned": 0,
        "universe_size": 0,
        "current_ticker": "",
    }
    st.session_state.pop(REC_SCAN_STOP_KEY, None)
    st.rerun()


def _handle_active_recommendation_scan(supabase_client: Optional[Client]) -> None:
    """Advance an in-progress scan by one step (universe build or single ticker)."""
    state = st.session_state.get(REC_SCAN_STATE_KEY)
    if not state:
        return

    if st.session_state.get(REC_SCAN_STOP_KEY):
        _complete_recommendation_scan(state, supabase_client, cancelled=True)
        st.rerun()
        return

    horizon = state["horizon"]
    title = state.get("title") or horizon

    if state.get("phase") == "universe":
        with st.spinner(f"Building universe for {title}…"):
            tickers = get_universe(horizon)  # type: ignore[arg-type]
        state["tickers"] = tickers
        state["universe_size"] = len(tickers)
        state["phase"] = "scan"
        st.rerun()
        return

    tickers: list[str] = state.get("tickers") or []
    i = int(state.get("i") or 0)
    if i >= len(tickers):
        _complete_recommendation_scan(state, supabase_client, cancelled=False)
        st.rerun()
        return

    ticker = tickers[i]
    state["current_ticker"] = ticker
    candidate, err, counted = process_ticker_for_recommendations(
        ticker,
        horizon=horizon,  # type: ignore[arg-type]
        min_confidence=float(state["min_confidence"]),
    )
    if counted:
        state["scanned"] = int(state.get("scanned") or 0) + 1
    if err:
        state.setdefault("errors", []).append(err)
    if candidate is not None:
        rank, analysis = candidate
        state.setdefault("candidates", []).append(
            {"rank": rank, "analysis": analysis.model_dump(mode="json")}
        )
    state["i"] = i + 1
    time.sleep(0.35)
    st.rerun()


def _render_active_scan_controls() -> bool:
    """Show progress and Stop while a scan is running. Returns True if a scan is active."""
    state = st.session_state.get(REC_SCAN_STATE_KEY)
    if not state:
        return False

    title = state.get("title") or "Recommendation"
    phase = state.get("phase") or "scan"
    total = len(state.get("tickers") or [])
    i = int(state.get("i") or 0)

    st.markdown(f"**Scan in progress:** {title}")
    if phase == "universe":
        st.progress(0.0, text="Building ticker universe…")
    else:
        pct = i / max(total, 1)
        current = state.get("current_ticker") or ""
        st.progress(pct, text=f"Analyzing {i}/{total}: {current}")

    if st.button("Stop scan", type="secondary", key="stop_rec_scan_global"):
        st.session_state[REC_SCAN_STOP_KEY] = True
        st.rerun()

    return True


def _render_recommendation_block(
    horizon: str,
    title: str,
    caption: str,
    score_label: str,
    session_key: str,
    supabase_client: Optional[Client],
    top_n: int,
    min_confidence: float,
) -> None:
    st.markdown(f"#### {title}")
    st.caption(caption)

    cached = st.session_state.get(session_key)
    if cached:
        as_of = cached.get("as_of") or cached.get("_supabase_created_at") or ""
        stopped = " (stopped early)" if cached.get("cancelled") else ""
        st.caption(
            f"Last scan: {as_of} — scanned {cached.get('scanned_count', '?')} of "
            f"{cached.get('universe_size', '?')}{stopped}"
        )

    scan_active = st.session_state.get(REC_SCAN_STATE_KEY) is not None
    this_scan = (
        scan_active
        and st.session_state[REC_SCAN_STATE_KEY].get("session_key") == session_key
    )
    other_scan = scan_active and not this_scan

    run_key = f"run_rec_{horizon}"
    if st.button(
        f"Run {title.lower()} scan",
        key=run_key,
        use_container_width=True,
        disabled=scan_active,
    ):
        _start_recommendation_scan(horizon, title, session_key, top_n, min_confidence)
    if other_scan:
        st.caption("Another scan is running — use **Stop scan** above.")

    cached = st.session_state.get(session_key)
    if not cached:
        st.info("No results yet. Run a scan to populate recommendations.")
        return

    if cached.get("cancelled"):
        st.warning("Previous scan was stopped early — picks are from partial progress only.")

    for err in cached.get("errors") or []:
        st.warning(err)

    picks = cached.get("picks") or []
    if not picks:
        st.info("No BUY picks matched your filters. Try lowering min confidence or re-run later.")
        return

    rec_df = build_recommendations_dataframe(picks, score_label)
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    tickers = ", ".join(str(p.get("ticker")) for p in picks if p.get("ticker"))
    if st.button(f"Load {title} picks into analyzer", key=f"load_rec_{horizon}"):
        st.session_state.ticker_input = tickers
        st.rerun()


def persist_watchlists(client: Optional[Client], watchlists: dict[str, list[str]]) -> None:
    """Save to Supabase when configured; otherwise to local file."""
    if client is not None:
        try:
            save_watchlists_to_supabase(client, watchlists)
            return
        except Exception:
            # Fallback keeps feature usable if remote persistence temporarily fails.
            pass
    save_watchlists(watchlists)


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

    supabase_client = _get_supabase_client()
    using_supabase = supabase_client is not None

    if "watchlists" not in st.session_state:
        if using_supabase:
            try:
                st.session_state.watchlists = load_watchlists_from_supabase(supabase_client)
            except Exception:
                st.session_state.watchlists = load_watchlists()
        else:
            st.session_state.watchlists = load_watchlists()
    if "ticker_input" not in st.session_state:
        st.session_state.ticker_input = "TSLA, AAPL, NVDA"

    if using_supabase and supabase_client is not None:
        for hz, key in (("long", "rec_long"), ("short", "rec_short")):
            if key not in st.session_state:
                loaded = load_recommendations_from_supabase(supabase_client, hz)
                if loaded:
                    st.session_state[key] = loaded

    st.subheader("Discover — stock recommendations")
    st.caption(
        "Automated heuristics for learning only, not financial advice. "
        "Long-term scans major US indexes (S&P 500/400/600, NASDAQ-100, Dow); "
        "Short-term scans the same major indexes, ranked by recent momentum. "
        "First long scan can take 30+ minutes; short scan is a smaller subset (~200 names)."
    )
    d1, d2, d3 = st.columns(3)
    with d1:
        rec_top_n = st.number_input("Top N picks", min_value=3, max_value=30, value=10, key="rec_top_n")
    with d2:
        rec_min_conf = st.slider(
            "Min confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="rec_min_conf",
        )
    with d3:
        st.caption("Scans run step-by-step; use **Stop scan** to cancel. Results cached 24h.")

    if _render_active_scan_controls():
        _handle_active_recommendation_scan(supabase_client if using_supabase else None)

    col_long, col_short = st.columns(2)
    with col_long:
        try:
            long_universe_n = len(load_large_cap_universe())
            long_universe_desc = major_index_universe_summary()
        except Exception:
            long_universe_n = "?"
            long_universe_desc = "S&P 500, S&P 400, S&P 600, NASDAQ-100, Dow 30"
        _render_recommendation_block(
            horizon="long",
            title="Long-term growth",
            caption=(
                f"Universe: {long_universe_desc} — {long_universe_n} symbols to scan. "
                "Scores favor fundamentals and market regime."
            ),
            score_label="Long score",
            session_key="rec_long",
            supabase_client=supabase_client if using_supabase else None,
            top_n=int(rec_top_n),
            min_confidence=float(rec_min_conf),
        )
    with col_short:
        try:
            short_universe_desc = short_term_universe_summary()
        except Exception:
            short_universe_desc = "Top momentum names from major US indexes"
        _render_recommendation_block(
            horizon="short",
            title="Short-term growth",
            caption=(
                f"{short_universe_desc}. "
                "Same indexes as long-term, filtered for 1–3 month price strength; "
                "scores favor technicals, sentiment, and session."
            ),
            score_label="Short score",
            session_key="rec_short",
            supabase_client=supabase_client if using_supabase else None,
            top_n=int(rec_top_n),
            min_confidence=float(rec_min_conf),
        )

    st.markdown("---")
    st.subheader("Analyze your tickers")

    with st.sidebar:
        st.subheader("Watchlist groups")
        st.caption(f"Storage: {'Supabase' if using_supabase else 'Local file'}")
        groups = sorted(st.session_state.watchlists.keys())
        new_group_name = st.text_input(
            "Create new group",
            placeholder="e.g. AI Leaders, Dividend Watch, Swing Setups",
        )
        if st.button("Create group", use_container_width=True):
            name = new_group_name.strip()
            if not name:
                st.warning("Enter a group name.")
            elif name in st.session_state.watchlists:
                st.warning("That group already exists.")
            else:
                st.session_state.watchlists[name] = []
                persist_watchlists(supabase_client, st.session_state.watchlists)
                st.success(f"Created {name}")
                st.rerun()

        st.markdown("---")
        selected_group = st.selectbox(
            "Select group",
            options=groups if groups else ["(none yet)"],
            disabled=not groups,
            key="selected_watchlist_group",
        )
        if groups:
            current_symbols = st.session_state.watchlists.get(selected_group, [])
            st.caption("Symbols in selected group")
            st.code(", ".join(current_symbols) if current_symbols else "(no symbols yet)")

            add_symbols = st.text_input(
                "Add ticker symbols to selected group",
                placeholder="e.g. MSFT, GOOGL, AMD",
            )
            a1, a2 = st.columns(2)
            if a1.button("Add symbols", use_container_width=True):
                added = parse_tickers(add_symbols)
                if not added:
                    st.warning("Enter at least one valid ticker.")
                else:
                    merged = parse_tickers(",".join(current_symbols + added))
                    st.session_state.watchlists[selected_group] = merged
                    persist_watchlists(supabase_client, st.session_state.watchlists)
                    st.success(f"Updated {selected_group} ({len(merged)} symbols)")
                    st.rerun()
            if a2.button("Delete group", use_container_width=True):
                del st.session_state.watchlists[selected_group]
                persist_watchlists(supabase_client, st.session_state.watchlists)
                st.rerun()

            if st.button("Load group into analyzer", use_container_width=True):
                st.session_state.ticker_input = ", ".join(current_symbols)
                st.success(f"Loaded {selected_group}")

            if st.button("Analyze selected group", use_container_width=True):
                st.session_state.ticker_input = ", ".join(current_symbols)
                st.session_state.run_group_analysis = True
                st.rerun()

    raw = st.text_area(
        "Tickers (comma or line separated)",
        key="ticker_input",
        height=100,
        placeholder="e.g. TSLA, AAPL, NVDA",
    )

    run_group_analysis = bool(st.session_state.get("run_group_analysis"))
    if run_group_analysis:
        st.session_state.run_group_analysis = False

    if st.button("Analyze", type="primary") or run_group_analysis:
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
                "Ticker": st.column_config.TextColumn("Ticker", help=_help_text(guide, "Ticker")),
                "Direction": st.column_config.TextColumn("Direction", help=_help_text(guide, "Direction")),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.0f%%",
                    min_value=0.0,
                    max_value=100.0,
                    help=_help_text(guide, "Confidence"),
                ),
                "Last": st.column_config.NumberColumn("Last", format="%.2f", help=_help_text(guide, "Last")),
                "Buy low": st.column_config.NumberColumn("Buy low", format="%.2f", help=_help_text(guide, "Buy low")),
                "Buy high": st.column_config.NumberColumn("Buy high", format="%.2f", help=_help_text(guide, "Buy high")),
                "Sell": st.column_config.NumberColumn("Sell", format="%.2f", help=_help_text(guide, "Sell")),
                "Day Δ %": st.column_config.NumberColumn("Day Δ %", format="%.2f", help=_help_text(guide, "Day Δ %")),
                "Δ to buy low %": st.column_config.NumberColumn(
                    "Δ to buy low %", format="%.2f", help=_help_text(guide, "Δ to buy low %")
                ),
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f", help=_help_text(guide, "RSI")),
                "Blended": st.column_config.NumberColumn("Blended", format="%.3f", help=_help_text(guide, "Blended")),
                "Tech": st.column_config.NumberColumn("Tech", format="%.2f", help=_help_text(guide, "Tech")),
                "Fund": st.column_config.NumberColumn("Fund", format="%.2f", help=_help_text(guide, "Fund")),
                "Sent": st.column_config.NumberColumn("Sent", format="%.2f", help=_help_text(guide, "Sent")),
                "Market": st.column_config.NumberColumn("Market", format="%.2f", help=_help_text(guide, "Market")),
                "Sess": st.column_config.NumberColumn("Sess", format="%.2f", help=_help_text(guide, "Sess")),
                "R:R": st.column_config.NumberColumn("R:R", format="%.2f", help=_help_text(guide, "R:R")),
                "R:R label": st.column_config.TextColumn("R:R label", help=_help_text(guide, "R:R label")),
                "Pre %": st.column_config.NumberColumn("Pre %", format="%.2f", help=_help_text(guide, "Pre %")),
                "AH %": st.column_config.NumberColumn("AH %", format="%.2f", help=_help_text(guide, "AH %")),
                "Buy lean": st.column_config.NumberColumn("Buy lean", format="%.3f", help=_help_text(guide, "Buy lean")),
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
