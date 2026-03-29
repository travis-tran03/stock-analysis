# Stock trade analysis app

**Streamlit** UI imports **`backend/`** directly (one process, no HTTP). Optional **FastAPI** (`backend/main.py`) exposes the same analysis for REST clients or tests.

Data: **yfinance** (OHLCV, fundamentals, news), **pandas_ta** (indicators), **TextBlob** + optional **RSS** (sentiment).

## Setup

1. Python 3.10+ recommended.

2. Install dependencies (from the project root `Stock_App`):

```bash
python -m pip install -r requirements.txt
```

3. Download TextBlob / NLTK data once:

```bash
python -m textblob.download_corpora
```

## Run locally (recommended: single app)

From the project root:

```bash
streamlit run streamlit_app.py
```

No separate API server is required.

## Optional: REST API

Same analysis logic, served over HTTP:

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs — `POST /analyze` with body `{ "tickers": ["TSLA", "AAPL"] }`.

## Deploy on Streamlit Community Cloud (free tier)

1. Push this repo to GitHub.
2. In [Streamlit Community Cloud](https://streamlit.io/cloud), create an app from the repo.
3. Set **Main file path** to `streamlit_app.py`.
4. Add **Secrets** only if you later introduce API keys (not required for this app).
5. After the first deploy, run **TextBlob corpora** once if sentiment fails: add a build step or use Streamlit’s “Manage app” → **Reboot** after adding `packages.txt` / a startup script if needed. Many deployments work if corpora download is triggered on first run; if you see NLTK errors, add to `requirements.txt` a post-install or use Streamlit’s **Advanced settings** → **Build command** to run `python -m textblob.download_corpora`.

If Yahoo Finance blocks cloud IPs intermittently, analysis may fail until you retry.

## Extending

- Add modules under `backend/` and wire them in `signals.py`.
- Shared entry: `backend/analysis_runner.run_analyze`.

## Disclaimer

Outputs are automated heuristics for learning and exploration, not financial advice.
