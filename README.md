# Stock trade analysis app

Full-stack app: **FastAPI** backend loads market data with **yfinance**, runs technical indicators (**pandas_ta**), fundamentals from `info`, and news sentiment (**TextBlob**), then returns JSON trade ideas (direction, entry range, stop, targets, risk/reward, rationale). **Streamlit** frontend calls the API and shows tables, cards, and expandable detail sections.

## Setup

1. Python 3.10+ recommended.

2. Create a virtual environment (optional) and install dependencies:

```bash
cd Stock_App
python -m pip install -r requirements.txt
```

3. Download TextBlob / NLTK data once (required for sentiment):

```bash
python -m textblob.download_corpora
```

## Run locally

Use **two terminals**, both with the project root as the current directory (`Stock_App`).

**Terminal A — API**

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs for interactive API docs.

**Terminal B — UI**

```bash
streamlit run frontend/streamlit_app.py
```

Optional: point the UI at a different API URL (sidebar) or set `API_BASE` before starting Streamlit:

```bash
set API_BASE=http://127.0.0.1:8000
streamlit run frontend/streamlit_app.py
```

## API

- `GET /health` — health check.
- `POST /analyze` — JSON body: `{ "tickers": ["TSLA", "AAPL"] }` — returns `{ "results": [...], "errors": [...] }`.

If every ticker fails, the API responds with **502** and an error detail. If the request has no valid symbols, **422** is returned.

## Extending

- Add modules under `backend/` and wire them in `signals.py`.
- EDGAR or alternative data sources can plug into `data_fetch.py` / `fundamental.py`.
- Replace Streamlit with any client that consumes the same JSON schema.

## Disclaimer

Outputs are automated heuristics for learning and exploration, not financial advice.
