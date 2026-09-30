# Stock Predictor

A stock price/sentiment predictor, ported from the architecture of
[soccer-predictor](../soccer-predictor) -- a companion project that
predicts football match outcomes from historical results + current-season
player stats. This project applies the same shape to equities: historical
daily prices (weighted toward recent data) plus recent news sentiment
("reputation") as a small, capped nudge on top.

## Phase 1 (current scope)

- Historical daily OHLCV prices from Yahoo Finance's public chart endpoint
  (free, keyless).
- News sentiment from Alpha Vantage's `NEWS_SENTIMENT` endpoint (free,
  25 requests/day -- see `config/watchlist.yaml`'s curated ~15-20 symbol
  list, sized to fit that budget).
- Fundamentals (P/E ratio, company profile) primarily from Finnhub's free
  tier (60 requests/minute, no daily cap), with Alpha Vantage's `OVERVIEW`
  kept as a fallback and as the source of the company description/address
  (see `ingest/fundamentals.py`).
- A per-symbol return-distribution fit (time-decay-weighted mean/
  volatility, Student-t tails) -- the Dixon-Coles analog, just simpler:
  each symbol is fit independently, no cross-symbol interaction.
- A 5-trading-day forward prediction: P(up)/P(flat)/P(down) + an expected
  return range.
- A Streamlit dashboard: a watchlist table and a per-symbol detail page
  with a price chart, sentiment reading, and the full prediction
  breakdown.

Deliberately out of scope for Phase 1: prediction-accuracy tracking, a
live intraday price banner, an admin panel, and cloud deployment -- see
the project plan for the reasoning (soccer-predictor itself was built
incrementally, not all at once).

## Setup

```bash
uv sync
cp .env.example .env   # fill in ALPHA_VANTAGE_API_KEY and FINNHUB_API_KEY
uv run python scripts/fetch_historical_data.py
uv run python scripts/refresh_live_data.py
uv run streamlit run src/stock_predictor/dashboard/app.py
```

## Tests

```bash
uv run pytest -q
```
