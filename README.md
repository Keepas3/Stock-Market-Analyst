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
- News sentiment from Finnhub's `company-news` endpoint + local VADER
  scoring (free, 60 requests/minute, no daily cap -- see
  `ingest/sentiment.py`), which is what let the watchlist grow to ~50
  curated symbols (see `config/watchlist.yaml`) after outgrowing Alpha
  Vantage's old 25-requests/day `NEWS_SENTIMENT` cap.
- Fundamentals (P/E ratio, company profile) primarily from Finnhub's free
  tier too, with Alpha Vantage's `OVERVIEW` kept as a fallback and as the
  source of the company description/address (see
  `ingest/fundamentals.py`). Quarterly financials (revenue/net income/EPS)
  still come from Alpha Vantage (see `ingest/financials.py`).
- A per-symbol return-distribution fit (time-decay-weighted mean/
  volatility, Student-t tails) -- the Dixon-Coles analog, just simpler:
  each symbol is fit independently, no cross-symbol interaction.
- A 5-trading-day forward prediction: P(up)/P(flat)/P(down) + an expected
  return range.
- A Streamlit dashboard with a left-hand nav: a Main page browsing every
  tracked symbol with a checkbox to add/remove it from your personal
  Watchlist, a Watchlist page showing only the ones you've checked, and a
  per-symbol detail page with a price chart, sentiment reading, and the
  full prediction breakdown.

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
