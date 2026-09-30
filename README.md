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
  tracked symbol with a button to add/remove it from your personal
  Watchlist, a Watchlist page showing only the ones you've added, and a
  per-symbol detail page with a price chart, sentiment reading, and the
  full prediction breakdown.
- Per-symbol price alerts: from the Watchlist page, set an upper and/or
  lower price threshold on any watchlisted symbol; a separate, lightweight
  GitHub Actions job (`.github/workflows/check-price-alerts.yml`, every
  ~15 min during US market hours) polls Finnhub's `/quote` endpoint for
  symbols with a threshold set and posts a Discord/Slack webhook alert the
  moment price crosses it (best-effort near-real-time, not a live feed --
  GitHub Actions cron has a ~5 minute floor). An alert fires only once per
  crossing, not on every check, and requires BOTH the symbol to be
  Watchlisted AND a threshold to be set -- removing either stops alerts
  for that symbol.

Deliberately out of scope for Phase 1: prediction-accuracy tracking, a
live intraday price banner, an admin panel, and cloud deployment -- see
the project plan for the reasoning (soccer-predictor itself was built
incrementally, not all at once).

## Setup

```bash
uv sync
cp .env.example .env   # fill in ALPHA_VANTAGE_API_KEY, FINNHUB_API_KEY, and (optionally) DISCORD_WEBHOOK_URL
uv run python scripts/fetch_historical_data.py
uv run python scripts/refresh_live_data.py
uv run streamlit run src/stock_predictor/dashboard/app.py
```

To check price alerts locally: `uv run python scripts/check_price_alerts.py`.

## GitHub Actions secrets

Both scheduled workflows (`.github/workflows/refresh-data.yml`,
`.github/workflows/check-price-alerts.yml`) read these from the repo's
Settings -> Secrets and variables -> Actions:

- `ALPHA_VANTAGE_API_KEY` -- used by `refresh-data.yml` only.
- `FINNHUB_API_KEY` -- used by both workflows.
- `DISCORD_WEBHOOK_URL` -- used by both workflows; required for the
  price-alert feature above, and also for `refresh-data.yml`'s existing
  recommendation-change alert to actually fire on the scheduled run.

## Tests

```bash
uv run pytest -q
```
