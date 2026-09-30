# Stock Predictor

A Streamlit dashboard that tracks ~50 stocks, combining daily price
history with news/social sentiment and fundamentals to produce a
5-trading-day forward prediction and a Buy/Hold/Sell recommendation for
each one.

## Features

- Browse every tracked stock and build a personal Watchlist
- Price chart, fundamentals (P/E), quarterly financials, and news/social
  sentiment for each symbol
- Price alerts: set an upper and/or lower price threshold on a watchlisted
  stock and get a Discord/Slack notification when it's crossed (checked
  every ~15 minutes during market hours)
- Optional password protection on Watchlist/alert changes, for running
  this publicly without letting visitors edit your data

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

Two scheduled workflows keep data and price alerts up to date
(`.github/workflows/refresh-data.yml`, `.github/workflows/check-price-alerts.yml`).
Add these as repo secrets (Settings -> Secrets and variables -> Actions):

- `ALPHA_VANTAGE_API_KEY`
- `FINNHUB_API_KEY`
- `DISCORD_WEBHOOK_URL` -- needed for both the recommendation-change and
  price-threshold alerts to actually fire

## Deploying publicly

There are no user accounts -- the database is shared, so without
`OWNER_PASSWORD` set, anyone with the URL can edit the Watchlist or price
alerts (viewing is always open to everyone). Set `OWNER_PASSWORD` as an
app secret before sharing a public link; Watchlist/alert changes will
then prompt for it once per browser session.

## Tests

```bash
uv run pytest -q
```
