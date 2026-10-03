# Stock Market Analyst

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
cp .env.example .env  
uv run python scripts/fetch_historical_data.py
uv run python scripts/refresh_live_data.py
uv run streamlit run src/stock_predictor/dashboard/app.py
```

To check price alerts locally: `uv run python scripts/check_price_alerts.py`.


- `ALPHA_VANTAGE_API_KEY`
- `FINNHUB_API_KEY`
- `DISCORD_WEBHOOK_URL` -- needed for both the recommendation-change and
  price-threshold alerts to actually fire
## Tests

```bash
uv run pytest -q
```
