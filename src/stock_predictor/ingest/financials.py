"""Quarterly financial statements (revenue, net income, EPS) via Alpha
Vantage's INCOME_STATEMENT and EARNINGS endpoints -- inspired by CNN's own
"Financials" panel, deliberately scoped down: a full clone would also need
Free Cash Flow (Alpha Vantage's CASH_FLOW endpoint) and Debt-to-Equity
(BALANCE_SHEET) -- two MORE calls on the same shared 25/day budget. Not
built this round; flagged here and in the dashboard rather than silently
dropped, so the gap is a documented decision, not an oversight.

Both endpoints share ingest/alpha_vantage_client.py's disk cache + daily
rate limiter with ingest/fundamentals.py's OVERVIEW fallback calls (same
account, same 25-requests/day budget -- news sentiment no longer draws
from it at all, see ingest/sentiment.py). Refreshed MONTHLY by
scripts/refresh_live_data.py -- real companies report quarterly (~every 90
days), so even a 30-day cadence catches a new quarter promptly; ~50
watchlist symbols x 2 endpoints / 30 days ~= 3.3 calls/day extra on top of
OVERVIEW's own ~7/day (weekly, ~50 symbols), comfortably under the shared
cap.

Confirmed live (2026-09, via Alpha Vantage's public `demo` key against
IBM, since this app's own key's daily quota was exhausted at the time):
INCOME_STATEMENT's `quarterlyReports[]` has `fiscalDateEnding`,
`totalRevenue`, `netIncome` (no EPS field there); EARNINGS'
`quarterlyEarnings[]` has `fiscalDateEnding`, `reportedEPS` -- joined here
on `fiscalDateEnding`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import requests

from stock_predictor.ingest import alpha_vantage_client

# See module docstring -- monthly, not weekly like ingest/fundamentals.py's
# P/E, since a quarterly-reported figure changes even less often.
CACHE_TTL_SECONDS = 30 * 24 * 3600

# Fetches one extra quarter beyond what's normally displayed (4, matching
# CNN's own "1-year income & revenue" chart) so a year-over-year
# comparison has something to compare against -- see
# dashboard/components.py::render_financials_section.
DEFAULT_QUARTERS = 5


@dataclass
class QuarterlyFinancials:
    fiscal_date_ending: dt.date
    total_revenue: float | None
    net_income: float | None
    eps: float | None

    @property
    def net_profit_margin(self) -> float | None:
        if self.total_revenue in (None, 0) or self.net_income is None:
            return None
        return self.net_income / self.total_revenue


def _parse_float(value: object) -> float | None:
    # Same "None" sentinel as ingest/fundamentals.py's _parse_float.
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_date(value: object) -> dt.date | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def fetch_quarterly_financials(ticker: str, limit: int = DEFAULT_QUARTERS) -> list[QuarterlyFinancials]:
    """The most recent `limit` quarters, OLDEST FIRST (same convention as
    ingest/price_history.py -- the natural order for a chart timeline).
    Empty list on any failure (missing key, network error, daily budget
    exhausted, unknown ticker, or no data for either endpoint) -- never
    raises, same contract as every other integration here.
    """
    try:
        income_data = alpha_vantage_client.get("INCOME_STATEMENT", {"symbol": ticker}, CACHE_TTL_SECONDS)
        earnings_data = alpha_vantage_client.get("EARNINGS", {"symbol": ticker}, CACHE_TTL_SECONDS)
    except (alpha_vantage_client.MissingApiKey, requests.RequestException, RuntimeError, ValueError):
        return []

    revenue_and_income_by_date: dict[dt.date, tuple[float | None, float | None]] = {}
    for report in income_data.get("quarterlyReports", []):
        date = _parse_date(report.get("fiscalDateEnding"))
        if date is None:
            continue
        revenue_and_income_by_date[date] = (
            _parse_float(report.get("totalRevenue")),
            _parse_float(report.get("netIncome")),
        )

    eps_by_date: dict[dt.date, float | None] = {}
    for entry in earnings_data.get("quarterlyEarnings", []):
        date = _parse_date(entry.get("fiscalDateEnding"))
        if date is None:
            continue
        eps_by_date[date] = _parse_float(entry.get("reportedEPS"))

    most_recent_dates = sorted(revenue_and_income_by_date, reverse=True)[:limit]
    quarters = [
        QuarterlyFinancials(
            fiscal_date_ending=date,
            total_revenue=revenue_and_income_by_date[date][0],
            net_income=revenue_and_income_by_date[date][1],
            eps=eps_by_date.get(date),
        )
        for date in most_recent_dates
    ]
    quarters.reverse()  # oldest-first
    return quarters
