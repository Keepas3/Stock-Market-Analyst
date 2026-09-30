from __future__ import annotations

import datetime as dt

import requests

from stock_predictor.ingest import alpha_vantage_client, financials


def _income_report(fiscal_date, total_revenue, net_income):
    return {"fiscalDateEnding": fiscal_date, "totalRevenue": str(total_revenue), "netIncome": str(net_income)}


def _earnings_entry(fiscal_date, eps):
    return {"fiscalDateEnding": fiscal_date, "reportedEPS": str(eps)}


def test_fetch_quarterly_financials_joins_income_and_eps_by_date(monkeypatch):
    def fake_get(function, params, cache_ttl_seconds):
        if function == "INCOME_STATEMENT":
            return {
                "quarterlyReports": [
                    _income_report("2026-06-30", 96220000000, 59690000000),
                    _income_report("2026-03-31", 81620000000, 59690000000),
                ]
            }
        if function == "EARNINGS":
            return {
                "quarterlyEarnings": [
                    _earnings_entry("2026-06-30", 2.46),
                    _earnings_entry("2026-03-31", 2.44),
                ]
            }
        raise AssertionError(f"unexpected function {function}")

    monkeypatch.setattr(alpha_vantage_client, "get", fake_get)

    quarters = financials.fetch_quarterly_financials("NVDA", limit=2)

    assert [q.fiscal_date_ending for q in quarters] == [dt.date(2026, 3, 31), dt.date(2026, 6, 30)]  # oldest-first
    latest = quarters[-1]
    assert latest.total_revenue == 96220000000
    assert latest.net_income == 59690000000
    assert latest.eps == 2.46


def test_fetch_quarterly_financials_respects_limit(monkeypatch):
    def fake_get(function, params, cache_ttl_seconds):
        if function == "INCOME_STATEMENT":
            return {"quarterlyReports": [_income_report(f"2026-0{i}-01", i * 100, i * 10) for i in range(1, 7)]}
        return {"quarterlyEarnings": []}

    monkeypatch.setattr(alpha_vantage_client, "get", fake_get)

    quarters = financials.fetch_quarterly_financials("NVDA", limit=3)
    assert len(quarters) == 3


def test_fetch_quarterly_financials_missing_eps_entry_is_none(monkeypatch):
    def fake_get(function, params, cache_ttl_seconds):
        if function == "INCOME_STATEMENT":
            return {"quarterlyReports": [_income_report("2026-06-30", 100, 10)]}
        return {"quarterlyEarnings": []}  # no matching EPS entry

    monkeypatch.setattr(alpha_vantage_client, "get", fake_get)

    quarters = financials.fetch_quarterly_financials("NVDA")
    assert quarters[0].eps is None


def test_fetch_quarterly_financials_returns_empty_list_on_missing_api_key(monkeypatch):
    def raise_missing_key(*a, **k):
        raise alpha_vantage_client.MissingApiKey("no key")

    monkeypatch.setattr(alpha_vantage_client, "get", raise_missing_key)
    assert financials.fetch_quarterly_financials("NVDA") == []


def test_fetch_quarterly_financials_returns_empty_list_on_request_exception(monkeypatch):
    def raise_error(*a, **k):
        raise requests.RequestException("network error")

    monkeypatch.setattr(alpha_vantage_client, "get", raise_error)
    assert financials.fetch_quarterly_financials("NVDA") == []


def test_quarterly_financials_net_profit_margin_computed():
    q = financials.QuarterlyFinancials(fiscal_date_ending=dt.date(2026, 6, 30), total_revenue=200.0, net_income=50.0, eps=1.0)
    assert q.net_profit_margin == 0.25


def test_quarterly_financials_net_profit_margin_none_when_revenue_missing():
    q = financials.QuarterlyFinancials(fiscal_date_ending=dt.date(2026, 6, 30), total_revenue=None, net_income=50.0, eps=1.0)
    assert q.net_profit_margin is None


def test_quarterly_financials_net_profit_margin_none_when_revenue_zero():
    q = financials.QuarterlyFinancials(fiscal_date_ending=dt.date(2026, 6, 30), total_revenue=0.0, net_income=50.0, eps=1.0)
    assert q.net_profit_margin is None
