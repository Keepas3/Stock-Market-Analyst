from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stock_predictor.model.return_model import (
    MIN_RETURNS_TO_FIT,
    compute_daily_log_returns,
    fit_symbol,
)


def _price_df(closes: list[float], start: dt.date = dt.date(2026, 1, 1)) -> pd.DataFrame:
    dates = [start + dt.timedelta(days=i) for i in range(len(closes))]
    return pd.DataFrame({"date": dates, "close": closes})


def test_compute_daily_log_returns_shape():
    df = _price_df([100.0, 110.0, 99.0])
    returns = compute_daily_log_returns(df)

    assert len(returns) == 2  # one fewer than input bars
    assert returns["log_return"].iloc[0] == pytest.approx(np.log(110.0 / 100.0))
    assert returns["log_return"].iloc[1] == pytest.approx(np.log(99.0 / 110.0))


def test_fit_symbol_none_when_not_enough_history():
    df = _price_df([100.0] * (MIN_RETURNS_TO_FIT))  # MIN_RETURNS_TO_FIT bars -> MIN_RETURNS_TO_FIT-1 returns
    assert fit_symbol(df) is None


def test_fit_symbol_recovers_synthetic_mean_and_volatility():
    rng = np.random.default_rng(42)
    true_mu, true_sigma = 0.0008, 0.02
    n = 500
    daily_returns = rng.normal(true_mu, true_sigma, n)
    closes = 100.0 * np.exp(np.cumsum(daily_returns))
    df = _price_df(list(closes))

    # xi=0 -- no time-decay, so every observation weighs equally, giving
    # the closest possible recovery of the true generating parameters.
    fit = fit_symbol(df, xi=0.0, as_of=df["date"].iloc[-1])

    assert fit is not None
    assert fit.mu == pytest.approx(true_mu, abs=0.001)
    assert fit.sigma == pytest.approx(true_sigma, rel=0.15)
    assert fit.n_bars == n - 1


def test_fit_symbol_recent_data_dominates_with_high_xi():
    # First half of history: strong negative drift. Second (recent) half:
    # strong positive drift. A high xi should recover something close to
    # the RECENT regime, not an average of both.
    rng = np.random.default_rng(7)
    old_returns = rng.normal(-0.01, 0.01, 200)
    recent_returns = rng.normal(0.01, 0.01, 200)
    all_returns = np.concatenate([old_returns, recent_returns])
    closes = 100.0 * np.exp(np.cumsum(all_returns))
    df = _price_df(list(closes))

    fit = fit_symbol(df, xi=0.05, as_of=df["date"].iloc[-1])  # short half-life

    assert fit is not None
    assert fit.mu > 0  # dominated by the recent positive-drift regime


def test_fit_symbol_carries_through_dof_and_xi():
    df = _price_df([100.0 + i for i in range(60)])
    fit = fit_symbol(df, xi=0.01, dof=7.0)
    assert fit.xi == 0.01
    assert fit.dof == 7.0
