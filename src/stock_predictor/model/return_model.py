"""Per-symbol daily log-return distribution fit -- the Dixon-Coles analog,
genuinely simpler: unlike football (every team's attack/defense fit
*jointly* within one league, against each other), each symbol here is
fit **independently** from nothing but its own price history, so this is
closed-form (weighted mean/variance), not an MLE optimizer.

DEFAULT_DOF is a fixed Student-t degrees-of-freedom constant, not fit per
symbol -- the direct analog of soccer-predictor's
model/scoreline_matrix.py::DEFAULT_DISPERSION decision: real daily equity
returns are well-known to be fat-tailed relative to Normal (the same
"plain-Poisson-underestimates-blowouts" lesson, applied here), and a fixed
heuristic constant is a documented, deliberate Phase 1 scope choice, not
an oversight -- per-symbol dof fitting is a natural Phase 2 refinement.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_predictor.model.time_weighting import DEFAULT_XI, bar_weights

DEFAULT_DOF = 5.0
# Below this many daily returns, a weighted mean/variance is too noisy to
# trust -- fit_symbol returns None rather than a wild estimate from a
# recently-added symbol with barely any history yet.
MIN_RETURNS_TO_FIT = 30


@dataclass
class ReturnModelFit:
    mu: float  # weighted mean daily log return
    sigma: float  # weighted std dev of daily log return
    dof: float
    xi: float
    n_bars: int  # daily returns used (== price bars used - 1)


def compute_daily_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """`price_df` needs at least `date`/`close` columns (see
    storage.repository.price_bars_for_symbol) -- returns `date`/`log_return`,
    one row shorter than the input (the first bar has no prior close to
    compare against).
    """
    df = price_df.sort_values("date").reset_index(drop=True)
    log_return = np.log(df["close"] / df["close"].shift(1))
    return pd.DataFrame({"date": df["date"], "log_return": log_return}).dropna().reset_index(drop=True)


def fit_symbol(
    price_df: pd.DataFrame,
    xi: float = DEFAULT_XI,
    dof: float = DEFAULT_DOF,
    as_of: dt.date | None = None,
) -> ReturnModelFit | None:
    """None if there isn't enough price history yet (MIN_RETURNS_TO_FIT) --
    callers (prediction/training.py) skip persisting a fit in that case,
    same "not enough data yet" degrade-gracefully contract as everywhere
    else in this app.
    """
    returns_df = compute_daily_log_returns(price_df)
    if len(returns_df) < MIN_RETURNS_TO_FIT:
        return None

    weights = bar_weights(returns_df["date"], as_of=as_of, xi=xi)
    weights = weights / weights.sum()

    mu = float(np.sum(weights * returns_df["log_return"]))
    variance = float(np.sum(weights * (returns_df["log_return"] - mu) ** 2))
    sigma = float(np.sqrt(max(variance, 0.0)))

    return ReturnModelFit(mu=mu, sigma=sigma, dof=dof, xi=xi, n_bars=len(returns_df))
