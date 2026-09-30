"""Refits one symbol's return-model params from its stored price history --
the direct analog of soccer-predictor's prediction/training.py, just per
SYMBOL instead of per LEAGUE (see model/return_model.py's own docstring
for why that's independently fit rather than jointly).
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from stock_predictor.model.return_model import ReturnModelFit, fit_symbol
from stock_predictor.storage.repository import price_bars_for_symbol, save_return_model_params


def train_symbol(session: Session, symbol_id: int) -> ReturnModelFit | None:
    """None if there isn't enough price history yet (see
    model.return_model.MIN_RETURNS_TO_FIT) -- no params row is written in
    that case, same "not enough data" contract as the rest of this app.
    """
    price_df = price_bars_for_symbol(session, symbol_id)
    fit = fit_symbol(price_df)
    if fit is None:
        return None
    save_return_model_params(session, symbol_id, fit.mu, fit.sigma, fit.dof, fit.xi, fit.n_bars)
    return fit
