"""The one module the dashboard calls into for predictions. Reads the
latest persisted return-model fit, applies the (cheap) sentiment
adjustment for the secondary Student-t forecast, and separately computes
the Phase 2 composite technical/fundamental/dual-sentiment vote (see
model/technical_score.py) -- the primary Buy/Hold/Sell signal as of Phase 2.
A sentiment/fundamentals/social refresh never requires re-fitting the
return model.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from stock_predictor.model.markets import PredictionBreakdown, StockPrediction, predict_markets
from stock_predictor.model.sentiment_adjustment import adjust_for_sentiment
from stock_predictor.model.technical_indicators import compute_moving_averages, ma_signal, pe_signal
from stock_predictor.model.technical_score import compute_recommendation, sentiment_vote
from stock_predictor.storage.repository import (
    latest_fundamentals,
    latest_return_model_params,
    latest_sentiment,
    latest_social_sentiment,
    price_bars_for_symbol,
)

HORIZON_DAYS = 5


def predict_symbol(session: Session, symbol_id: int) -> StockPrediction | None:
    """None if this symbol has no trained return model yet (see
    prediction.training.train_symbol) -- same "nothing to show yet"
    contract as soccer-predictor's own predict_fixture callers already
    handle (load_latest_params returning None).
    """
    params = latest_return_model_params(session, symbol_id)
    if params is None:
        return None

    sentiment_row = latest_sentiment(session, symbol_id)
    sentiment_score = sentiment_row.overall_sentiment_score if sentiment_row is not None else None

    adjusted_daily_mean = adjust_for_sentiment(params.mu, sentiment_score)

    horizon_mean = adjusted_daily_mean * HORIZON_DAYS
    horizon_sigma = params.sigma * (HORIZON_DAYS**0.5)

    prediction = predict_markets(horizon_mean, horizon_sigma, params.dof)
    prediction.breakdown = PredictionBreakdown(
        base_mean_return=params.mu,
        base_volatility=params.sigma,
        sentiment_score=sentiment_score,
        adjusted_mean_return=adjusted_daily_mean,
        horizon_days=HORIZON_DAYS,
        dof=params.dof,
        xi=params.xi,
        n_bars=params.n_bars,
        fitted_at=params.fitted_at.isoformat(),
    )

    price_df = price_bars_for_symbol(session, symbol_id)
    sma_short, sma_long = compute_moving_averages(price_df)
    ma_vote = ma_signal(sma_short, sma_long)

    fundamentals_row = latest_fundamentals(session, symbol_id)
    pe_ratio = fundamentals_row.pe_ratio if fundamentals_row is not None else None
    pe_vote = pe_signal(pe_ratio)

    news_vote = sentiment_vote(sentiment_score)

    social_row = latest_social_sentiment(session, symbol_id)
    social_score = social_row.overall_sentiment_score if social_row is not None else None
    social_vote = sentiment_vote(social_score)

    prediction.composite = compute_recommendation(ma_vote, pe_vote, news_vote, social_vote)
    return prediction
