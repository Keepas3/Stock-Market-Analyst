from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from stock_predictor.model.technical_indicators import SMA_LONG_WINDOW
from stock_predictor.prediction.service import HORIZON_DAYS, predict_symbol
from stock_predictor.storage.models import Base
from stock_predictor.storage.repository import (
    get_or_create_symbol,
    replace_fundamentals_snapshot,
    replace_sentiment_snapshot,
    replace_social_sentiment_snapshot,
    save_return_model_params,
    upsert_price_bar,
)


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_predict_symbol_none_without_a_trained_model(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    session.commit()

    assert predict_symbol(session, symbol.id) is None


def test_predict_symbol_uses_base_fit_when_no_sentiment(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    save_return_model_params(session, symbol.id, mu=0.001, sigma=0.02, dof=5.0, xi=0.008, n_bars=200)
    session.commit()

    prediction = predict_symbol(session, symbol.id)

    assert prediction is not None
    assert prediction.breakdown.sentiment_score is None
    assert prediction.breakdown.adjusted_mean_return == prediction.breakdown.base_mean_return
    assert prediction.expected_return == pytest.approx(0.001 * HORIZON_DAYS)


def test_predict_symbol_applies_sentiment_adjustment(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    save_return_model_params(session, symbol.id, mu=0.001, sigma=0.02, dof=5.0, xi=0.008, n_bars=200)
    replace_sentiment_snapshot(session, symbol.id, dt.date.today(), overall_sentiment_score=0.9, article_count=3)
    session.commit()

    prediction = predict_symbol(session, symbol.id)

    assert prediction.breakdown.sentiment_score == 0.9
    assert prediction.breakdown.adjusted_mean_return > prediction.breakdown.base_mean_return
    assert prediction.expected_return > 0.001 * HORIZON_DAYS


def test_predict_symbol_probabilities_sum_to_one(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    save_return_model_params(session, symbol.id, mu=0.0005, sigma=0.015, dof=5.0, xi=0.008, n_bars=200)
    session.commit()

    prediction = predict_symbol(session, symbol.id)

    assert prediction.p_up + prediction.p_flat + prediction.p_down == pytest.approx(1.0)


def test_predict_symbol_breakdown_carries_provenance(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    save_return_model_params(session, symbol.id, mu=0.0005, sigma=0.015, dof=5.0, xi=0.008, n_bars=321)
    session.commit()

    prediction = predict_symbol(session, symbol.id)

    assert prediction.breakdown.n_bars == 321
    assert prediction.breakdown.horizon_days == HORIZON_DAYS
    assert prediction.breakdown.dof == 5.0
    assert prediction.breakdown.xi == 0.008


def test_predict_symbol_composite_all_neutral_with_no_extra_data(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    save_return_model_params(session, symbol.id, mu=0.0005, sigma=0.015, dof=5.0, xi=0.008, n_bars=200)
    session.commit()

    prediction = predict_symbol(session, symbol.id)

    assert prediction.composite is not None
    assert prediction.composite.total == 0
    assert prediction.composite.recommendation == "Hold"


def test_predict_symbol_composite_reflects_golden_cross(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    save_return_model_params(session, symbol.id, mu=0.0005, sigma=0.015, dof=5.0, xi=0.008, n_bars=200)
    # A rising price series over SMA_LONG_WINDOW+some bars -- recent
    # (short-window) closes end up above the long-window average, a
    # genuine golden cross.
    base_date = dt.date(2020, 1, 1)
    for i in range(SMA_LONG_WINDOW + 10):
        upsert_price_bar(session, symbol.id, base_date + dt.timedelta(days=i), 1, 1, 1, 100.0 + i, 1)
    session.commit()

    prediction = predict_symbol(session, symbol.id)

    assert prediction.composite.ma_vote == 1


def test_predict_symbol_composite_reflects_pe_and_sentiment(session):
    symbol = get_or_create_symbol(session, "AAPL", "Apple Inc.")
    save_return_model_params(session, symbol.id, mu=0.0005, sigma=0.015, dof=5.0, xi=0.008, n_bars=200)
    replace_fundamentals_snapshot(session, symbol.id, dt.date.today(), pe_ratio=10.0, forward_pe=9.0)
    replace_sentiment_snapshot(session, symbol.id, dt.date.today(), overall_sentiment_score=0.5, article_count=3)
    replace_social_sentiment_snapshot(session, symbol.id, dt.date.today(), overall_sentiment_score=0.5, message_count=10)
    session.commit()

    prediction = predict_symbol(session, symbol.id)

    assert prediction.composite.pe_vote == 1
    assert prediction.composite.news_vote == 1
    assert prediction.composite.social_vote == 1
    assert prediction.composite.recommendation == "Buy"
