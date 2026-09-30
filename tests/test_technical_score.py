from __future__ import annotations

from stock_predictor.model.technical_score import (
    BUY_VOTE_THRESHOLD,
    SELL_VOTE_THRESHOLD,
    SENTIMENT_VOTE_BEARISH,
    SENTIMENT_VOTE_BULLISH,
    compute_recommendation,
    sentiment_vote,
)


def test_sentiment_vote_bullish_above_threshold():
    assert sentiment_vote(SENTIMENT_VOTE_BULLISH) == 1


def test_sentiment_vote_bearish_below_threshold():
    assert sentiment_vote(SENTIMENT_VOTE_BEARISH) == -1


def test_sentiment_vote_neutral_between_thresholds():
    assert sentiment_vote(0.0) == 0


def test_sentiment_vote_none_is_neutral():
    assert sentiment_vote(None) == 0


def test_compute_recommendation_buy_at_threshold():
    signal = compute_recommendation(ma_vote=1, pe_vote=1, news_vote=0, social_vote=0)
    assert signal.total == BUY_VOTE_THRESHOLD
    assert signal.recommendation == "Buy"


def test_compute_recommendation_sell_at_threshold():
    signal = compute_recommendation(ma_vote=-1, pe_vote=-1, news_vote=0, social_vote=0)
    assert signal.total == SELL_VOTE_THRESHOLD
    assert signal.recommendation == "Sell"


def test_compute_recommendation_hold_when_total_between_thresholds():
    signal = compute_recommendation(ma_vote=1, pe_vote=0, news_vote=0, social_vote=0)
    assert signal.recommendation == "Hold"


def test_compute_recommendation_all_bullish_is_buy():
    signal = compute_recommendation(ma_vote=1, pe_vote=1, news_vote=1, social_vote=1)
    assert signal.total == 4
    assert signal.recommendation == "Buy"


def test_compute_recommendation_all_bearish_is_sell():
    signal = compute_recommendation(ma_vote=-1, pe_vote=-1, news_vote=-1, social_vote=-1)
    assert signal.total == -4
    assert signal.recommendation == "Sell"


def test_compute_recommendation_all_missing_data_is_hold():
    signal = compute_recommendation(ma_vote=0, pe_vote=0, news_vote=0, social_vote=0)
    assert signal.total == 0
    assert signal.recommendation == "Hold"


def test_compute_recommendation_carries_individual_votes():
    signal = compute_recommendation(ma_vote=1, pe_vote=-1, news_vote=0, social_vote=1)
    assert (signal.ma_vote, signal.pe_vote, signal.news_vote, signal.social_vote) == (1, -1, 0, 1)
    assert signal.total == 1
