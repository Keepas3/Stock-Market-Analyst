from __future__ import annotations

from stock_predictor.model.sentiment_adjustment import SENTIMENT_IMPACT_CAP, adjust_for_sentiment


def test_none_leaves_return_unchanged():
    assert adjust_for_sentiment(0.001, None) == 0.001


def test_bullish_sentiment_raises_return():
    assert adjust_for_sentiment(0.001, sentiment_score=0.8) > 0.001


def test_bearish_sentiment_lowers_return():
    assert adjust_for_sentiment(0.001, sentiment_score=-0.8) < 0.001


def test_zero_sentiment_leaves_return_unchanged():
    assert adjust_for_sentiment(0.001, sentiment_score=0.0) == 0.001


def test_cap_respected_in_both_directions():
    assert adjust_for_sentiment(0.001, sentiment_score=5.0) == 0.001 + SENTIMENT_IMPACT_CAP
    assert adjust_for_sentiment(0.001, sentiment_score=-5.0) == 0.001 - SENTIMENT_IMPACT_CAP


def test_bullish_sentiment_correctly_signed_for_a_negative_base_return():
    # Regression guard: a naive multiplicative "+/-15% of base" design
    # would make a NEGATIVE base return MORE negative under bullish
    # sentiment (wrong sign) -- this must always move toward zero/positive.
    adjusted = adjust_for_sentiment(-0.002, sentiment_score=1.0)
    assert adjusted > -0.002


def test_bearish_sentiment_correctly_signed_for_a_negative_base_return():
    adjusted = adjust_for_sentiment(-0.002, sentiment_score=-1.0)
    assert adjusted < -0.002


def test_works_correctly_for_a_zero_base_return():
    assert adjust_for_sentiment(0.0, sentiment_score=1.0) == SENTIMENT_IMPACT_CAP
