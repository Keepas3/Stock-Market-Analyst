from __future__ import annotations

import pytest

from stock_predictor.model.sentiment_adjustment import SENTIMENT_IMPACT_CAP, adjust_for_sentiment
from stock_predictor.model.technical_indicators import pe_signal
from stock_predictor.model.technical_score import compute_recommendation, sentiment_vote
from stock_predictor.model.tuning import (
    PARAM_SPECS,
    TunableParams,
    coerce_value,
    combined_errors,
    resolve_params,
    sanitize_overrides,
)


def test_no_overrides_resolves_to_defaults():
    params = resolve_params({})
    assert params == TunableParams()
    assert params.overridden() == {}
    assert params.weights == (1.0, 1.0, 1.0, 1.0)


def test_zero_override_recommendation_matches_plain_vote_sum():
    for votes in [(1, 1, 0, 0), (-1, -1, 0, 0), (1, -1, 1, 1), (0, 0, 0, 0), (1, 1, 1, 1)]:
        plain = sum(votes)
        signal = compute_recommendation(*votes)
        assert signal.total == plain
        expected = "Buy" if plain >= 2 else "Sell" if plain <= -2 else "Hold"
        assert signal.recommendation == expected


def test_weights_change_the_verdict():
    assert compute_recommendation(1, 1, 0, 0).recommendation == "Buy"
    halved = compute_recommendation(1, 1, 0, 0, weights=(0.5, 0.5, 1.0, 1.0))
    assert halved.total == 1.0
    assert halved.recommendation == "Hold"


def test_quarter_step_weights_land_exactly_on_the_threshold():
    signal = compute_recommendation(1, 1, 1, 0, weights=(0.75, 0.75, 0.5, 1.0))
    assert signal.total == 2.0
    assert signal.recommendation == "Buy"


def test_custom_thresholds_are_used_and_recorded():
    signal = compute_recommendation(1, 0, 0, 0, buy_threshold=1.0, sell_threshold=-3.0)
    assert signal.recommendation == "Buy"
    assert signal.buy_threshold == 1.0
    assert signal.sell_threshold == -3.0


def test_pe_signal_defaults_unchanged_and_overridable():
    assert pe_signal(14) == 1
    assert pe_signal(20) == 0
    assert pe_signal(20, value_threshold=25) == 1
    assert pe_signal(35, expensive_threshold=40) == 0


def test_sentiment_vote_with_wider_cutoff():
    assert sentiment_vote(0.2) == 1
    assert sentiment_vote(0.2, 0.3, -0.3) == 0


def test_adjust_for_sentiment_cap_override():
    default = adjust_for_sentiment(0.0, 1.0)
    assert default == pytest.approx(SENTIMENT_IMPACT_CAP)
    assert adjust_for_sentiment(0.0, 1.0, cap=0.002) == pytest.approx(0.002)


def test_coerce_value_rejects_unknown_nonnumeric_and_out_of_range():
    with pytest.raises(ValueError, match="Unknown parameter"):
        coerce_value("nope", 1)
    with pytest.raises(ValueError, match="must be a number"):
        coerce_value("ma_weight", "high")
    with pytest.raises(ValueError, match="must be a number"):
        coerce_value("ma_weight", True)
    with pytest.raises(ValueError, match="between"):
        coerce_value("ma_weight", 5)
    with pytest.raises(ValueError, match="finite"):
        coerce_value("ma_weight", float("nan"))


def test_coerce_value_snaps_to_step():
    assert coerce_value("ma_weight", 0.6) == 0.5
    assert coerce_value("buy_threshold", 2.3) == 2.5


def test_sanitize_overrides_drops_junk_and_clamps():
    clean = sanitize_overrides({"ma_weight": 99, "bogus": 1, "pe_weight": "x", "news_weight": 0.5})
    assert clean == {"ma_weight": PARAM_SPECS["ma_weight"].maximum, "news_weight": 0.5}


def test_inconsistent_override_set_falls_back_to_defaults():
    assert combined_errors(TunableParams(pe_value_threshold=30, pe_expensive_threshold=20))
    assert resolve_params({"pe_value_threshold": 30, "pe_expensive_threshold": 20}) == TunableParams()


def test_overridden_lists_only_changed_fields():
    params = resolve_params({"news_weight": 0.5, "ma_weight": 1.0})
    assert params.overridden() == {"news_weight": 0.5}
