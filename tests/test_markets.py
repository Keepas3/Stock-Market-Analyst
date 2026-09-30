from __future__ import annotations

import pytest

from stock_predictor.model.markets import (
    FLAT_BAND,
    StockPrediction,
    _t_scale_for_target_std,
    predict_markets,
)


def test_probabilities_sum_to_one():
    prediction = predict_markets(horizon_mean=0.01, horizon_sigma=0.05, dof=5.0)
    assert prediction.p_up + prediction.p_flat + prediction.p_down == pytest.approx(1.0)


def test_positive_expected_return_favors_up():
    prediction = predict_markets(horizon_mean=0.05, horizon_sigma=0.03, dof=5.0)
    assert prediction.p_up > prediction.p_down


def test_negative_expected_return_favors_down():
    prediction = predict_markets(horizon_mean=-0.05, horizon_sigma=0.03, dof=5.0)
    assert prediction.p_down > prediction.p_up


def test_zero_mean_is_roughly_symmetric():
    prediction = predict_markets(horizon_mean=0.0, horizon_sigma=0.03, dof=5.0)
    assert prediction.p_up == pytest.approx(prediction.p_down, abs=1e-9)


def test_return_range_brackets_the_expected_return():
    prediction = predict_markets(horizon_mean=0.02, horizon_sigma=0.04, dof=5.0)
    lo, hi = prediction.return_range_80pct
    assert lo < prediction.expected_return < hi


def test_wider_volatility_widens_the_return_range():
    tight = predict_markets(horizon_mean=0.0, horizon_sigma=0.02, dof=5.0)
    wide = predict_markets(horizon_mean=0.0, horizon_sigma=0.10, dof=5.0)

    tight_width = tight.return_range_80pct[1] - tight.return_range_80pct[0]
    wide_width = wide.return_range_80pct[1] - wide.return_range_80pct[0]
    assert wide_width > tight_width


def test_lower_dof_gives_more_extreme_tail_probability_than_higher_dof():
    # Same mean/std dev, different degrees of freedom -- the same
    # "overdispersion widens the tail" lesson soccer-predictor already
    # applied to scorelines (see model/scoreline_matrix.py there). A
    # Student-t normalized to a FIXED std dev is actually *narrower* than
    # a thin-tailed one in its everyday 80% range (more of its variance
    # budget goes into rare extreme events, so the ordinary-day bulk is
    # more concentrated) -- confirmed live while writing this test, not
    # assumed -- so the real "fatter tails" claim has to be checked at a
    # genuinely extreme percentile, where the effect actually dominates.
    fat_tailed = predict_markets(horizon_mean=0.0, horizon_sigma=0.03, dof=3.0)
    thin_tailed = predict_markets(horizon_mean=0.0, horizon_sigma=0.03, dof=30.0)

    from stock_predictor.model.markets import _t_scale_for_target_std
    from scipy.stats import t as student_t

    fat_dist = student_t(df=3.0, scale=_t_scale_for_target_std(0.03, 3.0))
    thin_dist = student_t(df=30.0, scale=_t_scale_for_target_std(0.03, 30.0))

    assert fat_dist.ppf(0.999) > thin_dist.ppf(0.999)
    # Unambiguous either way: excess kurtosis is strictly higher for a
    # lower degrees-of-freedom Student-t, regardless of percentile choice.
    assert fat_dist.stats(moments="k") > thin_dist.stats(moments="k")


def test_flat_band_default_matches_module_constant():
    assert FLAT_BAND == 0.01


def test_t_scale_matches_requested_standard_deviation():
    from scipy.stats import t as student_t

    target_std = 0.05
    dof = 6.0
    scale = _t_scale_for_target_std(target_std, dof)
    actual_std = float(student_t(df=dof, scale=scale).std())
    assert actual_std == pytest.approx(target_std, rel=1e-6)


def test_t_scale_falls_back_for_low_dof():
    assert _t_scale_for_target_std(0.05, dof=2.0) == 0.05


def _prediction(p_up: float, p_down: float) -> StockPrediction:
    return StockPrediction(
        p_up=p_up,
        p_flat=1.0 - p_up - p_down,
        p_down=p_down,
        expected_return=0.0,
        return_range_80pct=(-0.01, 0.01),
    )


def test_composite_defaults_to_none():
    # StockPrediction no longer derives its own Buy/Hold/Sell from
    # P(Up)/P(Down) (see model/technical_score.py's CompositeSignal,
    # attached by prediction.service.predict_symbol instead) -- confirms
    # the field exists and is opt-in, not silently required.
    prediction = _prediction(p_up=0.6, p_down=0.1)
    assert prediction.composite is None
