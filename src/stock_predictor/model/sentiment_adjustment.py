"""Applies a small, capped nudge to the predicted mean return from a
symbol's latest news-sentiment reading (ingest/sentiment.py) -- the direct
analog of soccer-predictor's model/current_form_adjustment.py.

News sentiment moving a stock's price is a real, well-studied effect
*direction-wise*, but this app has no way to validate the *magnitude* any
more than soccer-predictor's own form_adjustment.py could validate its
hand-typed form notes -- treat this as a deliberately small, capped nudge,
not a modeled/backtested signal, and say so wherever it's shown in the UI,
not just here (same instruction that module's own docstring gives itself).
"""

from __future__ import annotations

# ADDITIVE, not a relative/multiplicative cap like current_form_adjustment.py's
# -- deliberately different, and not just for variety: a daily log RETURN
# can be positive, negative, or ~zero, so "+/-15% of the base value" is
# ill-defined (and actively wrong-signed for a negative base -- multiplying
# a negative return by a larger factor makes it MORE negative, the
# opposite of what a bullish nudge should do). A fixed absolute shift in
# return-space has no such sign problem. 0.001 = 10 basis points/day,
# modest but real; scaled up by the prediction horizon like the base mean
# return itself is (see prediction/service.py).
SENTIMENT_IMPACT_CAP = 0.001


def adjust_for_sentiment(
    base_mean_return: float, sentiment_score: float | None, cap: float = SENTIMENT_IMPACT_CAP
) -> float:
    """`sentiment_score` is VADER's compound score, -1 (bearish) to +1
    (bullish) (see ingest/sentiment.py::SentimentResult). None (no
    reading available) leaves `base_mean_return` untouched. `cap` defaults
    to SENTIMENT_IMPACT_CAP; model/tuning.py supplies per-company overrides.
    """
    if sentiment_score is None:
        return base_mean_return
    shift = max(-1.0, min(1.0, sentiment_score)) * cap
    return base_mean_return + shift
