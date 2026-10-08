"""Derives up/flat/down probabilities (and a predicted return range) from
a Student-t return distribution -- the direct analog of soccer-predictor's
model/markets.py (home_win/draw/away_win + top_scorelines from a Poisson/
negative-binomial scoreline grid). One dimension instead of a 2D grid
(there's no "opponent" here), but the same separation of concerns: this
module only derives probabilities from an already-parameterized
distribution, it never touches sentiment or persistence itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import t as student_t

from stock_predictor.model.technical_score import CompositeSignal

# +/-1% cumulative return over the horizon counts as "flat" -- the direct
# analog of a football draw. A tunable constant, not derived from anything.
FLAT_BAND = 0.01


@dataclass
class PredictionBreakdown:
    """Every number that fed the calculation for one symbol's prediction --
    attached by prediction.service.predict_symbol so the dashboard can show
    the actual math, mirroring soccer-predictor's own PredictionBreakdown.
    `base_mean_return`/`base_volatility` are the fitted DAILY figures
    straight from training; `adjusted_mean_return` is the daily figure
    after model.sentiment_adjustment's nudge (identical to base if no
    sentiment reading exists).
    """

    base_mean_return: float
    base_volatility: float
    sentiment_score: float | None
    adjusted_mean_return: float
    horizon_days: int
    dof: float
    xi: float  # time-decay rate used for the underlying fit
    n_bars: int  # daily returns used in the fit
    fitted_at: str  # ISO timestamp of that fit
    # The sentiment-nudge cap actually applied (SENTIMENT_IMPACT_CAP unless
    # overridden for this company -- see model/tuning.py).
    impact_cap: float | None = None


@dataclass
class StockPrediction:
    p_up: float
    p_flat: float
    p_down: float
    expected_return: float  # over the full horizon, not daily
    return_range_80pct: tuple[float, float]  # 10th/90th percentile horizon return -- the "top_scorelines" analog
    breakdown: PredictionBreakdown | None = None
    # The Phase 2 technical/fundamental/dual-sentiment composite (see
    # model/technical_score.py) -- attached by prediction.service.predict_symbol.
    # This, not this class's own P(Up)/P(Down) gap, is the primary
    # Buy/Hold/Sell surfaced on the dashboard as of Phase 2 (see
    # dashboard/components.py::render_composite_breakdown). None until a
    # trained return model exists for this symbol.
    composite: CompositeSignal | None = None


def _t_scale_for_target_std(target_std: float, dof: float) -> float:
    """scipy.stats.t's own `scale` parameter is NOT the distribution's
    standard deviation directly -- a Student-t's variance is
    dof/(dof-2) * scale^2 for dof > 2, so naively passing the target std
    dev as `scale` would make the distribution too wide. Solves for the
    `scale` that actually gives the desired std dev. Falls back to
    `target_std` unchanged for dof <= 2 (variance is undefined/infinite
    there -- shouldn't happen with this app's fixed DEFAULT_DOF, but keeps
    this function total rather than raising on a bad input).
    """
    if dof <= 2:
        return target_std
    return target_std * ((dof - 2) / dof) ** 0.5


def predict_markets(horizon_mean: float, horizon_sigma: float, dof: float, flat_band: float = FLAT_BAND) -> StockPrediction:
    scale = _t_scale_for_target_std(horizon_sigma, dof)
    dist = student_t(df=dof, loc=horizon_mean, scale=scale)

    p_down = float(dist.cdf(-flat_band))
    p_up = float(1.0 - dist.cdf(flat_band))
    p_flat = max(0.0, 1.0 - p_down - p_up)

    lo, hi = float(dist.ppf(0.1)), float(dist.ppf(0.9))

    return StockPrediction(
        p_up=p_up,
        p_flat=p_flat,
        p_down=p_down,
        expected_return=horizon_mean,
        return_range_80pct=(lo, hi),
    )
