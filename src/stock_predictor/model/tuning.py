"""The recommendation math's tunable knobs, their defaults, and their hard
bounds. Every default is imported from the module that already owns that
constant, so with zero overrides the app behaves exactly as it did before
tuning existed. Per-company overrides come from model/overrides_store.py
(a git-tracked YAML file, hand-editable) and are applied in
prediction/service.py::predict_symbol -- the one place every consumer
(Companies table, Symbol Detail, the Discord alert cron) gets its numbers.

Bounds are a safety rail for the AI assistant's write tools, not a claim
about what is "correct": a model that can set any number could trivially
force a Buy on anything. Values are snapped to each parameter's step so
fractional-weight sums never land on a float edge at a threshold.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields

from stock_predictor.model.sentiment_adjustment import SENTIMENT_IMPACT_CAP
from stock_predictor.model.technical_indicators import PE_EXPENSIVE_THRESHOLD, PE_VALUE_THRESHOLD
from stock_predictor.model.technical_score import (
    BUY_VOTE_THRESHOLD,
    SELL_VOTE_THRESHOLD,
    SENTIMENT_VOTE_BEARISH,
    SENTIMENT_VOTE_BULLISH,
)

# technical_score.py keeps separate bullish/bearish constants; tuning exposes
# one symmetric cutoff per sentiment signal, which only matches if they are.
assert SENTIMENT_VOTE_BULLISH == -SENTIMENT_VOTE_BEARISH


@dataclass(frozen=True)
class ParamSpec:
    name: str
    default: float
    minimum: float
    maximum: float
    step: float
    description: str


PARAM_SPECS: dict[str, ParamSpec] = {
    spec.name: spec
    for spec in (
        ParamSpec(
            "pe_value_threshold",
            PE_VALUE_THRESHOLD,
            5,
            40,
            1,
            "P/E below this casts a bullish (+1) 'cheap' vote.",
        ),
        ParamSpec(
            "pe_expensive_threshold",
            PE_EXPENSIVE_THRESHOLD,
            15,
            80,
            1,
            "P/E above this casts a bearish (-1) 'expensive' vote. Must exceed pe_value_threshold.",
        ),
        ParamSpec(
            "news_sentiment_cutoff",
            SENTIMENT_VOTE_BULLISH,
            0.02,
            0.6,
            0.01,
            "News sentiment score must reach +cutoff for a bullish vote or -cutoff for a bearish one.",
        ),
        ParamSpec(
            "social_sentiment_cutoff",
            SENTIMENT_VOTE_BULLISH,
            0.02,
            0.6,
            0.01,
            "Same as news_sentiment_cutoff, for the social-media sentiment score.",
        ),
        ParamSpec("ma_weight", 1.0, 0, 2, 0.25, "Weight of the moving-average (Golden/Death Cross) vote."),
        ParamSpec("pe_weight", 1.0, 0, 2, 0.25, "Weight of the P/E valuation vote."),
        ParamSpec("news_weight", 1.0, 0, 2, 0.25, "Weight of the news-sentiment vote."),
        ParamSpec("social_weight", 1.0, 0, 2, 0.25, "Weight of the social-sentiment vote."),
        ParamSpec(
            "buy_threshold",
            BUY_VOTE_THRESHOLD,
            0.5,
            4,
            0.5,
            "Weighted vote total at or above this is a Buy. Must be positive.",
        ),
        ParamSpec(
            "sell_threshold",
            SELL_VOTE_THRESHOLD,
            -4,
            -0.5,
            0.5,
            "Weighted vote total at or below this is a Sell. Must be negative.",
        ),
        ParamSpec(
            "sentiment_impact_cap",
            SENTIMENT_IMPACT_CAP,
            0,
            0.003,
            0.0001,
            "Max daily-return nudge (absolute) the news sentiment may add to the secondary statistical forecast.",
        ),
    )
}


@dataclass(frozen=True)
class TunableParams:
    pe_value_threshold: float = PARAM_SPECS["pe_value_threshold"].default
    pe_expensive_threshold: float = PARAM_SPECS["pe_expensive_threshold"].default
    news_sentiment_cutoff: float = PARAM_SPECS["news_sentiment_cutoff"].default
    social_sentiment_cutoff: float = PARAM_SPECS["social_sentiment_cutoff"].default
    ma_weight: float = PARAM_SPECS["ma_weight"].default
    pe_weight: float = PARAM_SPECS["pe_weight"].default
    news_weight: float = PARAM_SPECS["news_weight"].default
    social_weight: float = PARAM_SPECS["social_weight"].default
    buy_threshold: float = PARAM_SPECS["buy_threshold"].default
    sell_threshold: float = PARAM_SPECS["sell_threshold"].default
    sentiment_impact_cap: float = PARAM_SPECS["sentiment_impact_cap"].default

    @property
    def weights(self) -> tuple[float, float, float, float]:
        return (self.ma_weight, self.pe_weight, self.news_weight, self.social_weight)

    def overridden(self) -> dict[str, float]:
        """Only the fields that differ from their defaults."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) != PARAM_SPECS[f.name].default
        }


def snap(name: str, value: float) -> float:
    """Round to the parameter's step (and float-noise-free)."""
    step = PARAM_SPECS[name].step
    return round(round(value / step) * step, 6)


def coerce_value(name: str, value: object) -> float:
    """Strict validation for values coming from the AI or the UI: raises
    ValueError (with a message safe to hand back to the model) rather than
    silently clamping, so a bad request is visible instead of half-applied.
    """
    spec = PARAM_SPECS.get(name)
    if spec is None:
        raise ValueError(f"Unknown parameter '{name}'. Valid parameters: {', '.join(PARAM_SPECS)}.")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    if number < spec.minimum or number > spec.maximum:
        raise ValueError(f"{name} must be between {spec.minimum:g} and {spec.maximum:g}, got {number:g}.")
    return snap(name, number)


def combined_errors(params: TunableParams) -> list[str]:
    errors = []
    if params.pe_expensive_threshold <= params.pe_value_threshold:
        errors.append("pe_expensive_threshold must be greater than pe_value_threshold.")
    if params.buy_threshold <= 0:
        errors.append("buy_threshold must be positive.")
    if params.sell_threshold >= 0:
        errors.append("sell_threshold must be negative.")
    return errors


def sanitize_overrides(raw: Mapping[str, object] | None) -> dict[str, float]:
    """Lenient counterpart used when LOADING (the YAML file can be
    hand-edited): unknown names and non-numbers are dropped, out-of-range
    numbers are clamped, everything is snapped to its step.
    """
    clean: dict[str, float] = {}
    for name, value in (raw or {}).items():
        spec = PARAM_SPECS.get(name)
        if spec is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if not math.isfinite(number):
            continue
        clean[name] = snap(name, min(spec.maximum, max(spec.minimum, number)))
    return clean


def resolve_params(overrides: Mapping[str, object] | None) -> TunableParams:
    """Defaults + sanitized overrides. An override set that is internally
    inconsistent (e.g. hand-edited so P/E 'expensive' < 'cheap') is ignored
    wholesale rather than crashing a page or the cron.
    """
    params = TunableParams(**sanitize_overrides(overrides))
    if combined_errors(params):
        return TunableParams()
    return params
