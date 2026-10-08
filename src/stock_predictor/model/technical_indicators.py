"""Technical/fundamental signal derivation. Moving averages are computed
locally from already-ingested price history -- no new data source needed
-- and P/E gets a fixed-band heuristic. Each function here produces a
discrete vote (+1 bullish / -1 bearish / 0 neutral-or-no-data); see
model/technical_score.py for how these combine into one recommendation.
"""

from __future__ import annotations

import pandas as pd

SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200

# Below this P/E, treat the stock as statistically "cheap" (bullish vote);
# above the expensive threshold, "expensive" (bearish vote). Crude,
# fixed-band, NOT sector-adjusted -- no free per-sector P/E benchmark was
# found; tunable like every other threshold constant in this app (see
# model/sentiment_adjustment.py's own docstring for the same honesty).
PE_VALUE_THRESHOLD = 15
PE_EXPENSIVE_THRESHOLD = 30


def compute_moving_averages(price_df: pd.DataFrame) -> tuple[float | None, float | None]:
    """(sma_short, sma_long) over `price_df`'s `close` column (see
    storage.repository.price_bars_for_symbol). None for either window
    without enough history yet -- same "not enough data" contract as
    model/return_model.py's own MIN_RETURNS_TO_FIT guard.
    """
    closes = price_df["close"]
    sma_short = float(closes.tail(SMA_SHORT_WINDOW).mean()) if len(closes) >= SMA_SHORT_WINDOW else None
    sma_long = float(closes.tail(SMA_LONG_WINDOW).mean()) if len(closes) >= SMA_LONG_WINDOW else None
    return sma_short, sma_long


def ma_signal(sma_short: float | None, sma_long: float | None) -> int:
    """+1 golden cross (short > long, bullish), -1 death cross (short <
    long, bearish), 0 if either is None or they're exactly equal.
    """
    if sma_short is None or sma_long is None:
        return 0
    if sma_short > sma_long:
        return 1
    if sma_short < sma_long:
        return -1
    return 0


def pe_signal(
    pe_ratio: float | None,
    value_threshold: float = PE_VALUE_THRESHOLD,
    expensive_threshold: float = PE_EXPENSIVE_THRESHOLD,
) -> int:
    """+1 below `value_threshold`, -1 above `expensive_threshold`, 0
    between them or if `pe_ratio` is None/non-positive (a loss-making
    company's negative P/E isn't a "cheap" signal). Thresholds default to
    the module constants; model/tuning.py supplies per-company overrides.
    """
    if pe_ratio is None or pe_ratio <= 0:
        return 0
    if pe_ratio < value_threshold:
        return 1
    if pe_ratio > expensive_threshold:
        return -1
    return 0
