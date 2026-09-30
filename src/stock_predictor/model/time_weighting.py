"""Exponential recency weighting -- direct copy of soccer-predictor's own
model/time_weighting.py, just a shorter default half-life: stock
volatility regimes shift faster than a football team's underlying
strength, so a bar from a year ago shouldn't count nearly as much as a
match from a year ago did there. A documented, tunable heuristic, not
empirically fit -- easy to revisit once real predictions have accumulated
(see prediction/tracking.py's Phase 2 plan for how that could eventually
be checked).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

DEFAULT_XI = 0.008  # ~half-life of ~90 days (vs football's ~1 year)


def bar_weights(dates: pd.Series, as_of: dt.date | None = None, xi: float = DEFAULT_XI) -> np.ndarray:
    as_of = as_of or dates.max()
    days_since = (pd.Timestamp(as_of) - pd.to_datetime(dates)).dt.days.to_numpy()
    days_since = np.clip(days_since, a_min=0, a_max=None)
    return np.exp(-xi * days_since)
