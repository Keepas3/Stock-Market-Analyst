"""Combines four independent signals (moving-average cross, P/E, news
sentiment, social sentiment) into one Buy/Hold/Sell recommendation -- the
primary Watchlist/Symbol Detail signal as of Phase 2, replacing the old
P(Up)/P(Down)-gap-based StockPrediction.recommendation (see
model/markets.py, now removed). A documented, simple, equal-weighted
heuristic -- not fit/backtested, same honesty every other adjustment in
this app already commits to (see model/sentiment_adjustment.py's own
docstring for the pattern).
"""

from __future__ import annotations

from dataclasses import dataclass

# Same +1/0/-1 banding idea model/technical_indicators.py's pe_signal
# already uses, just for a roughly -1..+1 sentiment score instead of a P/E
# ratio (see ingest/sentiment.py::SentimentResult,
# ingest/social_sentiment.py::SocialSentimentResult).
SENTIMENT_VOTE_BULLISH = 0.15
SENTIMENT_VOTE_BEARISH = -0.15

# total = sum of the four votes (range -4..+4). A tunable threshold, not
# fit/validated -- same spirit as model/markets.py's old
# BUY_SELL_PROBABILITY_GAP it supersedes.
BUY_VOTE_THRESHOLD = 2
SELL_VOTE_THRESHOLD = -2


@dataclass
class CompositeSignal:
    ma_vote: int
    pe_vote: int
    news_vote: int
    social_vote: int
    total: int
    recommendation: str  # "Buy" | "Hold" | "Sell"


def sentiment_vote(
    score: float | None,
    bullish: float = SENTIMENT_VOTE_BULLISH,
    bearish: float = SENTIMENT_VOTE_BEARISH,
) -> int:
    """+1/0/-1 banding shared by news and social sentiment scores. 0 (no
    vote) when `score` is None (no reading available yet) -- so a missing
    signal naturally leans the composite toward Hold rather than needing a
    special "insufficient data" case in compute_recommendation.
    """
    if score is None:
        return 0
    if score >= bullish:
        return 1
    if score <= bearish:
        return -1
    return 0


def compute_recommendation(ma_vote: int, pe_vote: int, news_vote: int, social_vote: int) -> CompositeSignal:
    total = ma_vote + pe_vote + news_vote + social_vote
    if total >= BUY_VOTE_THRESHOLD:
        recommendation = "Buy"
    elif total <= SELL_VOTE_THRESHOLD:
        recommendation = "Sell"
    else:
        recommendation = "Hold"
    return CompositeSignal(
        ma_vote=ma_vote,
        pe_vote=pe_vote,
        news_vote=news_vote,
        social_vote=social_vote,
        total=total,
        recommendation=recommendation,
    )
