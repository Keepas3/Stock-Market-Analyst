"""Discord/Slack webhook delivery for recommendation-change alerts -- the
closest thing to "real-time" this architecture (a scheduled script, no
long-running server) can do without standing up an email service. See
scripts/refresh_live_data.py for the trigger condition (a symbol's
composite Recommendation, see model/technical_score.py, changed since the
last one recorded in storage.models.RecommendationLog).
"""

from __future__ import annotations

import requests

from stock_predictor.config import discord_webhook_url


def format_price_move(price: float, previous_close: float | None) -> str:
    """e.g. "$333.02 (up $3.62, +1.10% vs prev close $329.40)" -- plain
    words rather than arrows/emoji so it reads the same in Discord and
    Slack. Falls back to just the price if there's no previous close to
    compare against (or it's zero).
    """
    if not previous_close:
        return f"${price:.2f}"
    change = price - previous_close
    pct = change / previous_close
    direction = "up" if change > 0 else "down" if change < 0 else "unchanged"
    if direction == "unchanged":
        return f"${price:.2f} (unchanged vs prev close ${previous_close:.2f})"
    return f"${price:.2f} ({direction} ${abs(change):.2f}, {pct:+.2%} vs prev close ${previous_close:.2f})"


def send_webhook_alert(message: str) -> bool:
    """POSTs `message` as a Discord-shaped `{"content": ...}` payload
    (Slack's "Incoming Webhooks" integration also accepts this exact
    shape, so this one function covers either per the user's own choice).
    Returns False -- degrading silently, never raising -- if
    DISCORD_WEBHOOK_URL isn't set or the request fails, same contract as
    every other integration in this app.
    """
    url = discord_webhook_url()
    if not url:
        return False
    try:
        # allowed_mentions.parse=[] -- never ping @everyone/@here/roles even if
        # text from a tuning note or ticker field contained one.
        response = requests.post(
            url, json={"content": message, "allowed_mentions": {"parse": []}}, timeout=15
        )
        response.raise_for_status()
    except requests.RequestException:
        return False
    return True
