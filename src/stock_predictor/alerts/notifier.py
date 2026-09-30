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
        response = requests.post(url, json={"content": message}, timeout=15)
        response.raise_for_status()
    except requests.RequestException:
        return False
    return True
