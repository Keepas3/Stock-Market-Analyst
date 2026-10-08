from __future__ import annotations

import requests

from stock_predictor.alerts import notifier


def test_send_webhook_alert_returns_false_when_no_url_configured(monkeypatch):
    monkeypatch.setattr(notifier, "discord_webhook_url", lambda: None)
    assert notifier.send_webhook_alert("AAPL: Hold -> Buy") is False


def test_send_webhook_alert_posts_the_message_to_the_configured_url(monkeypatch):
    monkeypatch.setattr(notifier, "discord_webhook_url", lambda: "https://discord.example/webhook")
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(notifier.requests, "post", fake_post)

    result = notifier.send_webhook_alert("AAPL: Hold -> Buy")

    assert result is True
    assert captured["url"] == "https://discord.example/webhook"
    assert captured["json"] == {"content": "AAPL: Hold -> Buy", "allowed_mentions": {"parse": []}}


def test_send_webhook_alert_returns_false_on_request_exception(monkeypatch):
    monkeypatch.setattr(notifier, "discord_webhook_url", lambda: "https://discord.example/webhook")

    def raise_error(*a, **k):
        raise requests.RequestException("network error")

    monkeypatch.setattr(notifier.requests, "post", raise_error)

    assert notifier.send_webhook_alert("AAPL: Hold -> Buy") is False
