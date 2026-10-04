"""Visits the deployed Streamlit app in a real headless browser so Streamlit
Community Cloud counts it as a viewer and doesn't put it to sleep.

A plain HTTP request (curl) is not enough: the server returns the same
static page shell with a 200 whether the app is awake or asleep, and
"activity" only registers once the page's JavaScript opens the app's
websocket session. If the app is already asleep, this clicks the "Yes, get
this app back up!" button, waits for the app itself to render, then
lingers briefly so the session is registered.

Standalone on purpose (only needs `playwright`, installed by
.github/workflows/keep-alive.yml) -- it imports nothing from this project.

Usage:
    python scripts/keep_alive.py [URL]
"""

from __future__ import annotations

import re
import sys
import time

from playwright.sync_api import sync_playwright

DEFAULT_URL = "https://stock-market-analyst.streamlit.app/"
APP_FRAME = 'iframe[title="streamlitApp"]'  # Community Cloud renders the app inside this iframe
APP_MARKER = "Companies"  # sidebar nav entry -- only present once the real app has rendered
WAKE_BUTTON = re.compile(r"get this app back up", re.IGNORECASE)
TIMEOUT_SECONDS = 240  # a cold start can take a couple of minutes
LINGER_SECONDS = 20


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)

        deadline = time.monotonic() + TIMEOUT_SECONDS
        woke_it = False
        while time.monotonic() < deadline:
            wake_button = page.get_by_role("button", name=WAKE_BUTTON)
            if wake_button.count() and wake_button.first.is_visible():
                print("App was asleep -- clicking the wake-up button")
                wake_button.first.click()
                woke_it = True

            marker = page.frame_locator(APP_FRAME).get_by_text(APP_MARKER).first
            try:
                if marker.is_visible():
                    print(f"App is up{' (after waking it)' if woke_it else ''} -- holding the session open")
                    time.sleep(LINGER_SECONDS)
                    browser.close()
                    return 0
            except Exception:
                pass  # iframe not attached yet -- keep polling
            time.sleep(3)

        print(f"App did not render within {TIMEOUT_SECONDS}s", file=sys.stderr)
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
