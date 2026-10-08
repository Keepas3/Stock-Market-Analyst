"""Commits config/model_overrides.yaml back to the GitHub repo through the
Contents API, so a tuning change made on the deployed (Streamlit Cloud) app
survives the next redeploy and is visible to the cron workflows.

The content pushed is built from the REMOTE file with the turn's change
entries replayed onto it (overrides_store.apply_entries), not from the local
file -- so a hand edit made on GitHub since the app last deployed isn't
overwritten by a stale local copy. Never raises: failures come back as a
SyncResult the UI shows, with the entries kept so the owner can retry.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass

import requests

from stock_predictor import config
from stock_predictor.model import overrides_store

API_ROOT = "https://api.github.com"
FILE_PATH = f"config/{overrides_store.OVERRIDES_FILENAME}"
TIMEOUT_SECONDS = 20
_MAX_ATTEMPTS = 2  # one retry on a sha race (409/422)


@dataclass
class SyncResult:
    ok: bool
    message: str


def is_configured() -> bool:
    return bool(config.github_token() and config.github_repo())


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.github_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _commit_message(entries: list[dict]) -> str:
    tickers = sorted({e["ticker"] for e in entries})
    # Structured fields only -- never the model's free-text rationale.
    return f"AI tuning: {len(entries)} change(s) to {', '.join(tickers)}"[:100]


def commit_file(file_path: str, transform: Callable[[str], str], message: str) -> SyncResult:
    """Commit one repo file through the Contents API. `transform` receives the
    file's CURRENT text on GitHub ("" if it doesn't exist yet) and returns the
    new text -- so a change is always replayed onto the remote version, never a
    stale local copy. `message` must be built from structured fields only."""
    if not is_configured():
        return SyncResult(False, "Applied locally only: GITHUB_TOKEN and GITHUB_REPO aren't configured.")

    url = f"{API_ROOT}/repos/{config.github_repo()}/contents/{file_path}"
    branch = config.github_branch()
    last_error = "unknown error"
    for _ in range(_MAX_ATTEMPTS):
        try:
            got = requests.get(url, headers=_headers(), params={"ref": branch}, timeout=TIMEOUT_SECONDS)
            if got.status_code == 404:
                sha, remote_text = None, ""
            else:
                got.raise_for_status()
                body = got.json()
                sha = body["sha"]
                remote_text = base64.b64decode(body["content"]).decode("utf-8")

            payload = {
                "message": message[:100],
                "content": base64.b64encode(transform(remote_text).encode("utf-8")).decode("ascii"),
                "branch": branch,
            }
            if sha:
                payload["sha"] = sha
            put = requests.put(url, headers=_headers(), json=payload, timeout=TIMEOUT_SECONDS)
            if put.status_code in (409, 422):
                last_error = "the file changed on GitHub while saving"
                continue
            put.raise_for_status()
            return SyncResult(True, "Saved to git.")
        except (requests.RequestException, KeyError, ValueError) as exc:
            # str(exc) can embed the request URL but never the token (it's in a header).
            last_error = str(exc)[:200]
            if isinstance(exc, requests.RequestException):
                break
    return SyncResult(False, f"Applied locally, but not saved to git: {last_error}")


def commit_entries(entries: list[dict]) -> SyncResult:
    if not entries:
        return SyncResult(True, "Nothing to save.")

    def replay(remote_text: str) -> str:
        store = overrides_store.parse_store(remote_text)
        overrides_store.apply_entries(store, entries)
        return overrides_store.dump_store(store)

    return commit_file(FILE_PATH, replay, _commit_message(entries))
