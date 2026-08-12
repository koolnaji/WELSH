"""Shared, persistent coordination for every yt-dlp request in a run.

YouTube rate-limits an account/IP across endpoints.  Treating subtitle
metadata, timedtext downloads, and media downloads as independent services
causes a 429 from one endpoint to be immediately followed by requests to the
others.  This module provides one small coordinator for all of them.
"""
from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path


class YouTubeRateLimited(RuntimeError):
    """A request was rate-limited; ``retry_after`` is a conservative delay."""

    def __init__(self, retry_after: float, cause: BaseException | None = None):
        self.retry_after = max(0.0, float(retry_after))
        self.cause = cause
        super().__init__(f"YouTube rate-limited; retry after about {self.retry_after:.0f}s")


_LOCK = threading.RLock()
_STATE_PATH: Path | None = None
_not_before = 0.0
_last_request = 0.0

# One request at a time and modest pacing are substantially cheaper than
# repeatedly discovering a long server-side block.  The cap prevents a single
# batch from sleeping for hours; its persisted state still protects the next run.
MIN_REQUEST_INTERVAL = 2.5
RATE_LIMIT_DELAYS = (45.0, 120.0, 300.0)
MAX_IN_PROCESS_WAIT = 120.0


def configure(state_dir: Path) -> None:
    """Use a run-independent state file so an interrupted run remains polite."""
    global _STATE_PATH, _not_before
    with _LOCK:
        _STATE_PATH = Path(state_dir) / "youtube_rate_limit.json"
        try:
            data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
            _not_before = max(_not_before, float(data.get("not_before", 0)))
        except (OSError, ValueError, TypeError):
            pass


def is_rate_limit_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "http error 429" in message or "too many requests" in message or "rate limit" in message


def _save_state() -> None:
    if _STATE_PATH is None:
        return
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = _STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"not_before": _not_before}, indent=2), encoding="utf-8")
    temporary.replace(_STATE_PATH)


def _wait_for_turn() -> None:
    global _last_request
    with _LOCK:
        now = time.time()
        wait = max(_not_before - now, MIN_REQUEST_INTERVAL - (now - _last_request), 0.0)
    if wait > MAX_IN_PROCESS_WAIT:
        raise YouTubeRateLimited(wait)
    if wait:
        time.sleep(wait)
    with _LOCK:
        _last_request = time.time()


def _record_429(attempt: int) -> float:
    global _not_before
    # Full jitter avoids synchronized retries if callers later add workers.
    delay = RATE_LIMIT_DELAYS[min(attempt - 1, len(RATE_LIMIT_DELAYS) - 1)]
    delay *= random.uniform(0.8, 1.2)
    with _LOCK:
        _not_before = max(_not_before, time.time() + delay)
        _save_state()
    return delay


def call(operation: str, action, *, max_attempts: int = 3):
    """Run one yt-dlp action under the shared pace, cooldown and retry policy."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        _wait_for_turn()
        try:
            return action()
        except Exception as exc:
            last_error = exc
            if is_rate_limit_error(exc):
                delay = _record_429(attempt)
                if attempt == max_attempts or delay > MAX_IN_PROCESS_WAIT:
                    raise YouTubeRateLimited(delay, exc) from exc
            elif attempt == max_attempts:
                raise
            else:
                # Short, jittered retry only for ordinary transient failures.
                time.sleep(min(20.0, 2.0 ** attempt) * random.uniform(0.8, 1.2))
    raise last_error  # pragma: no cover - loop always returns or raises
