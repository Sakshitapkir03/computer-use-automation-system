"""
Bounded retry for transient locator failures during replay.

Applies to: click, type, read, wait_for (any Step action that resolves a Locator).
Does NOT apply to: navigate (no Locator resolution involved).

Retry schedule — RECOVERABLE_MAX_ATTEMPTS total attempts, with RECOVERABLE_BACKOFF_S
seconds of sleep inserted BEFORE attempt 2, 3, 4 respectively.  Total worst-case
added wait beyond the first attempt: 1 + 2 + 4 = 7 s (for 3 total attempts, only
the first two inter-attempt sleeps apply: 1 + 2 = 3 s).

After all attempts are exhausted the final LocatorResolutionError is re-raised
unchanged so the caller's existing business-outcome check and hard_failure
fallthrough in run_replay handle it exactly as they would without this wrapper.
"""
from __future__ import annotations

import time
from typing import Callable, TypeVar

from agent.actions import LocatorResolutionError

RECOVERABLE_MAX_ATTEMPTS: int = 3
RECOVERABLE_BACKOFF_S: list[float] = [1.0, 2.0, 4.0]

T = TypeVar("T")


def with_recoverable_retry(
    fn: Callable[[], T],
    log_fn: Callable[[dict], None],
) -> T:
    """
    Call fn() up to RECOVERABLE_MAX_ATTEMPTS times, sleeping between retries.

    On each LocatorResolutionError before the final attempt, emits a
    recoverable_retry event via log_fn and sleeps the corresponding backoff
    from RECOVERABLE_BACKOFF_S.  The event carries:
      attempt   — the attempt number that just failed (1-based)
      backoff_s — seconds slept before the next attempt
      error     — str(exc) from the failed LocatorResolutionError

    Steps that succeed on the first attempt pay only a function-call overhead —
    no logging, no sleeping.

    If all attempts raise LocatorResolutionError the final exception is
    re-raised so the caller's existing error path (business-outcome check /
    hard_failure) is unchanged.
    """
    last_exc: LocatorResolutionError | None = None
    for attempt in range(1, RECOVERABLE_MAX_ATTEMPTS + 1):
        try:
            return fn()
        except LocatorResolutionError as exc:
            last_exc = exc
            if attempt < RECOVERABLE_MAX_ATTEMPTS:
                backoff = RECOVERABLE_BACKOFF_S[attempt - 1]
                log_fn({
                    "event": "recoverable_retry",
                    "attempt": attempt,
                    "backoff_s": backoff,
                    "error": str(exc),
                })
                time.sleep(backoff)
    raise last_exc  # type: ignore[misc]
