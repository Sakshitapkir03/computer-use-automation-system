"""
Live recoverable-retry integration test.

Navigates to /reports/slow-summary, which streams HTTP headers immediately
but delays the HTML body by ~4 seconds (the server-side artificial delay).
page.goto(wait_until="commit") returns once headers are received, so the DOM
is still empty.  with_recoverable_retry wraps the do_read call:

  Attempt 1 — wait_for(attached, 2 s) times out; element not yet in DOM.
               recoverable_retry event logged; sleep 1 s.
  Attempt 2 — element has arrived (page delivered at ~4 s from navigation
               start, probe window opens at ~3 s); read succeeds.

Evidence is written to evidence/recoverable_<run_id>/steps.jsonl.

Run:
    python tests/run_recoverable_test.py

Expects the target app running at http://localhost:5001.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.actions import do_read
from artifact.evidence import EvidenceWriter
from artifact.schema import Locator
from replay.recoverable import with_recoverable_retry

SLOW_URL = "http://localhost:5001/reports/slow-summary"


def main() -> None:
    run_id = f"recoverable_{uuid.uuid4().hex[:8]}"
    writer = EvidenceWriter(run_id, "replay")

    def log(event: dict) -> None:
        writer.log(event)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate with wait_until="commit": returns once HTTP headers and the
        # initial streaming byte are received — before the 4-second body delay.
        page.goto(SLOW_URL, wait_until="commit", timeout=30_000)
        log({"event": "navigate", "action": "navigate", "url": SLOW_URL,
             "performed_by": "agent", "result": "ok"})

        # Target the section-title div rendered in slow_summary.html.
        # Single strategy, no fallback — probe_ms=2 s (default), so attempt 1
        # exhausts its window before the content arrives and raises
        # LocatorResolutionError, triggering the retry.
        loc = Locator(strategy="css_fallback", value=".section-title")

        value = with_recoverable_retry(
            lambda: do_read(page, loc),
            lambda ev: log({"performed_by": "agent", **ev}),
        )

        log({"event": "step", "action": "read", "locator": "css_fallback:.section-title",
             "performed_by": "agent", "result": "ok", "value": value})
        log({"event": "result", "status": "success",
             "outputs": {"summary_text": value}})

        browser.close()

    print(f"\nEvidence → {writer.path}\n")
    print("=" * 60)
    print("EVIDENCE TRAIL (raw JSONL)")
    print("=" * 60)
    for line in writer.path.read_text().splitlines():
        print(line)


if __name__ == "__main__":
    main()
