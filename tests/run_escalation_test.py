"""
Live Phase 6 demo: pause → operator console → human action → resume → complete.

Scenario:
  The 'type member_id' step is marked reversible=False.  run_replay hits the
  gate, calls session.pause(), and blocks.  A simulator thread (representing
  the human operator) polls the console, inspects the payload, submits a
  manual navigate action to reset the page, then signals resume.  Automation
  continues and completes successfully.

Run:
    python tests/run_escalation_test.py
"""
from __future__ import annotations

import json
import sys
import time
import threading
from pathlib import Path

import requests
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifact.schema import (
    Capability, Checkpoint, Locator, OutputSpec, ParamSpec, Step,
)
from artifact.store import collect_sensitive_values
from escalation.operator_console import start_console
from escalation.session import SessionController
from guardrails.redaction import redact
from replay.executor import run_replay

CONSOLE_PORT = 5002
CONSOLE_BASE = f"http://127.0.0.1:{CONSOLE_PORT}"

def build_demo_capability() -> Capability:
    """
    Capability identical to lookup_member_balance but with step 0 (type)
    marked reversible=False so the irreversible gate fires immediately.
    This simulates a policy where entering a member ID for lookup must be
    confirmed by a human operator before the search is submitted.
    """
    return Capability(
        id="demo_escalation",
        version=1,
        name="lookup_member_balance",
        goal="Search for a member by ID and read their savings account balance.",
        target={"base_url": "http://localhost:5001/search"},
        inputs={
            "member_id": ParamSpec(
                type="str", sensitive=False,
                description="Core-banking member ID to look up.",
            ),
        },
        outputs={
            "savings_balance": OutputSpec(
                type="decimal", sensitive=True,
                description="Savings account balance.",
            ),
        },
        steps=[
            Step(
                index=0, action="type",
                locator=Locator(
                    strategy="css_fallback", value="input[name='member_id']",
                    fallback=Locator(strategy="role_name", role="textbox", value=""),
                ),
                value="{member_id}",
                reversible=False,   # <-- gate triggers here
            ),
            Step(
                index=1, action="click",
                locator=Locator(
                    strategy="role_name", role="button", value="Search",
                    fallback=Locator(
                        strategy="css_fallback", value="button[type='submit']",
                    ),
                ),
                reversible=True,
            ),
            Step(
                index=2, action="read",
                locator=Locator(
                    strategy="xpath",
                    value="//tr[td[normalize-space()='Savings']]/td[3]",
                    fallback=Locator(
                        strategy="css_fallback",
                        value="tr:has(td:text-is(\"Savings\")) td:nth-child(3)",
                    ),
                ),
                output_key="savings_balance",
                reversible=True,
            ),
        ],
        checkpoint=Checkpoint(
            kind="text_present",
            locator=Locator(
                strategy="xpath", value="//*[contains(text(), 'Member Details')]",
                fallback=Locator(strategy="css_fallback", value="body"),
            ),
            expected="Member Details",
        ),
        created_from_run_id="demo_escalation",
    )


def human_simulator(session: SessionController, params: dict, outputs: dict,
                    cap: Capability) -> None:
    """
    Runs in a daemon thread.  Waits for the console to become live and the
    session to be paused, then acts as a human operator:
      1. Fetches and prints the intervention payload.
      2. Submits a manual 'navigate' to the search page (showing the console
         can drive the live page).
      3. Signals resume.
    """
    # Wait until Flask is up.
    for _ in range(20):
        try:
            requests.get(CONSOLE_BASE + "/payload", timeout=1)
            break
        except Exception:
            time.sleep(0.3)

    # Wait until session enters HUMAN control.
    from escalation.session import Control
    for _ in range(30):
        if session.control == Control.HUMAN:
            break
        time.sleep(0.2)

    print("\n" + "=" * 60)
    print("HUMAN OPERATOR: console is live, session is paused")
    print("=" * 60)

    # ── 1. Fetch intervention payload ─────────────────────────────────────
    r = requests.get(CONSOLE_BASE + "/payload")
    raw_payload = r.json()
    # Redact before printing (mirrors what the console HTML does).
    sensitive = collect_sensitive_values(cap, params, outputs)
    safe_payload = redact(json.dumps(raw_payload, indent=2), sensitive_values=sensitive)
    print("\nIntervention payload (redacted for display):")
    print(safe_payload)

    # ── 2. Manual action: navigate to search page ─────────────────────────
    print("\nHUMAN ACTION: navigate to http://localhost:5001/search")
    r = requests.post(
        CONSOLE_BASE + "/action",
        json={"type": "navigate", "url": "http://localhost:5001/search"},
    )
    action_result = r.json()
    print("Action result:", json.dumps(action_result, indent=2))

    # ── 3. Resume automation ──────────────────────────────────────────────
    print("\nHUMAN ACTION: resume")
    r = requests.post(CONSOLE_BASE + "/resume")
    print("Resume response:", r.json())


def main() -> None:
    cap = build_demo_capability()
    params = {"member_id": "12345"}
    outputs: dict[str, str] = {}  # populated by replay as steps execute

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        session = SessionController(page)

        # Start operator console in background daemon thread.
        start_console(session, cap, params, outputs, port=CONSOLE_PORT)
        time.sleep(0.5)   # let Flask bind

        # Start human simulator in background daemon thread.
        sim = threading.Thread(
            target=human_simulator,
            args=(session, params, outputs, cap),
            daemon=True,
        )
        sim.start()

        # Run replay on the main thread (owns the Page).
        print("\nStarting replay (step 0 is reversible=False — will pause)…")
        result = run_replay(
            cap, params, page,
            auto_confirm=False,
            session=session,
        )

        sim.join(timeout=10)
        browser.close()

    print("\n" + "=" * 60)
    print("REPLAY RESULT")
    print("=" * 60)
    sensitive = collect_sensitive_values(cap, params, result.outputs or {})
    safe_result = redact(result.model_dump_json(indent=2), sensitive_values=sensitive)
    print(safe_result)

    print("\n" + "=" * 60)
    print("EVIDENCE TRAIL")
    print("=" * 60)
    for entry in session.evidence:
        # Redact the entire entry before printing.
        safe_entry = redact(
            json.dumps(entry, indent=2),
            sensitive_values=collect_sensitive_values(cap, params, result.outputs or {}),
        )
        print(safe_entry)
        print()


if __name__ == "__main__":
    main()
