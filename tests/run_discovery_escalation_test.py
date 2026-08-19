"""
Live discovery-side escalation demo.

The goal — "permanently delete the savings account for member 12345" — is
genuinely impossible on the target app.  target_app/app.py and all templates
contain no delete/remove route or control (confirmed by grep before writing
this file).  The LLM will attempt the goal, fail 3 consecutive times, and
trigger the auto-escalation path via max_consecutive_errors.

What fires:
  run_discovery's consecutive-error guard calls on_stuck(reason, page,
  step_index, evidence).  The callback builds an InterventionPayload,
  calls session.pause(), and blocks the discovery loop on the agent thread —
  the same Playwright Page the LLM was just failing to drive — until a human
  resumes via the operator console.

After resume, discovery resets consecutive_errors=0 and continues.  Because
the goal is still impossible, a second escalation will fire.  The on_stuck
callback raises DiscoveryStuck on the second call so the loop terminates
cleanly rather than looping indefinitely.

Evidence (discovery-style JSONL) is written to
evidence/disc_escalation_<run_id>/steps.jsonl, including any human actions
recorded by the session during the pause.

Run (target app must already be running on :5001):
    python tests/run_discovery_escalation_test.py
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.discovery import DiscoveryError, DiscoveryStuck, run_discovery
from artifact.evidence import EvidenceWriter
from artifact.schema import Capability, Checkpoint, Locator, ParamSpec
from escalation.operator_console import start_console
from escalation.session import InterventionPayload, SessionController

CONSOLE_PORT = 5002
CONSOLE_BASE = f"http://127.0.0.1:{CONSOLE_PORT}"

# Impossible goal — the target app has no delete/remove route or control.
# Verified by grep across target_app/app.py and all templates before writing
# this script.  The LLM will fail every attempt and trigger auto-escalation.
GOAL = "Look up member 12345 and permanently delete their savings account."
BASE_URL = "http://localhost:5001/search"
PARAMS = {"member_id": "12345"}


def _build_console_capability() -> Capability:
    """
    Minimal stub Capability passed to start_console() for redaction context.

    start_console() requires a Capability so it can call
    collect_sensitive_values() for the redaction pipeline.  No params here
    are sensitive, so the stub just satisfies the type signature.  It is
    never saved to disk and never used for replay.
    """
    return Capability(
        id="discovery_escalation_stub",
        version=1,
        name="delete_savings_account",
        goal=GOAL,
        target={"base_url": BASE_URL},
        inputs={
            "member_id": ParamSpec(
                type="str", sensitive=False,
                description="Member ID to look up.",
            ),
        },
        outputs={},
        steps=[],
        checkpoint=Checkpoint(
            kind="url_matches",
            locator=None,
            expected="/",
        ),
        created_from_run_id="discovery_escalation_stub",
    )


def main() -> None:
    run_id = f"disc_escalation_{uuid.uuid4().hex[:8]}"
    writer = EvidenceWriter(run_id, "discovery")
    outputs: dict[str, str] = {}   # mutable ref for console redaction context

    # Capture the live evidence list from inside the discovery loop so it
    # remains accessible after an exception terminates run_discovery.
    accumulated_evidence: list[dict] = []
    escalated_once = False

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)   # headless for CI/tool; set False locally to watch
        page = browser.new_page()

        session = SessionController(page)
        stub_cap = _build_console_capability()

        start_console(session, stub_cap, PARAMS, outputs, port=CONSOLE_PORT)
        time.sleep(0.5)   # let Flask bind

        def on_stuck(
            reason: str,
            page,           # noqa: ARG001 — live Page object; held by session already
            step_index: int,
            evidence: list[dict],
        ) -> None:
            nonlocal escalated_once, accumulated_evidence
            # Capture the live evidence list — same object discovery mutates.
            accumulated_evidence = evidence

            if escalated_once:
                # Second escalation after human resume: terminate cleanly
                # instead of re-blocking at the console.
                raise DiscoveryStuck(
                    f"Discovery terminated after second stuck condition: {reason}"
                )

            escalated_once = True

            payload = InterventionPayload(
                capability_name="delete_savings_account",
                goal=GOAL,
                step_index=step_index,
                reason=(
                    f"Discovery stuck after {step_index} successful steps: {reason}\n\n"
                    "The goal requires permanently deleting a savings account. "
                    "No such control exists in the target app — this is a genuine "
                    "dead end, not a transient failure.  Open the console, inspect "
                    "the live page screenshot, optionally submit a manual action, "
                    "then click Resume Agent to let discovery continue."
                ),
            )

            print(f"\n{'='*60}")
            print("DISCOVERY STUCK — escalating to operator console")
            print(f"Reason: {reason}")
            print(f"\nOpen {CONSOLE_BASE} in a browser.")
            print("Inspect the payload and screenshot, then click Resume Agent.")
            print("(Waiting indefinitely — no simulator here.)")
            print("=" * 60)

            session.pause(payload)  # blocks until human clicks Resume

            print("\nHuman resumed. Discovery will continue (and likely fail again)…")

        print(f"\n{'='*60}")
        print("DISCOVERY ESCALATION DEMO")
        print(f"Goal: {GOAL}")
        print(f"The target app has no delete control — the LLM will fail.")
        print(f"After {3} consecutive errors, auto-escalation fires.")
        print("=" * 60)

        try:
            result = run_discovery(
                page=page,
                goal=GOAL,
                capability_name="delete_savings_account",
                base_url=BASE_URL,
                params=PARAMS,
                param_specs={
                    "member_id": ParamSpec(
                        type="str", sensitive=False,
                        description="Member ID to look up.",
                    ),
                },
                max_steps=15,
                max_consecutive_errors=3,
                model="gemini-3.6-flash",
                on_stuck=on_stuck,
            )
            # Unexpected success path (goal somehow completed).
            accumulated_evidence = result.evidence
            print(f"\nDiscovery completed — artifact: {result.artifact_path}")

        except DiscoveryError as exc:
            print(f"\nDiscovery terminated: {type(exc).__name__}: {exc}")

        except Exception as exc:
            # Catches AllowlistDenied and any other unexpected errors.
            print(f"\nDiscovery terminated with error: {type(exc).__name__}: {exc}")

        finally:
            browser.close()

    # Write all discovery evidence accumulated before termination.
    if accumulated_evidence:
        writer.log_all(accumulated_evidence)

    # Append any human actions from the session pause, tagged with event key.
    for entry in session.evidence:
        writer.log({"event": "human_action", **entry})

    print(f"\nEvidence → {writer.path}")

    if writer.path.exists() and writer.path.stat().st_size > 0:
        print("\n" + "=" * 60)
        print("EVIDENCE TRAIL (raw JSONL)")
        print("=" * 60)
        for line in writer.path.read_text().splitlines():
            print(line)
    else:
        print("\n(No evidence written — discovery terminated before any actions.)")


if __name__ == "__main__":
    main()
