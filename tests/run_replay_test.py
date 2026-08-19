"""
Live replay integration test — runs three scenarios against the real target app
using the saved lookup_member_balance_v1.json artifact.

Usage:
    python tests/run_replay_test.py

Expects the target app running at http://localhost:5001.
Evidence is written to evidence/<run_id>/steps.jsonl for each scenario.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent.parent))

from artifact.evidence import EvidenceWriter
from artifact.schema import Capability
from replay.executor import run_replay

ARTIFACT_PATH = Path(__file__).parent.parent / "artifact" / "store" / "lookup_member_balance_v1.json"


def load_capability() -> Capability:
    raw = json.loads(ARTIFACT_PATH.read_text())
    return Capability.model_validate(raw)


def run_scenario(label: str, member_id: str, auto_confirm: bool = True) -> None:
    print(f"\n{'='*60}")
    print(f"Scenario: {label}  (member_id={member_id!r})")
    print("=" * 60)

    cap = load_capability()
    params = {"member_id": member_id}
    run_id = f"replay_{uuid.uuid4().hex[:8]}"
    writer = EvidenceWriter(run_id, "replay")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            result = run_replay(cap, params, page, auto_confirm=auto_confirm, writer=writer)
        finally:
            browser.close()

    print(result.model_dump_json(indent=2))
    print(f"Evidence → {writer.path}")


if __name__ == "__main__":
    run_scenario("(a) Known member with Savings account", "12345")
    run_scenario("(b) Non-existent member", "99999")
    run_scenario("(c) Known member without Savings account", "67890")
