"""
Replay CLI — execute a saved capability artifact from the command line.

Usage:
    python -m replay.cli \\
        --capability lookup_member_balance \\
        --params '{"member_id": "12345"}' \\
        [--version 1] \\
        [--auto-confirm]

--capability is the capability name as stored in artifact/store/.
--params accepts a JSON object of name→value pairs.
--version selects the artifact version (default: 1).
--auto-confirm allows execution of steps marked reversible=False without
  pausing for human confirmation.  Omit this flag to stop at any
  irreversible step instead.

Expects the target app running at the URL recorded in the artifact.
Evidence is written to evidence/<run_id>/steps.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from playwright.sync_api import sync_playwright

from artifact.evidence import EvidenceWriter
from artifact.store import load_capability
from replay.executor import run_replay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a saved RPA capability artifact against the live target app."
    )
    parser.add_argument("--capability", required=True,
                        help="Capability name (e.g. lookup_member_balance).")
    parser.add_argument("--params", required=True,
                        help='JSON object of param name→value pairs, e.g. \'{"member_id": "12345"}\'.')
    parser.add_argument("--version", type=int, default=1,
                        help="Artifact version to load (default: 1).")
    parser.add_argument("--auto-confirm", action="store_true",
                        help="Execute reversible=False steps without pausing for confirmation.")
    args = parser.parse_args()

    try:
        params: dict[str, str] = json.loads(args.params)
    except json.JSONDecodeError as exc:
        parser.error(f"--params is not valid JSON: {exc}")

    try:
        capability = load_capability(args.capability, args.version)
    except FileNotFoundError:
        parser.error(
            f"No artifact found for capability={args.capability!r} version={args.version}. "
            f"Run discovery first to record it."
        )

    run_id = f"replay_{uuid.uuid4().hex[:8]}"
    writer = EvidenceWriter(run_id, "replay")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            result = run_replay(
                capability,
                params,
                page,
                auto_confirm=args.auto_confirm,
                writer=writer,
            )
        finally:
            browser.close()

    print(result.model_dump_json(indent=2))
    print(f"evidence: {writer.path}")

    if result.status != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
