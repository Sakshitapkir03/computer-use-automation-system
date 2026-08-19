"""
Discovery run — LLM-driven capability recording against the live target app.

Usage:
    python tests/run_discovery_test.py

Expects the target app running at http://localhost:5001 and GEMINI_API_KEY set
in .env (or the environment).  Evidence is written to
evidence/<run_id>/steps.jsonl after the run completes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from playwright.sync_api import sync_playwright

from agent.discovery import run_discovery
from artifact.evidence import EvidenceWriter
from artifact.schema import ParamSpec


def main() -> None:
    param_specs = {
        "member_id": ParamSpec(
            type="str",
            required=True,
            sensitive=False,
            description="Core-banking member ID to look up.",
        ),
    }

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            result = run_discovery(
                page=page,
                goal="Search for a member by ID and read their savings account balance.",
                capability_name="lookup_member_balance",
                base_url="http://localhost:5001/search",
                params={"member_id": "12345"},
                param_specs=param_specs,
                max_steps=20,
                model="gemini-3.5-flash",
            )
        except Exception as exc:
            print(f"\n=== DISCOVERY FAILED: {type(exc).__name__}: {exc} ===\n")
            browser.close()
            return

        browser.close()

    # Write evidence to disk now that the capability and outputs are known.
    # The writer applies collect_sensitive_values + redact() before every write.
    writer = EvidenceWriter(result.run_id, "discovery")
    writer.configure(result.capability, {"member_id": "12345"}, result.outputs)
    writer.log_all(result.evidence)

    print("\n=== DISCOVERY RESULT ===")
    print(f"run_id:        {result.run_id}")
    print(f"artifact_path: {result.artifact_path}")
    print(f"outputs:       {result.outputs}")
    print(f"steps_taken:   {result.steps_taken}")
    print(f"elapsed_s:     {result.elapsed_s}")
    print(f"evidence →     {writer.path}")

    print("\n=== EVIDENCE (step-by-step) ===")
    for i, ev in enumerate(result.evidence):
        print(f"\n  [{i}] type={ev.get('type')}")
        for k, v in ev.items():
            if k != "type":
                print(f"       {k}: {v}")

    print("\n=== SAVED ARTIFACT JSON ===")
    print(Path(result.artifact_path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
