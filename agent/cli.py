"""
Discovery CLI — run capability recording from the command line.

Usage:
    python -m agent.cli \\
        --goal "Search for a member by ID and read their savings account balance." \\
        --base-url "http://localhost:5001/search" \\
        --capability-name "lookup_member_balance" \\
        --params '{"member_id": "12345"}' \\
        [--param-specs '{"member_id": {"type": "str", "required": true, "sensitive": false, "description": "Member ID"}}'] \\
        [--model "gemini-3.5-flash"] \\
        [--max-steps 20]

--params accepts a JSON object of name→value pairs for this run.
--param-specs accepts a JSON object of name→ParamSpec dicts.  If omitted, every
  param is inferred as type=str, required=True, sensitive=False.

Expects the target app running and GEMINI_API_KEY set in .env or the environment.
Evidence is written to evidence/<run_id>/steps.jsonl after the run completes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from playwright.sync_api import sync_playwright

from agent.discovery import run_discovery
from artifact.evidence import EvidenceWriter
from artifact.schema import ParamSpec


def _infer_param_specs(params: dict[str, str]) -> dict[str, ParamSpec]:
    """Build minimal ParamSpec entries for params that have no explicit spec."""
    return {
        key: ParamSpec(
            type="str",
            required=True,
            sensitive=False,
            description=key,
        )
        for key in params
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a new RPA capability via LLM-driven browser discovery."
    )
    parser.add_argument("--goal", required=True,
                        help="Natural-language description of what to accomplish.")
    parser.add_argument("--base-url", required=True,
                        help="Starting URL for the capability (the page to open first).")
    parser.add_argument("--capability-name", required=True,
                        help="Stable identifier used to name the saved artifact.")
    parser.add_argument("--params", required=True,
                        help='JSON object of param name→value pairs, e.g. \'{"member_id": "12345"}\'.')
    parser.add_argument("--param-specs",
                        help="JSON object of param name→ParamSpec dicts.  "
                             "If omitted, all params are inferred as type=str/required=True/sensitive=False.")
    parser.add_argument("--model", default="gemini-3.5-flash",
                        help="Gemini model name (default: gemini-3.5-flash).")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Maximum LLM steps before the run is aborted (default: 20).")
    args = parser.parse_args()

    try:
        params: dict[str, str] = json.loads(args.params)
    except json.JSONDecodeError as exc:
        parser.error(f"--params is not valid JSON: {exc}")

    if args.param_specs:
        try:
            raw_specs = json.loads(args.param_specs)
            param_specs = {k: ParamSpec.model_validate(v) for k, v in raw_specs.items()}
        except (json.JSONDecodeError, Exception) as exc:
            parser.error(f"--param-specs is not valid: {exc}")
    else:
        param_specs = _infer_param_specs(params)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            result = run_discovery(
                page=page,
                goal=args.goal,
                capability_name=args.capability_name,
                base_url=args.base_url,
                params=params,
                param_specs=param_specs,
                max_steps=args.max_steps,
                model=args.model,
            )
        except Exception as exc:
            print(f"\n=== DISCOVERY FAILED: {type(exc).__name__}: {exc} ===\n", file=sys.stderr)
            browser.close()
            sys.exit(1)
        finally:
            browser.close()

    writer = EvidenceWriter(result.run_id, "discovery")
    writer.configure(result.capability, params, result.outputs)
    writer.log_all(result.evidence)

    print(f"run_id:        {result.run_id}")
    print(f"artifact_path: {result.artifact_path}")
    print(f"outputs:       {result.outputs}")
    print(f"steps_taken:   {result.steps_taken}")
    print(f"elapsed_s:     {result.elapsed_s}")
    print(f"evidence:      {writer.path}")


if __name__ == "__main__":
    main()
