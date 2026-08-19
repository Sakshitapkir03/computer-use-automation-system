"""
Deterministic replay of a saved Capability artifact.

Import boundary (enforced by convention, verified in Phase 7):
  MAY import:  agent/actions.py, artifact/schema.py, artifact/store.py,
               artifact/evidence.py, escalation/session.py
  MUST NOT import from:  agent/llm_client.py, agent/discovery.py,
                         agent/perception.py

No LLM is used during replay. Every targeting decision is encoded in the
Capability artifact. The executor is intentionally thin: it substitutes
{param_name} references, walks the step list, and verifies the checkpoint.

Status taxonomy:
  success          — all steps OK and the primary checkpoint passed.
  business_outcome — a step failed OR the primary checkpoint failed, AND a
                     declared BusinessOutcomeSpec positively matched the
                     current page state.  There is NO default business_outcome:
                     it is only returned on an explicit positive signal.
  hard_failure     — a step raised an exception or the primary checkpoint
                     failed, and no declared outcome spec matched.  This is
                     the default for any unrecognised failure state.

Escalation (Phase 6):
  When a SessionController is supplied and a reversible=False step is
  reached without auto_confirm, the executor calls session.pause() instead
  of returning hard_failure.  The agent thread blocks in pause() until the
  human operator signals resume via the operator console, then execution
  continues past the gate on the same Page object.

Evidence (Phase 7):
  Pass an EvidenceWriter to log each step and the final result to a JSONL
  file in evidence/<run_id>/steps.jsonl.  The writer applies redaction
  before every disk write.  If no writer is supplied, no disk I/O occurs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from playwright.sync_api import Page

from agent.actions import (
    ActionError,
    AllowlistDenied,
    LocatorResolutionError,
    do_click,
    do_navigate,
    do_read,
    do_type,
    do_wait_for,
    resolve_locator,
)
from artifact.schema import Capability, Checkpoint, ReplayResult
from replay.recoverable import with_recoverable_retry

if TYPE_CHECKING:
    from artifact.evidence import EvidenceWriter
    from escalation.session import SessionController


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve(template: str, params: dict[str, str]) -> str:
    """Substitute {param_name} references with actual run-time values."""
    for key, val in params.items():
        template = template.replace(f"{{{key}}}", val)
    return template


def _verify_checkpoint(page: Page, checkpoint: Checkpoint) -> tuple[bool, str]:
    """
    Assert one checkpoint condition.

    Returns (passed, observed) where observed is a short description of what
    was actually found — used to populate failure_observed / outcome_message.

    Checkpoint kinds:
      url_matches     — checkpoint.expected is a substring of the current URL.
      element_visible — the locator resolves and its first match is visible.
      text_present    — the locator resolves and checkpoint.expected is a
                        substring of the element's stripped text content.
    """
    if checkpoint.kind == "url_matches":
        current = page.url
        return checkpoint.expected in current, current

    if checkpoint.locator is None:
        return False, f"checkpoint kind {checkpoint.kind!r} requires a locator but none is set"

    try:
        loc = resolve_locator(page, checkpoint.locator)
    except LocatorResolutionError as exc:
        return False, f"locator not found: {exc}"

    if checkpoint.kind == "element_visible":
        visible = loc.first.is_visible()
        return visible, "element visible" if visible else "element not visible"

    if checkpoint.kind == "text_present":
        text = (loc.first.text_content() or "").strip()
        return checkpoint.expected in text, text

    return False, f"unknown checkpoint kind: {checkpoint.kind!r}"


def _check_business_outcomes(
    page: Page,
    capability: Capability,
    outputs: dict[str, str],
) -> ReplayResult | None:
    """
    Test each declared BusinessOutcomeSpec in order.

    Returns a ReplayResult(status="business_outcome") on the first spec whose
    checkpoint condition is positively matched on the current page, or None if
    no declared outcome is active.  Errors during individual checkpoint checks
    are treated as non-matches (not propagated).

    This is called both after a step failure and after the primary checkpoint
    fails, so that business outcomes can be detected regardless of whether the
    workflow reached its final step.
    """
    for spec in capability.business_outcomes:
        try:
            passed, _ = _verify_checkpoint(page, spec.checkpoint)
        except Exception:
            passed = False
        if passed:
            return ReplayResult(
                status="business_outcome",
                outputs=outputs,
                outcome_code=spec.outcome_code,
                outcome_message=spec.outcome_message,
            )
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_replay(
    capability: Capability,
    params: dict[str, str],
    page: Page,
    *,
    auto_confirm: bool = False,
    session: SessionController | None = None,
    writer: EvidenceWriter | None = None,
) -> ReplayResult:
    """
    Execute a Capability artifact against a live Playwright page.

    auto_confirm gates irreversible steps (reversible=False). Reaching one
    without auto_confirm=True stops the run immediately and returns
    hard_failure — the action is NOT executed. Re-run with auto_confirm=True
    to proceed past the gate.

    The executor navigates to capability.target["base_url"] before executing
    the recorded steps, mirroring the initial navigation that discovery
    performed before the LLM loop started.
    """
    outputs: dict[str, str] = {}
    base_url: str = capability.target.get("base_url", "")

    # Configure the writer with the live outputs dict (mutable reference so
    # the writer always sees the current state for redaction).
    if writer is not None:
        writer.configure(capability, params, outputs)

    def _log(event: dict) -> None:
        if writer is not None:
            writer.log(event)

    # Initial navigation to the capability's starting URL.
    try:
        do_navigate(page, base_url)
        _log({"event": "navigate", "action": "navigate",
              "url": base_url, "performed_by": "agent", "result": "ok"})
    except (ActionError, AllowlistDenied, Exception) as exc:
        result = ReplayResult(
            status="hard_failure",
            outputs=outputs,
            failure_step_index=-1,
            failure_expected=f"navigate to {base_url!r}",
            failure_observed=str(exc),
        )
        _log({"event": "result", **result.model_dump()})
        return result

    for step in capability.steps:
        # Irreversible gate — checked before any action is attempted.
        if not step.reversible and not auto_confirm:
            if session is not None:
                # Escalate: pause and wait for human confirmation.  When the
                # operator resumes, execution continues past the gate on the
                # same Page — no hard_failure is returned.
                from escalation.session import InterventionPayload
                payload = InterventionPayload(
                    capability_name=capability.name,
                    goal=capability.goal,
                    step_index=step.index,
                    reason=(
                        f"Step {step.index} ({step.action!r}) is marked "
                        "reversible=False and requires human confirmation "
                        "before execution. Resume via the operator console."
                    ),
                )
                _log({"event": "pause", "step_index": step.index,
                      "reason": payload.reason, "performed_by": "agent"})
                session.pause(payload)
                # Log any human actions that occurred during the pause.
                for human_entry in session.evidence:
                    _log(human_entry)
                _log({"event": "resume", "step_index": step.index,
                      "performed_by": "agent"})
                # Fall through: execute the step now that the human confirmed.
            else:
                result = ReplayResult(
                    status="hard_failure",
                    outputs=outputs,
                    failure_step_index=step.index,
                    failure_expected="auto_confirm=True to execute irreversible step",
                    failure_observed=(
                        f"Step {step.index} ({step.action!r}) is marked "
                        f"reversible=False but auto_confirm=False. "
                        f"Re-run with auto_confirm=True to proceed past this gate."
                    ),
                )
                _log({"event": "result", **result.model_dump()})
                return result

        try:
            if step.action == "navigate":
                url = _resolve(step.value, params)
                do_navigate(page, url)
                _log({"event": "step", "step_index": step.index, "action": "navigate",
                      "url": url, "performed_by": "agent", "result": "ok"})

            elif step.action == "click":
                with_recoverable_retry(
                    lambda: do_click(page, step.locator),
                    lambda ev: _log({"step_index": step.index, "action": step.action, **ev}),
                )
                _log({"event": "step", "step_index": step.index, "action": "click",
                      "locator": step.locator.strategy + ":" + step.locator.value,
                      "performed_by": "agent", "result": "ok"})

            elif step.action == "type":
                with_recoverable_retry(
                    lambda: do_type(page, step.locator, _resolve(step.value, params)),
                    lambda ev: _log({"step_index": step.index, "action": step.action, **ev}),
                )
                _log({"event": "step", "step_index": step.index, "action": "type",
                      "locator": step.locator.strategy + ":" + step.locator.value,
                      "value_template": step.value,
                      "performed_by": "agent", "result": "ok"})

            elif step.action == "read":
                outputs[step.output_key] = with_recoverable_retry(
                    lambda: do_read(page, step.locator),
                    lambda ev: _log({"step_index": step.index, "action": step.action, **ev}),
                )
                _log({"event": "step", "step_index": step.index, "action": "read",
                      "output_key": step.output_key,
                      "performed_by": "agent", "result": "ok"})

            elif step.action == "wait_for":
                with_recoverable_retry(
                    lambda: do_wait_for(page, step.locator),
                    lambda ev: _log({"step_index": step.index, "action": step.action, **ev}),
                )
                _log({"event": "step", "step_index": step.index, "action": "wait_for",
                      "locator": step.locator.strategy + ":" + step.locator.value,
                      "performed_by": "agent", "result": "ok"})

        except AllowlistDenied as exc:
            # Allowlist denial is a configuration failure, not a business outcome.
            result = ReplayResult(
                status="hard_failure",
                outputs=outputs,
                failure_step_index=step.index,
                failure_expected=f"step {step.index} ({step.action!r}) to be allowed",
                failure_observed=f"AllowlistDenied: {exc}",
            )
            _log({"event": "step", "step_index": step.index, "action": step.action,
                  "performed_by": "agent", "result": "error",
                  "error": f"AllowlistDenied: {exc}"})
            _log({"event": "result", **result.model_dump()})
            return result

        except (ActionError, Exception) as exc:
            _log({"event": "step", "step_index": step.index, "action": step.action,
                  "performed_by": "agent", "result": "error",
                  "error": f"{type(exc).__name__}: {exc}"})
            # Before declaring hard_failure, check for a positive business-outcome
            # signal — the page may have reached a known non-success state
            # (e.g. member not found redirect) that caused this step to fail.
            bo = _check_business_outcomes(page, capability, outputs)
            if bo is not None:
                _log({"event": "result", **bo.model_dump()})
                return bo
            result = ReplayResult(
                status="hard_failure",
                outputs=outputs,
                failure_step_index=step.index,
                failure_expected=f"step {step.index} ({step.action!r}) to succeed",
                failure_observed=f"{type(exc).__name__}: {exc}",
            )
            _log({"event": "result", **result.model_dump()})
            return result

    # All steps executed — verify the primary success checkpoint.
    passed, observed = _verify_checkpoint(page, capability.checkpoint)
    if passed:
        result = ReplayResult(status="success", outputs=outputs)
        _log({"event": "checkpoint", "passed": True, "url": page.url})
        _log({"event": "result", **result.model_dump()})
        return result

    # Primary checkpoint failed — check for a declared business outcome before
    # falling through to hard_failure.  hard_failure is the explicit default:
    # there is no fallback business_outcome category.
    _log({"event": "checkpoint", "passed": False, "observed": observed, "url": page.url})
    bo = _check_business_outcomes(page, capability, outputs)
    if bo is not None:
        _log({"event": "result", **bo.model_dump()})
        return bo

    result = ReplayResult(
        status="hard_failure",
        outputs=outputs,
        failure_step_index=None,
        failure_expected=(
            f"checkpoint {capability.checkpoint.kind!r} "
            f"with {capability.checkpoint.expected!r}"
        ),
        failure_observed=observed,
    )
    _log({"event": "result", **result.model_dump()})
    return result
