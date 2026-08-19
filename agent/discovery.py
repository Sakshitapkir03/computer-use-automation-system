"""
Discovery loop.

Import boundary (enforced by convention, verified by grep in Phase 7):
  MAY import:  agent/actions.py, agent/perception.py, agent/llm_client.py,
               artifact/schema.py, artifact/store.py, guardrails/allowlist.py
  MUST NOT import from:  replay/

File I/O constraint: this module writes NOTHING to disk directly.
  Capability artifacts → via artifact.store.save_capability(), which routes
                          through guardrails.redaction.redact() before disk.
  Evidence records     → accumulated in DiscoveryResult.evidence (in-memory).
                          The CLI entrypoint (Phase 7) persists them after
                          redaction is wired in. No pathlib/os import here.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from playwright.sync_api import Page

from agent.actions import (
    ActionError,
    AllowlistDenied,
    do_click,
    do_navigate,
    do_read,
    do_type,
    do_wait_for,
)
from agent.llm_client import LLMClient
from agent.perception import Observation, capture
from artifact.schema import (
    Capability,
    Checkpoint,
    Locator,
    OutputSpec,
    ParamSpec,
    Step,
)
from artifact.store import save_capability

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryResult:
    capability: Capability
    artifact_path: object          # pathlib.Path — typed as object to avoid import
    run_id: str
    outputs: dict                  # actual extracted values from this run
    evidence: list[dict]           # in-memory only; Phase 7 persists after redaction
    steps_taken: int
    elapsed_s: float


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DiscoveryError(RuntimeError):
    """Base for all discovery-loop failures."""

class DiscoveryTimeout(DiscoveryError):
    """Wall-clock timeout exceeded before goal_complete was called."""

class DiscoveryMaxSteps(DiscoveryError):
    """Step limit reached before goal_complete was called."""

class DiscoveryStuck(DiscoveryError):
    """LLM called report_stuck. Payload: reason string."""

class OutputKeyMismatch(DiscoveryError):
    """
    A read step declared an output_key not present in goal_complete outputs.
    The loop catches this and routes it to the stuck handler rather than
    silently patching the artifact.
    """


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _build_system_prompt(
    goal: str,
    params: dict[str, str],
    param_specs: dict[str, ParamSpec],
) -> str:
    param_lines = "\n".join(
        f"  {k}={v!r}  (type: {param_specs[k].type}, "
        f"{'SENSITIVE — never log raw value' if param_specs[k].sensitive else 'non-sensitive'})"
        for k, v in params.items()
    )
    return f"""You are a browser automation agent recording a reusable capability for a \
legacy banking back-office application. The application has no test IDs or semantic \
HTML attributes — interactive elements are plain HTML form controls inside table layouts.

GOAL: {goal}

PARAMETERS — substitute {{param_name}} in action values, never use the literal value:
{param_lines}

OBSERVATION FORMAT each turn:
  Accessibility Tree  — structural hierarchy with element roles and computed names
  Interactive Element Attributes — HTML name/id/label for every form control in \
document order. Use name= values to build css_fallback locators.

LOCATOR RULES (follow exactly):
1. Always supply a two-strategy locator: primary + fallback.
2. For any input with no accessible name (no associated <label>, no aria-label):
   PRIMARY must be css_fallback with input[name="x"] or select[name="x"].
   FALLBACK may be role_name/textbox or role_name/combobox.
3. For named buttons/links: PRIMARY role_name with role=button and value=<button text>.
   FALLBACK css_fallback with a CSS selector.
4. Never use positional CSS selectors (nth-child, nth-of-type). Prefer attribute selectors.
5. READ STEP LOCATORS — NEVER locate by the value you are reading. The value changes
   per run; a locator that contains the value will hard-fail on replay with different data.
   Use structural XPath or Playwright CSS to target the cell by its row label instead.

   WRONG (locates by the live value — breaks on any other member):
     strategy: text_exact   value: "$5,432.10"
     strategy: role_name    value: "$5,432.10"   role: cell

   CORRECT (locates by the row label — works for any member):
     PRIMARY:  strategy: xpath        value: //tr[td[normalize-space()='Savings']]/td[3]
     FALLBACK: strategy: css_fallback value: tr:has(td:text-is("Savings")) td:nth-child(3)

   General pattern for any label/value table row:
     xpath  //tr[td[normalize-space()='<Row Label>']]/td[<value column index>]
   Where <Row Label> is the static heading text visible in the adjacent cell.

OUTPUT KEY CONSISTENCY:
  Every output_key used in a read step must appear as a key in goal_complete outputs.
  Mismatch causes the run to fail — use the same string in both places.

REVERSIBILITY:
  reversible=true  — all navigations, reads, waits, form field entries.
  reversible=false — ONLY the final irreversible confirmation submit \
(e.g. clicking "Confirm — Open Account"). Replay gates this behind auto_confirm.

WHEN DONE: call goal_complete with every declared output and a checkpoint.
  The checkpoint should be a text_present check on a distinctive success indicator.

IF STUCK: call report_stuck with a clear reason. A human operator will take over.
"""


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def _obs_text(obs: Observation, step_index: int) -> str:
    return (
        f"Step {step_index} — Current page state:\n"
        f"URL: {obs.url}\n"
        f"Title: {obs.title}\n\n"
        + obs.for_llm()
    )


def _image_block(obs: Observation) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": obs.screenshot_b64(),
        },
    }


def _initial_user_message(obs: Observation) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": _obs_text(obs, 0)},
            _image_block(obs),
        ],
    }


def _followup_user_message(
    tool_use_id: str,
    result_text: str,
    obs: Observation,
    step_index: int,
    *,
    is_error: bool = False,
) -> dict:
    tool_result: dict = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": result_text,
    }
    if is_error:
        tool_result["is_error"] = True
    return {
        "role": "user",
        "content": [
            tool_result,
            {"type": "text", "text": _obs_text(obs, step_index)},
            _image_block(obs),
        ],
    }


# ---------------------------------------------------------------------------
# Action execution + step recording
# ---------------------------------------------------------------------------

def _resolve_value(template: str, params: dict[str, str]) -> str:
    """Substitute {param_name} references with actual values for live execution."""
    result = template
    for key, val in params.items():
        result = result.replace(f"{{{key}}}", val)
    return result


def _execute_action(
    page: Page,
    tool_name: str,
    tool_input: dict,
    params: dict[str, str],
    recorded_steps: list[Step],
    read_values: dict[str, str],
) -> tuple[str, bool]:
    """
    Execute a single tool call via actions.py and append a Step to recorded_steps.

    Returns (result_text, is_error).
    AllowlistDenied is re-raised immediately — it is never recoverable.
    All other ActionErrors are returned as error strings so the LLM can adapt.
    """
    try:
        if tool_name == "navigate":
            url: str = tool_input["url"]
            do_navigate(page, url)
            recorded_steps.append(Step(
                index=len(recorded_steps),
                action="navigate",
                value=url,
                reversible=True,
            ))
            return f"OK: Navigated to {url}", False

        elif tool_name == "click":
            locator = Locator.model_validate(tool_input["locator"])
            reversible: bool = bool(tool_input.get("reversible", True))
            do_click(page, locator)
            recorded_steps.append(Step(
                index=len(recorded_steps),
                action="click",
                locator=locator,
                reversible=reversible,
            ))
            return f"OK: Clicked {locator.strategy}:{locator.value!r}", False

        elif tool_name == "type":
            locator = Locator.model_validate(tool_input["locator"])
            template: str = tool_input["value"]
            reversible = bool(tool_input.get("reversible", True))
            # Execute with the resolved value; record the template so the
            # Capability is parameterised, not hard-coded.
            do_type(page, locator, _resolve_value(template, params))
            recorded_steps.append(Step(
                index=len(recorded_steps),
                action="type",
                locator=locator,
                value=template,
                reversible=reversible,
            ))
            return f"OK: Typed {template!r} into {locator.value!r}", False

        elif tool_name == "read":
            locator = Locator.model_validate(tool_input["locator"])
            output_key: str = tool_input["output_key"]
            text = do_read(page, locator)
            read_values[output_key] = text
            recorded_steps.append(Step(
                index=len(recorded_steps),
                action="read",
                locator=locator,
                output_key=output_key,
                reversible=True,
            ))
            return f"OK: Read output_key={output_key!r} value={text!r}", False

        elif tool_name == "wait_for":
            locator = Locator.model_validate(tool_input["locator"])
            do_wait_for(page, locator)
            recorded_steps.append(Step(
                index=len(recorded_steps),
                action="wait_for",
                locator=locator,
                reversible=True,
            ))
            return f"OK: Element visible {locator.value!r}", False

        else:
            return f"Unknown tool: {tool_name!r}", True

    except AllowlistDenied:
        raise  # hard failure — propagate immediately
    except ActionError as exc:
        return f"ActionError ({type(exc).__name__}): {exc}", True
    except Exception as exc:
        return f"Error ({type(exc).__name__}): {exc}", True


# ---------------------------------------------------------------------------
# Capability assembly
# ---------------------------------------------------------------------------

def _substitute_param_refs(steps: list[Step], params: dict[str, str]) -> list[Step]:
    """
    Replace any literal param values that slipped through as step values with
    their {param_name} references. Safety net for LLM non-compliance with the
    parameterisation instruction.
    """
    result: list[Step] = []
    for step in steps:
        new_value = step.value
        if new_value is not None:
            for key, val in params.items():
                new_value = new_value.replace(val, f"{{{key}}}")
        result.append(step.model_copy(update={"value": new_value}))
    return result


def _assemble_capability(
    *,
    capability_name: str,
    goal: str,
    base_url: str,
    params: dict[str, str],
    param_specs: dict[str, ParamSpec],
    recorded_steps: list[Step],
    goal_complete_input: dict,
    run_id: str,
) -> tuple[Capability, dict]:
    """
    Build a Capability from the recorded steps and goal_complete data.

    Returns (capability, actual_outputs).
    actual_outputs holds the real extracted values from this run — they belong
    in evidence, not in the Capability itself (which is parameterised).

    Raises OutputKeyMismatch if any read step declares an output_key not present
    in goal_complete outputs. The loop catches this and routes it to handle_stuck
    rather than silently patching the artifact.
    """
    # Normalise step values: replace any literal param values with {refs}.
    clean_steps = _substitute_param_refs(recorded_steps, params)

    # Normalise outputs: goal_complete sends a list [{key, value, type, description}].
    raw_outputs = goal_complete_input.get("outputs", [])
    if isinstance(raw_outputs, list):
        outputs_data: dict = {item["key"]: item for item in raw_outputs}
    else:
        outputs_data = raw_outputs  # defensive: accept legacy dict form too

    output_specs: dict[str, OutputSpec] = {
        key: OutputSpec(type=data["type"], description=data["description"])
        for key, data in outputs_data.items()
    }

    # Actual values extracted this run — returned separately, not stored in Capability.
    actual_outputs: dict[str, str] = {
        key: data["value"] for key, data in outputs_data.items()
    }

    # Fail loudly if the model used output_key names in read steps that it then
    # omitted from goal_complete outputs — silent auto-patching hides model errors.
    missing = {
        step.output_key
        for step in clean_steps
        if step.action == "read" and step.output_key and step.output_key not in output_specs
    }
    if missing:
        raise OutputKeyMismatch(
            f"Read steps reference output keys not declared in goal_complete outputs: {missing}. "
            f"Declared outputs: {set(output_specs)}. "
            f"Fix: ensure goal_complete outputs contains every key used in a read step."
        )

    # Parse checkpoint.
    ckpt_data: dict = goal_complete_input.get("checkpoint", {})
    ckpt_locator: Locator | None = None
    if "locator" in ckpt_data:
        ckpt_locator = Locator.model_validate(ckpt_data["locator"])
    checkpoint = Checkpoint(
        kind=ckpt_data["kind"],
        locator=ckpt_locator,
        expected=ckpt_data["expected"],
    )

    capability = Capability(
        id=run_id,
        version=1,
        name=capability_name,
        goal=goal,
        target={"base_url": base_url, "app": "mock_core_banking"},
        inputs=param_specs,
        outputs=output_specs,
        steps=clean_steps,
        checkpoint=checkpoint,
        created_from_run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return capability, actual_outputs


# ---------------------------------------------------------------------------
# Escalation hook
# ---------------------------------------------------------------------------

def _default_on_stuck(
    reason: str,
    page: Page,  # noqa: ARG001 — Phase 6 uses the live page object
    step_index: int,  # noqa: ARG001
    evidence: list[dict],  # noqa: ARG001
) -> None:
    """
    Default stuck handler — raises immediately.
    Phase 6 replaces this body with SessionController.pause() + operator console,
    which blocks until the human resumes, then returns so the loop can continue.
    The signature is fixed: Phase 6 only replaces the body.
    """
    raise DiscoveryStuck(reason)


# ---------------------------------------------------------------------------
# Evidence helpers (minimal metadata only — no raw observation text)
# ---------------------------------------------------------------------------

def _step_record(
    step_index: int,
    action: str,
    url_before: str,
    url_after: str,
    result: str,
    error: str | None,
) -> dict:
    return {
        "type": "action",
        "step_index": step_index,
        "action": action,
        "url_before": url_before,
        "url_after": url_after,
        "result": result,
        "error": error,
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "performed_by": "agent",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_discovery(
    page: Page,
    goal: str,
    capability_name: str,
    base_url: str,
    params: dict[str, str],
    param_specs: dict[str, ParamSpec],
    *,
    max_steps: int = 30,
    wall_clock_timeout_s: float = 300.0,
    max_consecutive_errors: int = 3,
    model: str = "gemini-3.6-flash",
    on_stuck: Callable[..., None] | None = None,
) -> DiscoveryResult:
    """
    Run the LLM-driven discovery loop against a live Playwright page.

    Stopping conditions (both are hard limits, whichever fires first):
      max_steps            — total successful action steps
      wall_clock_timeout_s — elapsed real time since run_discovery was called

    max_consecutive_errors consecutive action failures without any success
    triggers report_stuck automatically rather than burning steps on a stuck LLM.

    on_stuck: callable(reason, page, step_index, evidence) — called when the LLM
    invokes report_stuck or when consecutive error limit is hit. Default raises
    DiscoveryStuck. Phase 6 wires in SessionController.pause() here.
    """
    handle_stuck = on_stuck or _default_on_stuck
    run_id = f"disc_{uuid.uuid4().hex[:8]}"
    start_time = time.monotonic()
    llm = LLMClient(model=model)
    system_prompt = _build_system_prompt(goal, params, param_specs)

    recorded_steps: list[Step] = []
    read_values: dict[str, str] = {}
    evidence: list[dict] = []
    messages: list[dict] = []
    consecutive_errors = 0

    # Navigate to the start URL and take the initial observation.
    url_before = page.url
    do_navigate(page, base_url)
    obs: Observation = capture(page)
    messages.append(_initial_user_message(obs))
    evidence.append(_step_record(
        step_index=0,
        action="navigate",
        url_before=url_before,
        url_after=obs.url,
        result="ok",
        error=None,
    ))

    action_count = 0  # counts successful actions toward max_steps

    while True:
        # --- Wall-clock timeout check ---
        elapsed = time.monotonic() - start_time
        if elapsed > wall_clock_timeout_s:
            raise DiscoveryTimeout(
                f"Wall-clock timeout ({wall_clock_timeout_s}s) exceeded "
                f"after {action_count} successful steps."
            )

        # --- Step limit check ---
        if action_count >= max_steps:
            raise DiscoveryMaxSteps(
                f"Reached max_steps={max_steps} without completing the goal."
            )

        # --- Consecutive error guard ---
        if consecutive_errors >= max_consecutive_errors:
            handle_stuck(
                f"Automatic escalation: {consecutive_errors} consecutive action "
                f"failures without progress at step {action_count}.",
                page,
                action_count,
                evidence,
            )
            consecutive_errors = 0  # reset after human resumes (Phase 6)

        # --- Ask the LLM ---
        tool_call = llm.decide(messages, system=system_prompt)
        # Append the assistant turn to conversation history.
        messages.append({"role": "assistant", "content": [tool_call]})

        tool_name: str = tool_call.name
        tool_input: dict = tool_call.input  # type: ignore[union-attr]

        # --- goal_complete ---
        if tool_name == "goal_complete":
            try:
                capability, actual_outputs = _assemble_capability(
                    capability_name=capability_name,
                    goal=goal,
                    base_url=base_url,
                    params=params,
                    param_specs=param_specs,
                    recorded_steps=recorded_steps,
                    goal_complete_input=tool_input,
                    run_id=run_id,
                )
            except OutputKeyMismatch as exc:
                reason = (
                    f"goal_complete rejected: {exc} "
                    f"Retry goal_complete with output keys matching exactly what "
                    f"was declared in each read step."
                )
                evidence.append({
                    "type": "output_key_mismatch",
                    "reason": str(exc),
                    "step_index": action_count,
                    "timestamp_iso": datetime.now(timezone.utc).isoformat(),
                })
                handle_stuck(reason, page, action_count, evidence)
                continue
            artifact_path = save_capability(capability)
            evidence.append({
                "type": "goal_complete",
                "run_id": run_id,
                "artifact_path": artifact_path.name,
                "steps_taken": action_count,
                "elapsed_s": round(time.monotonic() - start_time, 2),
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            })
            return DiscoveryResult(
                capability=capability,
                artifact_path=artifact_path,
                run_id=run_id,
                outputs=actual_outputs,
                evidence=evidence,
                steps_taken=action_count,
                elapsed_s=round(time.monotonic() - start_time, 2),
            )

        # --- report_stuck ---
        if tool_name == "report_stuck":
            reason: str = tool_input.get("reason", "No reason provided.")
            evidence.append({
                "type": "stuck",
                "reason": reason,
                "step_index": action_count,
                "timestamp_iso": datetime.now(timezone.utc).isoformat(),
            })
            handle_stuck(reason, page, action_count, evidence)
            # If handle_stuck returns (Phase 6 human-resume path), take a fresh
            # observation and continue the loop without calling the LLM again
            # immediately — rebuild the last user message with the new page state.
            obs = capture(page)
            messages.append(_followup_user_message(
                tool_call.id,
                "Human operator took over and has resumed the session. "
                "The page state below reflects where they left off.",
                obs,
                action_count,
            ))
            continue

        # --- Action tools ---
        url_before_action = page.url
        result_text, is_error = _execute_action(
            page, tool_name, tool_input, params, recorded_steps, read_values
        )

        if is_error:
            consecutive_errors += 1
        else:
            consecutive_errors = 0
            action_count += 1

        url_after_action = page.url
        evidence.append(_step_record(
            step_index=action_count,
            action=tool_name,
            url_before=url_before_action,
            url_after=url_after_action,
            result="error" if is_error else "ok",
            error=result_text if is_error else None,
        ))

        # Capture fresh observation and append the follow-up user message.
        obs = capture(page)
        messages.append(_followup_user_message(
            tool_call.id,
            result_text,
            obs,
            action_count,
            is_error=is_error,
        ))
