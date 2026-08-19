"""
Unit tests for agent/discovery.py internal helpers.

These test the _assemble_capability function directly (imported by name from the
module) rather than running a full Playwright discovery loop.
"""
from __future__ import annotations

import pytest

from agent.discovery import OutputKeyMismatch, _assemble_capability
from artifact.schema import Checkpoint, Locator, ParamSpec, Step


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CHECKPOINT = {"kind": "url_matches", "expected": "/member/"}

_READ_STEP = Step(
    index=1,
    action="read",
    locator=Locator(
        strategy="xpath",
        value="//tr[td[normalize-space()='Savings']]/td[3]",
        fallback=Locator(strategy="css_fallback", value="td.balance"),
    ),
    output_key="savings_balance",
    reversible=True,
)

_NAV_STEP = Step(
    index=0,
    action="navigate",
    value="http://localhost:5001/search",
    reversible=True,
)

_BASE_KWARGS = dict(
    capability_name="lookup_member_balance",
    goal="Look up a member's savings balance.",
    base_url="http://localhost:5001/search",
    params={"member_id": "12345"},
    param_specs={
        "member_id": ParamSpec(type="str", description="Member to look up.")
    },
    recorded_steps=[_NAV_STEP, _READ_STEP],
    run_id="test_run_001",
)


# ---------------------------------------------------------------------------
# Typed outputs array
# ---------------------------------------------------------------------------

class TestTypedOutputsArray:
    """goal_complete now sends outputs as [{key, value, type, description}].
    _assemble_capability must normalise that list to the dict the Capability
    validator expects and return the correct actual_outputs."""

    def test_array_normalised_to_capability_output_specs(self):
        goal_complete_input = {
            "outputs": [
                {
                    "key": "savings_balance",
                    "value": "$5,432.10",
                    "type": "str",
                    "description": "Savings account current balance",
                }
            ],
            "checkpoint": _CHECKPOINT,
        }
        capability, actual_outputs = _assemble_capability(
            **_BASE_KWARGS, goal_complete_input=goal_complete_input
        )
        # OutputSpec was built from the array entry
        assert "savings_balance" in capability.outputs
        spec = capability.outputs["savings_balance"]
        assert spec.type == "str"
        assert spec.description == "Savings account current balance"

    def test_array_actual_outputs_contains_value(self):
        goal_complete_input = {
            "outputs": [
                {
                    "key": "savings_balance",
                    "value": "$5,432.10",
                    "type": "str",
                    "description": "Savings account current balance",
                }
            ],
            "checkpoint": _CHECKPOINT,
        }
        _, actual_outputs = _assemble_capability(
            **_BASE_KWARGS, goal_complete_input=goal_complete_input
        )
        # actual_outputs carries the live value (not stored in Capability itself)
        assert actual_outputs == {"savings_balance": "$5,432.10"}

    def test_dict_form_still_accepted(self):
        """Defensive: legacy dict format must still work."""
        goal_complete_input = {
            "outputs": {
                "savings_balance": {
                    "value": "$5,432.10",
                    "type": "str",
                    "description": "Savings account current balance",
                }
            },
            "checkpoint": _CHECKPOINT,
        }
        capability, actual_outputs = _assemble_capability(
            **_BASE_KWARGS, goal_complete_input=goal_complete_input
        )
        assert "savings_balance" in capability.outputs
        assert actual_outputs == {"savings_balance": "$5,432.10"}

    def test_array_roundtrip_serialises(self):
        """Capability built from an array output can be serialised to JSON."""
        goal_complete_input = {
            "outputs": [
                {
                    "key": "savings_balance",
                    "value": "$5,432.10",
                    "type": "str",
                    "description": "Savings account current balance",
                }
            ],
            "checkpoint": _CHECKPOINT,
        }
        capability, _ = _assemble_capability(
            **_BASE_KWARGS, goal_complete_input=goal_complete_input
        )
        serialised = capability.model_dump_json()
        assert "savings_balance" in serialised


# ---------------------------------------------------------------------------
# OutputKeyMismatch loud failure
# ---------------------------------------------------------------------------

class TestOutputKeyMismatch:
    """_assemble_capability must raise OutputKeyMismatch — not silently patch —
    when a read step declares an output_key absent from goal_complete outputs.
    This is distinct from the Pydantic ValidationError that Capability raises
    when it detects the same invariant violation after the fact."""

    def test_raises_when_output_key_missing_from_empty_outputs(self):
        goal_complete_input = {
            "outputs": [],       # read step declared savings_balance; not echoed here
            "checkpoint": _CHECKPOINT,
        }
        with pytest.raises(OutputKeyMismatch, match="savings_balance"):
            _assemble_capability(
                **_BASE_KWARGS, goal_complete_input=goal_complete_input
            )

    def test_raises_when_output_key_missing_from_partial_outputs(self):
        """Missing key is caught even when outputs has other entries."""
        goal_complete_input = {
            "outputs": [
                {
                    "key": "some_other_key",
                    "value": "x",
                    "type": "str",
                    "description": "irrelevant",
                }
            ],
            "checkpoint": _CHECKPOINT,
        }
        with pytest.raises(OutputKeyMismatch, match="savings_balance"):
            _assemble_capability(
                **_BASE_KWARGS, goal_complete_input=goal_complete_input
            )

    def test_error_message_names_missing_keys(self):
        goal_complete_input = {"outputs": [], "checkpoint": _CHECKPOINT}
        with pytest.raises(OutputKeyMismatch) as exc_info:
            _assemble_capability(
                **_BASE_KWARGS, goal_complete_input=goal_complete_input
            )
        msg = str(exc_info.value)
        assert "savings_balance" in msg
        assert "goal_complete" in msg

    def test_no_raise_when_all_keys_declared(self):
        """Sanity: no exception when outputs correctly covers all read steps."""
        goal_complete_input = {
            "outputs": [
                {
                    "key": "savings_balance",
                    "value": "$5,432.10",
                    "type": "str",
                    "description": "balance",
                }
            ],
            "checkpoint": _CHECKPOINT,
        }
        # Should not raise
        capability, _ = _assemble_capability(
            **_BASE_KWARGS, goal_complete_input=goal_complete_input
        )
        assert capability is not None

    def test_is_subclass_of_discovery_error(self):
        from agent.discovery import DiscoveryError
        assert issubclass(OutputKeyMismatch, DiscoveryError)
