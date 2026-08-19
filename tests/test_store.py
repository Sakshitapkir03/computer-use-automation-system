"""
Tests for artifact/store.py — focused on collect_sensitive_values().

save_capability / load_capability are integration-level (they touch disk);
collect_sensitive_values is pure logic and fully unit-testable.
"""
from __future__ import annotations

import pytest

from artifact.schema import Capability, Checkpoint, Locator, OutputSpec, ParamSpec, Step
from artifact.store import collect_sensitive_values
from guardrails.redaction import redact


# ---------------------------------------------------------------------------
# Minimal Capability fixture
# ---------------------------------------------------------------------------

def _make_cap(
    *,
    member_id_sensitive: bool = False,
    balance_sensitive: bool = False,
    ssn_sensitive: bool = False,
) -> Capability:
    steps = [
        Step(index=0, action="navigate", value="http://localhost:5001/search", reversible=True),
        Step(
            index=1,
            action="read",
            locator=Locator(strategy="css_fallback", value="td.balance"),
            output_key="savings_balance",
            reversible=True,
        ),
    ]
    outputs: dict = {
        "savings_balance": OutputSpec(
            type="decimal",
            sensitive=balance_sensitive,
            description="Savings account balance",
        ),
    }
    if ssn_sensitive:
        steps.append(
            Step(
                index=2,
                action="read",
                locator=Locator(strategy="css_fallback", value="td.ssn"),
                output_key="member_ssn",
                reversible=True,
            )
        )
        outputs["member_ssn"] = OutputSpec(
            type="str",
            sensitive=True,
            description="Member SSN",
        )

    return Capability(
        id="cap_test",
        version=1,
        name="test_cap",
        goal="Test capability",
        target={"base_url": "http://localhost:5001/search"},
        inputs={
            "member_id": ParamSpec(
                type="str",
                sensitive=member_id_sensitive,
                description="Member ID",
            ),
        },
        outputs=outputs,
        steps=steps,
        checkpoint=Checkpoint(kind="url_matches", expected="/member/"),
        created_from_run_id="run_test",
    )


_PARAMS = {"member_id": "12345"}
_OUTPUTS = {"savings_balance": "$5432.10"}


# ---------------------------------------------------------------------------
# collect_sensitive_values — param sensitivity
# ---------------------------------------------------------------------------

class TestCollectSensitiveValuesParams:
    def test_sensitive_param_value_included(self):
        cap = _make_cap(member_id_sensitive=True)
        result = collect_sensitive_values(cap, _PARAMS, _OUTPUTS)
        assert "12345" in result

    def test_non_sensitive_param_excluded(self):
        cap = _make_cap(member_id_sensitive=False)
        result = collect_sensitive_values(cap, _PARAMS, _OUTPUTS)
        assert "12345" not in result

    def test_sensitive_param_absent_from_runtime_dict_skipped(self):
        """Sensitive param with no runtime value must not add an empty string."""
        cap = _make_cap(member_id_sensitive=True)
        result = collect_sensitive_values(cap, {}, _OUTPUTS)
        assert result == [] or all(v for v in result)

    def test_sensitive_param_empty_string_value_skipped(self):
        cap = _make_cap(member_id_sensitive=True)
        result = collect_sensitive_values(cap, {"member_id": ""}, _OUTPUTS)
        assert "" not in result


# ---------------------------------------------------------------------------
# collect_sensitive_values — output sensitivity
# ---------------------------------------------------------------------------

class TestCollectSensitiveValuesOutputs:
    def test_sensitive_output_value_included(self):
        cap = _make_cap(balance_sensitive=True)
        result = collect_sensitive_values(cap, _PARAMS, _OUTPUTS)
        assert "$5432.10" in result

    def test_non_sensitive_output_excluded(self):
        cap = _make_cap(balance_sensitive=False)
        result = collect_sensitive_values(cap, _PARAMS, _OUTPUTS)
        assert "$5432.10" not in result

    def test_sensitive_output_absent_from_runtime_dict_skipped(self):
        cap = _make_cap(balance_sensitive=True)
        result = collect_sensitive_values(cap, _PARAMS, {})
        assert "$5432.10" not in result
        assert result == [] or all(v for v in result)

    def test_multiple_sensitive_outputs_all_included(self):
        cap = _make_cap(balance_sensitive=True, ssn_sensitive=True)
        outputs = {"savings_balance": "$5432.10", "member_ssn": "987-65-4321"}
        result = collect_sensitive_values(cap, _PARAMS, outputs)
        assert "$5432.10" in result
        assert "987-65-4321" in result

    def test_mix_sensitive_and_non_sensitive_outputs(self):
        cap = _make_cap(balance_sensitive=False, ssn_sensitive=True)
        outputs = {"savings_balance": "$5432.10", "member_ssn": "987-65-4321"}
        result = collect_sensitive_values(cap, _PARAMS, outputs)
        assert "$5432.10" not in result
        assert "987-65-4321" in result


# ---------------------------------------------------------------------------
# collect_sensitive_values — combined and edge cases
# ---------------------------------------------------------------------------

class TestCollectSensitiveValuesCombined:
    def test_nothing_sensitive_returns_empty(self):
        cap = _make_cap(member_id_sensitive=False, balance_sensitive=False)
        result = collect_sensitive_values(cap, _PARAMS, _OUTPUTS)
        assert result == []

    def test_both_sensitive_both_included(self):
        cap = _make_cap(member_id_sensitive=True, balance_sensitive=True)
        result = collect_sensitive_values(cap, _PARAMS, _OUTPUTS)
        assert "12345" in result
        assert "$5432.10" in result
        assert len(result) == 2

    def test_returns_list(self):
        cap = _make_cap()
        result = collect_sensitive_values(cap, _PARAMS, _OUTPUTS)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Integration: collect → redact removes all sensitive values from evidence
# ---------------------------------------------------------------------------

class TestCollectThenRedact:
    def test_sensitive_param_scrubbed_from_evidence(self):
        cap = _make_cap(member_id_sensitive=True)
        evidence = "Searching for member 12345 on page /search"
        safe = redact(evidence, sensitive_values=collect_sensitive_values(cap, _PARAMS, _OUTPUTS))
        assert "12345" not in safe
        assert "[REDACTED]" in safe

    def test_sensitive_output_scrubbed_from_evidence(self):
        cap = _make_cap(balance_sensitive=True)
        evidence = 'Aria snapshot: savings balance is "$5432.10" confirmed.'
        safe = redact(evidence, sensitive_values=collect_sensitive_values(cap, _PARAMS, _OUTPUTS))
        assert "$5432.10" not in safe
        assert "[REDACTED]" in safe

    def test_non_sensitive_values_survive_redaction(self):
        cap = _make_cap(member_id_sensitive=False, balance_sensitive=False)
        evidence = "member 12345 balance $5432.10"
        safe = redact(evidence, sensitive_values=collect_sensitive_values(cap, _PARAMS, _OUTPUTS))
        # Neither value is marked sensitive — structural patterns won't catch them either
        assert "12345" in safe   # short ID, below 9-digit threshold
        assert "$5432.10" in safe

    def test_sensitive_param_and_output_both_scrubbed(self):
        cap = _make_cap(member_id_sensitive=True, balance_sensitive=True)
        evidence = "member 12345 has savings balance $5432.10"
        safe = redact(evidence, sensitive_values=collect_sensitive_values(cap, _PARAMS, _OUTPUTS))
        assert "12345" not in safe
        assert "$5432.10" not in safe
        assert safe.count("[REDACTED]") == 2
