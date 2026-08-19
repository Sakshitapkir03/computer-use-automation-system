"""Round-trip serialization tests + JSON Schema export for artifact.schema."""
import json
import pytest
from artifact.schema import (
    Capability,
    Checkpoint,
    Locator,
    OutputSpec,
    ParamSpec,
    ReplayResult,
    Step,
)


# ---------------------------------------------------------------------------
# OutputSpec tests
# ---------------------------------------------------------------------------

class TestOutputSpec:
    def test_default_not_sensitive(self):
        spec = OutputSpec(type="str", description="balance")
        assert spec.sensitive is False

    def test_sensitive_true_accepted(self):
        spec = OutputSpec(type="decimal", sensitive=True, description="SSN")
        assert spec.sensitive is True

    def test_roundtrip_sensitive_false(self):
        spec = OutputSpec(type="str", description="balance")
        restored = OutputSpec.model_validate_json(spec.model_dump_json())
        assert restored == spec
        assert restored.sensitive is False

    def test_roundtrip_sensitive_true(self):
        spec = OutputSpec(type="str", sensitive=True, description="member ssn")
        restored = OutputSpec.model_validate_json(spec.model_dump_json())
        assert restored.sensitive is True

    def test_sensitive_field_present_in_capability_json(self):
        """OutputSpec.sensitive must be serialised into Capability JSON."""
        cap = _make_capability()
        j = cap.model_dump_json()
        data = json.loads(j)
        balance_spec = data["outputs"]["balance"]
        assert "sensitive" in balance_spec

    def test_sensitive_true_survives_capability_roundtrip(self):
        cap = _make_capability()
        # Rebuild with sensitive=True on the balance output.
        cap2 = cap.model_copy(
            update={"outputs": {"balance": OutputSpec(type="decimal", sensitive=True, description="balance")}}
        )
        restored = Capability.model_validate_json(cap2.model_dump_json())
        assert restored.outputs["balance"].sensitive is True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_locator(strategy="role_name", value="Search", role="button") -> Locator:
    return Locator(strategy=strategy, value=value, role=role)


def _make_locator_with_fallback() -> Locator:
    # Primary: css_fallback on the name attribute — specific to this element,
    # survives DOM reordering, and works even when multiple unlabeled textboxes
    # are present (e.g. the open-subaccount form has initial_deposit + purpose).
    # Last resort: empty-name role_name/textbox, only unambiguous when there is
    # exactly one textbox on the page — not a safe assumption in general.
    fallback = Locator(strategy="role_name", role="textbox", value="")
    return Locator(
        strategy="css_fallback",
        value="input[name='member_id']",
        fallback=fallback,
    )


def _make_capability() -> Capability:
    steps = [
        Step(
            index=0,
            action="navigate",
            value="http://localhost:5000/search",
            reversible=True,
        ),
        Step(
            index=1,
            action="type",
            locator=_make_locator_with_fallback(),
            value="{member_id}",
            reversible=True,
        ),
        Step(
            index=2,
            action="click",
            locator=_make_locator(),
            reversible=True,
        ),
        Step(
            index=3,
            action="read",
            locator=Locator(strategy="css_fallback", value="td.balance"),
            output_key="balance",
            reversible=True,
        ),
    ]
    return Capability(
        id="cap_001",
        version=1,
        name="lookup_member_balance",
        goal="Look up a member's savings account balance by member ID",
        target={"base_url": "http://localhost:5000", "app": "mock_core_banking"},
        inputs={
            "member_id": ParamSpec(type="str", required=True, description="Member ID"),
        },
        outputs={
            "balance": OutputSpec(type="decimal", description="Current savings balance"),
        },
        steps=steps,
        checkpoint=Checkpoint(
            kind="text_present",
            locator=Locator(strategy="css_fallback", value="td.balance"),
            expected="$",
        ),
        created_from_run_id="run_abc123",
    )


# ---------------------------------------------------------------------------
# Locator tests
# ---------------------------------------------------------------------------

class TestLocator:
    def test_role_name_roundtrip(self):
        loc = _make_locator()
        data = loc.model_dump()
        restored = Locator.model_validate(data)
        assert restored == loc

    def test_json_roundtrip(self):
        loc = _make_locator_with_fallback()
        j = loc.model_dump_json()
        restored = Locator.model_validate_json(j)
        assert restored == loc
        assert restored.fallback is not None
        assert restored.fallback.strategy == "role_name"

    def test_nested_fallback_preserved(self):
        inner = Locator(strategy="css_fallback", value="#balance")
        mid = Locator(strategy="aria_label", value="Balance", fallback=inner)
        outer = Locator(strategy="role_name", value="Balance cell", role="cell", fallback=mid)
        j = outer.model_dump_json()
        restored = Locator.model_validate_json(j)
        assert restored.fallback.fallback.strategy == "css_fallback"


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

class TestStep:
    def test_navigate_no_locator_valid(self):
        s = Step(index=0, action="navigate", value="http://localhost:5000", reversible=True)
        assert s.locator is None

    def test_navigate_with_locator_raises(self):
        with pytest.raises(Exception):
            Step(
                index=0,
                action="navigate",
                value="http://localhost:5000",
                locator=_make_locator(),
                reversible=True,
            )

    def test_navigate_without_value_raises(self):
        with pytest.raises(Exception):
            Step(index=0, action="navigate", reversible=True)

    def test_read_without_output_key_raises(self):
        with pytest.raises(Exception):
            Step(
                index=0,
                action="read",
                locator=Locator(strategy="css_fallback", value="td"),
                reversible=True,
            )

    def test_read_with_output_key_valid(self):
        s = Step(
            index=0,
            action="read",
            locator=Locator(strategy="css_fallback", value="td"),
            output_key="balance",
            reversible=True,
        )
        assert s.output_key == "balance"

    def test_roundtrip(self):
        s = Step(
            index=2,
            action="click",
            locator=_make_locator_with_fallback(),
            reversible=False,
        )
        restored = Step.model_validate_json(s.model_dump_json())
        assert restored == s


# ---------------------------------------------------------------------------
# Capability tests
# ---------------------------------------------------------------------------

class TestCapability:
    def test_roundtrip(self):
        cap = _make_capability()
        j = cap.model_dump_json()
        restored = Capability.model_validate_json(j)
        assert restored.id == cap.id
        assert restored.name == cap.name
        assert len(restored.steps) == len(cap.steps)
        assert restored.steps[1].locator.fallback.strategy == "role_name"

    def test_dict_roundtrip(self):
        cap = _make_capability()
        d = cap.model_dump()
        restored = Capability.model_validate(d)
        assert restored == cap

    def test_step_index_mismatch_raises(self):
        steps = [
            Step(index=0, action="navigate", value="http://localhost:5000", reversible=True),
            Step(index=5, action="click", locator=_make_locator(), reversible=True),
        ]
        with pytest.raises(Exception, match="index"):
            Capability(
                id="x",
                name="x",
                goal="x",
                target={},
                inputs={},
                outputs={},
                steps=steps,
                checkpoint=Checkpoint(kind="url_matches", expected="http://localhost:5000"),
                created_from_run_id="run_x",
            )

    def test_undeclared_output_key_raises(self):
        steps = [
            Step(index=0, action="navigate", value="http://localhost:5000", reversible=True),
            Step(
                index=1,
                action="read",
                locator=Locator(strategy="css_fallback", value="td"),
                output_key="undeclared_key",
                reversible=True,
            ),
        ]
        with pytest.raises(Exception, match="output keys"):
            Capability(
                id="x",
                name="x",
                goal="x",
                target={},
                inputs={},
                outputs={},
                steps=steps,
                checkpoint=Checkpoint(kind="url_matches", expected="http://localhost:5000"),
                created_from_run_id="run_x",
            )

    def test_json_schema_export(self):
        schema = Capability.model_json_schema()
        assert isinstance(schema, dict)
        assert schema.get("title") == "Capability"
        # Must have all top-level fields
        props = schema.get("properties", {})
        for field in ("id", "name", "goal", "steps", "inputs", "outputs", "checkpoint"):
            assert field in props, f"Missing field in JSON schema: {field}"

    def test_json_schema_is_serializable(self):
        schema = Capability.model_json_schema()
        # Must be valid JSON
        dumped = json.dumps(schema)
        assert isinstance(dumped, str)
        reloaded = json.loads(dumped)
        assert reloaded["title"] == "Capability"


# ---------------------------------------------------------------------------
# ReplayResult tests
# ---------------------------------------------------------------------------

class TestReplayResult:
    def test_success(self):
        r = ReplayResult(status="success", outputs={"balance": "5432.10"})
        assert r.status == "success"
        assert r.outcome_code is None

    def test_business_outcome(self):
        r = ReplayResult(
            status="business_outcome",
            outcome_code="MEMBER_NOT_FOUND",
            outcome_message="No member with ID 99999",
        )
        assert r.outcome_code == "MEMBER_NOT_FOUND"

    def test_hard_failure(self):
        r = ReplayResult(
            status="hard_failure",
            failure_step_index=3,
            failure_expected="Balance visible",
            failure_observed="Permission denied page",
            screenshot_path="/evidence/replay_run_001/failure.png",
        )
        assert r.failure_step_index == 3

    def test_roundtrip(self):
        r = ReplayResult(
            status="hard_failure",
            failure_step_index=2,
            failure_expected="text:$",
            failure_observed="text:Access denied",
        )
        restored = ReplayResult.model_validate_json(r.model_dump_json())
        assert restored == r

    def test_invalid_status_raises(self):
        with pytest.raises(Exception):
            ReplayResult(status="recoverable")
