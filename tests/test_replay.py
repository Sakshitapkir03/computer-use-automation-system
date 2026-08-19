"""
Unit tests for replay/executor.py.

All Playwright I/O is mocked — no browser or target app required.
The live-replay integration test lives in tests/run_replay_test.py.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from artifact.schema import (
    BusinessOutcomeSpec,
    Capability,
    Checkpoint,
    Locator,
    OutputSpec,
    ParamSpec,
    ReplayResult,
    Step,
)
from replay.executor import _resolve, _verify_checkpoint, run_replay


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_LOC = Locator(
    strategy="css_fallback",
    value="input[name='x']",
    fallback=Locator(strategy="role_name", value="", role="textbox"),
)

_URL_CHECKPOINT = Checkpoint(kind="url_matches", expected="/member/")
_TEXT_CHECKPOINT = Checkpoint(
    kind="text_present",
    locator=_LOC,
    expected="Member Details",
)
_VIS_CHECKPOINT = Checkpoint(kind="element_visible", locator=_LOC, expected="")

_NOT_FOUND_SPEC = BusinessOutcomeSpec(
    outcome_code="MEMBER_NOT_FOUND",
    outcome_message="Member ID was not found in the system.",
    checkpoint=Checkpoint(kind="url_matches", expected="/not-found"),
)


def _cap(
    steps: list[Step],
    *,
    outputs: dict | None = None,
    checkpoint: Checkpoint = _URL_CHECKPOINT,
    base_url: str = "http://localhost:5001/search",
    business_outcomes: list[BusinessOutcomeSpec] | None = None,
) -> Capability:
    """Build a minimal valid Capability for use in tests."""
    return Capability(
        id="test",
        version=1,
        name="test_cap",
        goal="test",
        target={"base_url": base_url},
        inputs={"member_id": ParamSpec(type="str", description="id")},
        outputs=outputs or {},
        steps=steps,
        checkpoint=checkpoint,
        business_outcomes=business_outcomes or [],
        created_from_run_id="test",
    )


def _mock_page(url: str = "http://localhost:5001/member/12345") -> MagicMock:
    page = MagicMock()
    type(page).url = property(lambda self: url)
    return page


# Patch targets — all action functions are imported into replay.executor's
# namespace, so that is where they must be patched.
_PATCH = "replay.executor.{}"


# ---------------------------------------------------------------------------
# _resolve
# ---------------------------------------------------------------------------

class TestResolve:
    def test_single_ref(self):
        assert _resolve("{member_id}", {"member_id": "12345"}) == "12345"

    def test_multiple_refs(self):
        result = _resolve("{a}/{b}", {"a": "foo", "b": "bar"})
        assert result == "foo/bar"

    def test_no_refs_unchanged(self):
        assert _resolve("http://localhost/search", {"member_id": "x"}) == "http://localhost/search"

    def test_unknown_ref_left_in_place(self):
        # Refs not in params are left as-is (not silently dropped).
        assert _resolve("{missing}", {"other": "x"}) == "{missing}"


# ---------------------------------------------------------------------------
# _verify_checkpoint
# ---------------------------------------------------------------------------

class TestVerifyCheckpoint:
    def test_url_matches_passes(self):
        page = _mock_page("http://localhost:5001/member/12345")
        passed, observed = _verify_checkpoint(page, _URL_CHECKPOINT)
        assert passed is True
        assert "/member/" in observed

    def test_url_matches_fails(self):
        page = _mock_page("http://localhost:5001/not-found")
        passed, _ = _verify_checkpoint(page, _URL_CHECKPOINT)
        assert passed is False

    @patch(_PATCH.format("resolve_locator"))
    def test_element_visible_passes(self, mock_resolve):
        loc_mock = MagicMock()
        loc_mock.first.is_visible.return_value = True
        mock_resolve.return_value = loc_mock
        page = _mock_page()
        passed, observed = _verify_checkpoint(page, _VIS_CHECKPOINT)
        assert passed is True
        assert observed == "element visible"

    @patch(_PATCH.format("resolve_locator"))
    def test_element_visible_fails(self, mock_resolve):
        loc_mock = MagicMock()
        loc_mock.first.is_visible.return_value = False
        mock_resolve.return_value = loc_mock
        page = _mock_page()
        passed, observed = _verify_checkpoint(page, _VIS_CHECKPOINT)
        assert passed is False
        assert observed == "element not visible"

    @patch(_PATCH.format("resolve_locator"))
    def test_element_visible_locator_not_found(self, mock_resolve):
        from agent.actions import LocatorResolutionError
        mock_resolve.side_effect = LocatorResolutionError("not found")
        page = _mock_page()
        passed, observed = _verify_checkpoint(page, _VIS_CHECKPOINT)
        assert passed is False
        assert "not found" in observed

    @patch(_PATCH.format("resolve_locator"))
    def test_text_present_passes(self, mock_resolve):
        loc_mock = MagicMock()
        loc_mock.first.text_content.return_value = "Member Details — John Smith"
        mock_resolve.return_value = loc_mock
        page = _mock_page()
        passed, observed = _verify_checkpoint(page, _TEXT_CHECKPOINT)
        assert passed is True
        assert "Member Details" in observed

    @patch(_PATCH.format("resolve_locator"))
    def test_text_present_fails(self, mock_resolve):
        loc_mock = MagicMock()
        loc_mock.first.text_content.return_value = "Member Not Found"
        mock_resolve.return_value = loc_mock
        page = _mock_page()
        passed, observed = _verify_checkpoint(page, _TEXT_CHECKPOINT)
        assert passed is False

    def test_element_visible_without_locator_returns_false(self):
        ckpt = Checkpoint(kind="element_visible", expected="x")  # no locator
        page = _mock_page()
        passed, observed = _verify_checkpoint(page, ckpt)
        assert passed is False
        assert "locator" in observed


# ---------------------------------------------------------------------------
# run_replay — happy path
# ---------------------------------------------------------------------------

class TestRunReplaySuccess:
    @patch(_PATCH.format("do_navigate"))
    @patch(_PATCH.format("do_click"))
    @patch(_PATCH.format("do_type"))
    def test_returns_success_when_checkpoint_passes(self, mock_type, mock_click, mock_nav):
        steps = [
            Step(index=0, action="type", locator=_LOC, value="{member_id}", reversible=True),
            Step(index=1, action="click", locator=_LOC, reversible=True),
        ]
        cap = _cap(steps, checkpoint=_URL_CHECKPOINT)
        page = _mock_page("http://localhost:5001/member/12345")
        result = run_replay(cap, {"member_id": "12345"}, page)
        assert result.status == "success"
        assert result.failure_step_index is None

    @patch(_PATCH.format("do_navigate"))
    @patch(_PATCH.format("do_read"))
    def test_read_step_populates_outputs(self, mock_read, mock_nav):
        mock_read.return_value = "$5,432.10"
        steps = [
            Step(
                index=0, action="read", locator=_LOC,
                output_key="savings_balance", reversible=True,
            ),
        ]
        cap = _cap(
            steps,
            outputs={"savings_balance": OutputSpec(type="str", description="balance")},
            checkpoint=_URL_CHECKPOINT,
        )
        page = _mock_page("http://localhost:5001/member/12345")
        result = run_replay(cap, {}, page)
        assert result.status == "success"
        assert result.outputs == {"savings_balance": "$5,432.10"}

    @patch(_PATCH.format("do_navigate"))
    @patch(_PATCH.format("do_type"))
    def test_param_refs_substituted_in_type_step(self, mock_type, mock_nav):
        steps = [
            Step(index=0, action="type", locator=_LOC, value="{member_id}", reversible=True),
        ]
        cap = _cap(steps)
        page = _mock_page("http://localhost:5001/member/12345")
        run_replay(cap, {"member_id": "12345"}, page)
        mock_type.assert_called_once_with(page, _LOC, "12345")

    @patch(_PATCH.format("do_navigate"))
    def test_param_refs_substituted_in_navigate_step(self, mock_nav):
        steps = [
            Step(
                index=0, action="navigate",
                value="http://localhost:5001/member/{member_id}",
                reversible=True,
            ),
        ]
        cap = _cap(steps)
        page = _mock_page("http://localhost:5001/member/12345")
        run_replay(cap, {"member_id": "12345"}, page)
        assert mock_nav.call_args_list == [
            call(page, "http://localhost:5001/search"),
            call(page, "http://localhost:5001/member/12345"),
        ]

    @patch(_PATCH.format("do_navigate"))
    @patch(_PATCH.format("do_wait_for"))
    @patch(_PATCH.format("do_click"))
    def test_all_step_types_dispatched(self, mock_click, mock_wait, mock_nav):
        steps = [
            Step(index=0, action="wait_for", locator=_LOC, reversible=True),
            Step(index=1, action="click", locator=_LOC, reversible=True),
        ]
        cap = _cap(steps)
        page = _mock_page("http://localhost:5001/member/12345")
        result = run_replay(cap, {}, page)
        assert result.status == "success"
        mock_wait.assert_called_once_with(page, _LOC)
        mock_click.assert_called_once_with(page, _LOC)


# ---------------------------------------------------------------------------
# run_replay — business_outcome (requires a declared BusinessOutcomeSpec)
# ---------------------------------------------------------------------------

class TestRunReplayBusinessOutcome:
    """
    business_outcome is ONLY returned when a declared BusinessOutcomeSpec
    positively matches the current page state.  It is never a default fallback.
    """

    @patch(_PATCH.format("do_navigate"))
    def test_checkpoint_fail_with_matching_outcome_spec(self, mock_nav):
        """Primary checkpoint fails; declared spec matches → business_outcome."""
        cap = _cap(
            steps=[],
            checkpoint=_URL_CHECKPOINT,          # expects /member/
            business_outcomes=[_NOT_FOUND_SPEC],
        )
        page = _mock_page("http://localhost:5001/not-found?member_id=99999")
        result = run_replay(cap, {}, page)
        assert result.status == "business_outcome"
        assert result.outcome_code == "MEMBER_NOT_FOUND"
        assert result.outcome_message == _NOT_FOUND_SPEC.outcome_message
        assert result.failure_step_index is None

    @patch(_PATCH.format("do_navigate"))
    def test_checkpoint_fail_without_any_outcome_spec_returns_hard_failure(self, mock_nav):
        """Primary checkpoint fails; no specs declared → hard_failure (not business_outcome)."""
        cap = _cap(
            steps=[],
            checkpoint=_URL_CHECKPOINT,
            business_outcomes=[],               # intentionally empty
        )
        page = _mock_page("http://localhost:5001/unknown-error")
        result = run_replay(cap, {}, page)
        assert result.status == "hard_failure"

    @patch(_PATCH.format("do_navigate"))
    def test_checkpoint_fail_with_non_matching_spec_returns_hard_failure(self, mock_nav):
        """Primary checkpoint fails; spec declared but page doesn't match it → hard_failure."""
        cap = _cap(
            steps=[],
            checkpoint=_URL_CHECKPOINT,         # expects /member/
            business_outcomes=[_NOT_FOUND_SPEC],  # expects /not-found
        )
        # Page matches neither /member/ nor /not-found — an unexpected error page
        page = _mock_page("http://localhost:5001/session-expired")
        result = run_replay(cap, {}, page)
        assert result.status == "hard_failure"

    @patch(_PATCH.format("do_navigate"))
    def test_step_failure_with_matching_outcome_spec(self, mock_nav):
        """Step fails; page is on a known outcome state → business_outcome (not hard_failure)."""
        from agent.actions import LocatorResolutionError
        cap = _cap(
            steps=[Step(index=0, action="click", locator=_LOC, reversible=True)],
            checkpoint=_URL_CHECKPOINT,
            business_outcomes=[_NOT_FOUND_SPEC],
        )
        page = _mock_page("http://localhost:5001/not-found?member_id=99999")
        with patch(_PATCH.format("do_click"), side_effect=LocatorResolutionError("no element")):
            result = run_replay(cap, {}, page)
        assert result.status == "business_outcome"
        assert result.outcome_code == "MEMBER_NOT_FOUND"

    @patch(_PATCH.format("do_navigate"))
    def test_step_failure_without_matching_spec_returns_hard_failure(self, mock_nav):
        """Step fails; page is NOT on any declared outcome state → hard_failure."""
        from agent.actions import LocatorResolutionError
        cap = _cap(
            steps=[Step(index=0, action="click", locator=_LOC, reversible=True)],
            checkpoint=_URL_CHECKPOINT,
            business_outcomes=[_NOT_FOUND_SPEC],  # expects /not-found
        )
        # Page is on member/67890 — an unexpected state, not the not-found page
        page = _mock_page("http://localhost:5001/member/67890")
        with patch(_PATCH.format("do_click"), side_effect=LocatorResolutionError("no savings row")):
            result = run_replay(cap, {}, page)
        assert result.status == "hard_failure"
        assert result.failure_step_index == 0

    @patch(_PATCH.format("do_navigate"))
    def test_outputs_preserved_on_business_outcome(self, mock_nav):
        """Outputs captured before a failing step are still returned on business_outcome."""
        with patch(_PATCH.format("do_read"), return_value="Savings"):
            steps = [
                Step(
                    index=0, action="read", locator=_LOC,
                    output_key="acct_type", reversible=True,
                ),
                Step(index=1, action="click", locator=_LOC, reversible=True),
            ]
            cap = _cap(
                steps,
                outputs={"acct_type": OutputSpec(type="str", description="type")},
                checkpoint=_URL_CHECKPOINT,
                business_outcomes=[_NOT_FOUND_SPEC],
            )
            from agent.actions import LocatorResolutionError
            page = _mock_page("http://localhost:5001/not-found")
            with patch(_PATCH.format("do_click"), side_effect=LocatorResolutionError("x")):
                result = run_replay(cap, {}, page)
        assert result.status == "business_outcome"
        assert result.outputs == {"acct_type": "Savings"}

    @patch(_PATCH.format("do_navigate"))
    def test_hard_failure_carries_observed_description(self, mock_nav):
        """When checkpoint fails and no spec matches, failure_observed is populated."""
        cap = _cap(steps=[], checkpoint=_URL_CHECKPOINT)
        page = _mock_page("http://localhost:5001/broken")
        result = run_replay(cap, {}, page)
        assert result.status == "hard_failure"
        assert result.failure_observed is not None
        assert "broken" in result.failure_observed  # the observed URL


# ---------------------------------------------------------------------------
# run_replay — hard_failure paths
# ---------------------------------------------------------------------------

class TestRunReplayHardFailure:
    @patch(_PATCH.format("do_navigate"))
    def test_initial_nav_failure_returns_hard_failure(self, mock_nav):
        from agent.actions import ActionError
        mock_nav.side_effect = ActionError("connection refused")
        cap = _cap([])
        page = _mock_page()
        result = run_replay(cap, {}, page)
        assert result.status == "hard_failure"
        assert result.failure_step_index == -1
        assert "connection refused" in result.failure_observed

    @patch(_PATCH.format("do_navigate"))
    @patch(_PATCH.format("do_click"))
    def test_action_error_in_step_returns_hard_failure(self, mock_click, mock_nav):
        from agent.actions import LocatorResolutionError
        mock_click.side_effect = LocatorResolutionError("button not found")
        steps = [Step(index=0, action="click", locator=_LOC, reversible=True)]
        cap = _cap(steps)
        page = _mock_page()
        result = run_replay(cap, {}, page)
        assert result.status == "hard_failure"
        assert result.failure_step_index == 0
        assert "button not found" in result.failure_observed

    @patch(_PATCH.format("do_navigate"))
    @patch(_PATCH.format("do_type"))
    def test_unexpected_exception_returns_hard_failure(self, mock_type, mock_nav):
        mock_type.side_effect = RuntimeError("unexpected crash")
        steps = [Step(index=0, action="type", locator=_LOC, value="x", reversible=True)]
        cap = _cap(steps)
        page = _mock_page()
        result = run_replay(cap, {}, page)
        assert result.status == "hard_failure"
        assert "RuntimeError" in result.failure_observed

    @patch(_PATCH.format("do_navigate"))
    def test_hard_failure_contains_step_index(self, mock_nav):
        from agent.actions import ActionError
        with patch(_PATCH.format("do_click"), side_effect=ActionError("x")):
            steps = [
                Step(index=0, action="click", locator=_LOC, reversible=True),
            ]
            cap = _cap(steps)
            result = run_replay(cap, {}, _mock_page())
        assert result.failure_step_index == 0

    @patch(_PATCH.format("do_navigate"))
    def test_allowlist_denied_returns_hard_failure(self, mock_nav):
        from agent.actions import AllowlistDenied
        with patch(_PATCH.format("do_click"), side_effect=AllowlistDenied("blocked")):
            steps = [Step(index=0, action="click", locator=_LOC, reversible=True)]
            cap = _cap(steps)
            result = run_replay(cap, {}, _mock_page())
        assert result.status == "hard_failure"
        assert "AllowlistDenied" in result.failure_observed


# ---------------------------------------------------------------------------
# run_replay — auto_confirm / irreversible gate
# ---------------------------------------------------------------------------

class TestAutoConfirmGate:
    @patch(_PATCH.format("do_navigate"))
    def test_irreversible_step_blocked_without_auto_confirm(self, mock_nav):
        steps = [
            Step(index=0, action="click", locator=_LOC, reversible=True),
            Step(index=1, action="click", locator=_LOC, reversible=False),
        ]
        cap = _cap(steps)
        with patch(_PATCH.format("do_click")) as mock_click:
            result = run_replay(cap, {}, _mock_page(), auto_confirm=False)
        assert result.status == "hard_failure"
        assert result.failure_step_index == 1
        assert "auto_confirm" in result.failure_observed
        assert mock_click.call_count == 1  # gate step was NOT executed

    @patch(_PATCH.format("do_navigate"))
    def test_irreversible_step_executes_with_auto_confirm(self, mock_nav):
        steps = [
            Step(index=0, action="click", locator=_LOC, reversible=False),
        ]
        cap = _cap(steps)
        with patch(_PATCH.format("do_click")) as mock_click:
            result = run_replay(cap, {}, _mock_page("http://localhost:5001/member/"), auto_confirm=True)
        assert result.status == "success"
        mock_click.assert_called_once()

    @patch(_PATCH.format("do_navigate"))
    def test_outputs_before_gate_are_preserved(self, mock_nav):
        with patch(_PATCH.format("do_read"), return_value="$1,000.00"):
            steps = [
                Step(
                    index=0, action="read", locator=_LOC,
                    output_key="bal", reversible=True,
                ),
                Step(index=1, action="click", locator=_LOC, reversible=False),
            ]
            cap = _cap(
                steps,
                outputs={"bal": OutputSpec(type="str", description="balance")},
            )
            result = run_replay(cap, {}, _mock_page(), auto_confirm=False)
        assert result.status == "hard_failure"
        assert result.failure_step_index == 1
        assert result.outputs == {"bal": "$1,000.00"}

    @patch(_PATCH.format("do_navigate"))
    def test_first_step_irreversible_is_blocked_before_execution(self, mock_nav):
        steps = [Step(index=0, action="click", locator=_LOC, reversible=False)]
        cap = _cap(steps)
        with patch(_PATCH.format("do_click")) as mock_click:
            result = run_replay(cap, {}, _mock_page(), auto_confirm=False)
        assert result.status == "hard_failure"
        mock_click.assert_not_called()
