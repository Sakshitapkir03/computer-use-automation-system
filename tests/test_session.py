"""
Unit tests for escalation/session.py.

All tests use a MagicMock Page so no real browser is needed.  Threading
tests use polling on session.control rather than arbitrary sleeps —
Queue guarantees that items put() before get() is called are still
delivered in order, so there is no race between the test thread and the
agent thread as long as the agent thread has been started.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, call, patch

import pytest

from escalation.session import Control, InterventionPayload, SessionController


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_page(url: str = "http://localhost:5001/search") -> MagicMock:
    page = MagicMock()
    page.screenshot.return_value = b"\x89PNG\r\n\x1a\n"  # minimal PNG header bytes
    page.url = url
    # locator().first.fill() / .click() return MagicMock — no exception raised
    return page


def _make_payload(**kwargs) -> InterventionPayload:
    defaults = dict(
        capability_name="lookup_member_balance",
        goal="Read savings balance.",
        step_index=1,
        reason="Step 1 is reversible=False.",
    )
    defaults.update(kwargs)
    return InterventionPayload(**defaults)


def _wait_human(session: SessionController, timeout: float = 2.0) -> None:
    """Poll until control transitions to HUMAN or timeout expires."""
    deadline = time.monotonic() + timeout
    while session.control != Control.HUMAN:
        if time.monotonic() > deadline:
            raise TimeoutError("session did not transition to HUMAN control")
        time.sleep(0.005)


def _start_pause(session: SessionController, payload: InterventionPayload) -> threading.Thread:
    """Start pause() on a daemon thread and return it."""
    t = threading.Thread(target=session.pause, args=(payload,), daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Control state transitions
# ---------------------------------------------------------------------------

class TestControlTransitions:
    def test_initial_control_is_agent(self):
        session = SessionController(_make_page())
        assert session.control == Control.AGENT

    def test_control_becomes_human_during_pause(self):
        session = SessionController(_make_page())
        t = _start_pause(session, _make_payload())
        _wait_human(session)
        assert session.control == Control.HUMAN
        session.request_resume()
        t.join(timeout=2)

    def test_control_returns_to_agent_after_resume(self):
        session = SessionController(_make_page())
        t = _start_pause(session, _make_payload())
        _wait_human(session)
        session.request_resume()
        t.join(timeout=2)
        assert session.control == Control.AGENT

    def test_pause_stores_payload(self):
        session = SessionController(_make_page())
        payload = _make_payload(step_index=3, reason="irreversible gate")
        t = _start_pause(session, payload)
        _wait_human(session)
        assert session.payload is payload
        assert session.payload.step_index == 3
        session.request_resume()
        t.join(timeout=2)


# ---------------------------------------------------------------------------
# pause() blocks until resume
# ---------------------------------------------------------------------------

class TestPauseBlocks:
    def test_pause_blocks_until_resume_is_called(self):
        session = SessionController(_make_page())
        finished = threading.Event()

        def agent():
            session.pause(_make_payload())
            finished.set()

        t = threading.Thread(target=agent, daemon=True)
        t.start()
        _wait_human(session)

        # pause() has not yet returned — agent is still blocked.
        assert not finished.is_set()

        session.request_resume()
        finished.wait(timeout=2)
        assert finished.is_set()
        t.join(timeout=2)

    def test_pause_returns_after_resume_not_before(self):
        """resume() is the only exit from the pause loop."""
        session = SessionController(_make_page())
        returned_at: list[float] = []

        def agent():
            session.pause(_make_payload())
            returned_at.append(time.monotonic())

        t = threading.Thread(target=agent, daemon=True)
        t.start()
        _wait_human(session)

        before_resume = time.monotonic()
        time.sleep(0.05)
        session.request_resume()
        t.join(timeout=2)

        # pause() must have returned after resume was called.
        assert returned_at and returned_at[0] >= before_resume


# ---------------------------------------------------------------------------
# screenshot captured into memory — never written to disk
# ---------------------------------------------------------------------------

class TestScreenshot:
    def test_screenshot_bytes_populated_during_pause(self):
        page = _make_page()
        page.screenshot.return_value = b"PNG_BYTES"
        session = SessionController(page)
        t = _start_pause(session, _make_payload())
        _wait_human(session)
        assert session.screenshot_bytes == b"PNG_BYTES"
        session.request_resume()
        t.join(timeout=2)

    def test_screenshot_called_without_path_argument(self):
        """page.screenshot() must never receive a path= kwarg."""
        page = _make_page()
        session = SessionController(page)
        t = _start_pause(session, _make_payload())
        _wait_human(session)
        session.request_resume()
        t.join(timeout=2)
        # Called exactly once, with no positional args and no path= kwarg.
        page.screenshot.assert_called_once_with()

    def test_screenshot_failure_does_not_abort_pause(self):
        """A screenshot error must not prevent pause from working."""
        page = _make_page()
        page.screenshot.side_effect = RuntimeError("renderer crash")
        session = SessionController(page)
        finished = threading.Event()

        def agent():
            session.pause(_make_payload())
            finished.set()

        t = threading.Thread(target=agent, daemon=True)
        t.start()
        _wait_human(session)
        assert session.screenshot_bytes is None  # captured as None, not raised
        session.request_resume()
        finished.wait(timeout=2)
        assert finished.is_set()
        t.join(timeout=2)


# ---------------------------------------------------------------------------
# submit_action() — executed on the agent thread, result returned to caller
# ---------------------------------------------------------------------------

class TestSubmitAction:
    def _run_with_action(self, action: dict, page: MagicMock | None = None):
        """Helper: pause, submit one action, resume. Returns (session, result)."""
        if page is None:
            page = _make_page()
        session = SessionController(page)
        result_holder: list[dict] = []

        t = _start_pause(session, _make_payload())
        _wait_human(session)

        result_holder.append(session.submit_action(action))
        session.request_resume()
        t.join(timeout=2)
        return session, result_holder[0]

    def test_navigate_calls_page_goto(self):
        page = _make_page()
        _, result = self._run_with_action(
            {"type": "navigate", "url": "http://localhost:5001/search"}, page
        )
        assert result["ok"] is True
        page.goto.assert_called_once_with(
            "http://localhost:5001/search",
            wait_until="domcontentloaded",
            timeout=10_000,
        )

    def test_navigate_returns_current_url(self):
        page = _make_page(url="http://localhost:5001/search")
        _, result = self._run_with_action(
            {"type": "navigate", "url": "http://localhost:5001/search"}, page
        )
        assert result["url"] == "http://localhost:5001/search"

    def test_click_calls_page_locator(self):
        page = _make_page()
        _, result = self._run_with_action(
            {"type": "click", "selector": "button[type='submit']"}, page
        )
        assert result["ok"] is True
        page.locator.assert_called_once_with("button[type='submit']")

    def test_type_calls_fill_on_locator(self):
        page = _make_page()
        _, result = self._run_with_action(
            {"type": "type", "selector": "input[name='member_id']", "value": "12345"}, page
        )
        assert result["ok"] is True
        page.locator.assert_called_once_with("input[name='member_id']")
        page.locator.return_value.first.fill.assert_called_once_with("12345", timeout=10_000)

    def test_unknown_action_type_returns_error(self):
        _, result = self._run_with_action({"type": "double_click", "selector": "td"})
        assert result["ok"] is False
        assert "unknown" in result["error"]

    def test_page_exception_returns_error_dict(self):
        page = _make_page()
        page.goto.side_effect = RuntimeError("net::ERR_CONNECTION_REFUSED")
        _, result = self._run_with_action(
            {"type": "navigate", "url": "http://does-not-exist"}, page
        )
        assert result["ok"] is False
        assert "ERR_CONNECTION_REFUSED" in result["error"]

    def test_multiple_actions_before_resume(self):
        """submit_action can be called multiple times before resume."""
        page = _make_page()
        session = SessionController(page)
        results: list[dict] = []

        t = _start_pause(session, _make_payload())
        _wait_human(session)

        for url in ("http://localhost:5001/search", "http://localhost:5001/not-found"):
            results.append(session.submit_action({"type": "navigate", "url": url}))

        session.request_resume()
        t.join(timeout=2)

        assert all(r["ok"] for r in results)
        assert page.goto.call_count == 2


# ---------------------------------------------------------------------------
# Evidence trail — performed_by tagging
# ---------------------------------------------------------------------------

class TestEvidenceTrail:
    def test_human_action_tagged_performed_by_human(self):
        session = SessionController(_make_page())
        t = _start_pause(session, _make_payload())
        _wait_human(session)

        session.submit_action({"type": "navigate", "url": "http://localhost:5001/search"})
        session.request_resume()
        t.join(timeout=2)

        assert len(session.evidence) == 1
        assert session.evidence[0]["performed_by"] == "human"

    def test_human_action_records_action_type(self):
        session = SessionController(_make_page())
        t = _start_pause(session, _make_payload())
        _wait_human(session)

        session.submit_action({"type": "navigate", "url": "http://localhost:5001/search"})
        session.request_resume()
        t.join(timeout=2)

        assert session.evidence[0]["action"] == "navigate"

    def test_agent_action_tagged_performed_by_agent(self):
        session = SessionController(_make_page())
        session.log_agent_action("click", {"selector": "button"})
        assert session.evidence[0]["performed_by"] == "agent"

    def test_agent_and_human_actions_distinguishable_in_order(self):
        session = SessionController(_make_page())
        session.log_agent_action("navigate", {"url": "http://localhost:5001/search"})
        session.log_agent_action("type", {"value": "12345"})

        t = _start_pause(session, _make_payload())
        _wait_human(session)
        session.submit_action({"type": "navigate", "url": "http://localhost:5001/search"})
        session.request_resume()
        t.join(timeout=2)

        session.log_agent_action("click", {"selector": "button"})

        tags = [e["performed_by"] for e in session.evidence]
        assert tags == ["agent", "agent", "human", "agent"]

    def test_no_evidence_before_any_action(self):
        session = SessionController(_make_page())
        assert session.evidence == []

    def test_resume_without_action_leaves_no_evidence(self):
        session = SessionController(_make_page())
        t = _start_pause(session, _make_payload())
        _wait_human(session)
        session.request_resume()
        t.join(timeout=2)
        assert session.evidence == []

    def test_evidence_property_returns_copy(self):
        """Mutating the returned list must not affect internal state."""
        session = SessionController(_make_page())
        session.log_agent_action("navigate", {})
        copy = session.evidence
        copy.clear()
        assert len(session.evidence) == 1
