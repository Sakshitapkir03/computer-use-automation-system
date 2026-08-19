"""
SessionController — the shared session object that lets agent and human
operate against the same live Playwright Page without handing over the
browser context.

Thread model:
  The Playwright sync API is NOT thread-safe.  Every Page call MUST happen
  on the thread that called sync_playwright() (the "agent thread").  The
  operator console runs on a separate Flask thread.  When a human submits
  an action through the console, the Flask thread puts it into _action_queue;
  the agent thread picks it up inside pause() and executes it on the Page,
  then puts the result into _result_queue for Flask to return to the caller.
  The sentinel value None in _action_queue signals resume.

Control states:
  AGENT — automated steps are executing; console is read-only.
  HUMAN — automation is paused; console may submit actions and resume.

Screenshot policy (Phase 6):
  Screenshots are captured into memory only (page.screenshot() returns bytes;
  no path= argument is passed so nothing is written to disk).  The in-memory
  bytes are held on the SessionController and served by the operator console
  for the duration of the pause.  They are discarded when the session
  controller object is garbage-collected.

  Phase 7 may decide to persist selected screenshots to disk, but that is an
  explicit decision made at that phase.  Phase 6 makes no disk writes at all.

  Note: redact() is text-only and cannot sanitize image content.  If
  screenshots are ever persisted, that must be treated as an accepted,
  documented risk (see REPORT.md) — not something that can be silently
  handled by the existing redaction pipeline.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from playwright.sync_api import Page


class Control(Enum):
    AGENT = "agent"
    HUMAN = "human"


@dataclass
class InterventionPayload:
    """
    Structured description of why automation paused.

    Does not carry a screenshot path or bytes — screenshots are held
    separately on SessionController so they never appear in serialised
    payloads or evidence JSON.
    """

    capability_name: str
    goal: str
    step_index: int | None
    reason: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SessionController:
    """
    Wraps a live Playwright Page and coordinates agent↔human hand-off.

    Usage (agent thread):
        session = SessionController(page)
        session.pause(payload)   # blocks here until operator resumes
        # ... continues on same Page object after resume

    Usage (Flask thread, via _action_queue/_result_queue):
        session.submit_action({"type": "navigate", "url": "..."})
        session.request_resume()
    """

    def __init__(self, page: Page) -> None:
        self._page = page
        self._control = Control.AGENT
        self._payload: InterventionPayload | None = None
        self._lock = threading.Lock()

        # Screenshot bytes captured at pause time — in memory only, never written to disk.
        self._screenshot_bytes: bytes | None = None

        # Queue pair for thread-safe Playwright access from Flask.
        self._action_queue: queue.Queue[dict | None] = queue.Queue()
        self._result_queue: queue.Queue[dict] = queue.Queue()

        # Shared evidence trail — all agent and human actions in order.
        self._evidence: list[dict] = []

    # ------------------------------------------------------------------
    # Public read properties
    # ------------------------------------------------------------------

    @property
    def page(self) -> Page:
        return self._page

    @property
    def control(self) -> Control:
        return self._control

    @property
    def payload(self) -> InterventionPayload | None:
        return self._payload

    @property
    def screenshot_bytes(self) -> bytes | None:
        """In-memory screenshot from the most recent pause. Never written to disk."""
        return self._screenshot_bytes

    @property
    def evidence(self) -> list[dict]:
        return list(self._evidence)

    # ------------------------------------------------------------------
    # Agent-thread interface
    # ------------------------------------------------------------------

    def pause(self, payload: InterventionPayload) -> None:
        """
        Switch to HUMAN control and block the agent thread until resume.

        Captures a screenshot into memory (no path= → no disk write), stores
        the payload, then enters a queue-drain loop: each dict popped from
        _action_queue is executed on the Page (agent thread owns the Page)
        and the result pushed to _result_queue.  A None sentinel breaks the
        loop and control returns to AGENT.

        This method must be called from the agent thread.
        """
        with self._lock:
            self._control = Control.HUMAN
            self._payload = payload

        # Capture page state into memory — no path argument, no disk write.
        try:
            self._screenshot_bytes = self._page.screenshot()
        except Exception:
            self._screenshot_bytes = None

        # Drain human actions until resume sentinel.
        while True:
            item = self._action_queue.get()
            if item is None:
                break
            result = self._execute_page_action(item)
            self._evidence.append({
                "performed_by": "human",
                "action": item.get("type"),
                "detail": {k: v for k, v in item.items() if k != "type"},
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            self._result_queue.put(result)

        with self._lock:
            self._control = Control.AGENT

    def log_agent_action(self, action: str, detail: dict[str, Any] | None = None) -> None:
        """Record an automated action in the evidence trail."""
        self._evidence.append({
            "performed_by": "agent",
            "action": action,
            "detail": detail or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # ------------------------------------------------------------------
    # Flask-thread interface (thread-safe, no Playwright calls here)
    # ------------------------------------------------------------------

    def submit_action(self, action: dict) -> dict:
        """
        Enqueue a human action and block until the agent thread executes it.

        Must only be called while control == HUMAN (i.e. inside pause()).
        Times out after 15 s and returns an error dict if the agent thread
        is unresponsive.
        """
        self._action_queue.put(action)
        try:
            return self._result_queue.get(timeout=15)
        except queue.Empty:
            return {"ok": False, "error": "agent thread did not respond within 15 s"}

    def request_resume(self) -> None:
        """Signal the agent thread to exit the pause loop."""
        self._action_queue.put(None)

    # ------------------------------------------------------------------
    # Internal — called only from the agent thread inside pause()
    # ------------------------------------------------------------------

    def _execute_page_action(self, action: dict) -> dict:
        """
        Execute one human-submitted action on the live Page.

        Supported types:
          navigate  {"type": "navigate", "url": "..."}
          click     {"type": "click",    "selector": "..."}
          type      {"type": "type",     "selector": "...", "value": "..."}

        Returns a result dict with ok: bool and an error key on failure.
        """
        try:
            kind = action.get("type")
            if kind == "navigate":
                self._page.goto(action["url"], wait_until="domcontentloaded", timeout=10_000)
                return {"ok": True, "url": self._page.url}
            if kind == "click":
                self._page.locator(action["selector"]).first.click(timeout=10_000)
                return {"ok": True, "url": self._page.url}
            if kind == "type":
                self._page.locator(action["selector"]).first.fill(
                    action.get("value", ""), timeout=10_000
                )
                return {"ok": True, "url": self._page.url}
            return {"ok": False, "error": f"unknown action type: {kind!r}"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
