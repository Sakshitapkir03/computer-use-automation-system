"""
Shared Playwright action primitives.

Import boundary (enforced by convention, checked in Phase 7):
  - MAY be imported by: agent/discovery.py, replay/executor.py
  - MUST NOT import from: agent/discovery.py, replay/executor.py

All public functions accept artifact.schema.Locator, never Playwright's
internal Locator type. This keeps the artifact schema surface-agnostic —
the same Capability JSON can drive a web browser today and a native desktop
accessibility tree tomorrow without changing the schema.

Locator resolution order:
  1. Primary strategy (role_name → get_by_role, aria_label → get_by_label,
     text_exact → get_by_text, css_fallback → locator)
  2. Walk .fallback chain until a strategy matches or all are exhausted.

Rationale for accessibility-tree-first:
  Role/name targeting survives CSS and layout rewrites, and maps directly
  to how native desktop apps expose interactive elements (AT-SPI, UIA).
  CSS fallback via attribute selectors (not positional indices) handles
  legacy inputs with no computable accessible name — e.g. <input name="x">
  inside a <td> with no associated <label>.
"""
from __future__ import annotations

import time

from playwright.sync_api import Locator as PWLocator
from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PWTimeout

from artifact.schema import Locator as CapLocator
from guardrails.allowlist import AllowlistDenied  # noqa: F401 — re-exported for callers
from guardrails.allowlist import check_allowlist

# Per-strategy probe when walking the fallback chain. Short enough that a
# multi-strategy chain doesn't stall on missing elements, long enough to
# survive a single JS render cycle on a dynamic page.
_PROBE_MS: int = 2_000

# Aggregate cap across the entire fallback chain. Without this, an N-deep chain
# could burn N × _PROBE_MS before raising, quietly eating into the discovery
# loop's wall-clock budget in a way that's invisible to max-step/timeout logic.
# Set to 2 × _PROBE_MS: enough headroom for a primary timeout + a fallback
# that needs its full probe, while bounding worst-case resolution to 4 s total.
_RESOLVE_TOTAL_MS: int = _PROBE_MS * 2

# Timeout for the actual interaction once a locator is resolved.
_ACTION_MS: int = 10_000


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------

class ActionError(RuntimeError):
    """Base for all action-layer errors."""


class LocatorResolutionError(ActionError):
    """Every strategy in the Locator fallback chain failed to match a live element."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_pw_locator(page: Page, loc: CapLocator) -> PWLocator:
    if loc.strategy == "role_name":
        # Omit name= when value is empty string — empty-name role_name is a
        # last-resort fallback for elements with no accessible name, and
        # passing name="" would filter to only nameless elements, which is
        # correct but only safe when used as a fallback after a specific
        # primary has already failed.
        kwargs: dict = {"name": loc.value} if loc.value else {}
        return page.get_by_role(loc.role, **kwargs)  # type: ignore[arg-type]
    if loc.strategy == "aria_label":
        return page.get_by_label(loc.value)
    if loc.strategy == "text_exact":
        return page.get_by_text(loc.value, exact=True)
    if loc.strategy == "css_fallback":
        return page.locator(loc.value)
    if loc.strategy == "xpath":
        return page.locator(f"xpath={loc.value}")
    raise ActionError(f"Unknown locator strategy: {loc.strategy!r}")


def resolve_locator(
    page: Page,
    loc: CapLocator,
    probe_ms: int = _PROBE_MS,
    total_ms: int = _RESOLVE_TOTAL_MS,
) -> PWLocator:
    """
    Walk the Locator fallback chain in order; return the first Playwright
    Locator whose .first element is attached to the DOM.

    Timing contract (two independent knobs):
      probe_ms  — max wait per strategy. Prevents a single absent element from
                  blocking the whole chain.
      total_ms  — aggregate deadline across ALL strategies in the chain.
                  Each strategy's actual probe is min(probe_ms, remaining_ms),
                  so a slow-failing primary cannot consume the full budget
                  before later fallbacks even get a chance. Once the deadline
                  is exceeded the loop stops and raises immediately, regardless
                  of how many strategies remain untried.

    Without total_ms an N-deep chain could burn N × probe_ms (e.g. 6 s for a
    3-deep chain at the default probe) before raising, silently eating into the
    discovery loop's wall-clock budget in a way that looks like normal latency.

    Uses .first throughout so a strategy that matches multiple elements
    (e.g. bare role_name with no name) resolves deterministically to the
    first match rather than raising an ambiguity error.

    Raises LocatorResolutionError if every strategy is exhausted or the
    aggregate deadline is exceeded before any strategy matches.
    """
    deadline = time.monotonic() + total_ms / 1000.0
    current: CapLocator | None = loc
    tried: list[str] = []
    while current is not None:
        remaining_ms = int((deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0:
            tried.append("<aggregate deadline exceeded>")
            break
        pw_loc = _build_pw_locator(page, current)
        try:
            pw_loc.first.wait_for(
                state="attached", timeout=min(probe_ms, remaining_ms)
            )
            return pw_loc
        except (PWTimeout, Exception):
            tried.append(f"{current.strategy}:{current.value!r}")
            current = current.fallback
    raise LocatorResolutionError(
        f"No locator strategy resolved. Tried (in order): {tried}"
    )


# ---------------------------------------------------------------------------
# Public action functions — one per Step action type
# ---------------------------------------------------------------------------

def do_navigate(page: Page, url: str) -> None:
    check_allowlist(url=url, action_type="navigate")
    page.goto(url, wait_until="domcontentloaded", timeout=_ACTION_MS)


def do_click(page: Page, loc: CapLocator) -> None:
    check_allowlist(url=page.url, action_type="click")
    resolve_locator(page, loc).first.click(timeout=_ACTION_MS)


def do_type(page: Page, loc: CapLocator, value: str) -> None:
    """
    Fill a text input or select an option from a <select> element.

    Both map to the same 'type' Step action at the artifact level. The
    distinction is resolved at runtime by inspecting the element's tag name:
      - <input> / <textarea>: fill(value) — replaces current content
      - <select>: select_option(label=value) — matches by visible text,
        not by the option's value attribute, so artifacts remain readable

    This means the 'value' field in a Step targeting a <select> should be
    the human-readable label (e.g. "Money Market"), not the underlying
    value attribute.
    """
    check_allowlist(url=page.url, action_type="type")
    pw_loc = resolve_locator(page, loc)
    tag: str = pw_loc.first.evaluate("el => el.tagName.toLowerCase()")
    if tag == "select":
        pw_loc.first.select_option(label=value, timeout=_ACTION_MS)
    else:
        pw_loc.first.fill(value, timeout=_ACTION_MS)


def do_read(page: Page, loc: CapLocator) -> str:
    """Return the stripped visible text content of the matched element."""
    check_allowlist(url=page.url, action_type="read")
    pw_loc = resolve_locator(page, loc)
    return (pw_loc.first.text_content(timeout=_ACTION_MS) or "").strip()


def do_wait_for(
    page: Page,
    loc: CapLocator,
    timeout_ms: int = _ACTION_MS,
) -> None:
    """Block until the element is visible (not just attached)."""
    check_allowlist(url=page.url, action_type="wait_for")
    resolve_locator(page, loc).first.wait_for(state="visible", timeout=timeout_ms)
