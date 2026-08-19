"""
Page observation for the discovery LLM.

Produces three signals from the current browser state:

  1. ax_tree_text — raw YAML-format accessibility tree from aria_snapshot().
     Gives structural context: which roles exist, their hierarchy, their
     computed accessible names (where the browser can compute one).

  2. element_attrs_text — flat JS-scraped list of every interactive element
     (input, select, textarea, button) in document order, with its HTML
     attributes (name, id, type, placeholder, aria-label).
     Gives the css_fallback data the ax tree alone cannot: a bare "textbox"
     entry in the YAML has no name attribute visible in the tree, so the LLM
     cannot derive input[name="initial_deposit"] without this signal.

  3. screenshot_bytes — PNG. Visual context only, never used as a targeting
     mechanism. Replay never receives or uses screenshots.

for_llm() combines (1) and (2) into the single string the discovery LLM
receives per turn. Keeping them as separate fields lets tests and evidence
logging inspect each signal independently without re-parsing the combined text.

Note on API: page.accessibility was removed in Playwright 1.47+.
aria_snapshot() (Playwright >= 1.49) is the replacement.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

from playwright.sync_api import Page

# JS query for interactive elements. Excludes hidden inputs (no user-facing
# role) and includes all form controls plus buttons.
#
# Label resolution order for each element:
#   1. <label for="element-id"> — standard HTML association
#   2. aria-label attribute
#   3. aria-labelledby reference
#   4. First <td>/<th> text in the nearest ancestor <tr> — handles legacy
#      table-based layouts where the label sits in an adjacent cell with no
#      formal association. This is the primary path for the mock target app.
#
# Having the label in element_attrs_text means the LLM cross-references by
# shared context ("Initial Deposit ($)" appears in both the ax tree row name
# and the element attrs label), not by document-order position.
_INTERACTIVE_JS = """
() => {
    function resolveLabel(el) {
        if (el.id) {
            const lbl = document.querySelector('label[for="' + el.id + '"]');
            if (lbl) return lbl.textContent.trim();
        }
        const ariaLabel = el.getAttribute('aria-label');
        if (ariaLabel) return ariaLabel;
        const ariaLabelledBy = el.getAttribute('aria-labelledby');
        if (ariaLabelledBy) {
            const ref = document.getElementById(ariaLabelledBy);
            if (ref) return ref.textContent.trim();
        }
        const row = el.closest('tr');
        if (row) {
            const myCell = el.closest('td, th');
            const cells = row.querySelectorAll('td, th');
            if (cells.length >= 2 && cells[0] !== myCell) {
                const t = cells[0].textContent.trim();
                if (t) return t;
            }
        }
        return null;
    }

    const sel = 'input:not([type=hidden]), select, textarea, button';
    return Array.from(document.querySelectorAll(sel)).map(el => {
        const o = { tag: el.tagName.toLowerCase() };
        for (const k of ['type', 'name', 'id', 'placeholder']) {
            const v = el.getAttribute(k);
            if (v !== null && v !== '') o[k] = v;
        }
        const al = el.getAttribute('aria-label');
        if (al) o['aria_label'] = al;
        const label = resolveLabel(el);
        if (label) o['label'] = label;
        // For buttons/submits with no name or id, surface text content so the
        // LLM can cross-reference with named button entries in the ax tree.
        const tag = el.tagName.toLowerCase();
        if ((tag === 'button' || el.getAttribute('type') === 'submit') &&
            !o['name'] && !o['id']) {
            const t = el.textContent?.trim();
            if (t) o['text'] = t.slice(0, 80);
        }
        return o;
    });
}
"""


@dataclass(frozen=True)
class Observation:
    url: str
    title: str
    ax_tree_text: str        # raw aria_snapshot() YAML — structural context
    element_attrs_text: str  # formatted DOM attributes — css_fallback data
    screenshot_bytes: bytes  # visual context; never used for targeting

    def screenshot_b64(self) -> str:
        """Base64-encoded PNG for LLM vision messages."""
        return base64.standard_b64encode(self.screenshot_bytes).decode()

    def for_llm(self) -> str:
        """
        Full observation text sent to the discovery LLM each turn.

        Two sections in one string so the LLM can cross-reference without
        a second tool call:
          - Accessibility Tree: structure, hierarchy, computed names
          - Interactive Element Attributes: name/id/type for css_fallback

        Example cross-reference: seeing "textbox" inside
        row "Initial Deposit ($)" in the tree, then finding
        input  type="text"  name="initial_deposit" in the attrs section,
        the LLM can emit css_fallback: input[name="initial_deposit"] in
        one turn.
        """
        return (
            "=== Accessibility Tree ===\n"
            + self.ax_tree_text
            + "\n\n=== Interactive Element Attributes ===\n"
            + "# HTML attributes in document order. Use name/id to build\n"
            + "# css_fallback locators. Cross-reference with the tree above.\n"
            + self.element_attrs_text
        )


def capture(page: Page) -> Observation:
    """Snapshot the current page state for the discovery LLM."""
    ax_tree_text: str = page.locator("body").aria_snapshot()
    element_attrs_text: str = _collect_element_attrs(page)
    return Observation(
        url=page.url,
        title=page.title(),
        ax_tree_text=ax_tree_text,
        element_attrs_text=element_attrs_text,
        screenshot_bytes=page.screenshot(type="png"),
    )


def _collect_element_attrs(page: Page) -> str:
    """
    Query DOM for all interactive elements and format their attributes.

    Each line includes the element's resolved label so the LLM can
    cross-reference by shared context (the same "Initial Deposit ($)" text
    appears in the ax tree's row name AND in the label= field here) rather
    than by document-order position.
    """
    elements: list[dict] = page.evaluate(_INTERACTIVE_JS)
    lines: list[str] = []
    for el in elements:
        parts: list[str] = [el.get("tag", "?")]
        for key in ("type", "name", "id", "placeholder", "aria_label"):
            if val := el.get(key):
                parts.append(f'{key}="{val}"')
        if text := el.get("text"):
            parts.append(f'text="{text}"')
        if label := el.get("label"):
            parts.append(f'label="{label}"')
        lines.append("  ".join(parts))
    return "\n".join(lines)
