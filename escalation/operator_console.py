"""
Operator console — MOCK UI (not production-grade; no auth, no CSRF protection).

Displays the current intervention payload and lets a human operator perform
one or more manual actions against the same live Playwright Page before
signalling resume.  Runs as a Flask dev server in a daemon thread so it
shares the process with the agent.

Redaction contract:
  Every string written to the HTML response or to the evidence log passes
  through redact(text, sensitive_values=collect_sensitive_values(...)) so
  that sensitive param/output values never appear in the UI or in logs.

Evidence contract:
  Human actions are appended to session.evidence with performed_by="human"
  by SessionController.pause() after the agent thread executes them.  The
  console itself does not write to evidence — it submits through the session
  so the write always happens on the agent thread in the correct order.
"""
from __future__ import annotations

import json
import threading
from typing import Any

from flask import Flask, jsonify, request

from artifact.schema import Capability
from artifact.store import collect_sensitive_values
from escalation.session import Control, SessionController
from guardrails.redaction import redact

# ---------------------------------------------------------------------------
# Module-level state — set by start_console() before the server starts.
# ---------------------------------------------------------------------------

_session: SessionController | None = None
_capability: Capability | None = None
_params: dict[str, str] = {}
_outputs: dict[str, str] = {}

app = Flask(__name__)
app.config["DEBUG"] = False


# ---------------------------------------------------------------------------
# Redaction helper
# ---------------------------------------------------------------------------

def _safe(text: str) -> str:
    """Redact sensitive values before any string reaches the response."""
    if _capability is None:
        return redact(text)
    return redact(text, sensitive_values=collect_sensitive_values(_capability, _params, _outputs))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# MOCK UI — minimal HTML inline (no templates directory needed for a mock)
_INDEX_TEMPLATE = """<!DOCTYPE html>
<html>
<head><title>Operator Console — MOCK UI</title>
<style>
  body {{ font-family: monospace; max-width: 900px; margin: 2em auto; background: #1a1a1a; color: #ccc; }}
  h1 {{ color: #f90; }} h2 {{ color: #8af; }}
  pre {{ background: #222; padding: 1em; border-radius: 4px; white-space: pre-wrap; }}
  .warn {{ color: #f44; font-weight: bold; }}
  form {{ margin: 1em 0; }}
  input, select {{ background: #333; color: #fff; border: 1px solid #555; padding: 4px 8px; width: 100%; box-sizing: border-box; margin: 4px 0; }}
  button {{ background: #f90; color: #000; border: none; padding: 8px 20px; cursor: pointer; font-weight: bold; margin: 4px 4px 4px 0; }}
  button.resume {{ background: #4c4; color: #000; }}
  .label {{ color: #888; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>&#x26A0; Operator Console <span class="warn">[MOCK UI]</span></h1>
<h2>Intervention Required</h2>
<pre>{payload_json}</pre>
{screenshot_tag}
<h2>Manual Action</h2>
<p class="label">Actions execute against the <strong>live</strong> browser session. All inputs are redacted before display.</p>
<form id="actionForm">
  <select id="actionType" onchange="toggleFields()">
    <option value="navigate">navigate</option>
    <option value="click">click</option>
    <option value="type">type</option>
  </select>
  <input id="url"      placeholder="URL (for navigate)"                   />
  <input id="selector" placeholder="CSS selector (for click / type)"      style="display:none"/>
  <input id="value"    placeholder="Value to type (for type)"             style="display:none"/>
  <button type="button" onclick="submitAction()">Execute</button>
</form>
<pre id="actionResult"></pre>
<h2>Resume Automation</h2>
<form method="post" action="/resume">
  <button class="resume" type="submit">Resume Agent</button>
</form>
<script>
function toggleFields() {{
  var t = document.getElementById('actionType').value;
  document.getElementById('url').style.display      = t==='navigate' ? '' : 'none';
  document.getElementById('selector').style.display = t!=='navigate' ? '' : 'none';
  document.getElementById('value').style.display    = t==='type'     ? '' : 'none';
}}
toggleFields();
async function submitAction() {{
  var t = document.getElementById('actionType').value;
  var body = {{type: t}};
  if (t==='navigate') body.url = document.getElementById('url').value;
  else body.selector = document.getElementById('selector').value;
  if (t==='type') body.value = document.getElementById('value').value;
  var resp = await fetch('/action', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body: JSON.stringify(body)}});
  var data = await resp.json();
  document.getElementById('actionResult').textContent = JSON.stringify(data, null, 2);
}}
</script>
</body>
</html>
"""


@app.route("/")
def index() -> Any:
    if _session is None:
        return "Console not configured.", 503

    payload = _session.payload
    if payload is None:
        return "<pre>No active intervention.</pre>", 200

    # Build a display-safe version of the payload — redact before render.
    safe_payload = {
        "capability_name":    _safe(payload.capability_name),
        "goal":               _safe(payload.goal),
        "step_index":         payload.step_index,
        "reason":             _safe(payload.reason),
        "timestamp":          payload.timestamp,
        "control":            _session.control.value,
        "screenshot_in_memory": _session.screenshot_bytes is not None,
    }
    payload_json = _safe(json.dumps(safe_payload, indent=2))

    screenshot_tag = ""
    if _session.screenshot_bytes is not None:
        screenshot_tag = (
            '<h2>Page State at Pause</h2>'
            '<img src="/screenshot" style="max-width:100%;border:1px solid #555">'
        )

    html = _INDEX_TEMPLATE.format(
        payload_json=payload_json,
        screenshot_tag=screenshot_tag,
    )
    return html, 200


@app.route("/screenshot")
def screenshot() -> Any:
    """Serve the pause screenshot from the in-memory bytes — never from disk."""
    if _session is None:
        return "", 404
    data = _session.screenshot_bytes
    if data is None:
        return "", 404
    return data, 200, {"Content-Type": "image/png"}


@app.route("/payload")
def payload_json_endpoint() -> Any:
    """JSON endpoint for the current intervention payload (for scripted access)."""
    if _session is None or _session.payload is None:
        return jsonify({"error": "no active intervention"}), 404

    p = _session.payload
    safe = {
        "capability_name":    _safe(p.capability_name),
        "goal":               _safe(p.goal),
        "step_index":         p.step_index,
        "reason":             _safe(p.reason),
        "timestamp":          p.timestamp,
        "control":            _session.control.value,
        "screenshot_in_memory": _session.screenshot_bytes is not None,
    }
    return jsonify(safe), 200


@app.route("/action", methods=["POST"])
def action() -> Any:
    """
    Execute a human-submitted action against the live Playwright page.

    The action is forwarded to the agent thread via session.submit_action(),
    which executes it on the Page (the only thread that may touch the Page)
    and returns the result.  The action and its result are logged to the
    session evidence trail by SessionController.pause() on the agent thread.
    """
    if _session is None:
        return jsonify({"ok": False, "error": "session not configured"}), 503
    if _session.control != Control.HUMAN:
        return jsonify({"ok": False, "error": "session is not paused"}), 409

    data = request.get_json(force=True, silent=True) or {}
    result = _session.submit_action(data)
    return jsonify(result), 200


@app.route("/resume", methods=["POST"])
def resume() -> Any:
    """Signal the agent thread to exit the pause loop and continue."""
    if _session is None:
        return jsonify({"ok": False, "error": "session not configured"}), 503
    _session.request_resume()
    return jsonify({"ok": True, "status": "resumed"}), 200


@app.route("/evidence")
def evidence() -> Any:
    """Return the full evidence trail (redacted) as JSON."""
    if _session is None:
        return jsonify([]), 200
    safe_entries = []
    for entry in _session.evidence:
        safe_entry = {
            k: _safe(str(v)) if isinstance(v, str) else v
            for k, v in entry.items()
        }
        safe_entries.append(safe_entry)
    return jsonify(safe_entries), 200


# ---------------------------------------------------------------------------
# Public setup function
# ---------------------------------------------------------------------------

def start_console(
    session: SessionController,
    capability: Capability,
    params: dict[str, str],
    outputs: dict[str, str],
    port: int = 5002,
) -> None:
    """
    Configure the console and start Flask in a daemon background thread.

    Must be called before any session.pause() so the console is ready when
    automation reaches a gate.  The daemon thread dies automatically when
    the main process exits.
    """
    global _session, _capability, _params, _outputs
    _session = session
    _capability = capability
    _params = dict(params)
    _outputs = dict(outputs)

    thread = threading.Thread(
        target=lambda: app.run(
            host="127.0.0.1",
            port=port,
            use_reloader=False,
            threaded=True,
        ),
        daemon=True,
    )
    thread.start()
