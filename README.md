# RPA Agent — Core Banking UI Automation

LLM-driven discovery + deterministic replay for browser-based RPA, with
guardrails, escalation, and structured evidence logging.

---

## Repository layout

```
rpa-agent/
├── agent/
│   ├── actions.py          # Playwright action primitives (navigate/click/type/read/wait_for)
│   ├── discovery.py        # LLM-driven capability recorder
│   ├── llm_client.py       # Gemini API wrapper
│   └── perception.py       # Page-state snapshot (aria + url + title)
├── artifact/
│   ├── evidence.py         # JSONL evidence writer with redaction
│   ├── schema.py           # Pydantic v2 models (Capability, Step, ReplayResult, …)
│   ├── store.py            # Artifact persistence + collect_sensitive_values
│   └── store/
│       └── lookup_member_balance_v1.json   # Pre-recorded capability artifact
├── escalation/
│   ├── session.py          # SessionController (pause / resume / submit_action)
│   └── operator_console.py # Flask operator UI served during pause
├── guardrails/
│   ├── allowlist.py        # URL allowlist enforcement
│   └── redaction.py        # redact() — three-layer PII scrubbing
├── replay/
│   └── executor.py         # Deterministic step executor
├── target_app/
│   └── app.py              # Mock core-banking Flask app (test target)
├── tests/
│   ├── test_*.py           # 147 unit tests (no browser required)
│   ├── run_discovery_test.py   # Live discovery demo
│   ├── run_replay_test.py      # Live replay demo (3 scenarios)
│   ├── run_escalation_test.py        # Escalation demo (simulated human)
│   └── run_escalation_interactive.py # Escalation demo (real human, no time limit)
├── evidence/               # JSONL run logs (created at runtime)
└── requirements.txt
```

---

## Setup

**Requirements:** Python ≥ 3.11 (the codebase uses `X | Y` union syntax from
Python 3.10+ in Pydantic models — Python 3.9 will fail at import time).
`python3 --version` must show 3.11 or higher before running the commands below.
On macOS the system `python3` is typically 3.9; use `python3.11`, `python3.12`,
or `python3.13` explicitly.

```bash
# 1. Create and activate a virtual environment (Python 3.11+ required)
python3.13 -m venv .venv           # adjust to python3.11 / python3.12 as available
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install Playwright browser binaries
playwright install chromium

# 4. Set the Gemini API key (required only for discovery)
echo "GEMINI_API_KEY=your_key_here" > .env
```

---

## Start the target app

The mock core-banking app must be running before executing any live test.
It binds to `http://localhost:5001`.

```bash
# In a separate terminal (leave it running)
.venv/bin/python target_app/app.py
```

Test data pre-loaded:

| Member ID | Has Savings? | Balance   |
|-----------|-------------|-----------|
| 12345     | Yes         | $5,432.10 |
| 67890     | No          | —         |
| 99999     | Not found   | —         |

---

## Unit tests (no browser, no API key required)

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: **147 passed**.

---

## CLI entrypoints

The system exposes two general-purpose CLI commands that accept arbitrary
goals and parameters at invocation time, separate from the fixed demo scripts.

### Discovery CLI

```bash
.venv/bin/python -m agent.cli \
    --goal "Search for a member by ID and read their savings account balance." \
    --base-url "http://localhost:5001/search" \
    --capability-name "lookup_member_balance" \
    --params '{"member_id": "12345"}' \
    --model "gemini-3.6-flash"
```

`--param-specs` is optional; if omitted, every key in `--params` is inferred
as `type=str / required=True / sensitive=False`.  Pass an explicit JSON object
to override type, sensitivity, or description for any param.

### Replay CLI

```bash
.venv/bin/python -m replay.cli \
    --capability lookup_member_balance \
    --params '{"member_id": "12345"}' \
    --auto-confirm
```

`--auto-confirm` allows steps marked `reversible=False` to execute without
pausing.  Omit it to stop at any irreversible step and return `hard_failure`.
`--version` selects the artifact version (default: 1).

---

## Demo command sequence

All commands assume the target app is running and `.venv` is activated.

### 1 — Discovery (requires GEMINI_API_KEY)

Records a new capability by having the LLM drive the browser.
Writes the artifact to `artifact/store/lookup_member_balance_v1.json`
and a JSONL evidence file to `evidence/<run_id>/steps.jsonl`.

```bash
.venv/bin/python tests/run_discovery_test.py
```

A pre-recorded artifact is already committed to `artifact/store/` so the
replay and escalation demos work without an API key.

### 2 — Replay (no API key required)

Runs three scenarios against the saved artifact:

```bash
.venv/bin/python tests/run_replay_test.py
```

Expected output:

```
Scenario: (a) Known member with Savings account  (member_id='12345')
  status: success
  outputs: {"savings_balance": "$5432.10"}

Scenario: (b) Non-existent member  (member_id='99999')
  status: business_outcome
  outcome_code: MEMBER_NOT_FOUND

Scenario: (c) Known member without Savings account  (member_id='67890')
  status: hard_failure
  failure_step_index: 2
```

Each scenario writes its evidence to `evidence/replay_<id>/steps.jsonl`.

Inspect a JSONL file:

```bash
cat evidence/replay_<id>/steps.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    print(json.dumps(json.loads(line), indent=2))
    print('---')
"
```

### 3 — Escalation (no API key required)

Starts a replay that pauses at an irreversible step and launches the
Flask operator console at `http://localhost:5002`.

There are two variants:

**Automated verification** — a simulator thread acts as the human operator
and resumes in under a second.  Good for CI or confirming the mechanism
works end-to-end without manual interaction.

```bash
.venv/bin/python tests/run_escalation_test.py
```

**Interactive** — no simulator.  The run blocks indefinitely at the
irreversible gate, waiting for a real person to open the console, inspect
the state, optionally submit manual actions, and click **Resume Agent**.
Use this to actually exercise the UI.

```bash
.venv/bin/python tests/run_escalation_interactive.py
```

#### What the console shows

When the run pauses, open `http://localhost:5002` in a browser.  The page
renders in three parts: at the top, the **intervention payload** — capability
name, goal, step index, and the reason the gate fired (e.g. "Step 0 is marked
reversible=False and requires human confirmation").  Below that, an **embedded
screenshot** of the browser at the exact moment automation paused, so the
operator can see what the page looked like without switching to the headless
browser.  At the bottom, a **manual-action form** (navigate / click / type)
that lets the operator drive the live page from the console, followed by a
**Resume Agent** button that unblocks the agent thread and lets replay
continue from where it stopped.  Scroll down to reach the form and button —
the screenshot can push them below the fold on smaller screens.

#### Console API routes

- `GET  /`           — Renders the intervention page described above
- `GET  /screenshot` — Raw PNG bytes (served in-memory, never written to disk)
- `POST /action`     — JSON `{"type": "navigate"|"click"|"type", ...}`
- `POST /resume`     — Releases the agent thread; returns `{"ok": true}`

---

## Evidence layout

```
evidence/
└── <run_id>/
    └── steps.jsonl     # One JSON object per line; seq is monotonic within run
```

Every record includes `seq`, `run_id`, `phase` (`discovery` or `replay`),
and `timestamp` (ISO-8601 UTC).  All sensitive param and output values are
redacted before any write via `guardrails/redaction.py`.

---

## Environment variables

| Variable        | Required for    | Description                        |
|----------------|-----------------|------------------------------------|
| `GEMINI_API_KEY` | Discovery only | Gemini API key (loaded from `.env`) |
