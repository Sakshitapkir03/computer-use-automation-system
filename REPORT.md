# Design Decisions and Accepted Risks

---

## 1. Architecture

The system separates three concerns that must never bleed into each other:

**Discovery** (`agent/discovery.py`) — a single-use, LLM-driven loop that
drives a real browser, observes page state via ARIA snapshots
(`agent/perception.py`), and asks the model what to do next.  The loop
exits when the model signals `goal_complete` or `max_steps` is exhausted.
Its only output is a `DiscoveryResult` carrying an in-memory `Capability`
object plus a list of evidence dicts.  Discovery writes nothing to disk
directly; the caller (test script or orchestrator) decides what to persist.

**Replay** (`replay/executor.py`) — a deterministic, LLM-free executor
that reads a saved `Capability` JSON and walks its `steps` list.  Every
targeting decision is already encoded in the artifact.  The executor
substitutes `{param_name}` templates, fires Playwright actions via
`agent/actions.py` primitives, and verifies the declared checkpoint.  No
model call occurs during replay.

**Guardrails** (`guardrails/`) — enforced at the `agent/actions.py`
layer, below both discovery and replay.  The URL allowlist and redaction
pipeline are not bypassed by either path.

```
┌─────────────────────────────────────────────┐
│  Discovery (LLM loop)                       │
│  ┌──────────────┐  ┌──────────────────────┐ │
│  │ llm_client   │  │ perception (ARIA snap)│ │
│  └──────┬───────┘  └────────────┬─────────┘ │
│         └──────────┬────────────┘           │
│               actions.py  ← allowlist gate  │
└───────────────────────────────────────────-─┘
         │ DiscoveryResult (in-memory Capability)
         ▼
   artifact/store.py  → JSON on disk

┌─────────────────────────────────────────────┐
│  Replay (deterministic)                     │
│  replay/executor.py                         │
│    └── actions.py  ← allowlist gate         │
└─────────────────────────────────────────────┘
         │ ReplayResult  +  EvidenceWriter → JSONL
         ▼
   evidence/<run_id>/steps.jsonl
```

The escalation layer (`escalation/session.py`, `escalation/operator_console.py`)
sits between the executor and the human operator.  It uses a two-queue design
so the Flask thread never calls Playwright directly; all page interactions
during a pause are routed back to the agent thread via `_action_queue` /
`_result_queue`.

---

## 2. Artifact schema

A `Capability` is a self-contained, versioned JSON document that encodes
everything replay needs without re-running discovery.

Key fields:

| Field               | Type                 | Purpose |
|--------------------|----------------------|---------|
| `id`               | `str`                | Unique run ID from the discovery that created this artifact |
| `version`          | `int`                | Integer monotonic; bump on breaking step-list changes |
| `name`             | `str`                | Stable key used to load the artifact (e.g. `lookup_member_balance`) |
| `goal`             | `str`                | Human-readable description of what the capability does |
| `target.base_url`  | `str`                | Starting URL; executor navigates here before step 0 |
| `inputs`           | `dict[str, ParamSpec]` | Named inputs; `sensitive=True` → values collected for redaction |
| `outputs`          | `dict[str, OutputSpec]` | Named outputs; `sensitive=True` → values collected for redaction |
| `steps`            | `list[Step]`         | Ordered action list; each step carries a `Locator` with fallback chain |
| `checkpoint`       | `Checkpoint`         | Primary success condition (url_matches / element_visible / text_present) |
| `business_outcomes` | `list[BusinessOutcomeSpec]` | Named non-success states with their own checkpoint signals |

**Locator fallback chain** — each `Locator` has an optional `fallback:
Locator` field, forming a linked list.  `resolve_locator()` in
`agent/actions.py` walks the chain in order (role_name → css_fallback →
xpath) and returns the first strategy whose selector resolves to at least
one visible element.  This allows one artifact to work across minor DOM
variations without re-recording.

**`reversible` flag** — each `Step` carries `reversible: bool = True`.
Steps marked `reversible=False` are gated by the executor: if
`auto_confirm=False`, the run pauses for human confirmation before
executing the step.  The flag is set by the LLM during discovery (it is
part of the structured output schema) and can be corrected by editing the
artifact JSON before replay.

**`OutputSpec.sensitive`** — if `True`, the runtime value of that output
is added to the redaction list so it is scrubbed from every JSONL record
before the line hits disk.  `ParamSpec.sensitive` works identically for
input values.

---

## 3. Determinism and error handling

**Discovery is non-deterministic** by design — the LLM may produce
different step sequences across runs, especially on novel pages.  The
artifact it produces is deterministic for all subsequent replays.

**Replay is fully deterministic** given the same artifact and the same
application state.  No model call is made.  `{param_name}` substitution
is a simple string replace.

**Three-status taxonomy** (enforced, not best-effort):

| Status | Trigger |
|--------|---------|
| `success` | All steps completed without exception AND the primary checkpoint passed |
| `business_outcome` | A `BusinessOutcomeSpec` checkpoint is positively matched — either after a step exception or after the primary checkpoint fails.  Never returned speculatively. |
| `hard_failure` | Any other outcome: unrecognised exception, primary checkpoint failed with no matching business outcome, allowlist denial.  This is the explicit default. |

`business_outcome` is only returned on a **positive signal** (the declared
checkpoint matched).  An empty `business_outcomes` list means any failure
becomes `hard_failure`.  This prevents false negatives where the system
guesses "member not found" from an ambiguous page state.

**Step exceptions and outcome ordering** — when a step raises, the
executor immediately tests business outcomes in declaration order before
returning `hard_failure`.  This catches the common pattern where an
application redirects to a known error page mid-flow, breaking a
subsequent locator resolution.

---

## 4. Heterogeneity and multi-tenant

The current system is single-capability and single-tenant.  The design
choices that make it extensible without a rewrite:

**Capability as a file** — `artifact/store.py` resolves capability names
to `artifact/store/<name>_v<N>.json`.  Adding a new capability means
running discovery once and dropping a new file.  No code change.

**`target.base_url` per artifact** — each capability encodes its own
starting URL.  Replaying `lookup_member_balance` against a staging
environment requires only changing `base_url` in the JSON; the executor
reads it at runtime.

**`inputs` dict, not positional args** — callers pass `params: dict[str,
str]` to `run_replay`.  A multi-capability orchestrator can call multiple
capabilities with their own param dicts without any shared state.

**Multi-tenant gap** — there is no per-tenant credential management or
session isolation.  All replays run in the same browser instance in the
current demo harness.  A production multi-tenant deployment would need to:
(a) provision a fresh browser context per tenant, (b) store credentials
separately from capability artifacts, and (c) scope evidence directories by
tenant to prevent cross-tenant evidence mixing.

---

## 5. Escalation and handoff

The escalation mechanism is designed around two invariants:

1. **All Playwright calls must happen on the thread that created the
   browser context.**  Flask runs on a different thread; it never touches
   `page` directly.

2. **The agent thread must not be blocked by I/O or network calls during
   a pause** — it is the only thread that can drive the browser.

These are satisfied by the two-queue design in `escalation/session.py`:

```
Flask thread:           submit_action(action_dict)
                            │  enqueue action
                            ▼
                    _action_queue (Queue)
                            │  dequeue
                            ▼
Agent thread (pause loop):  _execute_page_action()
                            │  enqueue result
                            ▼
                    _result_queue (Queue)
                            │  dequeue (blocking)
                            ▼
Flask thread:           returns result dict to caller
```

`session.pause(payload)` blocks the agent thread in a `while True` loop,
dequeuing and executing actions.  The loop exits only when a `None`
sentinel (the resume signal) is dequeued.  `request_resume()` enqueues
`None`.

**Evidence continuity** — during a pause, every human action is recorded
in `session.evidence` tagged `performed_by: human`.  When pause returns,
`run_replay` iterates `session.evidence` and writes each entry to the
JSONL file before appending the `resume` event.  The resulting JSONL
trail is a single interleaved sequence of agent and human events in
execution order, indistinguishable from the surrounding agent events
except for the `performed_by` field.

**Screenshot policy** — `page.screenshot()` is called with no `path=`
argument so Playwright returns bytes without writing to disk.  Bytes are
held in `SessionController._screenshot_bytes` for the duration of the
pause and served in-memory by `GET /screenshot` on the operator console.
They are discarded when the `SessionController` object is garbage-collected.
See Safety § Screenshot redaction for the accepted risk this carries.

---

## 6. Safety

### URL allowlist

`guardrails/allowlist.py` enforces an explicit list of allowed URL
prefixes.  `do_navigate` in `agent/actions.py` checks every navigation
target before calling `page.goto`.  A denied URL raises `AllowlistDenied`,
which the replay executor treats as `hard_failure` (not a business
outcome), preventing silent bypasses.  During discovery the LLM may
propose any action, but navigation is gated at the action layer; the LLM
cannot escape the allowlist by generating a step that calls `do_navigate`
with an unlisted host.

### Redaction pipeline

`guardrails/redaction.py` applies three layers in sequence before any text
reaches disk:

1. **Explicit sensitive values** — collected via `collect_sensitive_values()`
   from `ParamSpec.sensitive` and `OutputSpec.sensitive` fields.  Values
   marked sensitive are replaced with `[REDACTED]` as whole-string literals
   (not regex), so they are always found regardless of surrounding context.

2. **SSN pattern** — `\b\d{3}-\d{2}-\d{4}\b`

3. **Grouped-digit pattern** — `\b\d{4}[- ]\d{4}(?:[- ]\d{2,})+\b` —
   catches formatted account and card numbers (e.g. `5432-1098-7654`).
   ISO dates (`YYYY-MM-DD`) are intentionally excluded: the middle group
   is two digits (not four), so dates do not match this pattern.

4. **Long-digit pattern** — `\b\d{9,}\b` — catches unformatted 9+-digit
   sequences (account numbers, routing numbers, Social Security numbers
   stored without dashes).

`redact()` is applied inside `EvidenceWriter.log()` and inside
`operator_console._safe()`.  It is never applied to the in-memory
`ReplayResult` or `outputs` dict — callers receive real values; only disk
writes are scrubbed.

### Irreversible step gate

Steps marked `reversible=False` require `auto_confirm=True` or a human
resume signal before execution.  Reaching such a step without either stops
the run and returns `hard_failure` without executing the step.  This
prevents accidental form submissions, deletions, or fund transfers during
an automated run that has diverged from expected state.

### Screenshot redaction — accepted risk

`redact()` operates on text only.  A screenshot captured at the moment of
an escalation pause may contain PII (member name, balance, account number,
session tokens) rendered as image pixels.  This is currently unmitigated
for the following reasons:

- Screenshots are never written to disk in Phase 6.  They exist only in
  `SessionController._screenshot_bytes` and are served in-memory to the
  operator console for the duration of the pause.

- Any future decision to persist screenshots must be made explicitly.
  Acceptable mitigations would include Playwright `mask=` for known
  sensitive selectors (requires the capability schema to declare which
  locators contain PII — a future extension), or storing screenshots in
  an access-controlled location separate from text evidence with its own
  retention policy.

- This risk is treated as equivalent to a human operator looking directly
  at the browser window — which is the intent of the escalation feature.

---

## 7. Cuts (documented gaps)

These are known limitations accepted for the take-home scope.  Each would
need addressing before production use.

### 7a. Screenshot redaction

Covered in full under Safety § Screenshot redaction.  No automatic PII
scrubbing is possible for image content with the current pipeline.

### 7b. Column-index coupling in the Savings XPath locator

The `read` step targeting the savings balance uses:

```xpath
//tr[td[normalize-space()='Savings']]/td[3]
```

with CSS fallback:

```css
tr:has(td:text-is("Savings")) td:nth-child(3)
```

Both strategies depend on the balance appearing in column 3 of the
accounts table.  If the bank's UI adds or reorders columns (e.g. inserts
a "Currency" column before "Balance"), the locator returns the wrong cell
without raising an error — it silently reads the wrong value.

**Mitigation not yet implemented:** The capability schema has no way to
declare that a `read` step should target "the cell in the same row as the
'Savings' label in the column headed 'Balance'".  A future
`header_relative` locator strategy would resolve the row by label and the
column by a heading text match, making the read robust to column reordering.

Until then, any UI change to the accounts table requires re-running
discovery or manually updating the artifact JSON.

### 7c. Business-outcome specs are per-capability and manually declared

`BusinessOutcomeSpec` entries must be declared by the operator; discovery
does not automatically detect all possible exit states of a workflow.
Non-success paths the LLM did not encounter during its recording run have
no spec and fall through to `hard_failure`.

**`MEMBER_NOT_FOUND` is now baked into the artifact** —
`artifact/store/lookup_member_balance_v1.json` carries the spec in its
`business_outcomes` array (checkpoint: `url_matches /not-found`).
`run_replay_test.py` loads the artifact as-is with no in-memory mutation;
the `business_outcome` classification produced by scenario (b) is driven
entirely by the artifact on disk, not by test-time patching.  This
satisfies the "artifact is self-contained and reusable" premise.

The remaining gap is that the "member exists but has no Savings account"
case (scenario c) still returns `hard_failure` — no `BusinessOutcomeSpec`
has been declared for that state because the page does not navigate to a
distinct URL; it stays on `/member/67890` with the Savings row simply
absent.  A future `element_absent` checkpoint kind (not currently
supported) would allow this to be named.  For now it is an explicit
acknowledged gap: unrecognised failure states default to `hard_failure`.

### 7d. No per-tenant credential management or session isolation

All replays in the demo share a single browser instance and a single
Flask dev server.  Production multi-tenant use requires a fresh browser
context per run (to prevent cookie / local-storage leakage between
tenants), per-tenant credential storage, and scoped evidence directories.
None of this infrastructure exists yet.

### 7e. Discovery model string is not validated at startup

`agent/llm_client.py` accepts any model name string and defers validation
to the Gemini API — an invalid model ID causes the first `generate_content`
call to raise, after the browser has already been launched and the initial
navigation performed.  A production entrypoint should validate the model ID
against the API's model list (or a pinned allowlist) before opening the
browser so that configuration errors are surfaced immediately with a clear
message rather than mid-run after browser startup overhead.
