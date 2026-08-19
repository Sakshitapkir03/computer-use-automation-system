"""
JSONL evidence writer.

All writes pass through redact() so sensitive param and output values
never reach disk in plain text.  One EvidenceWriter covers one run
(discovery or replay).  Each call to log() appends one JSON object (one
line) to the run's steps.jsonl file.

Evidence directory layout:
  evidence/<run_id>/steps.jsonl

Every JSONL record has at minimum:
  seq       — monotonically increasing counter within this run
  run_id    — ties the record to a specific run
  phase     — "discovery" or "replay"
  timestamp — ISO-8601 UTC

The writer is intentionally separate from artifact/store.py so callers
can compose it independently: a discovery run writes its evidence list
after the fact; a replay run writes each step as it executes.

Redaction is applied at write time so records logged before configure()
still get structural PII treatment (SSN / long-digit patterns).  Sensitive
param and output values are added once configure() is called.  For replay
the capability is always known upfront; for discovery configure() is called
after goal_complete assembles the capability.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from artifact.store import collect_sensitive_values
from guardrails.redaction import redact

EVIDENCE_ROOT = Path(__file__).parent.parent / "evidence"


class EvidenceWriter:
    """
    Append-mode JSONL writer for a single discovery or replay run.

    Typical usage — replay:
        writer = EvidenceWriter(run_id, "replay")
        writer.configure(capability, params, outputs)   # outputs is mutated in place
        # ...
        writer.log({"event": "step", "action": "click", ...})

    Typical usage — discovery (batch at end):
        result = run_discovery(...)
        writer = EvidenceWriter(result.run_id, "discovery")
        writer.configure(result.capability, result.params, result.outputs)
        writer.log_all(result.evidence)
    """

    def __init__(self, run_id: str, phase: str) -> None:
        self._run_id = run_id
        self._phase = phase
        self._path = EVIDENCE_ROOT / run_id / "steps.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._capability = None
        self._params: dict[str, str] = {}
        self._outputs: dict[str, str] = {}  # live mutable reference — updated by caller

    @property
    def path(self) -> Path:
        return self._path

    def configure(
        self,
        capability,
        params: dict[str, str],
        outputs: dict[str, str],
    ) -> None:
        """
        Supply the Capability and runtime dicts for sensitive-value redaction.

        outputs must be the same dict object that the caller updates in place
        as read steps complete — the writer always sees the current values.
        """
        self._capability = capability
        self._params = params
        self._outputs = outputs

    def log(self, event: dict) -> None:
        """
        Redact and append one event to the JSONL file.

        seq, run_id, phase, and timestamp are added automatically.
        Non-JSON-serialisable values are coerced via str().

        Discovery note: when log_all() is used (batch write after a completed
        discovery run), every record in the file will share the same timestamp
        value — the moment log_all() was called — while the per-action
        timestamp_iso field inside each event reflects when the action
        actually occurred during the loop. This is expected: timestamp marks
        when the record was persisted, timestamp_iso marks when it happened.
        """
        record = {
            "seq": self._seq,
            "run_id": self._run_id,
            "phase": self._phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        self._seq += 1
        raw = json.dumps(record, default=str)
        safe = redact(raw, sensitive_values=self._sensitive_values())
        with self._path.open("a", encoding="utf-8") as f:
            f.write(safe + "\n")

    def log_all(self, events: list[dict]) -> None:
        """Write a pre-collected list of events in order (discovery post-run path)."""
        for event in events:
            self.log(event)

    def _sensitive_values(self) -> list[str]:
        if self._capability is None:
            return []
        return collect_sensitive_values(self._capability, self._params, self._outputs)
