"""
Capability persistence and redaction utilities.

save_capability() is the only path by which Capability artifacts reach disk.
All writes route through redact() so guardrails.redaction's implementation is
applied everywhere automatically — no call-site hunting required.

collect_sensitive_values() is the chokepoint for Phase 7 evidence writing.
Every evidence file (aria snapshots, screenshots metadata, step logs) must be
processed with:

    redact(text, sensitive_values=collect_sensitive_values(cap, params, outputs))

before touching disk, so that runtime values for params and outputs marked
sensitive=True are scrubbed regardless of where they appear in the evidence.
"""
from __future__ import annotations

from pathlib import Path

from artifact.schema import Capability
from guardrails.redaction import redact

STORE_DIR = Path(__file__).parent / "store"


def save_capability(capability: Capability) -> Path:
    """Serialise, redact, and write a Capability to the artifact store."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    raw = capability.model_dump_json(indent=2)
    clean = redact(raw)
    path = STORE_DIR / f"{capability.name}_v{capability.version}.json"
    path.write_text(clean, encoding="utf-8")
    return path


def load_capability(name: str, version: int = 1) -> Capability:
    """Load and validate a saved Capability by name and version."""
    path = STORE_DIR / f"{name}_v{version}.json"
    return Capability.model_validate_json(path.read_text(encoding="utf-8"))


def list_capabilities() -> list[Path]:
    """Return paths of all saved Capability JSON files, sorted by name."""
    if not STORE_DIR.exists():
        return []
    return sorted(STORE_DIR.glob("*.json"))


def collect_sensitive_values(
    capability: Capability,
    params: dict[str, str],
    outputs: dict[str, str],
) -> list[str]:
    """
    Collect runtime values that must be redacted from evidence text.

    Walks the Capability's input and output specs; for every entry whose
    ParamSpec.sensitive or OutputSpec.sensitive is True, appends the
    corresponding runtime value (if present in params / outputs) to the
    returned list.

    Callers pass this list directly to redact()'s sensitive_values argument:

        safe = redact(raw_text, sensitive_values=collect_sensitive_values(
            capability, params, outputs
        ))

    Keys that are declared sensitive in the spec but absent from the runtime
    dicts are silently skipped — the param may simply not have been provided
    for this run (e.g. an optional input).

    Args:
        capability: The Capability whose input/output specs define sensitivity.
        params:     Runtime param values keyed by param name.
        outputs:    Runtime output values keyed by output key (from ReplayResult
                    or discovery's actual_outputs).

    Returns:
        A list of non-empty strings to pass to redact(sensitive_values=...).
        May be empty if nothing is marked sensitive or no values are available.
    """
    sensitive: list[str] = []

    for key, spec in capability.inputs.items():
        if spec.sensitive:
            val = params.get(key, "")
            if val:
                sensitive.append(val)

    for key, spec in capability.outputs.items():
        if spec.sensitive:
            val = outputs.get(key, "")
            if val:
                sensitive.append(val)

    return sensitive
