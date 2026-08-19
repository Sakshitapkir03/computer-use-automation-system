"""
Allowlist enforcement.

Fail-closed design: any (domain, route_prefix, action_type) combination not
declared in the active AllowlistConfig raises AllowlistDenied. There is no
default-permit behaviour.

The module ships with a built-in config that covers the known target app
(localhost, routes /search / /member/ / /not-found, all five action types).
Use configure() to swap this for a deployment-specific config before running
any actions.

Route matching uses prefix semantics: an entry with route_prefix="/member/"
permits any path that starts with "/member/", including "/member/12345".
Domain matching is on the parsed hostname only (no port) so the same entry
covers HTTP and HTTPS regardless of port.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse


class AllowlistDenied(RuntimeError):
    """Raised when an action is blocked by the allowlist configuration."""


@dataclass(frozen=True)
class AllowlistEntry:
    """One permitted (domain, route_prefix, action) combination."""

    domain: str        # exact hostname match, e.g. "localhost" or "core.bank.internal"
    route_prefix: str  # path prefix, e.g. "/member/" matches /member/12345
    action: str        # one of: navigate / click / type / read / wait_for


@dataclass
class AllowlistConfig:
    """
    Allowlist for a single deployment target.

    An action on a URL is permitted when at least one entry satisfies:
      entry.domain    == parsed hostname of the URL   (exact)
      entry.route_prefix is a prefix of the URL path  (prefix)
      entry.action    == action_type                  (exact)

    An empty entries list means no actions are permitted (fail-closed by design).
    """

    entries: list[AllowlistEntry] = field(default_factory=list)

    def permits(self, url: str, action: str) -> bool:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        path = parsed.path or "/"
        return any(
            e.domain == hostname
            and path.startswith(e.route_prefix)
            and e.action == action
            for e in self.entries
        )


# ---------------------------------------------------------------------------
# Built-in default — covers the known mock target app at localhost:5001.
# ---------------------------------------------------------------------------

_ALL_ACTIONS = ("navigate", "click", "type", "read", "wait_for")

_DEFAULT_ENTRIES: list[AllowlistEntry] = [
    AllowlistEntry(domain="localhost", route_prefix=route, action=act)
    for route in ("/search", "/member/", "/not-found")
    for act in _ALL_ACTIONS
]

_config: AllowlistConfig = AllowlistConfig(entries=_DEFAULT_ENTRIES)


def configure(config: AllowlistConfig) -> None:
    """Replace the active AllowlistConfig for this process."""
    global _config
    _config = config


def reset_to_default() -> None:
    """Restore the built-in default config (useful in tests)."""
    global _config
    _config = AllowlistConfig(entries=_DEFAULT_ENTRIES)


# ---------------------------------------------------------------------------
# Enforcement entry point — called by every action in agent/actions.py
# ---------------------------------------------------------------------------

def check_allowlist(url: str, action_type: str) -> None:
    """
    Verify the action is permitted by the active AllowlistConfig.

    Args:
        url:         Full URL of the page at the moment the action is requested.
        action_type: One of navigate / click / type / read / wait_for.

    Raises:
        AllowlistDenied: Action is not in the allowlist. Callers treat this as
                         an immediate hard_failure — never retry.
    """
    if not _config.permits(url, action_type):
        parsed = urlparse(url)
        raise AllowlistDenied(
            f"Action {action_type!r} on host={parsed.hostname!r} "
            f"path={parsed.path!r} is not in the allowlist. "
            "Add an AllowlistEntry for this route+action combination."
        )
