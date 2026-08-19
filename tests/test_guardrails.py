"""
Unit tests for guardrails/allowlist.py and guardrails/redaction.py.
"""
from __future__ import annotations

import pytest

from guardrails.allowlist import (
    AllowlistConfig,
    AllowlistDenied,
    AllowlistEntry,
    check_allowlist,
    configure,
    reset_to_default,
)
from guardrails.redaction import redact


# ---------------------------------------------------------------------------
# Allowlist — AllowlistConfig.permits()
# ---------------------------------------------------------------------------

class TestAllowlistConfigPermits:
    """Unit tests for AllowlistConfig.permits() — no module-level state."""

    def _cfg(self, *entries: tuple[str, str, str]) -> AllowlistConfig:
        return AllowlistConfig(
            entries=[AllowlistEntry(domain=d, route_prefix=r, action=a) for d, r, a in entries]
        )

    def test_exact_match_permits(self):
        cfg = self._cfg(("localhost", "/search", "navigate"))
        assert cfg.permits("http://localhost:5001/search", "navigate")

    def test_route_prefix_match_permits(self):
        cfg = self._cfg(("localhost", "/member/", "read"))
        assert cfg.permits("http://localhost:5001/member/12345", "read")

    def test_wrong_domain_denies(self):
        cfg = self._cfg(("localhost", "/search", "navigate"))
        assert not cfg.permits("http://evil.com/search", "navigate")

    def test_wrong_route_denies(self):
        cfg = self._cfg(("localhost", "/search", "navigate"))
        assert not cfg.permits("http://localhost:5001/admin", "navigate")

    def test_wrong_action_denies(self):
        cfg = self._cfg(("localhost", "/search", "navigate"))
        assert not cfg.permits("http://localhost:5001/search", "click")

    def test_empty_config_denies_everything(self):
        cfg = AllowlistConfig(entries=[])
        assert not cfg.permits("http://localhost:5001/search", "navigate")

    def test_port_ignored_in_domain_match(self):
        """Domain match uses hostname only — port is irrelevant."""
        cfg = self._cfg(("localhost", "/search", "navigate"))
        assert cfg.permits("http://localhost:9999/search", "navigate")
        assert cfg.permits("https://localhost/search", "navigate")

    def test_multiple_entries_any_match_permits(self):
        cfg = self._cfg(
            ("localhost", "/search", "navigate"),
            ("localhost", "/member/", "read"),
        )
        assert cfg.permits("http://localhost/search", "navigate")
        assert cfg.permits("http://localhost/member/99", "read")
        assert not cfg.permits("http://localhost/member/99", "navigate")

    def test_root_route_prefix_matches_all_paths(self):
        cfg = self._cfg(("localhost", "/", "navigate"))
        assert cfg.permits("http://localhost/anything/goes", "navigate")

    def test_route_prefix_does_not_match_unrelated_sibling(self):
        """'/member/' must not match '/membership/' — prefix is correct but
        the test guards against an off-by-one with a non-slash-terminated prefix."""
        cfg = self._cfg(("localhost", "/member/", "click"))
        assert not cfg.permits("http://localhost/membership/enroll", "click")


# ---------------------------------------------------------------------------
# Allowlist — check_allowlist() (module-level state)
# ---------------------------------------------------------------------------

class TestCheckAllowlist:
    """Tests for the module-level check_allowlist() function."""

    def setup_method(self):
        """Restore default config before each test."""
        reset_to_default()

    def teardown_method(self):
        reset_to_default()

    def test_default_config_permits_localhost_search_navigate(self):
        check_allowlist("http://localhost:5001/search", "navigate")

    def test_default_config_permits_localhost_member_read(self):
        check_allowlist("http://localhost:5001/member/12345", "read")

    def test_default_config_permits_localhost_not_found_navigate(self):
        check_allowlist("http://localhost:5001/not-found?member_id=x", "navigate")

    def test_default_config_denies_external_domain(self):
        with pytest.raises(AllowlistDenied, match="evil.com"):
            check_allowlist("http://evil.com/search", "navigate")

    def test_default_config_denies_unlisted_route(self):
        with pytest.raises(AllowlistDenied):
            check_allowlist("http://localhost:5001/admin/users", "navigate")

    def test_configure_replaces_default(self):
        configure(AllowlistConfig(entries=[
            AllowlistEntry(domain="bank.internal", route_prefix="/teller/", action="navigate"),
        ]))
        check_allowlist("http://bank.internal/teller/dashboard", "navigate")
        with pytest.raises(AllowlistDenied):
            check_allowlist("http://localhost:5001/search", "navigate")

    def test_empty_config_raises_for_all_actions(self):
        configure(AllowlistConfig(entries=[]))
        with pytest.raises(AllowlistDenied):
            check_allowlist("http://localhost:5001/search", "navigate")

    def test_error_message_names_host_and_path(self):
        configure(AllowlistConfig(entries=[]))
        with pytest.raises(AllowlistDenied) as exc_info:
            check_allowlist("http://localhost:5001/forbidden", "click")
        msg = str(exc_info.value)
        assert "localhost" in msg
        assert "/forbidden" in msg


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

class TestRedactTypeGuard:
    def test_non_string_raises_type_error(self):
        with pytest.raises(TypeError, match="str"):
            redact({"key": "value"})  # type: ignore[arg-type]

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            redact(None)  # type: ignore[arg-type]


class TestRedactSSN:
    def test_hyphenated_ssn_redacted(self):
        assert redact("SSN: 123-45-6789") == "SSN: [REDACTED]"

    def test_ssn_in_json_redacted(self):
        result = redact('{"ssn": "987-65-4321", "name": "Alice"}')
        assert "[REDACTED]" in result
        assert "987-65-4321" not in result

    def test_partial_ssn_not_redacted(self):
        """Two-segment NNN-NN is not a full SSN."""
        assert redact("ref: 123-45") == "ref: 123-45"

    def test_ssn_word_boundary_respected(self):
        """A digit that immediately follows is not part of a word boundary."""
        assert redact("code-123-45-67890") == "code-123-45-67890"


class TestRedactGroupedDigits:
    """Hyphenated/spaced account and card number formats."""

    def test_hyphenated_account_number_redacted(self):
        """5432-1098-76 is a 10-digit account number in grouped form."""
        assert redact("5432-1098-76") == "[REDACTED]"

    def test_sixteen_digit_card_redacted(self):
        assert redact("card: 4111-1111-1111-1111") == "card: [REDACTED]"

    def test_space_separated_card_redacted(self):
        assert redact("card: 4111 1111 1111 1111") == "card: [REDACTED]"

    def test_iso_date_not_redacted(self):
        """YYYY-MM-DD has a 2-digit middle group — does not match NNNN-NNNN prefix."""
        assert redact("since 2019-08-15") == "since 2019-08-15"

    def test_year_alongside_account_number(self):
        """Year left intact; 9-digit account number caught by long-digits rule."""
        result = redact("Member since 2019, account 543210987")
        assert "2019" in result
        assert "543210987" not in result


class TestRedactLongDigits:
    def test_nine_digit_number_redacted(self):
        assert redact("routing: 123456789") == "routing: [REDACTED]"

    def test_ten_digit_number_redacted(self):
        assert redact("acct: 1234567890") == "acct: [REDACTED]"

    def test_eight_digit_number_not_redacted(self):
        """8 digits is below the threshold — short IDs should not be masked."""
        assert redact("id: 12345678") == "id: 12345678"

    def test_masked_account_unchanged(self):
        """****7890 is already masked — the non-digit chars break the pattern."""
        assert redact("account ****7890 active") == "account ****7890 active"


class TestRedactSensitiveValues:
    def test_single_sensitive_value_redacted(self):
        result = redact("member_id=12345", sensitive_values=["12345"])
        assert result == "member_id=[REDACTED]"

    def test_multiple_sensitive_values_all_redacted(self):
        result = redact(
            "id=abc, token=xyz99",
            sensitive_values=["abc", "xyz99"],
        )
        assert "abc" not in result
        assert "xyz99" not in result
        assert result.count("[REDACTED]") == 2

    def test_sensitive_value_all_occurrences_replaced(self):
        result = redact("val=12345 and again val=12345", sensitive_values=["12345"])
        assert "12345" not in result
        assert result.count("[REDACTED]") == 2

    def test_empty_sensitive_value_skipped(self):
        """Replacing "" would corrupt the entire string."""
        result = redact("hello world", sensitive_values=[""])
        assert result == "hello world"

    def test_sensitive_value_applied_before_patterns(self):
        """A sensitive value that's also 9 digits gets caught by layer 1."""
        result = redact("acct=123456789", sensitive_values=["123456789"])
        assert result == "acct=[REDACTED]"

    def test_no_sensitive_values_clean_text_unchanged(self):
        result = redact("name: Alice, balance: $500.00")
        assert result == "name: Alice, balance: $500.00"

    def test_generator_accepted_as_sensitive_values(self):
        """sensitive_values is Iterable — generators must work."""
        result = redact("x=abc", sensitive_values=(v for v in ["abc"]))
        assert "abc" not in result
