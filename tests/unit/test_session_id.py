"""Phase 4.5: chatbot.py used to read `?sid=` straight from the URL and use
it directly as the Redis key, with no check that the server ever issued it —
editing the query param to any string, guessed or fabricated, was accepted as
a previously-issued session. Signing closes that: a session is only accepted
back if its signature verifies, while a legitimately shared link (the
documented "sessions are shareable via URL" feature) still works, since it
carries a real signature.
"""

from __future__ import annotations

from unittest.mock import patch

import chatbot


class _FakeSessionState(dict):
    """Minimal stand-in for st.session_state: dict `in`/[] plus attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _run(query_params: dict) -> tuple[str, dict, _FakeSessionState]:
    """Call _get_or_create_session_id with query_params/session_state faked out."""
    session_state = _FakeSessionState()
    with patch("chatbot.st") as st_mock:
        st_mock.query_params = query_params
        st_mock.session_state = session_state
        sid = chatbot._get_or_create_session_id()
    return sid, query_params, session_state


# ---------------------------------------------------------------------------
# _sign_session_id / _verify_session_id — the pure signing logic
# ---------------------------------------------------------------------------


def test_a_signed_id_verifies_back_to_the_original():
    signed = chatbot._sign_session_id("abc-123")
    assert chatbot._verify_session_id(signed) == "abc-123"


def test_an_unsigned_arbitrary_string_does_not_verify():
    assert chatbot._verify_session_id("just-a-guess") is None


def test_a_tampered_signature_does_not_verify():
    signed = chatbot._sign_session_id("abc-123")
    sid, _, mac = signed.rpartition(".")
    tampered = f"{sid}.{'0' * len(mac)}"
    assert tampered != signed  # sanity: the tamper actually changed something
    assert chatbot._verify_session_id(tampered) is None


def test_a_signature_for_a_different_id_does_not_transfer():
    """The signature must bind to that specific id, not just prove *some*
    id was once signed — otherwise swapping the id half back out reopens
    exactly the IDOR this exists to close."""
    mac = chatbot._sign_session_id("victim-session").rpartition(".")[2]
    forged = f"attacker-chosen-id.{mac}"
    assert chatbot._verify_session_id(forged) is None


def test_empty_string_does_not_verify():
    assert chatbot._verify_session_id("") is None


def test_a_configured_session_secret_changes_the_signature():
    """Proof SESSION_SECRET is actually consulted, not just present as an
    unused option — a link signed under one secret must not verify under
    another (simulating what should happen across a redeploy that rotates it,
    versus a deployment that pins one on purpose for continuity)."""
    with patch.dict("os.environ", {"SESSION_SECRET": "secret-one"}):
        signed_with_one = chatbot._sign_session_id("abc-123")
    with patch.dict("os.environ", {"SESSION_SECRET": "secret-two"}):
        verified_under_two = chatbot._verify_session_id(signed_with_one)
        signed_with_two = chatbot._sign_session_id("abc-123")

    assert verified_under_two is None
    assert signed_with_one != signed_with_two


# ---------------------------------------------------------------------------
# _get_or_create_session_id — the Streamlit-facing behaviour
# ---------------------------------------------------------------------------


def test_a_fresh_visit_mints_a_new_signed_session():
    sid, query_params, _state = _run({})
    assert sid
    assert query_params["sid"] == chatbot._sign_session_id(sid)


def test_a_validly_signed_returning_link_is_honoured():
    """The documented feature this must not break: a real, previously-issued
    link still opens the same session."""
    original_sid = "00000000-0000-0000-0000-000000000000"
    signed = chatbot._sign_session_id(original_sid)

    sid, _query_params, _state = _run({"sid": signed})

    assert sid == original_sid


def test_a_fabricated_sid_is_rejected_and_a_fresh_session_is_issued():
    """The actual vulnerability: editing ?sid= to an arbitrary string used to
    be accepted outright and used as the Redis key as-is."""
    sid, query_params, _state = _run({"sid": "someone-elses-guessed-id"})

    assert sid != "someone-elses-guessed-id"
    assert query_params["sid"] == chatbot._sign_session_id(sid)


def test_a_forged_id_with_a_stolen_signature_is_rejected():
    stolen_mac = chatbot._sign_session_id("victim-session").rpartition(".")[2]
    sid, _query_params, _state = _run({"sid": f"attacker-chosen-id.{stolen_mac}"})

    assert sid != "attacker-chosen-id"


def test_session_state_is_only_set_once_per_session():
    session_state = _FakeSessionState()
    query_params = {}
    with patch("chatbot.st") as st_mock:
        st_mock.query_params = query_params
        st_mock.session_state = session_state
        first = chatbot._get_or_create_session_id()
        # A second call in the same "session" (session_state persists,
        # query_params now carries the signed link from the first call).
        second = chatbot._get_or_create_session_id()

    assert first == second
