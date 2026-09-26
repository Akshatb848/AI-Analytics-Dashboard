"""Optional sign-in: the app is open by default and gated once [auth] is configured."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui import auth

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
AUTH_SECRETS = {
    "redirect_uri": "http://localhost:8501/oauth2callback",
    "cookie_secret": "test-cookie-secret",
    "client_id": "test-client",
    "client_secret": "test-secret",
    "server_metadata_url": "https://accounts.google.com/.well-known/openid-configuration",
}


@pytest.mark.parametrize("configured, logged_in, email, emails, domains, expected", [
    (False, False, None, set(), set(), "open"),
    (True, False, None, set(), set(), "login"),
    (True, True, "ana@acme.com", set(), set(), "allowed"),
    (True, True, "ana@acme.com", {"ana@acme.com"}, set(), "allowed"),
    (True, True, "Ana@Acme.com", {"ana@acme.com"}, set(), "allowed"),
    (True, True, "raj@acme.com", set(), {"acme.com"}, "allowed"),
    (True, True, "eve@evil.com", {"ana@acme.com"}, {"acme.com"}, "denied"),
    (True, True, "eve@acme.com.evil.com", set(), {"acme.com"}, "denied"),
    (True, True, None, set(), {"acme.com"}, "denied"),
])
def test_access_decision(configured, logged_in, email, emails, domains, expected):
    assert auth.access_decision(configured, logged_in, email, emails, domains) == expected


def test_parse_list():
    assert auth.parse_list(" Ana@Acme.com, @acme.com ,,") == {"ana@acme.com", "acme.com"}
    assert auth.parse_list(None) == set()


def run_app(monkeypatch, secrets=None, user=(False, None), env=None):
    for name in ("ALLOWED_EMAILS", "ALLOWED_EMAIL_DOMAINS", "ZAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(auth, "current_user", lambda: user)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    for key, value in (secrets or {}).items():
        at.secrets[key] = value
    at.run()
    return at


def test_app_is_open_without_auth_configured(monkeypatch):
    at = run_app(monkeypatch)
    assert not at.exception
    assert len(at.tabs) == 7
    assert not any(b.label == "Sign in" for b in at.button)


def test_signed_out_visitors_only_see_the_sign_in_page(monkeypatch):
    at = run_app(monkeypatch, secrets={"auth": AUTH_SECRETS})
    assert not at.exception
    assert any(b.label == "Sign in" for b in at.button)
    assert len(at.tabs) == 0  # nothing else rendered


def test_signed_in_users_get_the_app_and_a_sign_out_button(monkeypatch):
    at = run_app(monkeypatch, secrets={"auth": AUTH_SECRETS}, user=(True, "ana@acme.com"),
                 env={"ALLOWED_EMAIL_DOMAINS": "acme.com"})
    assert not at.exception
    assert len(at.tabs) == 7
    assert any(b.label == "Sign out" for b in at.sidebar.button)
    assert any("Signed in as ana@acme.com" in c.value for c in at.sidebar.caption)


def test_signed_in_users_outside_the_allow_list_are_refused(monkeypatch):
    at = run_app(monkeypatch, secrets={"auth": AUTH_SECRETS}, user=(True, "eve@evil.com"),
                 env={"ALLOWED_EMAILS": "ana@acme.com"})
    assert not at.exception
    assert len(at.tabs) == 0
    assert any("isn't allowed" in m.value for m in at.markdown)
    assert any(b.label == "Sign out" for b in at.button)
