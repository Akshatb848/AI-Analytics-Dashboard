"""Optional sign-in using Streamlit's built-in OpenID Connect support (st.login).

Sign-in is off unless an `[auth]` section exists in the app's secrets, so the app
stays open by default. With it configured, visitors must sign in with the
configured provider (Google, Microsoft, Auth0, Okta, ...). Access can then be
limited with, in the environment or secrets:

    ALLOWED_EMAILS         comma-separated addresses, e.g. "ana@acme.com,raj@acme.com"
    ALLOWED_EMAIL_DOMAINS  comma-separated domains, e.g. "acme.com"

With neither set, anyone who can sign in with the provider gets access.
"""
import os
from collections.abc import Mapping
from typing import Optional, Set, Tuple

import streamlit as st

from analytics.formatting import safe_html


def _secret(name: str) -> Optional[str]:
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file
        return None
    return str(value) if value else None


def _setting(name: str) -> Optional[str]:
    return os.environ.get(name) or _secret(name)


def parse_list(value: Optional[str]) -> Set[str]:
    return {item.strip().lower().lstrip("@") for item in (value or "").split(",") if item.strip()}


def auth_settings() -> Optional[Mapping]:
    """The [auth] secrets section, or None when sign-in isn't configured."""
    try:
        section = st.secrets.get("auth")
    except Exception:
        return None
    return section if isinstance(section, Mapping) and section else None


def current_user() -> Tuple[bool, Optional[str]]:
    """(signed in?, email). Kept separate so tests can stand in for a signed-in user."""
    user = st.user
    return bool(getattr(user, "is_logged_in", False)), user.get("email")


def access_decision(auth_configured: bool, logged_in: bool, email: Optional[str],
                    allowed_emails: Set[str], allowed_domains: Set[str]) -> str:
    """'open' (no sign-in configured), 'login', 'denied' or 'allowed'."""
    if not auth_configured:
        return "open"
    if not logged_in:
        return "login"
    if not allowed_emails and not allowed_domains:
        return "allowed"
    address = (email or "").strip().lower()
    domain = address.rsplit("@", 1)[-1] if "@" in address else ""
    return "allowed" if address in allowed_emails or domain in allowed_domains else "denied"


def require_login() -> None:
    """Gate the page when sign-in is configured; stops the script for visitors without access."""
    settings = auth_settings()
    logged_in, email = current_user() if settings else (False, None)
    decision = access_decision(
        settings is not None, logged_in, email,
        parse_list(_setting("ALLOWED_EMAILS")), parse_list(_setting("ALLOWED_EMAIL_DOMAINS")),
    )
    if decision == "open":
        return

    if decision == "login":
        # A named provider section like [auth.google] selects that provider
        providers = [name for name, value in settings.items() if isinstance(value, Mapping)]
        st.markdown("## 🔒 Sign in to AI Analytics Dashboard")
        st.caption("This dashboard is private. Sign in to continue.")
        st.button("Sign in", type="primary", on_click=st.login,
                  args=(providers[0],) if providers else ())
        st.stop()

    if decision == "denied":
        st.markdown("## 🚫 No access")
        st.markdown(f"**{safe_html(email or 'This account')}** isn't allowed to use this dashboard. "
                    "Ask the owner to add you, or sign in with a different account.",
                    unsafe_allow_html=True)
        st.button("Sign out", on_click=st.logout)
        st.stop()

    with st.sidebar:
        st.caption(f"Signed in as {email or 'unknown user'}")
        st.button("Sign out", on_click=st.logout, key="sign_out")
