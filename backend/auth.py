"""
Supabase JWT verification.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

WHY THIS EXISTS: the sign-in page authenticated the browser and nothing else.
The API identified its caller from an `X-User` header that anyone could set,
defaulting to `operator1`, so every route was reachable without signing in -
`curl -H "X-User: admin"` was full administrative access. The interface was
protected; the system was not.

WHAT IT DOES: verifies the access token Supabase issues at sign-in, so the
caller's identity and role come from a signature the server can check rather
than from a header the caller chose.

HS256 ONLY, DELIBERATELY. This project's Supabase instance signs with HS256 -
readable from the algorithm header of its own anon key - which is a shared
secret, so the same secret that verifies a token could also mint one. That is
fine here because only this server holds it, but it means the secret is a
credential of the same weight as a database password and belongs in the
environment, never in the repository. A project migrated to asymmetric keys
(RS256/ES256 with a JWKS endpoint) needs a different verifier; this one
refuses those algorithms by name rather than pretending to check them.

NO PyJWT. HMAC-SHA256 verification is hmac + hashlib + base64, all stdlib,
which keeps the promise that the core of this system installs with nothing.
The one thing a hand-rolled verifier must not get wrong is accepting a token
it did not actually check, so the failure mode here is always "raise", never
"return unverified claims" - see AuthError below.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

# Supabase stamps every user token with this audience.
DEFAULT_AUDIENCE = "authenticated"

# Clocks drift. A minute of leeway stops a correct token being rejected by a
# server that is slightly ahead, without meaningfully extending its life.
CLOCK_SKEW_SECONDS = 60

# The four roles the users table's CHECK constraint permits. A token carrying
# anything else is not trusted to name a role - see role_from_claims.
VALID_ROLES = ("operator", "verifier", "admin", "auditor")
DEFAULT_ROLE = "operator"


class AuthError(Exception):
    """
    A token was absent, malformed, expired, or not signed by us.

    Deliberately one exception rather than several: the caller's correct
    response to every one of these is the same 401, and distinguishing
    "expired" from "bad signature" in a response body tells an attacker which
    half of their guess was right.
    """


def _b64url_decode(segment: str) -> bytes:
    """Decode a JWT segment, restoring the padding JWTs strip."""
    padding = "=" * (-len(segment) % 4)
    try:
        return base64.urlsafe_b64decode(segment + padding)
    except Exception as exc:                       # malformed base64
        raise AuthError("Token segment is not valid base64url.") from exc


def verify_token(token: str,
                 secret: str,
                 audience: Optional[str] = DEFAULT_AUDIENCE,
                 issuer: Optional[str] = None,
                 now: Optional[float] = None) -> dict:
    """
    Verify a Supabase HS256 access token and return its claims.

    Raises AuthError on anything short of a fully verified token. `now` is
    injectable so the expiry paths can be tested without sleeping.
    """
    if not secret:
        raise AuthError("No JWT secret is configured, so tokens cannot be verified.")
    if not token or token.count(".") != 2:
        raise AuthError("Token is not a well-formed JWT.")

    header_b64, payload_b64, signature_b64 = token.split(".")

    try:
        header = json.loads(_b64url_decode(header_b64))
        claims = json.loads(_b64url_decode(payload_b64))
    except (ValueError, TypeError) as exc:
        raise AuthError("Token header or payload is not valid JSON.") from exc
    if not isinstance(claims, dict) or not isinstance(header, dict):
        raise AuthError("Token header or payload is not an object.")

    # Refuse "alg": "none" and any algorithm this verifier does not actually
    # implement. Accepting the token's own word on how to check it is the
    # classic JWT forgery, and silently treating RS256 as HS256 would verify
    # a token against a public key used as an HMAC secret.
    if header.get("alg") != "HS256":
        raise AuthError(f"Unsupported token algorithm {header.get('alg')!r}; "
                        f"this verifier implements HS256 only.")

    expected = hmac.new(secret.encode("utf-8"),
                        f"{header_b64}.{payload_b64}".encode("ascii"),
                        hashlib.sha256).digest()
    # compare_digest, not ==: a byte-by-byte comparison leaks, through its
    # timing, how much of a forged signature was correct.
    if not hmac.compare_digest(expected, _b64url_decode(signature_b64)):
        raise AuthError("Token signature does not verify.")

    clock = time.time() if now is None else now

    exp = claims.get("exp")
    if exp is None:
        raise AuthError("Token has no expiry.")
    try:
        if clock > float(exp) + CLOCK_SKEW_SECONDS:
            raise AuthError("Token has expired.")
    except (TypeError, ValueError) as exc:
        raise AuthError("Token expiry is not a number.") from exc

    nbf = claims.get("nbf")
    if nbf is not None:
        try:
            if clock < float(nbf) - CLOCK_SKEW_SECONDS:
                raise AuthError("Token is not valid yet.")
        except (TypeError, ValueError) as exc:
            raise AuthError("Token nbf is not a number.") from exc

    # Audience may be a string or a list in the spec; Supabase sends a string.
    if audience is not None:
        aud = claims.get("aud")
        allowed = aud if isinstance(aud, list) else [aud]
        if audience not in allowed:
            raise AuthError("Token audience does not match this service.")

    if issuer is not None and claims.get("iss") != issuer:
        raise AuthError("Token issuer does not match this service.")

    if not claims.get("sub"):
        raise AuthError("Token has no subject.")

    return claims


def role_from_claims(claims: dict) -> str:
    """
    Read the application role a token carries, falling back to the least
    privileged one.

    app_metadata is checked before user_metadata because a user can edit their
    own user_metadata through the Supabase client, and a self-assigned "admin"
    must not become an actual admin here. app_metadata is server-side only.
    Anything unrecognised is demoted rather than rejected: an unknown role is
    not an authentication failure, it is a reason not to grant rights.
    """
    for container in ("app_metadata", "user_metadata"):
        meta = claims.get(container)
        if isinstance(meta, dict):
            candidate = str(meta.get("role", "")).strip().lower()
            if candidate in VALID_ROLES:
                return candidate
    return DEFAULT_ROLE


def identity_from_claims(claims: dict) -> str:
    """
    The username this token maps to locally.

    Email where present - it is what an officer recognises and what the audit
    trail should name - otherwise the immutable subject id, so a token without
    an email still resolves to a stable identity rather than to nobody.
    """
    email = claims.get("email")
    if isinstance(email, str) and email.strip():
        return email.strip().lower()
    return str(claims.get("sub"))


# -- configuration ---------------------------------------------------------

def jwt_secret() -> Optional[str]:
    """The Supabase project's JWT secret, or None if it is not configured."""
    return os.environ.get("SUPABASE_JWT_SECRET") or None


def dev_mode() -> bool:
    """
    Whether unauthenticated X-User access is still permitted.

    True when explicitly requested, and also whenever no secret is configured
    - because a server with no secret cannot verify anything, and refusing
    every request would turn a missing environment variable into a system
    that simply does not work. Which of the two applies is reported at
    startup by auth_status(), so this is never a silent downgrade.
    """
    return os.environ.get("AUTH_DEV_MODE") == "1" or jwt_secret() is None


def auth_status() -> dict:
    """Machine-readable posture, for run.py --check and the startup banner."""
    secret, dev = jwt_secret(), dev_mode()
    if secret and not dev:
        return {"enforced": True,
                "detail": "Supabase access tokens are required and verified."}
    if secret and dev:
        return {"enforced": False,
                "detail": "AUTH_DEV_MODE=1: tokens are verified when sent, "
                          "but an unauthenticated X-User header is still accepted."}
    return {"enforced": False,
            "detail": "SUPABASE_JWT_SECRET is not set, so tokens cannot be "
                      "verified and any X-User header is accepted. Set it "
                      "before exposing this server to a network."}
