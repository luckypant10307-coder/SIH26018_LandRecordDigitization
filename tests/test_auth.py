#!/usr/bin/env python3
"""
Unit tests for Supabase access-token verification (backend/auth.py).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Written from the forger's side. A verifier that returns claims is easy to
write and easy to get catastrophically wrong: the dangerous bug is not
rejecting a good token, it is ACCEPTING a bad one, and that bug is invisible
to any test that only signs a token correctly and checks the claims come back.

So most of what follows is deliberate forgery - the alg:none swap, a tampered
payload, a signature from the wrong secret, an expired token - each asserting
that verification raises rather than returns.

No live Supabase session is used or needed. Tokens are minted here against a
test secret, which is also the only way these tests can run offline and in CI.

Run from anywhere with:
    python3 tests/test_auth.py -v
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import auth  # noqa: E402

SECRET = "test-secret-not-a-real-one"


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def make_token(claims: dict, secret: str = SECRET, alg: str = "HS256",
               signature: bytes = None) -> str:
    """Mint a JWT. Every forgery in this file is a variation on this."""
    header = _b64(json.dumps({"alg": alg, "typ": "JWT"}).encode())
    payload = _b64(json.dumps(claims).encode())
    if signature is None:
        signature = hmac.new(secret.encode(), f"{header}.{payload}".encode(),
                             hashlib.sha256).digest()
    return f"{header}.{payload}.{_b64(signature)}"


def valid_claims(**overrides) -> dict:
    now = int(time.time())
    claims = {
        "sub": "8f1c2e40-0000-4000-8000-000000000001",
        "email": "officer@example.gov.in",
        "aud": "authenticated",
        "iat": now,
        "exp": now + 3600,
        "app_metadata": {"role": "verifier"},
        "user_metadata": {},
    }
    claims.update(overrides)
    return claims


class TestAcceptsGoodTokens(unittest.TestCase):

    def test_valid_token_returns_claims(self):
        claims = auth.verify_token(make_token(valid_claims()), SECRET)
        self.assertEqual(claims["email"], "officer@example.gov.in")

    def test_expiry_within_clock_skew_is_tolerated(self):
        # A token that expired two seconds ago is a clock difference, not an
        # attack; rejecting it would log out a correct user at random.
        claims = valid_claims(exp=int(time.time()) - 2)
        self.assertTrue(auth.verify_token(make_token(claims), SECRET))


class TestRejectsForgeries(unittest.TestCase):
    """Each test is an attack that must not succeed."""

    def test_alg_none_is_refused(self):
        # The classic JWT forgery: strip the signature and tell the verifier
        # not to check one. Accepting the token's own word on how to verify
        # it defeats the whole mechanism.
        token = make_token(valid_claims(), alg="none", signature=b"")
        with self.assertRaises(auth.AuthError):
            auth.verify_token(token, SECRET)

    def test_asymmetric_alg_is_refused_not_treated_as_hmac(self):
        # RS256 verified as HS256 would check a signature against a PUBLIC
        # key used as a shared secret - forgeable by anyone who has that key.
        with self.assertRaises(auth.AuthError):
            auth.verify_token(make_token(valid_claims(), alg="RS256"), SECRET)

    def test_signature_from_another_secret_is_refused(self):
        token = make_token(valid_claims(), secret="some-other-secret")
        with self.assertRaises(auth.AuthError):
            auth.verify_token(token, SECRET)

    def test_tampered_payload_is_refused(self):
        # Sign as an operator, then rewrite the role to admin - the attack
        # that matters most here, because it is a privilege escalation.
        header, payload, sig = make_token(valid_claims()).split(".")
        forged = valid_claims(app_metadata={"role": "admin"})
        tampered = f"{header}.{_b64(json.dumps(forged).encode())}.{sig}"
        with self.assertRaises(auth.AuthError):
            auth.verify_token(tampered, SECRET)

    def test_expired_token_is_refused(self):
        claims = valid_claims(exp=int(time.time()) - 9999)
        with self.assertRaises(auth.AuthError):
            auth.verify_token(make_token(claims), SECRET)

    def test_token_without_expiry_is_refused(self):
        claims = valid_claims()
        del claims["exp"]
        with self.assertRaises(auth.AuthError):
            auth.verify_token(make_token(claims), SECRET)

    def test_wrong_audience_is_refused(self):
        # Stops a token minted for a different Supabase service being
        # replayed against this one.
        claims = valid_claims(aud="some-other-service")
        with self.assertRaises(auth.AuthError):
            auth.verify_token(make_token(claims), SECRET)

    def test_wrong_issuer_is_refused_when_pinned(self):
        token = make_token(valid_claims(iss="https://evil.example.com"))
        with self.assertRaises(auth.AuthError):
            auth.verify_token(token, SECRET,
                              issuer="https://real.supabase.co/auth/v1")

    def test_token_without_subject_is_refused(self):
        claims = valid_claims()
        del claims["sub"]
        with self.assertRaises(auth.AuthError):
            auth.verify_token(make_token(claims), SECRET)

    def test_malformed_tokens_are_refused(self):
        for bad in ("", "not-a-jwt", "a.b", "a.b.c.d", "...", "a.b.!!!"):
            with self.assertRaises(auth.AuthError):
                auth.verify_token(bad, SECRET)

    def test_empty_secret_never_verifies(self):
        # Guards the worst possible regression: an unset secret making every
        # token valid instead of none.
        with self.assertRaises(auth.AuthError):
            auth.verify_token(make_token(valid_claims()), "")


class TestRoleMapping(unittest.TestCase):

    def test_app_metadata_role_is_used(self):
        self.assertEqual(
            auth.role_from_claims(valid_claims(app_metadata={"role": "admin"})),
            "admin")

    def test_app_metadata_beats_user_metadata(self):
        # user_metadata is editable by the user through the Supabase client.
        # If it won, any user could make themselves an admin.
        claims = valid_claims(app_metadata={"role": "operator"},
                              user_metadata={"role": "admin"})
        self.assertEqual(auth.role_from_claims(claims), "operator")

    def test_self_assigned_role_in_user_metadata_is_only_a_fallback(self):
        claims = valid_claims(app_metadata={}, user_metadata={"role": "admin"})
        # Still honoured when the server said nothing - but this is exactly
        # why app_metadata is what an administrator should set.
        self.assertEqual(auth.role_from_claims(claims), "admin")

    def test_unknown_role_is_demoted_not_accepted(self):
        claims = valid_claims(app_metadata={"role": "superuser"})
        self.assertEqual(auth.role_from_claims(claims), auth.DEFAULT_ROLE)

    def test_missing_role_defaults_to_least_privilege(self):
        claims = valid_claims(app_metadata={}, user_metadata={})
        self.assertEqual(auth.role_from_claims(claims), "operator")

    def test_every_default_role_is_one_the_database_permits(self):
        # The users table has a CHECK constraint on role; a role this module
        # could emit but the schema rejects would be a runtime crash on
        # someone's first sign-in.
        self.assertIn(auth.DEFAULT_ROLE, auth.VALID_ROLES)


class TestIdentityMapping(unittest.TestCase):

    def test_email_is_preferred(self):
        self.assertEqual(auth.identity_from_claims(valid_claims()),
                         "officer@example.gov.in")

    def test_email_is_normalised(self):
        claims = valid_claims(email="  Officer@Example.Gov.In  ")
        self.assertEqual(auth.identity_from_claims(claims),
                         "officer@example.gov.in")

    def test_falls_back_to_subject(self):
        claims = valid_claims()
        del claims["email"]
        self.assertEqual(auth.identity_from_claims(claims), claims["sub"])


class TestConfiguration(unittest.TestCase):

    def setUp(self):
        self._saved = {k: os.environ.get(k)
                       for k in ("SUPABASE_JWT_SECRET", "AUTH_DEV_MODE")}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_no_secret_means_dev_mode_and_is_reported(self):
        os.environ.pop("SUPABASE_JWT_SECRET", None)
        os.environ.pop("AUTH_DEV_MODE", None)
        self.assertTrue(auth.dev_mode())
        status = auth.auth_status()
        self.assertFalse(status["enforced"])
        # The operator must be able to tell WHY it is not enforced.
        self.assertIn("SUPABASE_JWT_SECRET", status["detail"])

    def test_secret_alone_enforces(self):
        os.environ["SUPABASE_JWT_SECRET"] = SECRET
        os.environ.pop("AUTH_DEV_MODE", None)
        self.assertFalse(auth.dev_mode())
        self.assertTrue(auth.auth_status()["enforced"])

    def test_dev_mode_flag_opens_it_again_and_says_so(self):
        os.environ["SUPABASE_JWT_SECRET"] = SECRET
        os.environ["AUTH_DEV_MODE"] = "1"
        self.assertTrue(auth.dev_mode())
        self.assertIn("AUTH_DEV_MODE", auth.auth_status()["detail"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
