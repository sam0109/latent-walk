from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from functools import lru_cache

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError

COOKIE_NAME = "latent_walk_session"
SESSION_SECONDS = 30 * 24 * 60 * 60
PASSWORD_HASH_ENV = "LATENT_WALK_PASSWORD_HASH"
SESSION_SECRET_ENV = "LATENT_WALK_SESSION_SECRET"

_password_hasher = PasswordHasher()


@lru_cache(maxsize=1)
def _session_secret() -> bytes:
    configured = os.environ.get(SESSION_SECRET_ENV)
    if configured:
        secret = configured.encode()
    else:
        password_hash = os.environ.get(PASSWORD_HASH_ENV)
        if not password_hash:
            raise RuntimeError("Session signing key is unavailable")
        secret = hmac.new(
            password_hash.encode(),
            b"latent-walk/session-signing/v1",
            hashlib.sha256,
        ).digest()
    if len(secret) < 32:
        raise RuntimeError("Session signing key is unavailable")
    return secret


def verify_password(password: str) -> bool:
    password_hash = os.environ.get(PASSWORD_HASH_ENV)
    if not password_hash:
        raise RuntimeError(f"{PASSWORD_HASH_ENV} is not configured")
    try:
        return _password_hasher.verify(password_hash, password)
    except VerificationError:
        return False


def issue_session() -> str:
    expires = int(time.time()) + SESSION_SECONDS
    payload = f"{expires}.{secrets.token_urlsafe(18)}".encode()
    signature = hmac.new(_session_secret(), payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(payload + signature).decode()


def verify_session(token: str | None) -> bool:
    if not token:
        return False
    try:
        decoded = base64.urlsafe_b64decode(token.encode())
        payload, signature = decoded[:-32], decoded[-32:]
        expected = hmac.new(_session_secret(), payload, hashlib.sha256).digest()
        expires = int(payload.split(b".", 1)[0])
    except (RuntimeError, ValueError, TypeError):
        return False
    return hmac.compare_digest(signature, expected) and expires >= int(time.time())
