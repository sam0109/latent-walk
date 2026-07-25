from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError

COOKIE_NAME = "latent_walk_session"
SESSION_SECONDS = 12 * 60 * 60
PASSWORD_HASH_ENV = "LATENT_WALK_PASSWORD_HASH"

_password_hasher = PasswordHasher()
_session_secret = secrets.token_bytes(32)


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
    signature = hmac.new(_session_secret, payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(payload + signature).decode()


def verify_session(token: str | None) -> bool:
    if not token:
        return False
    try:
        decoded = base64.urlsafe_b64decode(token.encode())
        payload, signature = decoded[:-32], decoded[-32:]
        expected = hmac.new(_session_secret, payload, hashlib.sha256).digest()
        expires = int(payload.split(b".", 1)[0])
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(signature, expected) and expires >= int(time.time())
