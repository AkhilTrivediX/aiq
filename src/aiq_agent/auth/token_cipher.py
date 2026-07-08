# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Authenticated symmetric encryption for stored auth tokens.

Refresh tokens are long-lived secrets and must never be persisted in plaintext.
``TokenCipher`` wraps AES-256-GCM with a versioned, self-describing envelope and
binds each ciphertext to a caller-supplied AAD (the session id), so a ciphertext
row cannot be silently moved to a different session.

The key comes from ``AIQ_TOKEN_ENCRYPTION_KEY`` (base64-encoded 32 bytes),
mirroring the separate ``AIQ_CONTENT_ENCRYPTION_KEY`` used for job content. The
two keys are intentionally distinct so token and content encryption can be
rotated and scoped independently.
"""

from __future__ import annotations

import base64
import binascii
import json
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_ENV_KEY = "AIQ_TOKEN_ENCRYPTION_KEY"
_ENVELOPE_VERSION = 1
_ALGORITHM = "AES-256-GCM"
_KEY_BYTES = 32
_NONCE_BYTES = 12


class TokenEncryptionError(Exception):
    """Base error for token encryption problems."""


class TokenEncryptionConfigError(TokenEncryptionError, ValueError):
    """The encryption key is missing or malformed."""


class TokenEncryptionInvalidData(TokenEncryptionError):
    """Stored ciphertext is malformed or failed authentication."""


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value + padding)
    except (binascii.Error, ValueError) as exc:
        raise TokenEncryptionInvalidData("token envelope field is not valid base64url") from exc


def load_token_encryption_key(raw_key: str | None = None) -> bytes:
    """Return the 32-byte AES key from ``AIQ_TOKEN_ENCRYPTION_KEY``.

    Args:
        raw_key: Base64 key override; falls back to the env var when None.

    Raises:
        TokenEncryptionConfigError: if the key is absent or not 32 bytes.
    """
    raw = raw_key if raw_key is not None else os.environ.get(_ENV_KEY)
    if not raw:
        raise TokenEncryptionConfigError(f"{_ENV_KEY} is required to encrypt stored tokens")
    try:
        key = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise TokenEncryptionConfigError(f"{_ENV_KEY} must be valid base64") from exc
    if len(key) != _KEY_BYTES:
        raise TokenEncryptionConfigError(
            f"{_ENV_KEY} must decode to {_KEY_BYTES} bytes (got {len(key)}) for AES-256-GCM"
        )
    return key


def token_encryption_configured(raw_key: str | None = None) -> bool:
    """Return True when a usable token encryption key is configured."""
    try:
        load_token_encryption_key(raw_key)
        return True
    except TokenEncryptionConfigError:
        return False


class TokenCipher:
    """Encrypt/decrypt secret strings with AES-256-GCM and AAD binding."""

    def __init__(self, key: bytes | None = None):
        """Build a cipher from an explicit key or the configured env key."""
        self._key = key if key is not None else load_token_encryption_key()
        if len(self._key) != _KEY_BYTES:
            raise TokenEncryptionConfigError(f"token encryption key must be {_KEY_BYTES} bytes")

    def encrypt(self, plaintext: str, aad: str) -> str:
        """Encrypt ``plaintext``, binding the ciphertext to ``aad``."""
        nonce = os.urandom(_NONCE_BYTES)
        sealed = AESGCM(self._key).encrypt(nonce, plaintext.encode("utf-8"), aad.encode("utf-8"))
        envelope = {
            "v": _ENVELOPE_VERSION,
            "alg": _ALGORITHM,
            "nonce": _b64url_encode(nonce),
            "ct": _b64url_encode(sealed),
        }
        return json.dumps(envelope, separators=(",", ":"))

    def decrypt(self, stored: str, aad: str) -> str:
        """Decrypt an envelope produced by :meth:`encrypt`, checking ``aad``."""
        try:
            envelope = json.loads(stored)
        except (json.JSONDecodeError, TypeError) as exc:
            raise TokenEncryptionInvalidData("stored token is not a valid envelope") from exc
        if not isinstance(envelope, dict) or envelope.get("v") != _ENVELOPE_VERSION:
            raise TokenEncryptionInvalidData("unsupported token envelope version")
        nonce = _b64url_decode(envelope.get("nonce", ""))
        sealed = _b64url_decode(envelope.get("ct", ""))
        try:
            plaintext = AESGCM(self._key).decrypt(nonce, sealed, aad.encode("utf-8"))
        except InvalidTag as exc:
            raise TokenEncryptionInvalidData("stored token failed authentication") from exc
        except ValueError as exc:
            raise TokenEncryptionInvalidData("stored token is malformed") from exc
        return plaintext.decode("utf-8")
