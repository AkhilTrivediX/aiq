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

"""Tests for TokenCipher (AES-256-GCM with AAD binding)."""

from __future__ import annotations

import base64
import os

import pytest

from aiq_agent.auth.token_cipher import TokenCipher
from aiq_agent.auth.token_cipher import TokenEncryptionConfigError
from aiq_agent.auth.token_cipher import TokenEncryptionInvalidData
from aiq_agent.auth.token_cipher import load_token_encryption_key
from aiq_agent.auth.token_cipher import token_encryption_configured

_KEY = base64.b64encode(os.urandom(32)).decode()
_OTHER_KEY = base64.b64encode(os.urandom(32)).decode()


def _cipher(key_b64: str = _KEY) -> TokenCipher:
    return TokenCipher(key=base64.b64decode(key_b64))


def test_round_trip():
    """Encrypt then decrypt with the same key and AAD returns the plaintext."""
    c = _cipher()
    token = "refresh-token-abc123"
    envelope = c.encrypt(token, aad="session-1")
    assert token not in envelope  # ciphertext must not leak the plaintext
    assert c.decrypt(envelope, aad="session-1") == token


def test_aad_binding_rejects_wrong_session():
    """A ciphertext bound to one session must not decrypt under another."""
    c = _cipher()
    envelope = c.encrypt("secret", aad="session-1")
    with pytest.raises(TokenEncryptionInvalidData):
        c.decrypt(envelope, aad="session-2")


def test_wrong_key_fails_authentication():
    """A different key cannot decrypt the envelope."""
    envelope = _cipher(_KEY).encrypt("secret", aad="s")
    with pytest.raises(TokenEncryptionInvalidData):
        _cipher(_OTHER_KEY).decrypt(envelope, aad="s")


def test_tampered_ciphertext_is_rejected():
    """Flipping a byte in the ciphertext breaks the GCM tag."""
    c = _cipher()
    envelope = c.encrypt("secret", aad="s")
    tampered = envelope.replace('"ct":"', '"ct":"A')
    with pytest.raises(TokenEncryptionInvalidData):
        c.decrypt(tampered, aad="s")


def test_malformed_envelope_is_rejected():
    """Non-JSON / wrong-version envelopes raise a clear error, not a crash."""
    c = _cipher()
    with pytest.raises(TokenEncryptionInvalidData):
        c.decrypt("not-json", aad="s")
    with pytest.raises(TokenEncryptionInvalidData):
        c.decrypt('{"v":999,"nonce":"x","ct":"y"}', aad="s")


def test_tampered_alg_on_valid_ciphertext_is_rejected():
    """A genuine ciphertext whose alg label is swapped must be refused.

    This isolates the algorithm guard: the ciphertext is otherwise valid, so
    without the alg check it would decrypt successfully (a downgrade).
    """
    import json

    c = _cipher()
    envelope = json.loads(c.encrypt("secret", aad="s"))
    envelope["alg"] = "AES-128-GCM"
    with pytest.raises(TokenEncryptionInvalidData):
        c.decrypt(json.dumps(envelope), aad="s")


@pytest.mark.parametrize(
    "envelope",
    [
        '{"v":1,"nonce":"AAAAAAAAAAAAAAAA","ct":"AAAA"}',  # missing alg
        '{"v":1,"alg":"AES-256-GCM","nonce":123,"ct":"AAAA"}',  # non-string nonce (would TypeError)
        '{"v":1,"alg":"AES-256-GCM","nonce":"AAAAAAAAAAAAAAAA","ct":456}',  # non-string ct (would TypeError)
        '{"v":1,"alg":"AES-256-GCM","nonce":"AAAAAAAAAAAAAAAA"}',  # missing ct
        '{"v":1,"alg":"AES-256-GCM","nonce":"AA","ct":"AAAA"}',  # wrong nonce length
    ],
)
def test_envelope_metadata_is_validated(envelope):
    """Missing/non-string/wrong-length fields are rejected cleanly, not as TypeError."""
    c = _cipher()
    with pytest.raises(TokenEncryptionInvalidData):
        c.decrypt(envelope, aad="s")


def test_load_key_validates_length():
    """A key that does not decode to 32 bytes is rejected."""
    with pytest.raises(TokenEncryptionConfigError):
        load_token_encryption_key(base64.b64encode(os.urandom(16)).decode())


def test_load_key_rejects_non_base64():
    """A non-base64 key string is rejected."""
    with pytest.raises(TokenEncryptionConfigError):
        load_token_encryption_key("!!!not base64!!!")


def test_missing_key_reports_unconfigured(monkeypatch):
    """token_encryption_configured is False when the env var is absent."""
    monkeypatch.delenv("AIQ_TOKEN_ENCRYPTION_KEY", raising=False)
    assert token_encryption_configured() is False
    with pytest.raises(TokenEncryptionConfigError):
        load_token_encryption_key()


def test_env_key_is_used(monkeypatch):
    """A valid env key makes the feature report as configured."""
    monkeypatch.setenv("AIQ_TOKEN_ENCRYPTION_KEY", _KEY)
    assert token_encryption_configured() is True
    c = TokenCipher()
    assert c.decrypt(c.encrypt("t", aad="a"), aad="a") == "t"
