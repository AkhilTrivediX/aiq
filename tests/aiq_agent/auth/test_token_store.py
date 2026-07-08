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

"""Tests for the SQL-backed TokenStore. Uses a real temp SQLite DB."""

from __future__ import annotations

import base64
import itertools
import os
import tempfile
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy import text

from aiq_agent.auth.token_cipher import TokenCipher
from aiq_agent.auth.token_store import SqlTokenStore
from aiq_agent.auth.token_store import StoredToken
from aiq_agent.auth.token_store import get_token_store
from aiq_agent.auth.token_store import reset_token_store_cache_for_tests

_KEY_B64 = base64.b64encode(os.urandom(32)).decode()

# mkdtemp creates a 0700 directory atomically (unlike the TOCTOU-prone mktemp);
# unique filenames keep each store's DB isolated.
_TMPDIR = tempfile.mkdtemp(prefix="aiq_token_store_test_")
_counter = itertools.count()


def _new_db_url() -> str:
    return f"sqlite:///{Path(_TMPDIR) / f'tokens_{next(_counter)}.db'}"


@pytest.fixture(autouse=True)
def _clear_store_cache():
    """Reset the module-level store cache/flags around every test."""
    reset_token_store_cache_for_tests()
    yield
    reset_token_store_cache_for_tests()


def _store() -> tuple[SqlTokenStore, str]:
    db_url = _new_db_url()
    cipher = TokenCipher(key=base64.b64decode(_KEY_B64))
    return SqlTokenStore(db_url, cipher=cipher), db_url


def _record(session_id: str = "sess-1", refresh_token: str = "refresh-xyz") -> StoredToken:
    return StoredToken(
        session_id=session_id,
        user_sub="user-123",
        refresh_token=refresh_token,
        id_token_expires_at=datetime.now(UTC) + timedelta(minutes=5),
        refresh_token_expires_at=datetime.now(UTC) + timedelta(days=30),
    )


@pytest.mark.asyncio
async def test_put_then_get_round_trip():
    """A stored record round-trips with its refresh token intact."""
    store, _ = _store()
    await store.put(_record())
    got = await store.get("sess-1")
    assert got is not None
    assert got.user_sub == "user-123"
    assert got.refresh_token == "refresh-xyz"
    assert got.id_token_expires_at is not None
    assert got.refresh_token_expires_at is not None


@pytest.mark.asyncio
async def test_refresh_token_is_encrypted_at_rest():
    """The raw DB column must not contain the plaintext refresh token."""
    store, db_url = _store()
    await store.put(_record(refresh_token="SUPER-SECRET-TOKEN"))
    engine = create_engine(db_url)
    with engine.connect() as conn:
        raw = conn.execute(text("SELECT refresh_token_encrypted FROM auth_token_store")).scalar_one()
    assert "SUPER-SECRET-TOKEN" not in raw
    assert raw.strip().startswith("{")  # it's an encryption envelope


@pytest.mark.asyncio
async def test_put_is_idempotent_replace():
    """Putting the same session id twice updates rather than duplicates."""
    store, db_url = _store()
    await store.put(_record(refresh_token="first"))
    await store.put(_record(refresh_token="second"))
    got = await store.get("sess-1")
    assert got.refresh_token == "second"
    engine = create_engine(db_url)
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM auth_token_store")).scalar_one()
    assert count == 1


@pytest.mark.asyncio
async def test_get_missing_returns_none():
    """An unknown session id returns None, not an error."""
    store, _ = _store()
    assert await store.get("nope") is None


@pytest.mark.asyncio
async def test_delete_removes_record():
    """Delete removes the row; a second delete is a no-op."""
    store, _ = _store()
    await store.put(_record())
    await store.delete("sess-1")
    assert await store.get("sess-1") is None
    await store.delete("sess-1")  # idempotent


@pytest.mark.asyncio
async def test_records_are_isolated_by_session():
    """Two sessions keep independent tokens, each AAD-bound to its id."""
    store, _ = _store()
    await store.put(_record(session_id="a", refresh_token="tok-a"))
    await store.put(_record(session_id="b", refresh_token="tok-b"))
    assert (await store.get("a")).refresh_token == "tok-a"
    assert (await store.get("b")).refresh_token == "tok-b"


@pytest.mark.asyncio
async def test_swapped_ciphertext_rows_fail_to_decrypt():
    """AAD binding: moving one session's ciphertext onto another row must fail.

    Each refresh token is encrypted with its session_id as AAD, so an attacker
    who swaps the encrypted blobs between two DB rows cannot make either decrypt
    as the other session — the read raises rather than returning a wrong token.
    """
    from aiq_agent.auth.token_cipher import TokenEncryptionInvalidData

    store, db_url = _store()
    await store.put(_record(session_id="a", refresh_token="tok-a"))
    await store.put(_record(session_id="b", refresh_token="tok-b"))

    engine = create_engine(db_url)
    with engine.begin() as conn:
        blob_a = conn.execute(
            text("SELECT refresh_token_encrypted FROM auth_token_store WHERE session_id='a'")
        ).scalar_one()
        # Overwrite session b's ciphertext with session a's (a swap/copy attack).
        conn.execute(
            text("UPDATE auth_token_store SET refresh_token_encrypted=:v WHERE session_id='b'"),
            {"v": blob_a},
        )

    with pytest.raises(TokenEncryptionInvalidData):
        await store.get("b")


def test_factory_disabled_without_key(monkeypatch):
    """get_token_store returns None (feature off) when no encryption key is set."""
    monkeypatch.delenv("AIQ_TOKEN_ENCRYPTION_KEY", raising=False)
    assert get_token_store() is None


def test_factory_enabled_with_key(monkeypatch):
    """A configured key yields a usable SqlTokenStore."""
    monkeypatch.setenv("AIQ_TOKEN_ENCRYPTION_KEY", _KEY_B64)
    store = get_token_store(_new_db_url())
    assert isinstance(store, SqlTokenStore)


def test_factory_fails_loudly_on_invalid_key(monkeypatch):
    """A key that is set but malformed must raise, not silently disable."""
    from aiq_agent.auth.token_cipher import TokenEncryptionConfigError

    monkeypatch.setenv("AIQ_TOKEN_ENCRYPTION_KEY", "not-valid-base64!!!")
    with pytest.raises(TokenEncryptionConfigError):
        get_token_store(_new_db_url())


def test_factory_caches_store_per_db_url(monkeypatch):
    """Repeat calls for the same db_url return the same cached store/engine."""
    monkeypatch.setenv("AIQ_TOKEN_ENCRYPTION_KEY", _KEY_B64)
    url = _new_db_url()
    first = get_token_store(url)
    second = get_token_store(url)
    assert first is second
