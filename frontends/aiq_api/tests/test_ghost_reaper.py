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

"""Tests for the ghost-job reaper's stale-job detection (_find_stale_jobs).

These use a real SQLite database so the SQL runs exactly as it would in
production. The key regression: a job that entered RUNNING but never stored an
event (worker crash/OOM before the first event flush) must still be reaped;
before the fix, the INNER JOIN on job_events made such jobs invisible.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aiq_api.routes.jobs import GHOST_JOB_TIMEOUT_SECONDS  # noqa: E402
from aiq_api.routes.jobs import _find_stale_jobs  # noqa: E402

RUNNING = "running"


def _make_db() -> str:
    """Create a temp SQLite DB with the job_info and job_events tables."""
    db_path = tempfile.mktemp(suffix=".db")
    db_url = f"sqlite:///{db_path}"

    # job_info comes from NAT's ORM model.
    from nat.front_ends.fastapi.async_jobs.job_store import JobInfo

    engine = create_engine(db_url)
    JobInfo.__table__.metadata.create_all(engine)

    # job_events is created by aiq's EventStore.
    from aiq_api.jobs.event_store import EventStore

    EventStore._ensure_table_exists(db_url)
    return db_url


def _insert_job(db_url: str, job_id: str, *, status: str, updated_ago_seconds: float) -> None:
    """Insert a job_info row with updated_at set to now minus the given age."""
    from nat.front_ends.fastapi.async_jobs.job_store import JobInfo

    ts = datetime.now(UTC) - timedelta(seconds=updated_ago_seconds)
    engine = create_engine(db_url)
    with Session(engine) as s:
        s.add(JobInfo(job_id=job_id, status=status, expiry_seconds=3600, created_at=ts, updated_at=ts))
        s.commit()


def _insert_event(db_url: str, job_id: str, *, created_ago_seconds: float) -> None:
    """Insert a job_events row with created_at set to now minus the given age."""
    ts = (datetime.now(UTC) - timedelta(seconds=created_ago_seconds)).strftime("%Y-%m-%d %H:%M:%S")
    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO job_events (job_id, event_type, event_data, created_at) VALUES (:j, :t, :d, :c)"),
            {"j": job_id, "t": "test.event", "d": "{}", "c": ts},
        )


OLD = GHOST_JOB_TIMEOUT_SECONDS + 60
RECENT = 5


def test_zero_event_running_job_past_timeout_is_reaped():
    """A RUNNING job with no events, started long ago, is a ghost and reaped.

    This is the regression: the old INNER JOIN made zero-event jobs invisible.
    """
    db = _make_db()
    _insert_job(db, "ghost", status=RUNNING, updated_ago_seconds=OLD)
    assert _find_stale_jobs(db, RUNNING) == ["ghost"]


def test_zero_event_running_job_within_timeout_is_not_reaped():
    """A freshly-started RUNNING job with no events yet must not be reaped."""
    db = _make_db()
    _insert_job(db, "fresh", status=RUNNING, updated_ago_seconds=RECENT)
    assert _find_stale_jobs(db, RUNNING) == []


def test_running_job_with_stale_last_event_is_reaped():
    """Existing behavior preserved: events present but last one is old."""
    db = _make_db()
    _insert_job(db, "stalled", status=RUNNING, updated_ago_seconds=OLD)
    _insert_event(db, "stalled", created_ago_seconds=OLD)
    assert _find_stale_jobs(db, RUNNING) == ["stalled"]


def test_running_job_with_recent_event_is_not_reaped():
    """A job actively emitting events is healthy, even if it started long ago."""
    db = _make_db()
    _insert_job(db, "active", status=RUNNING, updated_ago_seconds=OLD)
    _insert_event(db, "active", created_ago_seconds=RECENT)
    assert _find_stale_jobs(db, RUNNING) == []


def test_non_running_job_is_never_reaped():
    """Only RUNNING jobs are candidates; a completed job is left alone."""
    db = _make_db()
    _insert_job(db, "done", status="success", updated_ago_seconds=OLD)
    assert _find_stale_jobs(db, RUNNING) == []


def test_missing_tables_returns_empty():
    """No job_info/job_events tables (fresh deployment) -> nothing to reap."""
    db_path = tempfile.mktemp(suffix=".db")
    assert _find_stale_jobs(f"sqlite:///{db_path}", RUNNING) == []


def test_mixed_fleet_reaps_only_ghosts():
    """A realistic mix: only the two ghosts (old zero-event + stalled) return."""
    db = _make_db()
    _insert_job(db, "ghost-zero", status=RUNNING, updated_ago_seconds=OLD)
    _insert_job(db, "fresh-zero", status=RUNNING, updated_ago_seconds=RECENT)
    _insert_job(db, "stalled", status=RUNNING, updated_ago_seconds=OLD)
    _insert_event(db, "stalled", created_ago_seconds=OLD)
    _insert_job(db, "healthy", status=RUNNING, updated_ago_seconds=OLD)
    _insert_event(db, "healthy", created_ago_seconds=RECENT)
    _insert_job(db, "done", status="success", updated_ago_seconds=OLD)

    assert sorted(_find_stale_jobs(db, RUNNING)) == ["ghost-zero", "stalled"]
