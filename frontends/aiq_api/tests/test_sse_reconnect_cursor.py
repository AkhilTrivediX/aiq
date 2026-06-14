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

"""Tests for the SSE reconnect cursor behaviour in jobs.py.

Background: the historical ``format_sse`` closure incremented a synthetic
``sequence_id`` for every event that did not carry a real DB event id, and
always emitted an ``id:`` line. Browsers persist the last seen ``id:`` as
``EventSource.lastEventId`` and pass it back on reconnect, where the server
uses it as the DB cursor ``WHERE id > :after_id``. The bug was that synthetic
control events (``stream.mode``, ``job.status``, ``job.shutdown``,
``job.error``) emitted incrementing ids that could overshoot the next real DB
event's ``_id``, causing the reconnect query to silently skip real events.

The fix moves ``format_sse`` to module level and only emits the ``id:`` line
when an event has a real DB-backed event id; synthetic events emit only
``event:`` and ``data:`` lines so the browser's ``lastEventId`` stays anchored
to the last persisted event.
"""

from __future__ import annotations

import json
import re
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from aiq_api.routes.jobs import _format_sse
from aiq_api.routes.jobs import _resolve_start_event_id


def _parse_sse_id(frame: str) -> int | None:
    """Return the integer value of the ``id:`` line in an SSE frame, or None.

    Anchored to start-of-line (``re.MULTILINE``) so JSON payloads that happen
    to contain the substring ``id:`` (for example ``{"id": 1}`` -> ``"id":1``,
    or any ``"x_id":...`` key) cannot match.
    """
    match = re.search(r"^id:\s*(\d+)\s*$", frame, flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def _last_emitted_id(frames: list[str]) -> int | None:
    """Walk frames in order and return the id the browser would persist.

    Models the WHATWG EventSource behaviour: the "last event ID" buffer is
    updated only by an ``id:`` field; frames without one leave it unchanged.
    """
    last_id: int | None = None
    for frame in frames:
        seen = _parse_sse_id(frame)
        if seen is not None:
            last_id = seen
    return last_id


class TestFormatSseBasics:
    """Basic shape of frames emitted by _format_sse."""

    def test_real_event_emits_id_line(self) -> None:
        """A real DB-backed event emits an id: line carrying its event id."""
        frame = _format_sse("job.event", {"value": 1}, event_id=42)
        assert _parse_sse_id(frame) == 42
        assert "event: job.event" in frame
        # Payload round-trips through json.loads on the data: line.
        data_line = re.search(r"^data:\s*(.+)$", frame, flags=re.MULTILINE)
        assert data_line is not None
        assert json.loads(data_line.group(1)) == {"value": 1}

    def test_synthetic_event_omits_id_line(self) -> None:
        """A synthetic event omits the id: line entirely."""
        # Use an event_id substring in the payload to catch any substring-coupled
        # assertion regression.
        frame = _format_sse("stream.mode", {"mode": "pubsub", "id": 1}, event_id=None)
        assert _parse_sse_id(frame) is None, "synthetic events must not emit an id: line"
        assert "event: stream.mode" in frame

    def test_event_id_zero_is_treated_as_real(self) -> None:
        """event_id 0 is a real cursor, not the synthetic (None) sentinel."""
        # event_id 0 is a legitimate first DB row in some stores; the only
        # 'no real id' sentinel is None. Tests against any future regression
        # to ``if not event_id`` truthy-check semantics.
        frame = _format_sse("job.event", {"x": 0}, event_id=0)
        assert _parse_sse_id(frame) == 0

    def test_default_event_id_is_synthetic(self) -> None:
        """Omitting event_id yields the synthetic shape with no id: line."""
        # Callers that omit event_id (job.error, job.shutdown, ...) get the
        # synthetic shape - no id: line. Also smuggles "id: 99" inside the
        # payload to make sure the parser only sees real headers.
        frame = _format_sse("job.shutdown", {"message": "bye", "id_hint": 99})
        assert _parse_sse_id(frame) is None
        assert "event: job.shutdown" in frame

    @pytest.mark.parametrize(
        "frame",
        [
            _format_sse("job.event", {}, event_id=1),
            _format_sse("stream.mode", {}),
        ],
        ids=["real", "synthetic"],
    )
    def test_frame_terminator(self, frame: str) -> None:
        """Every SSE frame ends with a blank-line delimiter."""
        # SSE frames must end with a blank line; the browser uses it to
        # delimit events.
        assert frame.endswith("\n\n")

    def test_data_is_json_encoded(self) -> None:
        """The data: payload round-trips through json.loads."""
        frame = _format_sse("job.event", {"q": 'hello "world"', "n": 7}, event_id=9)
        data_line = re.search(r"^data:\s*(.+)$", frame, flags=re.MULTILINE)
        assert data_line is not None
        assert json.loads(data_line.group(1)) == {"q": 'hello "world"', "n": 7}


class TestSseReconnectCursor:
    """End-to-end semantic test for the reconnect cursor.

    Simulates the exact flow that produced the bug: a sequence of real DB
    events interleaved with synthetic control events, followed by a client
    disconnect after a synthetic event. The client's persisted lastEventId
    must equal the last real DB event id, never a synthetic placeholder.
    """

    @pytest.mark.parametrize(
        "synth_type, synth_data, real_id",
        [
            ("stream.mode", {"mode": "pubsub", "channel": "job_events_x"}, 102),
            ("job.status", {"status": "RUNNING"}, 200),
            ("job.shutdown", {"message": "Server shutting down"}, 500),
            ("job.error", {"error": "Internal server error"}, 10),
        ],
        ids=["stream.mode", "job.status", "job.shutdown", "job.error"],
    )
    def test_disconnect_after_synthetic_keeps_real_cursor(
        self,
        synth_type: str,
        synth_data: dict,
        real_id: int,
    ) -> None:
        """After a synthetic control frame, lastEventId stays on the last real DB id."""
        # Real DB event id ``real_id`` is the last persisted event; the
        # browser then receives a synthetic control frame and disconnects.
        # The reconnect endpoint expects ``last_event_id == real_id``.
        frames = [
            _format_sse("job.event", {"step": "a"}, event_id=real_id),
            _format_sse(synth_type, synth_data),
        ]
        cursor = _last_emitted_id(frames)
        assert cursor == real_id, (
            f"after a {synth_type} synthetic frame, browser lastEventId "
            f"must equal the last real DB event id ({real_id}), got {cursor}"
        )

    def test_synthetic_only_stream_leaves_cursor_unanchored(self) -> None:
        """A synthetic-only stream leaves lastEventId unset (resume from start)."""
        # If a stream emits only synthetic events (e.g., job-not-found error
        # before any historical event), the client's lastEventId must remain
        # unset so the reconnect query starts from after_id=0 - not from a
        # synthetic id that would skip the entire job history.
        frames = [
            _format_sse("stream.mode", {"mode": "pubsub"}),
            _format_sse("job.error", {"error": "Job not found"}),
        ]
        cursor = _last_emitted_id(frames)
        assert cursor is None, (
            f"synthetic-only stream must leave lastEventId unset; got {cursor} "
            f"which would cause the reconnect endpoint to query WHERE id > {cursor} "
            f"and skip the entire job history"
        )

    def test_interleaved_real_and_synthetic_keeps_real_cursor(self) -> None:
        """Interleaved frames advance the cursor only on real events."""
        # Real DB events interleaved with synthetic events - the cursor must
        # only advance on real ones, regardless of order.
        frames = [
            _format_sse("job.event", {"step": "a"}, event_id=300),
            _format_sse("stream.mode", {"mode": "pubsub"}),
            _format_sse("job.event", {"step": "b"}, event_id=301),
            _format_sse("job.status", {"status": "RUNNING"}),
            _format_sse("job.event", {"step": "c"}, event_id=302),
            _format_sse("job.shutdown", {"message": "bye"}),
        ]
        cursor = _last_emitted_id(frames)
        assert cursor == 302, (
            f"interleaved real+synthetic frames must leave lastEventId on the last real DB id (302), got {cursor}"
        )


class TestOldBuggyFormatterRegressionGuard:
    """Pin the bug as a permanent guard against regressing the closure pattern.

    Recreates the pre-fix ``format_sse`` closure semantics (nonlocal
    ``sequence_id`` incremented for synthetic events) and demonstrates that it
    WOULD have polluted the reconnect cursor on the exact scenarios the fix
    addresses. If a future refactor accidentally restores the bug, the same
    fixture used here will catch it.
    """

    @staticmethod
    def _old_buggy_formatter(start_event_id: int = 0):
        """Re-create the old closure verbatim (pre-fix behaviour)."""
        sequence_id = start_event_id

        def format_sse(event_type: str, data: dict, event_id: int | None = None) -> str:
            """Old buggy formatter: always emits id:, incrementing for synthetic frames."""
            nonlocal sequence_id
            if event_id is not None:
                sequence_id = event_id
            else:
                sequence_id += 1
            return f"id: {sequence_id}\nevent: {event_type}\ndata: {json.dumps(data)}\n\n"

        return format_sse

    def test_old_formatter_polluted_cursor_after_stream_mode(self) -> None:
        """The old formatter advanced the cursor past the last real id (the bug)."""
        old = self._old_buggy_formatter()
        frames = [
            old("job.event", {"step": "a"}, event_id=100),
            old("job.event", {"step": "b"}, event_id=101),
            old("job.event", {"step": "c"}, event_id=102),
            old("stream.mode", {"mode": "pubsub"}),
        ]
        bug_cursor = _last_emitted_id(frames)
        # The bug: synthetic stream.mode emits id: 103, which is >= the next
        # real DB row id (also 103 in production). On reconnect the WHERE
        # id > 103 clause would skip that next real row.
        assert bug_cursor == 103, (
            "old formatter must have advanced past last real id (this assertion "
            "documents the bug that the fix prevents)"
        )

        # And the current fix avoids that exact pollution.
        fixed = [
            _format_sse("job.event", {"step": "a"}, event_id=100),
            _format_sse("job.event", {"step": "b"}, event_id=101),
            _format_sse("job.event", {"step": "c"}, event_id=102),
            _format_sse("stream.mode", {"mode": "pubsub"}),
        ]
        fixed_cursor = _last_emitted_id(fixed)
        assert fixed_cursor == 102, (
            f"current _format_sse must leave cursor anchored to last real DB id (102), got {fixed_cursor}"
        )

    @pytest.mark.parametrize(
        "synth_type, synth_data",
        [
            ("stream.mode", {"mode": "pubsub"}),
            ("job.status", {"status": "RUNNING"}),
            ("job.shutdown", {"message": "bye"}),
            ("job.error", {"error": "boom"}),
        ],
        ids=["stream.mode", "job.status", "job.shutdown", "job.error"],
    )
    def test_old_formatter_polluted_for_every_synthetic_type(self, synth_type: str, synth_data: dict) -> None:
        """The old formatter polluted the cursor for every synthetic event type."""
        # The bug applied uniformly across every synthetic event type.
        old = self._old_buggy_formatter()
        frames = [
            old("job.event", {"step": "a"}, event_id=500),
            old(synth_type, synth_data),
        ]
        assert _last_emitted_id(frames) == 501, (
            f"old formatter advanced cursor past 500 on {synth_type}; "
            f"the new _format_sse leaves it at 500 (verified above)"
        )


class TestGeneratorLevelReconnect:
    """Drive the real ``_sse_generator_polling`` with fakes and verify no real
    DB event is silently skipped after a synthetic-frame disconnect+reconnect.

    This catches call-site bugs that helper-level tests cannot reach: if any
    yield site in the generator accidentally passes a synthetic ``event_id``
    or wires the cursor incorrectly, the simulated client will see an event
    gap on reconnect.
    """

    @staticmethod
    def _make_fake_event_store(events: list[dict]):
        """Return a fake EventStore module with cursor-respecting fetch."""

        async def get_events_async(db_url, job_id, after_id, limit):
            """Fake EventStore.get_events_async honoring WHERE id > after_id."""
            # Mirror the real WHERE id > :after_id ORDER BY id semantics.
            matching = [e for e in events if e["_id"] > after_id]
            return [dict(e) for e in matching[:limit]]

        async def get_event_by_id_async(db_url, event_id):
            """Fake EventStore.get_event_by_id_async returning a row by exact id."""
            for e in events:
                if e["_id"] == event_id:
                    return dict(e)
            return None

        fake = MagicMock()
        fake.is_postgres = MagicMock(return_value=False)
        fake.get_events_async = AsyncMock(side_effect=get_events_async)
        fake.get_event_by_id_async = AsyncMock(side_effect=get_event_by_id_async)
        return fake

    @staticmethod
    def _make_fake_job_store(status: str):
        """Build a fake job store whose job reports the given status."""
        job = MagicMock()
        job.status = status
        job.error = None
        store = MagicMock()
        store.get_job = AsyncMock(return_value=job)
        return store

    @staticmethod
    def _make_fake_connection_manager():
        """Build a fake SSE connection manager that never signals shutdown."""
        cm = MagicMock()
        cm.is_shutting_down = False

        class _Tracker:
            async def __aenter__(self_inner):
                """Enter the no-op async connection-tracking context."""
                return self_inner

            async def __aexit__(self_inner, exc_type, exc, tb):
                """Exit the no-op async connection-tracking context."""
                return False

        cm.track_connection = MagicMock(return_value=_Tracker())
        cm.wait_or_shutdown = AsyncMock(return_value=False)
        return cm

    @staticmethod
    async def _drain(generator, *, max_frames: int = 100) -> list[str]:
        """Consume an SSE generator with a hard frame cap.

        The polling generator only terminates when the job reaches a terminal
        status. The cap is a test-only safety net: if a future change breaks
        that exit condition the test fails fast instead of hanging CI.
        """
        frames: list[str] = []
        async for frame in generator:
            frames.append(frame)
            if len(frames) > max_frames:
                raise AssertionError(
                    f"SSE generator did not terminate within {max_frames} frames - terminal-status break likely broken"
                )
        return frames

    @pytest.mark.asyncio
    async def test_polling_reconnect_after_disconnect_skips_no_events(self) -> None:
        """A polling reconnect cycle re-emits no real event and skips none."""
        # 5 real DB events. Client receives them all plus the polling
        # generator's synthetic frames, then disconnects after the last
        # synthetic frame. On reconnect, the server queries from the
        # browser's lastEventId; we assert every remaining real event is
        # emitted across both passes combined (i.e. no skip).
        from nat.front_ends.fastapi.async_jobs.job_store import JobStatus

        events = [{"_id": i, "type": "job.event", "step": chr(ord("a") + i - 1)} for i in range(1, 6)]

        fake_store = self._make_fake_event_store(events)
        # Use the real terminal-status value so the generator actually exits.
        fake_job_store = self._make_fake_job_store(status=JobStatus.SUCCESS.value)
        fake_cm = self._make_fake_connection_manager()

        from aiq_api.routes.jobs import _sse_generator_polling

        with (
            patch(
                "aiq_api.jobs.event_store.EventStore.get_events_async",
                AsyncMock(side_effect=fake_store.get_events_async),
            ),
            patch(
                "aiq_api.jobs.event_store.EventStore.get_event_by_id_async",
                AsyncMock(side_effect=fake_store.get_event_by_id_async),
            ),
            patch(
                "aiq_api.jobs.event_store.EventStore.is_postgres",
                MagicMock(return_value=False),
            ),
            patch(
                "aiq_api.jobs.connection_manager.get_connection_manager",
                return_value=fake_cm,
            ),
        ):
            # First connection: drain all frames until the terminal status
            # closes the generator.
            first_frames = await self._drain(
                _sse_generator_polling(fake_job_store, "job-x", "sqlite:///:memory:", start_event_id=0)
            )

            first_cursor = _last_emitted_id(first_frames)
            # All 5 real events should have been delivered in the first pass;
            # the cursor must point at the last real id (5), not past it.
            assert first_cursor == 5, (
                f"after first pass, browser lastEventId must equal the last real DB id (5); got {first_cursor}"
            )

            # Reconnect with the cursor we observed. Because no new DB events
            # arrived, the second pass should emit no real events; in
            # particular no real event must be re-emitted or skipped.
            second_frames = await self._drain(
                _sse_generator_polling(fake_job_store, "job-x", "sqlite:///:memory:", start_event_id=first_cursor)
            )

        # Across both passes, every real DB event appears exactly once.
        all_real_ids = [_parse_sse_id(f) for f in first_frames + second_frames]
        real_ids = [i for i in all_real_ids if i is not None]
        assert real_ids == [1, 2, 3, 4, 5], f"real DB events must appear exactly once across reconnect; got {real_ids}"


class TestResolveStartEventId:
    """The reconnect-cursor resolver: path param vs Last-Event-ID header.

    Native EventSource sends the cursor in the Last-Event-ID header on auto
    reconnect; clients that rebuild the URL pass it as a path value. The
    resolver honors both and treats the value as untrusted (non-int/negative
    fall back to 0 = replay from start).
    """

    def test_fresh_connection_no_path_no_header(self):
        """No path and no header resolves to a fresh start (cursor 0)."""
        assert _resolve_start_event_id(None, None) == 0

    def test_header_used_when_no_path(self):
        """The Last-Event-ID header is used when no path cursor is given."""
        assert _resolve_start_event_id(None, "42") == 42

    def test_header_whitespace_tolerated(self):
        """Surrounding whitespace in the header value is tolerated."""
        assert _resolve_start_event_id(None, "  7 ") == 7

    def test_path_value_preferred_over_header(self):
        """An explicit path cursor takes precedence over the header."""
        assert _resolve_start_event_id("10", "42") == 10

    def test_path_zero_is_fresh(self):
        """A path cursor of 0 means start from the beginning."""
        assert _resolve_start_event_id("0", None) == 0

    def test_negative_path_clamped_to_zero(self):
        """A negative path cursor is clamped to 0."""
        assert _resolve_start_event_id("-5", None) == 0

    def test_garbage_path_falls_back_to_zero(self):
        """A non-integer path segment degrades to 0 instead of raising 422.

        The path parameter is declared as a string in the route so a malformed
        cursor reaches this resolver and replays from the beginning, matching the
        documented graceful-degradation contract.
        """
        assert _resolve_start_event_id("not-an-int", None) == 0

    def test_path_whitespace_tolerated(self):
        """Surrounding whitespace in the path value is tolerated."""
        assert _resolve_start_event_id("  9 ", None) == 9

    def test_garbage_header_falls_back_to_zero(self):
        """A non-integer header value falls back to 0."""
        assert _resolve_start_event_id(None, "not-an-int") == 0

    def test_negative_header_clamped_to_zero(self):
        """A negative header value is clamped to 0."""
        assert _resolve_start_event_id(None, "-3") == 0

    def test_empty_header_is_fresh(self):
        """An empty header value (the reset case) resolves to a fresh start."""
        assert _resolve_start_event_id(None, "") == 0

    def test_empty_path_is_fresh(self):
        """An empty path value resolves to a fresh start."""
        assert _resolve_start_event_id("", None) == 0


class TestGeneratorIdEdgeCases:
    """Generator behavior when a DB row's _id is missing or zero.

    A missing _id must not emit an id: line (so the browser cursor can't advance
    past a row whose id we lost); these are the edge cases flagged in review.
    """

    def test_missing_id_emits_no_id_line(self):
        """A missing _id (event_id None) emits no id: line."""
        # _format_sse with event_id=None (what event.pop("_id", None) yields when
        # a row lacks _id) must omit the id: line entirely.
        frame = _format_sse("job.event", {"step": "x"}, event_id=None)
        assert _parse_sse_id(frame) is None

    def test_id_zero_is_emitted_as_real(self):
        """A real _id of 0 is emitted as id: 0."""
        # A legitimate _id of 0 is a real cursor, not the synthetic sentinel.
        frame = _format_sse("job.event", {"step": "x"}, event_id=0)
        assert _parse_sse_id(frame) == 0
