import sqlite3
from datetime import datetime, timezone

import pytest

from factories import USER
from cedar.contracts import RecordedEvent, RoundStarted
from cedar.store import ConcurrencyConflict, SQLiteEventStore


def event(version: int, investigation_id: str = "inv-1", key: str | None = None) -> RecordedEvent:
    return RecordedEvent(
        event_id=f"evt-{investigation_id}-{version}",
        investigation_id=investigation_id,
        stream_version=version,
        occurred_at=datetime(2026, 10, 21, tzinfo=timezone.utc),
        actor=USER,
        round=0,
        plan_version=1,
        idempotency_key=key,
        payload=RoundStarted(round=version, reason="synthetic"),
    )


def test_events_round_trip_and_survive_reopening(store, tmp_path):
    store.append([event(1), event(2)], expected_version=0)
    reopened = SQLiteEventStore(tmp_path / "events.sqlite")
    assert reopened.read_stream("inv-1") == [event(1), event(2)]
    assert reopened.read_stream("inv-1", after_version=1) == [event(2)]
    assert reopened.current_version("inv-1") == 2


def test_stale_writer_is_rejected(store):
    store.append([event(1)], expected_version=0)
    with pytest.raises(ConcurrencyConflict):
        store.append([event(1).model_copy(update={"event_id": "other"})], expected_version=0)
    assert store.current_version("inv-1") == 1


def test_versions_must_be_contiguous(store):
    with pytest.raises(ValueError, match="contiguous"):
        store.append([event(2)], expected_version=0)


def test_batch_is_atomic(store):
    store.append([event(1, key="k")], expected_version=0)
    with pytest.raises(ConcurrencyConflict):
        store.append([event(2), event(3, key="k")], expected_version=1)
    assert store.current_version("inv-1") == 1


def test_events_cannot_be_updated_or_deleted(store):
    store.append([event(1)], expected_version=0)
    conn = sqlite3.connect(store.path)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("UPDATE events SET event_type = 'x'")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("DELETE FROM events")
    finally:
        conn.close()


def test_streams_are_independent_and_globally_ordered(store):
    store.append([event(1, "inv-a")], expected_version=0)
    store.append([event(1, "inv-b")], expected_version=0)
    store.append([event(2, "inv-a")], expected_version=1)
    assert store.investigation_ids() == ["inv-a", "inv-b"]
    seqs, events = zip(*store.read_all())
    assert [(e.investigation_id, e.stream_version) for e in events] == [("inv-a", 1), ("inv-b", 1), ("inv-a", 2)]
    assert [e.investigation_id for _, e in store.read_all(after_seq=seqs[0])] == ["inv-b", "inv-a"]


def test_lookup_by_idempotency_key(store):
    store.append([event(1, key="slurm:1001:RUNNING"), event(2)], expected_version=0)
    found = store.find_by_idempotency_keys("inv-1", ["slurm:1001:RUNNING", "missing"])
    assert list(found) == ["slurm:1001:RUNNING"] and found["slurm:1001:RUNNING"] == event(1, key="slurm:1001:RUNNING")
    assert store.find_by_idempotency_keys("inv-2", ["slurm:1001:RUNNING"]) == {}
