"""Append-only SQLite event store.

One stream per investigation. ``stream_version`` numbers each stream's events
contiguously, and appends state the version they expect (optimistic concurrency), so two
writers acting on the same stale state cannot both succeed. Triggers forbid UPDATE and
DELETE.

The default rollback journal works on network filesystems such as NFS; WAL mode is
faster but needs shared memory and should only be enabled on local disks.
"""

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from cedar.contracts.events import RecordedEvent, content_hash

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    global_seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id         TEXT NOT NULL UNIQUE,
    investigation_id TEXT NOT NULL,
    stream_version   INTEGER NOT NULL,
    event_type       TEXT NOT NULL,
    idempotency_key  TEXT,
    content_hash     TEXT NOT NULL,
    occurred_at      TEXT NOT NULL,
    envelope         TEXT NOT NULL,
    UNIQUE (investigation_id, stream_version),
    UNIQUE (investigation_id, idempotency_key)
);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
PRAGMA user_version = 1;
"""


class ConcurrencyConflict(RuntimeError):
    """The stream changed since the caller read it; reload and retry."""


class IdempotencyConflict(RuntimeError):
    """An idempotency key was reused for different content."""


class SQLiteEventStore:
    def __init__(self, path: str | Path, *, wal: bool = False, timeout: float = 30.0) -> None:
        self.path = Path(path)
        self._timeout = timeout
        conn = self._connect()
        try:
            if wal:
                conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(_SCHEMA)
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=self._timeout, isolation_level=None)

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")
        finally:
            conn.close()

    def append(self, events: Sequence[RecordedEvent], *, expected_version: int) -> None:
        """Append events atomically if the stream is still at ``expected_version``."""
        if not events:
            raise ValueError("nothing to append")
        investigation_id = events[0].investigation_id
        for offset, event in enumerate(events, start=1):
            if event.investigation_id != investigation_id:
                raise ValueError("all events in one append must belong to the same investigation")
            if event.stream_version != expected_version + offset:
                raise ValueError("stream versions must continue contiguously from expected_version")

        with self._write_transaction() as conn:
            current = self._current_version(conn, investigation_id)
            if current != expected_version:
                raise ConcurrencyConflict(
                    f"{investigation_id}: expected version {expected_version}, found {current}"
                )
            try:
                conn.executemany(
                    "INSERT INTO events (event_id, investigation_id, stream_version, event_type,"
                    " idempotency_key, content_hash, occurred_at, envelope)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            e.event_id,
                            e.investigation_id,
                            e.stream_version,
                            e.type,
                            e.idempotency_key,
                            content_hash(e.actor, e.payload),
                            e.occurred_at.isoformat(),
                            e.model_dump_json(),
                        )
                        for e in events
                    ],
                )
            except sqlite3.IntegrityError as exc:
                # A concurrent writer recorded the same idempotency key or event id.
                raise ConcurrencyConflict(str(exc)) from exc

    def current_version(self, investigation_id: str) -> int:
        conn = self._connect()
        try:
            return self._current_version(conn, investigation_id)
        finally:
            conn.close()

    @staticmethod
    def _current_version(conn: sqlite3.Connection, investigation_id: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(stream_version), 0) FROM events WHERE investigation_id = ?",
            (investigation_id,),
        ).fetchone()
        return row[0]

    def read_stream(self, investigation_id: str, *, after_version: int = 0) -> list[RecordedEvent]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT envelope FROM events WHERE investigation_id = ? AND stream_version > ?"
                " ORDER BY stream_version",
                (investigation_id, after_version),
            ).fetchall()
        finally:
            conn.close()
        return [RecordedEvent.model_validate_json(row[0]) for row in rows]

    def read_all(self, *, after_seq: int = 0, limit: int = 1000) -> list[tuple[int, RecordedEvent]]:
        """Events across all investigations in commit order, for tailing by monitors and UIs."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT global_seq, envelope FROM events WHERE global_seq > ? ORDER BY global_seq LIMIT ?",
                (after_seq, limit),
            ).fetchall()
        finally:
            conn.close()
        return [(seq, RecordedEvent.model_validate_json(envelope)) for seq, envelope in rows]

    def find_by_idempotency_keys(
        self, investigation_id: str, keys: Sequence[str]
    ) -> dict[str, RecordedEvent]:
        if not keys:
            return {}
        placeholders = ", ".join("?" for _ in keys)
        conn = self._connect()
        try:
            rows = conn.execute(
                f"SELECT idempotency_key, envelope FROM events"
                f" WHERE investigation_id = ? AND idempotency_key IN ({placeholders})",
                (investigation_id, *keys),
            ).fetchall()
        finally:
            conn.close()
        return {key: RecordedEvent.model_validate_json(envelope) for key, envelope in rows}

    def investigation_ids(self) -> list[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT investigation_id FROM events GROUP BY investigation_id ORDER BY MIN(global_seq)"
            ).fetchall()
        finally:
            conn.close()
        return [row[0] for row in rows]
