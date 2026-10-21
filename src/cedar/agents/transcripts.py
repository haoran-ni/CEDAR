"""Append-only conversation transcripts, one JSON line per message, per agent session.

A transcript is the operational memory an agent resumes from after waiting for a job.
It is never rewritten: some providers require earlier turns to come back exactly as they
were. It is not the research record; that is the event log.
"""

import os
from pathlib import Path

from pydantic import TypeAdapter

from cedar.contracts.base import Identifier
from cedar.models.types import Message

_SESSION = TypeAdapter(Identifier)


class TranscriptStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{_SESSION.validate_python(session_id)}.jsonl"

    def load(self, session_id: str) -> list[Message]:
        path = self._path(session_id)
        if not path.exists():
            return []
        return [Message.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]

    def append(self, session_id: str, message: Message) -> None:
        path = self._path(session_id)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a") as stream:
            stream.write(message.model_dump_json() + "\n")
