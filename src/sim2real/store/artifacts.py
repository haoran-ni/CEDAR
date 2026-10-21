"""Content-addressed, write-once storage for raw outputs and input structures."""

import hashlib
import os
import uuid
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from sim2real.contracts.base import ArtifactRef

_CHUNK = 1 << 20


class ArtifactStore:
    """Stores each file once under ``objects/<aa>/<sha256>`` and makes it read-only."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_file(self, source: str | Path, *, media_type: str | None = None) -> ArtifactRef:
        source = Path(source)
        with source.open("rb") as stream:
            return self._put_stream(stream, label=source.name, media_type=media_type)

    def put_bytes(self, data: bytes, *, label: str | None = None, media_type: str | None = None) -> ArtifactRef:
        return self._put_stream(BytesIO(data), label=label, media_type=media_type)

    def _put_stream(self, stream: BinaryIO, *, label: str | None, media_type: str | None) -> ArtifactRef:
        objects = self.root / "objects"
        objects.mkdir(exist_ok=True)
        tmp = objects / f".incoming-{uuid.uuid4().hex}"
        digest = hashlib.sha256()
        size = 0
        try:
            with tmp.open("xb") as out:
                while chunk := stream.read(_CHUNK):
                    digest.update(chunk)
                    out.write(chunk)
                    size += len(chunk)
            sha = digest.hexdigest()
            relative = f"objects/{sha[:2]}/{sha}"
            target = self.root / relative
            if target.exists():
                tmp.unlink()
            else:
                target.parent.mkdir(exist_ok=True)
                os.chmod(tmp, 0o444)
                os.replace(tmp, target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return ArtifactRef(path=relative, sha256=sha, size_bytes=size, media_type=media_type, label=label)

    def resolve(self, ref: ArtifactRef) -> Path:
        path = (self.root / ref.path).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"artifact path escapes the store: {ref.path!r}")
        return path

    def verify(self, ref: ArtifactRef) -> bool:
        """True if the stored file still matches the reference's size and hash."""
        path = self.resolve(ref)
        if not path.is_file() or path.stat().st_size != ref.size_bytes:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(_CHUNK):
                digest.update(chunk)
        return digest.hexdigest() == ref.sha256
