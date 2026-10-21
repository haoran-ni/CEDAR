import hashlib
import os
import stat

import pytest

from sim2real.contracts import ArtifactRef


def test_content_is_stored_once_read_only_and_verifiable(artifacts):
    first = artifacts.put_bytes(b"SCF converged\n", label="aims.out")
    second = artifacts.put_bytes(b"SCF converged\n", label="copy.out")
    assert first.sha256 == hashlib.sha256(b"SCF converged\n").hexdigest()
    assert first.path == second.path == f"objects/{first.sha256[:2]}/{first.sha256}"
    path = artifacts.resolve(first)
    assert path.read_bytes() == b"SCF converged\n"
    assert not os.stat(path).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    assert artifacts.verify(first)
    assert not list((artifacts.root / "objects").glob(".incoming-*"))


def test_put_file_keeps_the_original_name_as_label(artifacts, tmp_path):
    source = tmp_path / "geometry.in"
    source.write_bytes(b"atom 0 0 0 Cu\n" * 1000)
    ref = artifacts.put_file(source, media_type="text/plain")
    assert ref.label == "geometry.in" and ref.size_bytes == source.stat().st_size
    assert artifacts.verify(ref)


def test_tampering_is_detected(artifacts):
    ref = artifacts.put_bytes(b"original")
    path = artifacts.resolve(ref)
    os.chmod(path, 0o644)
    path.write_bytes(b"modified")
    assert not artifacts.verify(ref)


def test_missing_artifact_fails_verification(artifacts):
    ref = ArtifactRef(path="objects/ff/" + "f" * 64, sha256="f" * 64, size_bytes=1)
    assert not artifacts.verify(ref)


def test_symlink_cannot_escape_the_store(artifacts, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (artifacts.root / "objects" / "zz").mkdir(parents=True)
    (artifacts.root / "objects" / "zz" / "link").symlink_to(outside)
    ref = ArtifactRef(path="objects/zz/link", sha256="0" * 64, size_bytes=6)
    with pytest.raises(ValueError, match="escapes"):
        artifacts.resolve(ref)
