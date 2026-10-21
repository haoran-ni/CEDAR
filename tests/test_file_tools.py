import pytest

from cedar.agents.toolkits import FileAccess, file_tools


@pytest.fixture
def tree(tmp_path):
    runs = tmp_path / "runs" / "calc-1"
    runs.mkdir(parents=True)
    (runs / "aims.out").write_text("".join(f"line {i}\n" for i in range(1, 501)) + "  | Total energy : -1.5 eV\n")
    (runs / "control.in").write_text("xc pbe\n")
    (runs / "restart.bin").write_bytes(b"\x00\x01binary")
    secrets = tmp_path / "runs" / "config"
    secrets.mkdir()
    (secrets / "credentials.toml").write_text("[anthropic]\napi_key = \"x\"\n")
    outside = tmp_path / "outside.txt"
    outside.write_text("private\n")
    (runs / "link.txt").symlink_to(outside)
    return tmp_path


@pytest.fixture
def tools(tree):
    access = FileAccess(roots=[tree / "runs"], deny=[tree / "runs" / "config"])
    return {t.name: t for t in file_tools(access)}


def run(tools, name, **args):
    return tools[name].invoke(args)


def test_read_numbers_lines_and_reads_in_ranges(tools, tree):
    out = run(tools, "read_file", path=str(tree / "runs/calc-1/aims.out"), start_line=499, max_lines=5)
    assert not out.is_error
    assert out.content.splitlines()[:3] == ["499: line 499", "500: line 500", "501:   | Total energy : -1.5 eV"]
    assert out.content.endswith("[lines 499-501 of 501]")


def test_search_returns_matching_lines_with_numbers(tools, tree):
    out = run(tools, "search_file", path=str(tree / "runs/calc-1/aims.out"), pattern=r"Total energy", context_lines=1)
    assert out.content.splitlines()[:2] == ["500: line 500", "501:   | Total energy : -1.5 eV"]
    assert "[file has 501 lines]" in out.content


def test_list_and_find(tools, tree):
    listing = run(tools, "list_directory", path=str(tree / "runs/calc-1")).content
    assert "aims.out  (" in listing and "control.in" in listing
    found = run(tools, "find_files", root=str(tree / "runs"), name_pattern="*.in").content
    assert found.strip().endswith("calc-1/control.in")


@pytest.mark.parametrize("path", ["runs/calc-1/aims.out", "/etc/passwd"])
def test_paths_must_be_absolute_and_inside_the_roots(tools, tree, path):
    out = run(tools, "read_file", path=path)
    assert out.is_error and ("absolute path" in out.content or "outside the readable roots" in out.content)


def test_symlinks_cannot_escape(tools, tree):
    out = run(tools, "read_file", path=str(tree / "runs/calc-1/link.txt"))
    assert out.is_error and "outside the readable roots" in out.content


def test_credentials_are_never_readable(tools, tree):
    for name, args in [("read_file", {"path": str(tree / "runs/config/credentials.toml")}),
                       ("list_directory", {"path": str(tree / "runs/config")})]:
        out = run(tools, name, **args)
        assert out.is_error and "not readable by agents" in out.content
    found = run(tools, "find_files", root=str(tree / "runs"), name_pattern="*.toml").content
    assert "credentials" not in found


def test_binary_files_are_refused(tools, tree):
    out = run(tools, "read_file", path=str(tree / "runs/calc-1/restart.bin"))
    assert out.is_error and "binary" in out.content


def test_bad_regular_expression_is_reported(tools, tree):
    out = run(tools, "search_file", path=str(tree / "runs/calc-1/aims.out"), pattern="(")
    assert out.is_error and "invalid regular expression" in out.content


def test_find_reports_matches_it_did_not_show(tools, tree):
    for i in range(5):
        (tree / "runs" / "calc-1" / f"extra{i}.in").write_text("x\n")
    out = run(tools, "find_files", root=str(tree / "runs"), name_pattern="*.in", max_results=2).content
    assert len(out.splitlines()) == 3 and out.endswith("[4 more matches not shown]")
