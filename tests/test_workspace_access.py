"""What an agent may read and write: any file but credentials, and only its own workspace.

Everything here is a SYNTHETIC directory tree under the test's temporary directory; the FakeScheduler TEST
DOUBLE runs nothing.
"""

import pytest
from factories import AGENT
from sites import co_on_cu111, declared, geometry_text
from worlds import World

import cedar.agents.toolkits.files as files_module
from cedar.agents.toolkits import ResearchContext, agent_access, file_tools, research_tools, workspace_tools
from cedar.contracts import CalculationState


@pytest.fixture
def world(ledger, store, artifacts, site):
    return World(ledger, store, artifacts, site)


@pytest.fixture
def tree(tmp_path, monkeypatch, world):
    """The researcher's home with secrets, their earlier work, CEDAR's data directory and two workspaces."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_ed25519").write_text("PRIVATE KEY\n")
    earlier = home / "projects" / "co-cu111"
    earlier.mkdir(parents=True)
    (earlier / "notes.txt").write_text("PBE prefers fcc\n")
    data = tmp_path / "data"
    (data / "workspaces" / "inv-1" / world.hyp).mkdir(parents=True)
    (data / "workspaces" / "inv-1" / "hyp-b").mkdir(parents=True)
    (data / "workspaces" / "inv-1" / "hyp-b" / "their-notes.txt").write_text("hyp-b's work\n")
    (data / "events.sqlite").write_text("log\n")
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "credentials.toml").write_text("api_key = 'x'\n")
    return {"home": home, "earlier": earlier, "data": data, "workspace": data / "workspaces" / "inv-1" / world.hyp,
            "config": tmp_path / "config"}


def access_for(world, tree):
    return agent_access(deny=[tree["config"]])


def tools(access):
    return {t.name: t for t in file_tools(access)}


def read(access, path):
    return tools(access)["read_file"].invoke({"path": str(path)})


def test_agents_read_the_researchers_files_anywhere(world, tree):
    out = read(access_for(world, tree), tree["earlier"] / "notes.txt")
    assert not out.is_error and "1: PBE prefers fcc" in out.content


@pytest.mark.parametrize("secret", ["home/.ssh/id_ed25519", "config/credentials.toml"])
def test_credentials_are_never_readable(world, tree, tmp_path, secret):
    out = read(access_for(world, tree), tmp_path / secret)
    assert out.is_error and "not readable by agents" in out.content


@pytest.mark.parametrize("record", ["data/events.sqlite", "data/workspaces/inv-1/hyp-b/their-notes.txt"])
def test_cedars_records_and_the_other_agents_workspaces_are_readable(world, tree, tmp_path, record):
    assert not read(access_for(world, tree), tmp_path / record).is_error


def test_pseudo_filesystems_are_not_files(world, tree):
    access = access_for(world, tree)
    for path in ("/proc/self/environ", "/dev/zero"):
        out = read(access, path)
        assert out.is_error and "not readable by agents" in out.content


def test_searches_find_everything_but_credentials(world, tree):
    (tree["workspace"] / "slab.in").write_text("atom 0 0 0 Cu\n")
    access = access_for(world, tree)
    found = tools(access)["find_files"].invoke({"root": str(tree["data"].parent), "name_pattern": "*.*"}).content
    assert str((tree["workspace"] / "slab.in").resolve()) in found
    assert "their-notes" in found and "events.sqlite" in found and "credentials" not in found
    listing = tools(access)["list_directory"].invoke({"path": str(tree["data"].parent)}).content
    assert "data/" in listing and "config  (not readable by agents)" in listing


def test_a_link_out_of_the_workspace_reaches_nothing_denied(world, tree):
    (tree["workspace"] / "key").symlink_to(tree["home"] / ".ssh" / "id_ed25519")
    out = read(access_for(world, tree), tree["workspace"] / "key")
    assert out.is_error and "not readable by agents" in out.content


def test_large_files_are_streamed(world, tree, monkeypatch):
    big = tree["earlier"] / "aims.out"
    big.write_text("".join(f"line {i}\n" for i in range(1, 2001)) + "  | Total energy : -1.5 eV\n")
    access = access_for(world, tree)
    monkeypatch.setattr(files_module, "MAX_COUNT_BYTES", 1000)
    out = tools(access)["read_file"].invoke({"path": str(big), "start_line": 3, "max_lines": 2}).content
    assert out.startswith("3: line 3\n4: line 4") and "too many to count" in out
    monkeypatch.setattr(files_module, "MAX_SCAN_BYTES", 5000)
    found = tools(access)["search_file"].invoke({"path": str(big), "pattern": "Total energy"}).content
    assert found.startswith("no matching lines") and "stopped after scanning 5000 bytes" in found


def test_lines_are_numbered_as_the_validator_numbers_them(world, tree):
    odd = tree["earlier"] / "odd.out"
    odd.write_bytes(b"one\rtwo\x0cthree\r\nfour\n")
    out = read(access_for(world, tree), odd).content
    assert out.splitlines()[:4] == ["1: one", "2: two", "3: three", "4: four"]
    assert odd.read_bytes().decode().splitlines() == ["one", "two", "three", "four"]


# -- writing in the workspace ----------------------------------------------------------------------

def writing(tree):
    return {t.name: t for t in workspace_tools(tree["workspace"])}


def test_agents_write_and_delete_in_their_workspace(tree):
    tools = writing(tree)
    assert not tools["write_file"].invoke({"path": "structures/slab.in", "text": "atom 0 0 0 Cu\n"}).is_error
    assert (tree["workspace"] / "structures" / "slab.in").read_text() == "atom 0 0 0 Cu\n"
    again = tools["write_file"].invoke({"path": "structures/slab.in", "text": "x\n"})
    assert again.is_error and "overwrite=true" in again.content
    assert not tools["write_file"].invoke({"path": str(tree["workspace"] / "structures/slab.in"), "text": "y\n",
                                           "overwrite": True}).is_error
    assert not tools["delete_file"].invoke({"path": "structures"}).is_error
    assert not (tree["workspace"] / "structures").exists()


@pytest.mark.parametrize("path", ["../hyp-b/their-notes.txt", "/tmp/elsewhere.txt", "out/x.txt"])
def test_agents_write_nowhere_else(tree, path):
    (tree["workspace"] / "out").symlink_to(tree["earlier"])
    outcome = writing(tree)["write_file"].invoke({"path": path, "text": "x\n", "overwrite": True})
    assert outcome.is_error and "outside your workspace" in outcome.content
    assert (tree["data"] / "workspaces/inv-1/hyp-b/their-notes.txt").read_text() == "hyp-b's work\n"
    assert not (tree["earlier"] / "x.txt").exists()


def test_a_structure_written_in_the_workspace_can_be_requested(world, tree):
    access = access_for(world, tree)
    ctx = ResearchContext(ledger=world.ledger, investigation_id=world.inv, hypothesis_id=world.hyp, actor=AGENT,
                          registry=world.registry, artifacts=world.artifacts, engine=world.engine,
                          validator=world.validator, files=access, analysis=world.analysis,
                          workspace=tree["workspace"])
    agent = {t.name: t for t in research_tools(ctx)}
    slab, bottom = co_on_cu111()
    assert not agent["write_file"].invoke({"path": "atop/geometry.in",
                                           "text": geometry_text(slab, fixed=bottom)}).is_error
    outcome = agent["request_calculation"].invoke({
        "protocol_name": "fhiaims.run", "protocol_version": "0.3", "observable_name": "total energy",
        "observable_definition": "final total energy", "observable_unit": "eV", "estimand": "E(atop)",
        "system_label": "CO/Cu(111) atop", "structure_path": str(tree["workspace"] / "atop/geometry.in"),
        "rationale": "baseline", **declared()})
    assert not outcome.is_error, outcome.content
    (calc,) = world.state().calculations.values()
    assert calc.state is CalculationState.SPECIFIED and calc.examination.agent_review  # for the reviewer
    assert world.artifacts.read(calc.task.structure).decode() == geometry_text(slab, fixed=bottom)
    assert not read(access_for(world, tree), tree["workspace"] / "atop/geometry.in").is_error  # any agent reads it
