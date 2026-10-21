from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from cedar.contracts import FindingSeverity
from cedar.hpc import ComputeProfile, ProfileNotApproved, ProfileStore, check_profile, file_source

NOW = datetime(2026, 10, 21, tzinfo=timezone.utc)


def with_template(site, template: str) -> dict:
    return {**site.model_dump(), "job_template": template}


class TestTemplate:
    def test_unknown_placeholder_is_rejected(self, site):
        with pytest.raises(ValidationError, match="unknown template placeholders"):
            ComputeProfile.model_validate(with_template(site, site.job_template + "echo {{account}}\n"))

    def test_exactly_one_launch_line(self, site):
        no_launch = "\n".join(l for l in site.job_template.splitlines() if "{{binary}}" not in l)
        with pytest.raises(ValidationError, match="exactly one template line"):
            ComputeProfile.model_validate(with_template(site, no_launch))

    def test_shebang_required(self, site):
        with pytest.raises(ValidationError, match="shebang"):
            ComputeProfile.model_validate(with_template(site, site.job_template.split("\n", 1)[1]))

    def test_shell_variables_are_not_placeholders(self, site):
        ComputeProfile.model_validate(with_template(site, site.job_template + 'echo "${HOME}"\n'))


class TestChecks:
    def test_a_correct_profile_has_no_errors(self, site):
        assert [f for f in check_profile(site) if f.severity is FindingSeverity.ERROR] == []

    def test_missing_or_unusable_paths_are_errors(self, site, tmp_path):
        code = site.codes["fhi-aims"]
        broken = site.model_copy(update={"codes": {"fhi-aims": code.model_copy(update={
            "binary": str(tmp_path / "nowhere.x"),
            "data_paths": {"species_defaults": str(tmp_path / "missing")},
        })}})
        assert {f.code for f in check_profile(broken)} >= {"missing_binary", "missing_data_path"}

    def test_binary_must_be_executable(self, site):
        from pathlib import Path

        Path(site.codes["fhi-aims"].binary).chmod(0o644)
        assert "binary_not_executable" in {f.code for f in check_profile(site)}

    def test_profiles_should_say_where_values_came_from(self, site):
        assert "no_sources" in {f.code for f in check_profile(site.model_copy(update={"sources": ()}))}


class TestApproval:
    def test_profile_takes_effect_only_after_approval(self, site, tmp_path):
        store = ProfileStore(tmp_path / "profiles")
        store.save(site)
        assert store.load(site.name) == site
        with pytest.raises(ProfileNotApproved, match="not been approved"):
            store.load_approved(site.name)
        store.approve(site.name, "researcher", NOW)
        assert store.load_approved(site.name) == site

    def test_changing_an_approved_profile_withdraws_approval(self, site, tmp_path):
        store = ProfileStore(tmp_path / "profiles")
        store.save(site)
        store.approve(site.name, "researcher", NOW)
        store.save(site.model_copy(update={"limits": site.limits.model_copy(update={"max_nodes": 99})}))
        with pytest.raises(ProfileNotApproved, match="changed after"):
            store.load_approved(site.name)


def test_file_sources_record_content_hashes(tmp_path):
    path = tmp_path / "submit.sh"
    path.write_text("#!/bin/bash\n")
    source = file_source(path, note="existing job script")
    assert source.kind == "file" and source.reference == str(path) and len(source.sha256) == 64
