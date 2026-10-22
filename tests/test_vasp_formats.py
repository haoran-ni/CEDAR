import os
from pathlib import Path

import pytest
from ase.build import bulk
from sites import dataset_text, kpoints_text, poscar_text, vasp_outcar

from cedar.codes.vasp import (
    element_of,
    fatal_error,
    library_dataset,
    parse_incar,
    parse_kpoints,
    parse_outcar,
    parse_poscar,
    parse_potcar,
    render_potcar,
)


class TestIncar:
    def test_tags_as_vasp_reads_them(self):
        incar = parse_incar("encut = 400   # cutoff\nISMEAR = 0; SIGMA = 0.05 ! narrow\nMAGMOM = 2*1.0 \\\n  0.5\n")
        assert [(t.name, t.value, t.line) for t in incar.tags] == [
            ("ENCUT", "400", 1), ("ISMEAR", "0", 2), ("SIGMA", "0.05", 2), ("MAGMOM", "2*1.0 0.5", 3)]
        assert incar.number("encut") == 400 and incar.get("MAGMOM").value == "2*1.0 0.5"

    def test_lines_without_a_tag_are_ignored_by_vasp_and_kept(self):
        incar = parse_incar("Static run\nENCUT 520\nPREC = Accurate\n# a comment\n\n")
        assert incar.ignored_lines == ((1, "Static run"), (2, "ENCUT 520"))
        assert [t.name for t in incar.tags] == ["PREC"]

    def test_logicals_in_every_spelling(self):
        incar = parse_incar("LWAVE = .FALSE.\nLCHARG = F\nLSORBIT = .TRUE. (spin-orbit)\nLORBIT = 11\n")
        assert (incar.flag("LWAVE"), incar.flag("LCHARG"), incar.flag("LSORBIT")) == (False, False, True)
        assert incar.flag("LORBIT") is None and incar.flag("LNONCOLLINEAR") is None

    def test_quoted_values_may_span_lines(self):
        incar = parse_incar('SYSTEM = "two\nlines"\nENCUT = 300\n')
        assert [t.name for t in incar.tags] == ["SYSTEM", "ENCUT"]


class TestPoscar:
    def test_species_counts_and_cartesian_positions(self):
        atoms = bulk("Cu", "fcc", a=3.6) * (2, 1, 1)
        poscar = parse_poscar(poscar_text(atoms, fixed={0}))
        assert poscar.species == ("Cu",) and poscar.counts == (2,) and poscar.selective
        assert [a.fixed for a in poscar.atoms] == [(True, True, True), (False, False, False)]
        assert poscar.to_atoms(["Cu"]).get_distance(0, 1) == pytest.approx(atoms.get_distance(0, 1))

    def test_direct_coordinates_and_a_volume_as_scale(self):
        text = "rocksalt\n-64.0\n1 0 0\n0 1 0\n0 0 1\nNa Cl\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n"
        poscar = parse_poscar(text)
        assert poscar.lattice[0][0] == pytest.approx(4.0)
        assert poscar.atoms[1].position == pytest.approx((2.0, 2.0, 2.0))

    def test_vasp4_files_name_no_species(self):
        poscar = parse_poscar("old\n1.0\n3 0 0\n0 3 0\n0 0 3\n1 1\nDirect\n0 0 0\n0.5 0.5 0.5\n")
        assert poscar.species is None and poscar.counts == (1, 1) and len(poscar.atoms) == 2

    def test_what_cannot_be_read_is_reported(self):
        poscar = parse_poscar("x\n1.0\n3 0 0\n0 3 0\n0 0 3\nCu\n2\nDirect\n0 0 0\n")
        assert poscar.problems == ((0, "2 atoms are counted, but only 1 positions follow"),)
        assert parse_poscar("x\n1.0\n3 0\n").problems

    def test_element_of_a_species_or_dataset_name(self):
        assert [element_of(n) for n in ("Fe", "Fe_pv", "Fe_sv/1a2b", "H.75", "1x")] == ["Fe", "Fe", "Fe", "H", None]


class TestKpoints:
    def test_generated_meshes(self):
        gamma = parse_kpoints(kpoints_text((6, 6, 1)))
        assert (gamma.mode, gamma.mesh, gamma.summary) == ("gamma", (6, 6, 1), "Γ-centred 6×6×1 mesh")
        shifted = parse_kpoints("k\n0\nMonkhorst-Pack\n4 4 4\n0.5 0.5 0.5\n")
        assert shifted.summary == "Monkhorst-Pack 4×4×4 mesh, shifted by 0.5 0.5 0.5" and not shifted.gamma_only
        assert parse_kpoints("k\n0\nAuto\n25\n").summary == "fully automatic mesh, length 25"
        assert parse_kpoints(kpoints_text((1, 1, 1))).gamma_only

    def test_lists_and_band_paths(self):
        explicit = parse_kpoints("k\n2\nReciprocal lattice\n0 0 0 1\n0.5 0 0 1\n")
        assert explicit.summary == "2 explicit k-points (reciprocal coordinates)" and not explicit.gamma_only
        assert parse_kpoints("k\n1\nReciprocal\n0 0 0 1\n").gamma_only
        path = parse_kpoints("k\n40\nLine-mode\nReciprocal\n0 0 0\n0.5 0 0\n\n0.5 0 0\n0.5 0.5 0\n")
        assert path.summary == "band path in line mode, 40 points per segment, 2 segments"


class TestPotcar:
    def test_datasets_with_their_names_elements_and_valence(self, tmp_path):
        potcar = parse_potcar(dataset_text("Cu_pv") + dataset_text("O"))
        assert [(d.name, d.element, d.valence, d.line) for d in potcar.datasets] == [("Cu_pv", "Cu", 17, 1),
                                                                                    ("O", "O", 6, 10)]
        assert potcar.datasets[0].title == "PAW_PBE Cu_pv 01Jan2000" and not potcar.problems

    def test_written_from_a_library_and_compared_with_it(self, tmp_path):
        for name in ("Cu", "O"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "POTCAR").write_text(dataset_text(name))
        text = render_potcar(tmp_path, ["O", "Cu"])
        potcar = parse_potcar(text)
        assert [d.name for d in potcar.datasets] == ["O", "Cu"]
        assert potcar.datasets[1].same_as((tmp_path / "Cu" / "POTCAR").read_text() + "\n  ")
        assert not potcar.datasets[1].same_as(dataset_text("Cu").replace("synthetic", "edited"))
        with pytest.raises(LookupError):
            render_potcar(tmp_path, ["Cu", "Zn"])
        assert library_dataset(tmp_path, "../Cu") is None

    def test_a_truncated_dataset_is_reported(self):
        assert parse_potcar(dataset_text("O") + "  PAW_PBE C 01Jan2000\n").problems


class TestOutcar:
    def test_healthy_static_run(self):
        out = parse_outcar(vasp_outcar(version="6.4.2", energies=(-12.5,)))
        assert out.version_string == "6.4.2" and out.parameters == {"NELM": 3, "NSW": 0, "IBRION": -1}
        assert out.electronic_steps == {1: 2} and out.unconverged_steps == [] and not out.relaxes
        assert out.free_energies[-1].value == -12.5 and "free  energy   TOTEN" in out.free_energies[-1].text
        assert out.final_sigma0.value == pytest.approx(-12.4995) and out.normal_termination is not None

    def test_vasp5_version_and_relaxation(self):
        out = parse_outcar(vasp_outcar(version="5.4.4.18Apr17-6-g9f103f2a35", energies=(-1.0, -1.2, -1.3),
                                       relax=True))
        assert out.version_string == "5.4.4.18Apr17-6-g9f103f2a35" and out.relaxes
        assert len(out.free_energies) == 3 and out.geometry_converged is not None

    def test_unconverged_electronic_loops_in_vasp5_and_vasp6(self):
        assert parse_outcar(vasp_outcar(energies=(-1.0, -1.1), unconverged=(2,))).unconverged_steps == [2]
        six = parse_outcar(vasp_outcar(energies=(-1.0,), unconverged=(1,), vasp6_marker=True))
        assert six.unconverged_steps == [1] and six.scf_failures

    def test_fatal_errors_in_the_standard_output(self):
        assert fatal_error(" running on 4 nodes\n F= -.1E+02\n") is None
        marker = fatal_error(" DAV: 1\n VERY BAD NEWS! internal error in subroutine IBZKPT\n")
        assert marker.line == 2


@pytest.mark.skipif("CEDAR_REAL_VASP_RUNS" not in os.environ,
                    reason="set CEDAR_REAL_VASP_RUNS to a directory of real finished VASP runs")
def test_real_runs_parse():
    """Optional check against real VASP runs available on the current machine: every OUTCAR below the directory,
    with the inputs beside it."""
    outcars = sorted(Path(os.environ["CEDAR_REAL_VASP_RUNS"]).rglob("OUTCAR"))
    assert outcars
    for path in outcars:
        out = parse_outcar(path.read_text(errors="replace"))
        assert out.version_string is not None and "NELM" in out.parameters, path
        if out.normal_termination is not None:
            assert out.free_energies and out.final_sigma0 is not None and out.electronic_steps, path
        directory = path.parent
        if (directory / "POSCAR").is_file() and (directory / "POTCAR").is_file():
            poscar = parse_poscar((directory / "POSCAR").read_text(errors="replace"))
            potcar = parse_potcar((directory / "POTCAR").read_text(errors="replace"))
            assert not poscar.problems and not potcar.problems, directory
            assert len(potcar.datasets) == len(poscar.counts), directory
        if (directory / "INCAR").is_file():
            assert parse_incar((directory / "INCAR").read_text(errors="replace")).tags, directory
