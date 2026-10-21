import os
from pathlib import Path

import pytest
from ase.build import molecule
from sites import aims_output, co_on_cu111, geometry_text

from cedar.codes.fhiaims import (
    minimum_distance,
    parse_control,
    parse_geometry,
    parse_output,
    render_control,
    vacuum_along_c,
)


class TestControl:
    TEXT = """# comment line
xc    pbe   # trailing comment
KS_method   parallel
k_grid 4 4 1

species Cu
  nucleus 29   # comment
#  hydro 4 f 7.4
species  O
  nucleus 8
"""

    def test_keywords_stop_at_the_first_species_block(self):
        control = parse_control(self.TEXT)
        assert [(k.name, k.args, k.line) for k in control.keywords] == [
            ("xc", ("pbe",), 2), ("ks_method", ("parallel",), 3), ("k_grid", ("4", "4", "1"), 4)]
        assert [b.species for b in control.species] == ["Cu", "O"]
        assert control.species_block("Cu").normalized() == ("species Cu", "nucleus 29")

    def test_render_then_parse_round_trips(self):
        text = render_control(["header"], [("xc", "pbe"), ("k_grid", "2 2 1")], ["species H\n  nucleus 1\n"])
        control = parse_control(text)
        assert [k.name for k in control.keywords] == ["xc", "k_grid"]
        assert control.species_block("H").normalized() == ("species H", "nucleus 1")


class TestGeometry:
    def test_reads_cell_atoms_constraints_and_moments(self):
        slab, bottom = co_on_cu111()
        geometry = parse_geometry(geometry_text(slab, fixed=bottom, moments={0: 0.5}))
        assert geometry.periodic and len(geometry.atoms) == len(slab)
        assert geometry.elements == ("Cu", "C", "O")
        assert sum(a.constrained for a in geometry.atoms) == len(bottom)
        assert geometry.atoms[0].initial_moment == 0.5
        assert geometry.unknown_lines == () and geometry.malformed_lines == ()

    def test_fractional_coordinates_are_converted(self):
        text = "lattice_vector 2 0 0\nlattice_vector 0 2 0\nlattice_vector 0 0 2\natom_frac 0.5 0.5 0.5 H\n"
        atoms = parse_geometry(text).to_atoms()
        assert atoms.positions[0].tolist() == [1.0, 1.0, 1.0]

    def test_unknown_and_malformed_lines_are_reported(self):
        geometry = parse_geometry("atom 0 0 0 H\nhessian_block 1 2\natom 0 0 X H\nlattice_vector 1 0 0\n")
        assert [line for line, _ in geometry.unknown_lines] == [2]
        assert {line for line, _ in geometry.malformed_lines} == {3, 0}

    def test_vacuum_and_distances(self):
        slab, _ = co_on_cu111(vacuum=7.5)
        atoms = parse_geometry(geometry_text(slab)).to_atoms()
        assert vacuum_along_c(atoms) == pytest.approx(15.0 - 1.9 - 1.15, abs=0.1)
        assert minimum_distance(atoms) == pytest.approx(1.15, abs=0.05)

    def test_molecule_without_cell(self):
        geometry = parse_geometry(geometry_text(molecule("CO")))
        assert not geometry.periodic
        assert minimum_distance(geometry.to_atoms()) == pytest.approx(1.13, abs=0.05)


class TestOutput:
    def test_successful_relaxation(self):
        out = parse_output(aims_output(relax=True, energy=-100.5))
        assert out.version_string == "test-1.0" and out.parallel_tasks == 4
        assert len(out.scf_converged) == 2 and out.scf_failures == ()
        assert out.relaxation_steps == 1 and out.geometry_converged is not None
        assert out.final_total_energy.value == -100.5
        assert "Total energy of the DFT" in out.final_total_energy.text
        assert out.normal_termination is not None and out.killed is None

    def test_run_that_ended_early(self):
        out = parse_output(aims_output(relax=True, geometry_converged=False, normal_end=False))
        assert out.normal_termination is None and out.geometry_converged is None
        assert out.final_total_energy is None

    def test_scf_failure_is_detected(self):
        out = parse_output(aims_output(scf_failure=True))
        assert len(out.scf_failures) == 2 and out.normal_termination is None

    def test_killed_run(self):
        text = aims_output(normal_end=False) + "=   KILLED BY SIGNAL: 9 (Killed)\n"
        assert parse_output(text).killed is not None

    def test_fortran_exponents(self):
        from cedar.codes.fhiaims.output import numbers_in

        assert numbers_in("| Total energy : -0.200529726686124E+07 eV and 1.5D-03") == [-2005297.26686124, 1.5e-3]


@pytest.mark.skipif("CEDAR_REAL_AIMS_OUTPUTS" not in os.environ,
                    reason="set CEDAR_REAL_AIMS_OUTPUTS to a directory of real finished aims.out files")
def test_real_outputs_parse():
    """Optional check against real FHI-aims outputs available on the current machine."""
    files = sorted(Path(os.environ["CEDAR_REAL_AIMS_OUTPUTS"]).rglob("aims.out"))
    assert files
    for path in files:
        out = parse_output(path.read_text(errors="replace"))
        assert out.version_string is not None, path
        if out.normal_termination is not None:
            assert out.final_total_energy is not None, path
