"""SYNTHETIC test environment: a made-up site, code installation and FHI-aims outputs.

Nothing here describes a real cluster. The binary is never executed; species files and
outputs only reproduce the line formats the examiner and parser rely on.
"""

import os
from pathlib import Path

from ase import Atoms
from ase.build import add_adsorbate, fcc111

from sim2real.hpc.profile import CodeInstallation, ComputeProfile, ProfileSource, SchedulerLimits

ELEMENTS = {"H": 1, "C": 6, "O": 8, "Cu": 29}
VERSION = "test-1.0"

JOB_TEMPLATE = """#!/bin/bash
#SBATCH --partition={{partition}}
#SBATCH --nodes={{nodes}}
#SBATCH --ntasks={{ntasks}}
#SBATCH --time={{walltime}}
#SBATCH --exclusive

ulimit -s unlimited
source ENV_SCRIPT
export OMP_NUM_THREADS=1
mpirun -np {{ntasks}} {{binary}} > {{output}}
"""


def species_text(element: str, level: str) -> str:
    return (
        "################################################################################\n"
        f"#  SYNTHETIC {level} test species file for {element}; not a real FHI-aims default\n"
        "################################################################################\n"
        f"  species        {element}\n"
        f"    nucleus      {ELEMENTS[element]}\n"
        f"    l_hartree    {4 if level == 'light' else 6}\n"
        "    radial_base  40 5.0\n"
        "#     hydro 4 f 7.4\n"
    )


def build_site(root: Path) -> ComputeProfile:
    """Create a synthetic installation under ``root`` and return a profile describing it."""
    software = root / "software"
    binary = software / "bin" / "fake-aims.x"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\necho 'synthetic test binary; never executed' >&2\nexit 1\n")
    binary.chmod(0o755)
    env_script = software / "env.sh"
    env_script.write_text("# synthetic environment script\n")
    species = software / "species_defaults"
    for level in ("light", "tight"):
        (species / level).mkdir(parents=True)
        for element, z in ELEMENTS.items():
            (species / level / f"{z:02d}_{element}_default").write_text(species_text(element, level))
    return ComputeProfile(
        name="synthetic-site",
        scheduler="slurm",
        limits=SchedulerLimits(partitions=("debug", "compute"), default_partition="compute", cores_per_node=64,
                               default_ntasks=32, max_nodes=2, default_walltime_minutes=60,
                               max_walltime_minutes=24 * 60),
        codes={"fhi-aims": CodeInstallation(binary=str(binary), version=VERSION,
                                            data_paths={"species_defaults": str(species)})},
        job_template=JOB_TEMPLATE.replace("ENV_SCRIPT", str(env_script)),
        runs_root=str(root / "runs"),
        sources=(ProfileSource(kind="user_answer", reference="synthetic test site"),),
    )


def geometry_text(atoms, fixed: set[int] = frozenset(), moments: dict[int, float] | None = None) -> str:
    lines = ["# SYNTHETIC test structure"]
    if atoms.pbc.any():
        lines += [f"lattice_vector {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}" for v in atoms.cell]
    for index, atom in enumerate(atoms):
        x, y, z = atom.position
        lines.append(f"atom {x:.8f} {y:.8f} {z:.8f} {atom.symbol}")
        if index in fixed:
            lines.append("    constrain_relaxation .true.")
        if moments and index in moments:
            lines.append(f"    initial_moment {moments[index]}")
    return "\n".join(lines) + "\n"


def co_on_cu111(vacuum: float = 7.5):
    """CO bound C-down on the atop site of a 2x2, three-layer Cu(111) slab (illustrative geometry)."""
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=vacuum)
    add_adsorbate(slab, Atoms("CO", positions=[(0, 0, 0), (0, 0, 1.15)]), 1.9, "ontop", mol_index=0)
    bottom = {a.index for a in slab if a.tag == 3}
    return slab, bottom


def declared(**overrides):
    """A complete declaration for a CO/Cu(111) slab, as an agent would state it."""
    interventions = {"xc": "pbe"}
    controls = {"spin": "none", "charge": 0, "relativistic": "atomic_zora scalar"}
    numerical = {"species_defaults": "light", "k_grid": [4, 4, 1], "occupation_type": "gaussian 0.1",
                 "sc_accuracy_rho": 1e-5, "sc_accuracy_eev": 1e-3, "sc_accuracy_etot": 1e-6}
    for group, values in (("interventions", interventions), ("controls", controls), ("numerical", numerical)):
        for key, value in overrides.pop(group, {}).items():
            if value is None:
                values.pop(key, None)
            else:
                values[key] = value
    return {"interventions": interventions, "controls": controls, "numerical_settings": numerical}


def aims_output(*, version: str = VERSION, energy: float = -1234.567890123, relax: bool = False,
                geometry_converged: bool = True, normal_end: bool = True, scf_failure: bool = False) -> str:
    """SYNTHETIC aims.out reproducing the marker formats of real FHI-aims output."""
    corrected = energy - 0.000387
    lines = [
        "------------------------------------------------------------",
        "          Invoking FHI-aims ...",
        "  SYNTHETIC TEST OUTPUT: the marker lines follow FHI-aims formats; every number is made up.",
        "------------------------------------------------------------",
        f"  FHI-aims version      : {version}",
        "  Using        4 parallel tasks.",
        "",
    ]
    steps = 2 if relax else 1
    for step in range(1, steps + 1):
        if scf_failure and step == steps:
            lines += ["  * SELF-CONSISTENCY CYCLE DID NOT CONVERGE",
                      " *** scf_solver: SCF cycle not converged.", ""]
            break
        lines += [
            "  Self-consistency cycle converged.",
            "",
            "  Energy and forces in a compact form:",
            f"  | Total energy uncorrected      :         {energy + step * 0.001:.15E} eV",
            f"  | Total energy corrected        :         {corrected + step * 0.001:.15E} eV  "
            "<-- do not rely on this value for anything but (periodic) metals",
            f"  | Electronic free energy        :         {energy - 0.0007 + step * 0.001:.15E} eV",
        ]
        if relax:
            last = step == steps
            if last and geometry_converged:
                lines += ["  Maximum force component is  0.357140E-03 eV/A.", "  Present geometry is converged."]
            else:
                lines += ["  Maximum force component is  0.344656E+00 eV/A.", "  Present geometry is not yet converged.",
                          "", f"  Relaxation step number      {step}: Predicting new coordinates."]
    if normal_end and not scf_failure:
        lines += [
            "",
            "  Final output of selected total energy values:",
            "",
            f"  | Total energy of the DFT / Hartree-Fock s.c.f. calculation      :       {energy:.9f} eV",
            f"  | Final zero-broadening corrected energy (caution - metals only) :       {corrected:.9f} eV",
            "",
            "          Leaving FHI-aims.",
            "",
            "          Have a nice day.",
            "------------------------------------------------------------",
        ]
    return "\n".join(lines) + "\n"


def make_writable(path: Path) -> None:
    os.chmod(path, 0o644)
