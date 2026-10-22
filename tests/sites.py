"""SYNTHETIC test environment: a made-up site, code installation and FHI-aims outputs.

Nothing here describes a real cluster. The binary is never executed; species files and
outputs only reproduce the line formats the examiner and parser rely on.
"""

import os
from pathlib import Path

from ase import Atoms
from ase.build import add_adsorbate, fcc111

from cedar.hpc.profile import CodeInstallation, ComputeProfile, ProfileSource, SchedulerLimits

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


def control_text(site: ComputeProfile, geometry: str, **overrides) -> str:
    """A control.in for ``geometry`` that sets ``declared(**overrides)`` with the site's species defaults, as an agent
    writes one for fhiaims.run 0.4: by copying one generated under 0.3."""
    from cedar.contracts import ProtocolRef, ScientificTaskSpec
    from cedar.protocols import TOTAL_ENERGY, FhiAimsRunReviewed

    task = ScientificTaskSpec(protocol=ProtocolRef(name="fhiaims.run", version="0.3"), observable=TOTAL_ENERGY,
                              estimand="for its control.in", system_label="for its control.in", **declared(**overrides))
    generated, _ = FhiAimsRunReviewed(site).generate(task, {"geometry.in": geometry.encode()})
    return generated["control.in"].decode()


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


# -- VASP ------------------------------------------------------------------------------------

VASP_VERSION = "test-vasp-6.4.2"
VALENCE = {"H": 1, "C": 4, "O": 6, "Cu": 11, "Cu_pv": 17}
VASP_TEMPLATE = """#!/bin/bash
#SBATCH --partition={{partition}}
#SBATCH --nodes={{nodes}}
#SBATCH --ntasks={{ntasks}}
#SBATCH --time={{walltime}}

ulimit -s unlimited
source VASP_ENV
mpirun -np {{ntasks}} {{binary}} > {{output}}
"""
VASP_SOURCE = """! SYNTHETIC stand-in for a VASP source file: only the quoted tag names matter to the examiner
      CALL RDATAB(LOPEN,INCAR,IU5,'ENCUT','=','#',';','F',IDUM,ENCUT,CDUM,LDUM,CHARAC,N,1,IERR)
      CALL PROCESS_INCAR(LOPEN, IU0, IU5, 'EDIFF', EDIFF, IERR, WRITEXMLINCAR)
"""
VASP_TAGS = ("ENCUT", "EDIFF", "EDIFFG", "ISMEAR", "SIGMA", "PREC", "GGA", "ISPIN", "MAGMOM", "IBRION", "NSW",
             "ISIF", "LWAVE", "LCHARG", "LSORBIT", "LNONCOLLINEAR", "KSPACING", "NCORE", "ALGO", "SYSTEM", "NELM")


def dataset_text(name: str) -> str:
    """A SYNTHETIC PAW dataset reproducing the lines the examiner reads; not a real potential."""
    element = name.split("_")[0]
    return (f"  PAW_PBE {name} 01Jan2000\n"
            f" {VALENCE[name]:.16f}\n"
            " parameters from PSCTR are:\n"
            f"   VRHFIN ={element}: synthetic\n"
            "   LEXCH  = PE\n"
            f"   TITEL  = PAW_PBE {name} 01Jan2000\n"
            f"   POMASS =    1.000; ZVAL   =   {VALENCE[name]:.3f}    mass and valenz\n"
            f"  SYNTHETIC test dataset for {element}; not a real PAW potential\n"
            " End of Dataset\n")


def add_vasp(site: ComputeProfile, root: Path, *, own_template: bool = True, variants: bool = True,
             source: bool = True) -> ComputeProfile:
    """``site`` with a synthetic VASP installation under ``root``: three builds that are never executed, a PAW
    library, a source directory naming the tags in VASP_TAGS and, by default, a job template of its own."""
    software = root / "vasp"
    builds = {}
    for build in ("vasp_std", "vasp_gam", "vasp_ncl"):
        path = software / "bin" / build
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\necho 'synthetic test binary; never executed' >&2\nexit 1\n")
        path.chmod(0o755)
        builds[build] = str(path)
    library = software / "potpaw_PBE"
    for name in VALENCE:
        (library / name).mkdir(parents=True)
        (library / name / "POTCAR").write_text(dataset_text(name))
    if source:
        (software / "src").mkdir()
        tags = "".join(f"      DATA '{tag}'\n" for tag in VASP_TAGS)
        (software / "src" / "reader.F").write_text(VASP_SOURCE + tags)
    env = software / "vasp_env.sh"
    env.write_text("# synthetic VASP environment script\n")
    vasp = CodeInstallation(binary=builds["vasp_std"], version=VASP_VERSION, data_paths={"potcars": str(library)},
                            variants={"gam": builds["vasp_gam"], "ncl": builds["vasp_ncl"]} if variants else {},
                            job_template=VASP_TEMPLATE.replace("VASP_ENV", str(env)) if own_template else None)
    return site.model_copy(update={"codes": {**site.codes, "vasp": vasp}})


def incar_text(**tags) -> str:
    lines = ["# SYNTHETIC test INCAR", "SYSTEM = CO on Cu(111)"]
    values = {"ENCUT": 400, "EDIFF": "1E-6", "ISMEAR": 1, "SIGMA": 0.1, "GGA": "PE", "LWAVE": ".FALSE.",
              "LCHARG": ".FALSE."}
    for name, value in tags.items():
        if value is None:
            values.pop(name, None)
        else:
            values[name] = value
    return "\n".join([*lines, *(f"{name} = {value}" for name, value in values.items())]) + "\n"


def poscar_text(atoms, fixed: set[int] = frozenset(), names: bool = True) -> str:
    """POSCAR for ``atoms``, grouped by species in order of first appearance, with the indices in ``fixed`` held
    fixed by selective dynamics."""
    species = list(dict.fromkeys(atoms.get_chemical_symbols()))
    order = [i for s in species for i, atom in enumerate(atoms) if atom.symbol == s]
    lines = ["SYNTHETIC test structure", "1.0"]
    lines += [f"  {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}" for v in atoms.cell]
    if names:
        lines.append("  " + " ".join(species))
    lines.append("  " + " ".join(str(atoms.get_chemical_symbols().count(s)) for s in species))
    if fixed:
        lines.append("Selective dynamics")
    lines.append("Cartesian")
    for i in order:
        x, y, z = atoms[i].position
        flags = (" F F F" if i in fixed else " T T T") if fixed else ""
        lines.append(f"  {x:.10f} {y:.10f} {z:.10f}{flags}")
    return "\n".join(lines) + "\n"


def kpoints_text(mesh=(4, 4, 1), style: str = "Gamma") -> str:
    return f"SYNTHETIC k-points\n0\n{style}\n{mesh[0]} {mesh[1]} {mesh[2]}\n0 0 0\n"


def vasp_outcar(*, version: str = VASP_VERSION, energies: tuple[float, ...] = (-123.456789,), relax: bool = False,
                geometry_converged: bool = True, normal_end: bool = True, unconverged: tuple[int, ...] = (),
                vasp6_marker: bool = False, potcars: tuple[str, ...] = ("Cu", "C", "O"), nelm: int = 3) -> str:
    """SYNTHETIC OUTCAR reproducing the marker formats of real VASP output, one ionic step per energy; the ionic
    steps in ``unconverged`` run NELM electronic steps without reaching EDIFF."""
    lines = [f" vasp.{version} 20Jul23 (build Sep 26 2023 11:38:54) complex", "  ",
             " running on    4 total cores",
             " SYNTHETIC TEST OUTPUT: the marker lines follow VASP formats; every number is made up.", "",
             " INCAR:", *(f" POTCAR:    PAW_PBE {name} 01Jan2000" for name in potcars), "",
             f"   NELM   =    {nelm:3d};   NELMIN=  2; NELMDL= -5     # of ELM steps",
             "   EDIFF  = 0.1E-05   stopping-criterion for ELM",
             f"   NSW    =    {len(energies) + 3 if relax else 0:3d}    number of steps for IOM",
             f"   IBRION =    {2 if relax else -1:3d}    ionic relax: 0-MD 1-quasi-New 2-CG", ""]
    for step, energy in enumerate(energies, start=1):
        electronic = nelm if step in unconverged else 2
        for k in range(1, electronic + 1):
            lines += [f"--------------------------------------- Iteration {step:6d}({k:4d})  "
                      "---------------------------------------", "",
                      f"  free energy    TOTEN  =      {energy + 1 / k:14.8f} eV", "",
                      f"  energy without entropy =     {energy + 1 / k + 0.001:14.8f}  energy(sigma->0) =      "
                      f"{energy + 1 / k + 0.0005:14.8f}", ""]
        if step not in unconverged:
            lines.append("------------------------ aborting loop because EDIFF is reached "
                         "----------------------------------------")
        elif vasp6_marker:
            lines.append("------------------------ aborting loop EDIFF was not reached (unconverged)  "
                         "----------------------------")
        lines += ["", "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
                  "  ---------------------------------------------------",
                  f"  free  energy   TOTEN  =      {energy:14.8f} eV", "",
                  f"  energy  without entropy=      {energy + 0.001:14.8f}  energy(sigma->0) =      "
                  f"{energy + 0.0005:14.8f}", ""]
        if relax and step == len(energies) and geometry_converged:
            lines.append(" reached required accuracy - stopping structural energy minimisation")
    if normal_end:
        lines += ["", " General timing and accounting informations for this job:",
                  " ========================================================", "",
                  "                  Total CPU time used (sec):        1.000"]
    return "\n".join(lines) + "\n"
