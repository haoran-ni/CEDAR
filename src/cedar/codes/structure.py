"""Measurements of a structure that the examiners of every code record."""

import numpy as np
from ase import Atoms


def minimum_distance(atoms: Atoms) -> float:
    """Shortest interatomic distance, using periodic images where the cell is periodic."""
    if len(atoms) < 2:
        return float("inf")
    distances = atoms.get_all_distances(mic=bool(atoms.pbc.any()))
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def vacuum_along_c(atoms: Atoms) -> float:
    """Largest empty gap along the third lattice vector, measured perpendicular to the a-b plane."""
    a, b, c = atoms.cell
    normal = np.cross(a, b)
    normal /= np.linalg.norm(normal)
    height = abs(float(np.dot(c, normal)))
    fractions = np.sort(atoms.get_scaled_positions(wrap=True)[:, 2])
    gaps = np.diff(np.concatenate([fractions, [fractions[0] + 1.0]]))
    return float(gaps.max() * height)
