"""Moving candidates out of crowded neighbourhoods before refinement.

Candidates that start within `neighbor_radius` of one another tend to refine to the same place, so a
dense clump spends many candidates on one try. `redistribute_xnn` caps every neighbourhood at
`max_neighbors`: it repeatedly takes the most crowded candidate, moves the excess of its neighbours
next to candidates in sparse neighbourhoods, and perturbs them by about one neighbour radius.
"""
import numpy as np
import scipy.spatial

from mlindex.utilities.Reindexing import reindex_entry_basic
from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM
from mlindex.utilities.UnitCellTools import fix_unphysical
from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import get_xnn_from_reciprocal_unit_cell


def redistribute_xnn(xnn, bravais_lattice, max_neighbors, neighbor_radius, rng,
                     minimum_unit_cell, maximum_unit_cell):
    """A copy of `xnn` with no candidate having more than `max_neighbors` within `neighbor_radius`.

    `xnn` is (n_candidates, n_parameters) in the lattice system's partial xnn. At most twenty
    passes are made, so a cloud too dense to spread may come back still over the cap.
    """
    lattice_system = BL_TO_LATTICE_SYSTEM[bravais_lattice]
    redistributed_xnn = xnn.copy()
    iteration = 0
    largest_neighborhood = max_neighbors + 1
    from_indices = None
    while largest_neighborhood > max_neighbors and iteration < 20:
        # The first distance calculation is the expensive one. After it, only the rows and columns
        # of the candidates that moved are recomputed.
        if from_indices is None:
            distance = scipy.spatial.distance.cdist(redistributed_xnn, redistributed_xnn)
            neighbor_array = distance < neighbor_radius
        else:
            distance_0 = scipy.spatial.distance.cdist(redistributed_xnn[from_indices], redistributed_xnn)
            distance[from_indices, :] = distance_0
            distance[:, from_indices] = distance_0.T
            neighbor_array[from_indices, :] = distance[from_indices, :] < neighbor_radius
            neighbor_array[:, from_indices] = distance[:, from_indices] < neighbor_radius
        neighbor_count = np.sum(neighbor_array, axis=1)
        largest_neighborhood = neighbor_count.max()
        if largest_neighborhood > max_neighbors:
            # Take the candidate with the most neighbours and move a random subset of them, so that
            # it is left with exactly max_neighbors.
            highest_density_index = np.argmax(neighbor_count)
            neighbor_indices = np.where(neighbor_array[highest_density_index])[0]
            excess_neighbors = neighbor_indices.size - max_neighbors
            from_indices = neighbor_indices[
                rng.choice(neighbor_indices.size, size=excess_neighbors, replace=False)
                ]

            # The excess goes next to candidates below the cap, chosen with probability rising as
            # their neighbourhood empties.
            low_density_indices = np.where(neighbor_count < max_neighbors)[0]
            if low_density_indices.size > 0:
                prob = max_neighbors - neighbor_count[low_density_indices]
                prob = prob / prob.sum()
                if excess_neighbors <= low_density_indices.size:
                    replace = False
                else:
                    replace = True
                to_indices = low_density_indices[rng.choice(
                    low_density_indices.size, size=excess_neighbors, replace=replace, p=prob
                    )]
                norm_factor = 1
            else:
                # No neighbourhood is below the cap: use the least crowded candidates, and move
                # twice as far from them.
                to_indices = np.argsort(neighbor_count)[:excess_neighbors]
                norm_factor = 2
            redistributed_xnn = _move_next_to(
                redistributed_xnn, from_indices, to_indices, norm_factor*neighbor_radius, rng,
                bravais_lattice, lattice_system, minimum_unit_cell, maximum_unit_cell
                )
        iteration += 1
    return redistributed_xnn


def _move_next_to(xnn, from_indices, to_indices, step, rng, bravais_lattice, lattice_system,
                  minimum_unit_cell, maximum_unit_cell):
    """Put each `from_indices` candidate a distance `step` from its `to_indices` partner."""
    perturbation = rng.uniform(low=-1, high=1, size=(from_indices.size, xnn.shape[1]))
    perturbation *= (step / np.linalg.norm(perturbation, axis=1))[:, np.newaxis]
    xnn[from_indices] = xnn[to_indices] + perturbation
    xnn[from_indices] = fix_unphysical(
        xnn=xnn[from_indices],
        rng=rng,
        minimum_unit_cell=minimum_unit_cell,
        maximum_unit_cell=maximum_unit_cell,
        lattice_system=lattice_system
        )

    # Reindexing enforces the lattice's cell conventions, and is expensive, so only the moved
    # candidates are reindexed.
    reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
        xnn, partial_unit_cell=True, lattice_system=lattice_system
        )
    reciprocal_unit_cell[from_indices] = reindex_entry_basic(
        reciprocal_unit_cell[from_indices],
        lattice_system=lattice_system,
        bravais_lattice=bravais_lattice,
        space='reciprocal'
        )
    return get_xnn_from_reciprocal_unit_cell(
        reciprocal_unit_cell, partial_unit_cell=True, lattice_system=lattice_system
        )
