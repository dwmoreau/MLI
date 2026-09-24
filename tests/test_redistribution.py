"""`Redistribution.redistribute_xnn` is the optimizer method it replaced, moved.

`_Before` holds the two methods verbatim as they stood on OptimizerManager before the move, so the
comparison is against the original code and not against a second call of the new function.
"""
import numpy as np
import pytest
import scipy.spatial

from mlindex.utilities.Redistribution import redistribute_xnn
from mlindex.utilities.Reindexing import reindex_entry_basic
from mlindex.utilities.UnitCellTools import BL_TO_LATTICE_SYSTEM
from mlindex.utilities.UnitCellTools import fix_unphysical
from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import get_xnn_from_reciprocal_unit_cell
from mlindex.utilities.UnitCellTools import get_xnn_from_unit_cell


class _Before:
    def __init__(self, bravais_lattice, max_neighbors, neighbor_radius, seed, unit_cell_length):
        self.bravais_lattice = bravais_lattice
        self.lattice_system = BL_TO_LATTICE_SYSTEM[bravais_lattice]
        self.opt_params = {'max_neighbors': max_neighbors, 'neighbor_radius': neighbor_radius,
                           'minimum_uc': 2, 'maximum_uc': 500}
        self.rng = np.random.default_rng(seed)
        self.unit_cell_length = unit_cell_length

    def redistribute_xnn(self, xnn):
        # This function is meant to be called only once before optimization starts
        redistributed_xnn = xnn.copy()
        n_redistributed = 0
        iteration = 0
        # Capping the number of iterations is arbitrary.
        # Just an attempt to prevent an excessively long loop
        largest_neighborhood = self.opt_params['max_neighbors'] + 1
        from_indices = None
        while largest_neighborhood > self.opt_params['max_neighbors'] and iteration < 20:
            # This initial distance calculation is time intensive.
            # After the first iteration, only calculate distances after they have been updated.
            if from_indices is None:
                distance = scipy.spatial.distance.cdist(redistributed_xnn, redistributed_xnn)
                neighbor_array = distance < self.opt_params['neighbor_radius']
            else:
                distance_0 = scipy.spatial.distance.cdist(redistributed_xnn[from_indices], redistributed_xnn)
                distance[from_indices, :] = distance_0
                distance[:, from_indices] = distance_0.T
                neighbor_array[from_indices, :] = distance[from_indices, :] < self.opt_params['neighbor_radius']
                neighbor_array[:, from_indices] = distance[:, from_indices] < self.opt_params['neighbor_radius']
            neighbor_count = np.sum(neighbor_array, axis=1)
            largest_neighborhood = neighbor_count.max()
            if largest_neighborhood > self.opt_params['max_neighbors']:
                # This gets the candidate that has the most nearest neighbors and redistributes
                # a subsample of its neighbors such that it has the correct amount of neighbors
                highest_density_index = np.argmax(neighbor_count)
                neighbor_indices = np.where(neighbor_array[highest_density_index])[0]
                excess_neighbors = neighbor_indices.size - self.opt_params['max_neighbors']
                from_indices = neighbor_indices[
                    self.rng.choice(neighbor_indices.size, size=excess_neighbors, replace=False)
                    ]
                n_redistributed += excess_neighbors

                # We want to redistribute the excess only to regions where the density is low
                # Find candidates that have fewer than the number of maximum neighbors and
                # redistribute excess to neighborhoods near these candidates
                low_density_indices = np.where(neighbor_count < self.opt_params['max_neighbors'])[0]
                if low_density_indices.size > 0:
                    # Bias the redistribution to the lowest density regions by probabalistly sampling
                    # the low density regions.
                    prob = self.opt_params['max_neighbors'] - neighbor_count[low_density_indices]
                    prob = prob / prob.sum()
                    if excess_neighbors <= low_density_indices.size:
                        replace = False
                    else:
                        replace = True
                    to_indices = low_density_indices[self.rng.choice(
                        low_density_indices.size, size=excess_neighbors, replace=replace, p=prob
                        )]
                    norm_factor = 1
                else:
                    # In the case that there are no low density regions, perturb by selecting the
                    # lowest density indices, then perturb by a larger amount.
                    to_indices = np.argsort(neighbor_count)[:excess_neighbors]
                    norm_factor = 2
                redistributed_xnn = self.redistribute_and_perturb_xnn(
                    redistributed_xnn, from_indices, to_indices, norm_factor
                    )
            iteration += 1
        return redistributed_xnn

    def redistribute_and_perturb_xnn(self, xnn, from_indices, to_indices, norm_factor):
        n_indices = from_indices.size
        perturbation = self.rng.uniform(low=-1, high=1, size=(n_indices, self.unit_cell_length))
        perturbation *= (
            norm_factor*self.opt_params['neighbor_radius'] / np.linalg.norm(perturbation, axis=1)
            )[:, np.newaxis]
        xnn[from_indices] = xnn[to_indices] + perturbation
        xnn[from_indices] = fix_unphysical(
            xnn=xnn[from_indices],
            rng=self.rng,
            minimum_unit_cell=self.opt_params['minimum_uc'],
            maximum_unit_cell=self.opt_params['maximum_uc'],
            lattice_system=self.lattice_system
            )

        # Enforce the constraints on the unit cells by reindexing
        reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        # This reindexing is time intensive. Only reindex entries that were updated.
        reciprocal_unit_cell[from_indices] = reindex_entry_basic(
            reciprocal_unit_cell[from_indices],
            lattice_system=self.lattice_system,
            bravais_lattice=self.bravais_lattice,
            space='reciprocal'
            )
        xnn = get_xnn_from_reciprocal_unit_cell(
            reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        return xnn


CELLS = {
    'cP': np.array([5.1, 5.1, 5.1, 90.0, 90.0, 90.0]),
    'oP': np.array([5.1, 7.3, 9.2, 90.0, 90.0, 90.0]),
    'mP': np.array([5.1, 7.3, 9.2, 90.0, 104.0, 90.0]),
    'aP': np.array([5.1, 7.3, 9.2, 81.0, 104.0, 97.0]),
    }


def _cloud(bravais_lattice, n, spread, seed):
    """n candidates scattered around one real cell, in the lattice system's partial xnn."""
    rng = np.random.default_rng(seed)
    centre = get_xnn_from_unit_cell(CELLS[bravais_lattice][np.newaxis], partial_unit_cell=True,
                                    lattice_system=BL_TO_LATTICE_SYSTEM[bravais_lattice])
    return centre + spread*rng.standard_normal((n, centre.shape[1]))


@pytest.mark.parametrize('bravais_lattice', sorted(CELLS))
@pytest.mark.parametrize('spread, max_neighbors', [
    (2e-3, 5),     # clumps and sparse neighbourhoods both present
    (1e-6, 20),    # everything in one clump: no neighbourhood is below the cap
    (5e-2, 3),     # sparse: little or nothing to move
    ])
def test_the_moved_function_is_bit_identical_to_the_method(bravais_lattice, spread, max_neighbors):
    radius = 1e-3
    xnn = _cloud(bravais_lattice, 300, spread, seed=7)
    before = _Before(bravais_lattice, max_neighbors, radius, seed=11, unit_cell_length=xnn.shape[1])
    expected = before.redistribute_xnn(xnn)
    rng = np.random.default_rng(11)
    got = redistribute_xnn(xnn, bravais_lattice, max_neighbors, radius, rng,
                           minimum_unit_cell=2, maximum_unit_cell=500)
    np.testing.assert_array_equal(got, expected)
    # Both leave the generator in the same state, so everything drawn after is unchanged too.
    assert rng.bit_generator.state == before.rng.bit_generator.state


@pytest.mark.parametrize('spread, max_neighbors', [(2e-3, 5), (1e-6, 20)])
def test_the_dense_cases_actually_move_candidates(spread, max_neighbors):
    """Otherwise the comparison above would pass on a function that does nothing."""
    xnn = _cloud('oP', 300, spread, seed=7)
    got = redistribute_xnn(xnn, 'oP', max_neighbors, 1e-3, np.random.default_rng(11), 2, 500)
    assert np.sum(np.any(got != xnn, axis=1)) > 50
