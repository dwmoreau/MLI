"""Starting clouds whose candidates come in groups a set distance apart.

The convergence curve measures one candidate in isolation. It says nothing about two candidates
that start near each other, and they are not independent: they share their first Miller-index
assignment and much of their refinement. Measuring that needs a starting cloud built differently
from the shipped one -- candidates in groups, the members of a group a controlled distance apart --
and this manager is the only change needed to produce it.

At `separation_ratio` 0 the members of a group are the same cell, which is the identical-start
case. Above it they are displaced isotropically by that fraction of the shell radius.
"""
import numpy as np

from mlindex.optimization.MPIOptimizer import OptimizerManager
from mlindex.utilities.UnitCellTools import fix_unphysical


class SeparatedStartManager(OptimizerManager):
    """A convergence-testing cloud of groups. `group_size` and `separation_ratio` are set on the
    instance after construction; at group size 1 this class is its parent.

    After a run, `last_distance` holds every candidate's distance from the true cell and
    `last_separation` its distance from its own group's centre -- both measured on the cloud that
    was actually refined, not assumed from what was asked for.
    """

    group_size = 1
    separation_ratio = 0.0
    chunk_tag = None
    last_distance = None
    last_separation = None

    def _reseed_for_pattern(self):
        """The shipped re-keying, then a distinct stream per chunk.

        The parent keys the generator on the peak list, the lattice and the rank, deliberately, so
        a pattern gets the same search however many patterns run beside it. It does not key on
        which distances are being measured, so two calls for one crystal would draw the same
        perturbation directions and the shells of a curve would share them.
        """
        super()._reseed_for_pattern()
        if self.chunk_tag is not None:
            self.rng = np.random.default_rng(
                [int(self.rng.integers(1 << 62)), int(self.chunk_tag)])

    def _generate_candidates_xnn(self):
        if self.group_size == 1:
            return super()._generate_candidates_xnn()
        total = self.opt_params['convergence_candidates']
        if total % self.group_size:
            raise ValueError(
                f'convergence_candidates {total} is not a multiple of group size '
                f'{self.group_size}, so the distance blocks would not divide into whole groups')
        n_groups = total // self.group_size

        # perturb_xnn draws one direction per candidate, so ask the parent for one candidate per
        # GROUP and repeat afterwards. The repeat comes after the parent's own repair, or two
        # members of one group could be repaired differently and stop sharing a starting cell.
        self.opt_params['convergence_candidates'] = n_groups
        try:
            xnn = super()._generate_candidates_xnn()
        finally:
            self.opt_params['convergence_candidates'] = total
        radii = np.asarray(self.opt_params['convergence_distances'], dtype=float)
        n_distances = radii.size
        block = np.repeat(xnn.reshape(n_distances, n_groups, -1), self.group_size, axis=1)
        block = block.reshape(n_distances, n_groups, self.group_size, -1)

        if self.separation_ratio > 0.0:
            # Isotropic, in every direction including the radial one: two candidates a fixed
            # distance apart in a real pool differ radially too, so a clump displaced only
            # tangentially is one no pool contains. The cost is that members drift off the shell,
            # since E|x+u|^2 = r^2 + m^2 -- about 1 % at a tenth of the radius and 6 % at a
            # seventh of it. That is why the realised distance of every member is recorded: the
            # reduction prices each one at where it actually is.
            offsets = self.rng.uniform(-1.0, 1.0, size=block.shape)
            offsets /= np.linalg.norm(offsets, axis=-1, keepdims=True)
            offsets *= (0.5*self.separation_ratio*radii)[:, np.newaxis, np.newaxis, np.newaxis]
            block = block + offsets
            block = fix_unphysical(
                xnn=block.reshape(-1, block.shape[-1]), rng=self.rng,
                minimum_unit_cell=self.opt_params['minimum_uc'],
                maximum_unit_cell=self.opt_params['maximum_uc'],
                lattice_system=self.lattice_system,
                ).reshape(block.shape)
        elif np.abs(np.diff(block, axis=2)).max(initial=0.0) != 0.0:
            raise RuntimeError(
                'group members differ before refinement at separation 0; they must be the same '
                'starting cell or the measured group rate is not a correlation')

        flat = block.reshape(-1, block.shape[-1])
        self.last_distance = np.linalg.norm(flat - self.xnn_true[np.newaxis], axis=1)
        self.last_separation = np.linalg.norm(
            block - block.mean(axis=2, keepdims=True), axis=-1)
        return flat
