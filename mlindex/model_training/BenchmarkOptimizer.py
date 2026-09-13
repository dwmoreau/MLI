"""The optimizer a benchmark run uses: an ordinary manager that keeps what it found.

The shipped indexer reports twenty candidates per Bravais lattice and forgets the rest. A
benchmark needs all of them -- the question it answers is where in the pooled ranking the correct
cell sits, and a cell truncated away has no rank at all. It also needs three quantities that exist
nowhere else, because they are bookkeeping about the search rather than properties of a cell:
`n_entering` (how large the pool was before near-duplicates were collapsed), `final_rank` (the rank
over every survivor, so truncation is a column rather than a missing row) and `in_top_n`.

`OptimizerManager._on_downsample` is the observation point and does nothing on the shipped path.
Everything this module knows about columns, schemas and identity stays here; the optimizer knows
only that an observation point exists.

The records come back as one dict of columns per (pattern, Bravais lattice), which
`Benchmark.records_to_frame` concatenates. Columns rather than rows because a triclinic pattern
produces several thousand survivors and a pool holds hundreds of patterns.
"""

import numpy as np

from mlindex.optimization.MPOptimizer import MPOptimizerManager
from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import get_unit_cell_volume


class BenchmarkOptimizer(MPOptimizerManager):
    """A manager that buffers every survivor of every pattern it is given.

    The driver sets `dump_context` to identify the pattern before each run -- the optimizer is
    never told which crystal it is looking at, and a record that cannot name its pattern cannot be
    joined to the truth. It then calls `drain` to take the buffer and reset it, which is what keeps
    memory bounded over a pool of hundreds of patterns.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dump_context = None
        self._records = []

    def drain(self):
        """Hand over the buffered records and start a new buffer."""
        records = self._records
        self._records = []
        return records

    def _identified(self):
        """The pattern these candidates belong to, refused rather than guessed.

        The digest in particular cannot be computed here. A manager truncates the peak list to
        the number of lines its own lattice system is fitted on -- ten for cubic, twenty for the
        rest -- so `self.q2_obs` is a different array per lattice and digesting it would give a
        pattern several different identities, each of which parses and joins to nothing. The
        driver holds the peak list as given and supplies the digest of that.
        """
        context = self.dump_context or {}
        missing = [key for key in ('entry_id', 'condition_bundle', 'q2_digest')
                   if not context.get(key)]
        if missing:
            raise ValueError(
                f'dump_context is missing {missing}. A candidate that cannot name its pattern '
                'cannot be joined to the truth, and a wrong name joins silently to the wrong '
                'crystal -- set dump_context before running a pattern.'
                )
        return context

    def _on_downsample(self, survivors, order, n_entering, n_top_candidates):
        if self.zero_error:
            raise NotImplementedError(
                'A benchmark run cannot use zero-error refinement: the per-candidate zeropoint '
                'stays with the worker that fitted it and never reaches the manager, so the '
                'recorded cells would not reproduce the M20 the pipeline computed.'
                )

        xnn = survivors['xnn']
        n_candidates = xnn.shape[0]

        # Rank by descending M20 over ALL survivors, from the order the truncation below is about
        # to use. Ranking over the retained twenty instead would make truncation a missing row,
        # and a rank recomputed later from whichever rows a file happens to hold drifts every time
        # rows are dropped.
        final_rank = np.empty(n_candidates, dtype=np.int64)
        final_rank[order] = np.arange(n_candidates)

        unit_cell = get_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=self.lattice_system)
        reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
            xnn, partial_unit_cell=True, lattice_system=self.lattice_system)

        context = self._identified()
        self._records.append({
            'entry_id': context['entry_id'],
            'condition_bundle': context['condition_bundle'],
            # The join-integrity check: the entry table carries the digest of the same peak list,
            # and a shard whose rows disagree with it is describing a different pattern.
            'q2_digest': context['q2_digest'],
            'bravais_lattice': self.bravais_lattice,
            'lattice_system': self.lattice_system,
            'n_peaks': int(self.n_peaks),
            'hkl_ref_length': int(self.hkl_ref_length),
            'n_entering': int(n_entering),
            'assignment_threshold': float(self.opt_params['assignment_threshold']),
            'downsample_radius': float(self.opt_params['downsample_radius']),
            'prune_threshold': float(self.opt_params.get('prune_m20_threshold', 5.0)),
            # Position in this shard, which is the order deduplication returned them in. It is the
            # last tie-break the reduction falls back to, so it only has to be stable.
            'candidate_id': np.arange(n_candidates, dtype=np.int64),
            'xnn': np.array(xnn, dtype=np.float64, copy=True),
            'unit_cell': np.array(unit_cell, dtype=np.float64, copy=True),
            'volume': get_unit_cell_volume(
                unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system),
            'reciprocal_volume': get_unit_cell_volume(
                reciprocal_unit_cell, partial_unit_cell=True,
                lattice_system=self.lattice_system),
            'spacegroup': list(survivors['spacegroup']),
            'M20': np.array(survivors['M20'], dtype=np.float64, copy=True),
            'n_indexed': np.array(survivors['n_indexed'], dtype=np.int64, copy=True),
            'final_rank': final_rank,
            'in_top_n': final_rank < int(n_top_candidates),
            })
