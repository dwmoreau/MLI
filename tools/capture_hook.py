"""The hook the two figure-of-merit reproducers capture their real arguments through.

DEVELOPMENT TOOL -- not part of the installed package.

On branch `fom` both reproducers patched `Candidates._retention_fom_values`, a hook that
exists only there. Campaign 2 does not port the multi-merit iterate retention that method
belongs to -- it buys four entries of 972 (F-155) -- so there is nothing here to patch, and
adding the hook to `Candidates.py` to suit a development tool would be the tail wagging the
dog. The capture point on this branch is the module-level name
`mlindex.optimization.Candidates.get_M20` instead.

Three things this has to get right, which is why it is a module rather than four lines
inlined twice.

**`get_M20` mutates its input.** It calls `np.putmask` on `q2_ref_calc`, zeroing every
reference line above the cut-off. A capture taken afterwards would hold an array full of
holes, and every reverse or symmetric merit scored on it would be wrong. So the arrays are
copied *before* the real function is called.

**The module-level name carries no `self`.** The capture needs `xnn`, `lattice_system` and
`bravais_lattice` off the live `Candidates` instance -- `compute_all`, `get_bic`,
`get_zone_dominance` and the de Wolff terms all take `xnn`. `assign_hkls` is wrapped to
publish the instance for the duration of its own call, which needs no frame introspection
and leaves `Candidates.py` untouched.

**Only one of five call sites is the right one.** `get_M20` is called from `assign_hkls`,
`correct_zero_error`, `refine_cell`, `correct_off_by_two` and `assign_extinction_group`.
Only `assign_hkls` hands over the arrays the optimiser actually scored, row-aligned with
`self.xnn`; recording anywhere else would pair a merit with the wrong cell. Publishing the
instance only inside `assign_hkls` is what confines the recorder to it.
"""

import contextlib

__all__ = ['capture_get_M20']


class _Record:
    """What one capture run saw: every invocation's shape, and the largest one's arrays."""

    def __init__(self):
        self.shapes = []          # (n_candidates, n_ref, n_peaks) per invocation
        self.rows = -1            # the largest UNtruncated candidate count seen
        self.arrays = None        # (q2_obs, q2_calc, q2_ref_calc, xnn), truncated
        self.lattice_system = None
        self.bravais_lattice = None


@contextlib.contextmanager
def capture_get_M20(max_rows):
    """Patch `Candidates.get_M20` to record the largest call `assign_hkls` makes.

    Yields a `_Record`; both patches are restored on the way out, including on an
    exception, so a failed run cannot leave the optimiser wrapped.
    """
    from mlindex.optimization import Candidates as candidates_module
    from mlindex.optimization.Candidates import Candidates

    record = _Record()
    holder = {'owner': None}
    original_assign = Candidates.assign_hkls
    original_get_M20 = candidates_module.get_M20

    def assign_hkls_wrapper(self):
        holder['owner'] = self
        try:
            return original_assign(self)
        finally:
            holder['owner'] = None

    def get_M20_recorder(q2_obs, q2_calc, q2_ref_calc):
        owner = holder['owner']
        if owner is not None:
            record.shapes.append((q2_calc.shape[0], q2_ref_calc.shape[1], q2_calc.shape[1]))
            if q2_calc.shape[0] > record.rows:
                rows = min(q2_calc.shape[0], max_rows)
                record.rows = q2_calc.shape[0]
                # Copied before delegating: the real get_M20 is about to putmask q2_ref_calc.
                record.arrays = (q2_obs.copy(), q2_calc[:rows].copy(),
                                 q2_ref_calc[:rows].copy(), owner.xnn[:rows].copy())
                record.lattice_system = owner.lattice_system
                record.bravais_lattice = owner.bravais_lattice
        return original_get_M20(q2_obs, q2_calc, q2_ref_calc)

    Candidates.assign_hkls = assign_hkls_wrapper
    candidates_module.get_M20 = get_M20_recorder
    try:
        yield record
    finally:
        Candidates.assign_hkls = original_assign
        candidates_module.get_M20 = original_get_M20
