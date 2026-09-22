"""Stable digests, and stable seeds, for things that must not change between processes.

Two callers need the same peak list reduced to bytes the same way: the search re-keys its random
generator on the pattern it is about to index, and a benchmark carries a short digest in both the
candidate and the entry table so a mis-joined shard is detectable. A mis-join is otherwise silent
-- every column still parses, the numbers are simply attached to the wrong pattern.

`hash()` will not do for either: it is salted per process, so the same pattern would digest
differently on every run. The dtype is pinned little-endian so a peak list gives the same bytes on
every machine, which matters because these digests are compared across machines -- a benchmark is
generated on one and analysed on another.
"""

import hashlib

import numpy as np


def derived_seed(key, base_seed):
    """A stable seed for `key`. `hash()` will not do: it is salted per process."""
    digest = hashlib.sha256(f'{base_seed}:{key}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big')


def peak_list_bytes(q2):
    """A peak list as canonical bytes: contiguous, float64, little-endian."""
    return np.ascontiguousarray(q2, dtype='<f8').tobytes()


def q2_digest(q2, digest_size=8):
    """A short hexadecimal digest of a peak list, for checking a join rather than securing one."""
    return hashlib.blake2b(peak_list_bytes(q2), digest_size=digest_size).hexdigest()
