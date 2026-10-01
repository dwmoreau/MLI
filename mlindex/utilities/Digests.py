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


def file_digest(path):
    """The sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root, subdirectories):
    """The sha256 of every file under `root/<subdirectory>`, names and contents together.

    Files are taken in sorted order of their path relative to `root`, written with forward
    slashes, so the same tree digests the same on every operating system. Returns
    (hexdigest, number of files).
    """
    from pathlib import Path

    root = Path(root)
    names = sorted(path.relative_to(root).as_posix() for subdirectory in subdirectories
                   for path in (root / subdirectory).rglob('*') if path.is_file())
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode('utf-8'))
        digest.update(bytes.fromhex(file_digest(root.joinpath(*name.split('/')))))
    return digest.hexdigest(), len(names)
