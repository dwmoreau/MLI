"""Model directory locations, shared by the downloader and the optimizer.

The downloader (`mlindex.command_line.download_models`) and the resolver
(`mlindex.optimization.UtilitiesOptimizer._resolve_models_dir`) must agree on where
models live. Keeping the search path in one place stops the two from drifting apart.

A "models directory" is the directory that *directly* contains the per-lattice-system
subdirectories: cubic_1/, hexagonal_1/, monoclinic_1/, orthorhombic_1/, rhombohedral_1/,
tetragonal_1/, triclinic_1/.
"""

import os
from pathlib import Path

# Name of the environment variable users set to point at their models.
ENV_VAR = 'MLINDEX_MODELS_DIR'

# Matches the per-lattice-system directories (cubic_1, hexagonal_1, ...).
LATTICE_DIR_GLOB = '*_1'


def default_models_dir():
    """Return the default install location: <XDG_DATA_HOME>/mlindex/models."""
    xdg_data = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))
    return xdg_data / 'mlindex' / 'models'


def models_dir_from_env():
    """Return the path in MLINDEX_MODELS_DIR, or None if it is unset or blank.

    Surrounding whitespace and a wrapping pair of quotes are stripped: setting an
    environment variable through the Windows GUI routinely produces values like
    '"D:\\models"' or a path with a trailing space.
    """
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return None
    value = raw.strip()
    for quote in ('"', "'"):
        if len(value) >= 2 and value.startswith(quote) and value.endswith(quote):
            value = value[1:-1].strip()
            break
    if not value:
        return None
    return Path(value)


def looks_like_models_dir(path):
    """Return True if `path` holds downloaded models rather than just bundled data.

    The wheel always ships a *partial* mlindex/models/ tree (the hkl_ref_*.npy files
    used by the analytic optimizer), so the presence of `models/` proves nothing.
    cubic_1/integral_filter/ only exists after a real download or a training run.
    """
    return (Path(path) / 'cubic_1' / 'integral_filter').is_dir()
