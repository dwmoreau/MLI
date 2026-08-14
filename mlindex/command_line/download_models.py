"""Download ML models for mlindex from the Hugging Face Hub.

Models are placed in the XDG data directory (~/.local/share/mlindex/models/) by default,
or a custom path specified via --models-dir or the MLINDEX_MODELS_DIR environment variable.

The download requires no git and no git-lfs -- only the huggingface_hub package, which
is installed as a normal dependency of mlindex.

A legacy git-lfs download from the GitHub repository is still available via
--source github, for networks that block huggingface.co. It requires git and git-lfs,
and will be removed in a future release.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from mlindex import paths

DEFAULT_HF_REPO_ID = 'dwmoreau/mlindex-models'
DEFAULT_GITHUB_REPO_URL = 'https://github.com/dwmoreau/MLI.git'
DEFAULT_GITHUB_BRANCH = 'main'

# Approximate download size, for the progress banner.
DOWNLOAD_SIZE_HINT = '~545 MB, 784 files'


def _hf_imports():
    """Import huggingface_hub lazily; returns (snapshot_download, errors module).

    Imported inside a function so that --help works without the dependency, and so
    tests can replace this seam without touching the network. `errors` is imported as
    a module because that is the one form valid across huggingface_hub 0.25 -> 1.x.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub import errors as hf_errors
    return snapshot_download, hf_errors


def _default_models_dir():
    return paths.default_models_dir()


def _load_metadata():
    """Read model_metadata.json bundled with the package."""
    try:
        import mlindex
        meta_path = Path(mlindex.__file__).parent / 'model_metadata.json'
        with open(meta_path) as f:
            return json.load(f)
    except (OSError, ValueError, ImportError) as exc:
        print(
            f"WARNING: could not read model_metadata.json ({exc}); using built-in defaults.",
            file=sys.stderr,
        )
        return {}


def _resolve_target_dir(models_dir_arg):
    """Return the directory to install into: --models-dir > env var > default."""
    if models_dir_arg:
        return Path(models_dir_arg)
    from_env = paths.models_dir_from_env()
    if from_env is not None:
        return from_env
    return _default_models_dir()


def _existing_install_state(models_dir):
    """Classify what is already at models_dir.

    Returns 'empty', 'previous_download' (safe to resume/refresh over), or
    'foreign_data' (unrelated files we should not touch without --force).
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return 'empty'
    entries = [p for p in models_dir.iterdir() if not p.name.startswith('.')]
    if (models_dir / '.cache' / 'huggingface').is_dir():
        return 'previous_download'
    if any(models_dir.glob(paths.LATTICE_DIR_GLOB)):
        return 'previous_download'
    return 'foreign_data' if entries else 'empty'


def _is_rate_limited(exc):
    """True if the exception looks like an HTTP 429 from the Hub."""
    text = str(exc)
    return '429' in text or 'Too Many Requests' in text


def _is_network_error(exc):
    """True if the exception looks like a transport failure rather than a disk problem."""
    text = str(exc).lower()
    return any(marker in text for marker in (
        'network', 'connection', 'timed out', 'timeout', 'http status', 'ssl', 'proxy',
    ))


def _download_hf(repo_id, revision, models_dir, redownload=False):
    """Download the model snapshot from the Hugging Face Hub into models_dir."""
    try:
        snapshot_download, hf_errors = _hf_imports()
    except ImportError as exc:
        print(
            f"ERROR: huggingface_hub is required to download models ({exc}).\n"
            "Install it with:\n"
            "    pip install 'huggingface_hub>=0.25'\n"
            "or re-run with --source github to use the legacy git-lfs download.",
            file=sys.stderr,
        )
        sys.exit(1)

    models_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type='model',
            revision=revision,
            local_dir=str(models_dir),
            force_download=redownload,
            max_workers=8,
        )
    except hf_errors.RevisionNotFoundError:
        print(
            f"\nERROR: revision '{revision}' was not found in {repo_id}.\n"
            "This usually means your installed mlindex version pins a model revision\n"
            "that no longer exists. Upgrade mlindex ('pip install -U mlindex'), or\n"
            "re-run with '--revision main' to take the latest models.",
            file=sys.stderr,
        )
        sys.exit(1)
    except hf_errors.RepositoryNotFoundError:
        print(
            f"\nERROR: repository '{repo_id}' was not found, or is private/gated.\n"
            "Check the name with --repo-id, and if the repository is private set an\n"
            "access token in the HF_TOKEN environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)
    except hf_errors.LocalEntryNotFoundError:
        print(
            "\nERROR: could not reach huggingface.co and no local copy is available.\n"
            "Check your network connection or proxy settings. If your network blocks\n"
            "huggingface.co, set HF_ENDPOINT to a mirror, or re-run with --source github.",
            file=sys.stderr,
        )
        sys.exit(1)
    except hf_errors.HfHubHTTPError as exc:
        print(f"\nERROR: the Hugging Face Hub returned an error:\n{exc}", file=sys.stderr)
        sys.exit(1)
    except OSError as exc:
        # huggingface_hub wraps transport failures (including Xet transfer errors) in
        # OSError subclasses, so an OSError here is not necessarily a disk problem.
        if _is_rate_limited(exc):
            print(
                f"\nERROR: the Hugging Face Hub rate-limited this download ({exc}).\n"
                "Anonymous downloads have a low rate limit and this repository has 780 files.\n"
                "Re-run the same command to resume -- files already downloaded are kept and\n"
                "will not be fetched again. To raise the limit, log in first ('hf auth login')\n"
                "or set the HF_TOKEN environment variable.",
                file=sys.stderr,
            )
        elif _is_network_error(exc):
            print(
                f"\nERROR: the download failed with a network error ({exc}).\n"
                "Re-run the same command to resume -- already downloaded files are kept.",
                file=sys.stderr,
            )
        else:
            print(
                f"\nERROR: could not write to {models_dir} ({exc}).\n"
                "About 600 MB of free space is required.",
                file=sys.stderr,
            )
        sys.exit(1)

    print(f"Models installed to: {models_dir}")


def _check_git_lfs():
    """Verify git and git-lfs are available; exit with instructions if not."""
    for cmd in (['git', '--version'], ['git', 'lfs', 'version']):
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(
                f"ERROR: '{' '.join(cmd)}' failed.\n"
                "git and git-lfs must be installed to use --source github.\n"
                "Install git-lfs: https://git-lfs.com\n"
                "Alternatively, use the default Hugging Face download, which needs neither.",
                file=sys.stderr,
            )
            sys.exit(1)


def _run(cmd, **kwargs):
    """Run a subprocess command, streaming output to the terminal."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def _download_git_lfs(repo_url, branch, models_dir):
    """Legacy path: clone the repo sparsely and pull LFS model files into models_dir."""
    _check_git_lfs()
    with tempfile.TemporaryDirectory(prefix='mlindex_download_') as tmpdir:
        tmpdir = Path(tmpdir)
        print("\nCloning repository (sparse, no checkout) ...")
        _run(
            ['git', 'clone', '--no-checkout',
             '--depth=1', '--branch', branch, repo_url, str(tmpdir / 'repo')]
        )
        repo = tmpdir / 'repo'

        print("Configuring sparse checkout for mlindex/models/ ...")
        _run(['git', '-C', str(repo), 'sparse-checkout', 'init', '--cone'])
        _run(['git', '-C', str(repo), 'sparse-checkout', 'set', 'mlindex/models'])

        print("\nChecking out model files ...")
        _run(['git', '-C', str(repo), 'checkout', 'HEAD'])

        print("Fetching LFS objects ...")
        _run(['git', '-C', str(repo), 'lfs', 'pull', '--include', 'mlindex/models/**'])

        src = repo / 'mlindex' / 'models'
        if not src.exists():
            print(f"ERROR: expected {src} after download", file=sys.stderr)
            sys.exit(1)

        print(f"\nMoving models to {models_dir} ...")
        models_dir.parent.mkdir(parents=True, exist_ok=True)
        if models_dir.exists():
            shutil.rmtree(models_dir)
        shutil.move(str(src), str(models_dir))

    print(f"Models installed to: {models_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download ML models for mlindex from the Hugging Face Hub."
    )
    parser.add_argument(
        '--models-dir',
        default=None,
        help=(
            "Directory to install models into (will contain cubic_1/, hexagonal_1/, etc.). "
            "Defaults to MLINDEX_MODELS_DIR env var, or ~/.local/share/mlindex/models/."
        ),
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help=(
            "Write into the target directory even if it already holds unrelated files. "
            "Does not re-download files that are already present -- see --redownload."
        ),
    )
    parser.add_argument(
        '--redownload',
        action='store_true',
        help="Re-fetch every file even if it is already present and up to date.",
    )
    parser.add_argument(
        '--revision',
        default=None,
        help="Model revision (tag, branch, or commit) to download. Defaults to the "
             "revision pinned by this mlindex version.",
    )
    parser.add_argument(
        '--repo-id',
        default=None,
        help=f"Hugging Face repository to download from (default: {DEFAULT_HF_REPO_ID}).",
    )
    parser.add_argument(
        '--source',
        choices=('hf', 'github'),
        default='hf',
        help="Where to download from. 'hf' (default) uses the Hugging Face Hub; "
             "'github' uses the legacy git-lfs clone and requires git and git-lfs.",
    )
    args = parser.parse_args()

    meta = _load_metadata()
    repo_id = args.repo_id or meta.get('hf_repo_id') or DEFAULT_HF_REPO_ID
    revision = args.revision or meta.get('model_revision')
    repo_url = meta.get('repo_url', DEFAULT_GITHUB_REPO_URL)
    branch = meta.get('models_branch', DEFAULT_GITHUB_BRANCH)

    models_dir = _resolve_target_dir(args.models_dir)

    state = _existing_install_state(models_dir)
    if state == 'foreign_data' and not args.force:
        print(
            f"ERROR: {models_dir} already exists and contains unrelated files.\n"
            "Use --force to install into it anyway, or choose another --models-dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.source == 'github':
        print(f"Downloading mlindex models from {repo_url} (branch: {branch})")
        print(f"Target directory: {models_dir}\n")
        _download_git_lfs(repo_url, branch, models_dir)
    else:
        print(f"Downloading mlindex models from https://huggingface.co/{repo_id}")
        print(f"Revision: {revision or 'main (unpinned)'}")
        print(f"Target directory: {models_dir}")
        print(f"Download size: {DOWNLOAD_SIZE_HINT}\n")
        if state == 'previous_download':
            print("Existing model directory found; verifying and completing download ...")
            print("(Files already present and up to date are not re-downloaded. To start")
            print(" completely fresh, delete the directory and re-run.)\n")
        try:
            _download_hf(repo_id, revision, models_dir, redownload=args.redownload)
        except KeyboardInterrupt:
            print(
                "\nInterrupted. The partial download has been kept -- "
                "re-run the same command to resume.",
                file=sys.stderr,
            )
            sys.exit(130)

    print("\nDownload complete.")
    if models_dir != _default_models_dir():
        print("To use these models, set:")
        print(f"  export {paths.ENV_VAR}={models_dir}")
    else:
        print("Models are installed to the default location and will be found automatically.")


if __name__ == '__main__':
    main()
