"""Download ML models for mlindex from the git LFS repository.

Models are placed in the XDG data directory (~/.local/share/mlindex/models/) by default,
or a custom path specified via --models-dir or the MLINDEX_MODELS_DIR environment variable.

Requires git and git-lfs to be installed:
    https://git-lfs.com
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _check_git_lfs():
    """Verify git and git-lfs are available; exit with instructions if not."""
    for cmd in (['git', '--version'], ['git', 'lfs', 'version']):
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(
                f"ERROR: '{' '.join(cmd)}' failed.\n"
                "git and git-lfs must be installed before downloading models.\n"
                "Install git-lfs: https://git-lfs.com",
                file=sys.stderr,
            )
            sys.exit(1)


def _default_models_dir():
    xdg_data = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))
    return xdg_data / 'mlindex' / 'models'


def _load_metadata():
    """Read model_metadata.json bundled with the package."""
    try:
        import mlindex
        meta_path = Path(mlindex.__file__).parent / 'model_metadata.json'
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return {}


def _run(cmd, **kwargs):
    """Run a subprocess command, streaming output to the terminal."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)


def _verify_no_lfs_pointers(models_dir: Path) -> None:
    """Exit with a clear error if any model files are still LFS pointer stubs."""
    lfs_marker = b'version https://git-lfs'
    pointers = []
    for f in models_dir.rglob('*'):
        if f.is_file() and f.stat().st_size <= 200:
            try:
                with open(f, 'rb') as fh:
                    if fh.read(22) == lfs_marker:
                        pointers.append(f.relative_to(models_dir))
            except OSError:
                pass
    if pointers:
        sample = '\n  '.join(str(p) for p in pointers[:5])
        print(
            f"\nERROR: {len(pointers)} model file(s) are still LFS pointer stubs after download.\n"
            f"  {sample}\n"
            "Ensure git-lfs is installed ('git lfs version') and re-run with --force.",
            file=sys.stderr,
        )
        sys.exit(1)


def _download(repo_url, branch, models_dir):
    """Clone the repo sparsely and pull LFS model files into models_dir."""
    with tempfile.TemporaryDirectory(prefix='mlindex_download_') as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"\nCloning repository (sparse, no checkout) ...")
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

    _verify_no_lfs_pointers(models_dir)
    print(f"Models installed to: {models_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download ML models for mlindex from git LFS."
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
        help="Overwrite existing models directory without prompting.",
    )
    args = parser.parse_args()

    meta = _load_metadata()
    repo_url = meta.get('repo_url', 'https://github.com/dwmoreau/MLI.git')
    branch = meta.get('models_branch', 'main')

    if args.models_dir:
        models_dir = Path(args.models_dir)
    elif os.environ.get('MLINDEX_MODELS_DIR'):
        models_dir = Path(os.environ['MLINDEX_MODELS_DIR'])
    else:
        models_dir = _default_models_dir()

    if models_dir.exists() and any(models_dir.iterdir()):
        if not args.force:
            print(
                f"ERROR: {models_dir} already exists and is non-empty.\n"
                "Use --force to overwrite.",
                file=sys.stderr,
            )
            sys.exit(1)

    _check_git_lfs()

    print(f"Downloading mlindex models from {repo_url} (branch: {branch})")
    print(f"Target directory: {models_dir}\n")

    _download(repo_url, branch, models_dir)

    print("\nDownload complete.")
    print(
        f"To use these models, set:\n"
        f"  export MLINDEX_MODELS_DIR={models_dir}"
        if models_dir != _default_models_dir() else
        "\nModels are installed to the default location and will be found automatically."
    )


if __name__ == '__main__':
    main()

