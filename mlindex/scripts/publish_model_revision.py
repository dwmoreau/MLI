"""Publish the repository's model tree to the Hugging Face hub as a new tagged revision.

The repository's `mlindex/models/` is the source of truth. This compares it, file by file and by
content, with a revision already on the hub, and turns the difference into ONE hub commit: files
whose content is unchanged are left alone, a git-lfs file whose content already exists elsewhere
on the hub is copied server-side rather than uploaded again, anything else is uploaded, and a
model path the repository no longer has is deleted. A rename is therefore a copy plus a delete and
costs no upload. The hub's own top-level files -- the model card and its `.gitattributes` -- are
never touched, except that `--model-card` replaces the card in the same commit, so a tagged
revision describes what it holds.

Writing to the hub needs a token with write access, so the default is a dry run that only prints
the plan. The three steps, in the order PROTOCOL rule 7 fixes:

    # 1. see what would change, against the hub's main branch (no token needed)
    python -m mlindex.scripts.publish_model_revision --base main --model-card path/to/README.md

    # 2. apply it as one commit and tag it (needs a write token)
    python -m mlindex.scripts.publish_model_revision --base main --model-card path/to/README.md \
        --apply --tag v2

    # 3. check the tag holds exactly the repository's tree, BEFORE setting model_revision
    python -m mlindex.scripts.publish_model_revision --verify v2

Only after step 3 passes does `model_revision` in `mlindex/model_metadata.json` change. A checkout
naming a tag the hub does not have breaks every fresh install at its first `download_models`.

The commit is made against the exact base commit the plan was computed from, so if anyone pushes
to the hub branch in between, the commit is refused rather than silently applied on top.
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

DEFAULT_REPO = 'dwmoreau/mlindex-models'
# The per-lattice-system directories this script manages; everything else on the hub is left alone.
MODEL_DIR_GLOB = '*_1'


def local_tree(models_dir):
    """{path: (sha256, git blob id)} for every file under the lattice-system directories.

    Both digests are needed because the hub identifies a git-lfs file by its sha256 and a regular
    file by its git blob id, and which one a file is depends on the hub, not on this checkout.
    """
    tree = {}
    for path in sorted(Path(models_dir).glob(f'{MODEL_DIR_GLOB}/**/*')):
        if not path.is_file() or path.name == '.DS_Store':
            continue
        data = path.read_bytes()
        relative = path.relative_to(models_dir).as_posix()
        tree[relative] = (hashlib.sha256(data).hexdigest(), git_blob_id(data))
    return tree


def hub_tree(api, repo_id, revision):
    """{path: (kind, id)}, kind 'lfs' with a sha256 or 'git' with a blob id, and the commit sha."""
    tree = {}
    for entry in api.list_repo_tree(repo_id, revision=revision, recursive=True, expand=True):
        if not hasattr(entry, 'size'):
            continue
        if entry.lfs is not None:
            tree[entry.path] = ('lfs', entry.lfs.sha256)
        else:
            tree[entry.path] = ('git', entry.blob_id)
    commit = api.list_repo_commits(repo_id, revision=revision)[0].commit_id
    return tree, commit


def git_blob_id(data):
    """The id git, and so the hub, gives a regular file with this content."""
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def model_card_changes(card, hub):
    """Whether publishing `card` (bytes) as README.md would change the hub's model card."""
    return hub.get('README.md', (None, None))[1] != git_blob_id(card)


def is_model_path(path, model_dirs):
    return '/' in path and path.split('/', 1)[0] in model_dirs


def plan_sync(local, hub):
    """The operations that turn the hub's model tree into the local one.

    Returns a dict of lists: `keep` (same content at the same path), `copy` as (source, destination)
    pairs of git-lfs content already on the hub, `upload` (content the hub does not have as a
    git-lfs file at any path -- including a regular file that merely moved, since the hub cannot
    copy those), and `delete` (model paths the local tree no longer has).
    """
    model_dirs = {path.split('/', 1)[0] for path in local}
    lfs_source = {}
    for path, (kind, oid) in sorted(hub.items()):
        if kind == 'lfs':
            lfs_source.setdefault(oid, path)

    plan = {'keep': [], 'copy': [], 'upload': [], 'delete': []}
    for path, (sha256, git_oid) in sorted(local.items()):
        on_hub = hub.get(path)
        if on_hub is not None and on_hub[1] in (sha256, git_oid):
            plan['keep'].append(path)
        elif sha256 in lfs_source:
            plan['copy'].append((lfs_source[sha256], path))
        else:
            plan['upload'].append(path)
    plan['delete'] = sorted(path for path in hub
                            if is_model_path(path, model_dirs) and path not in local)
    return plan


def verify(local, hub):
    """Discrepancies between a hub revision's model tree and the local one; empty means identical."""
    model_dirs = {path.split('/', 1)[0] for path in local}
    problems = []
    for path, (sha256, git_oid) in sorted(local.items()):
        on_hub = hub.get(path)
        if on_hub is None:
            problems.append(f'missing on the hub: {path}')
        elif on_hub[1] not in (sha256, git_oid):
            problems.append(f'different content on the hub: {path}')
    for path in sorted(hub):
        if is_model_path(path, model_dirs) and path not in local:
            problems.append(f'on the hub but not in the repository: {path}')
    return problems


def _summarise(plan, local, hub):
    for name in ('keep', 'copy', 'upload', 'delete'):
        print(f'  {name:<7} {len(plan[name]):>5}')
    for name in ('copy', 'upload', 'delete'):
        for item in plan[name][:5]:
            print(f'    {name}: {" -> ".join(item) if isinstance(item, tuple) else item}')
        if len(plan[name]) > 5:
            print(f'    ... and {len(plan[name]) - 5} more')
    untouched = sorted(path for path in hub if '/' not in path)
    print(f'  untouched top-level files: {untouched}')
    print(f'  the new revision will hold {len(local) + len(untouched)} files')


def build_parser():
    parser = argparse.ArgumentParser(
        description='Publish the repository model tree to the Hugging Face hub as a tagged '
                    'revision. Dry run unless --apply is given.')
    parser.add_argument('--repo-id', default=DEFAULT_REPO,
                        help=f'Hub model repository (default: {DEFAULT_REPO}).')
    parser.add_argument('--models-dir', default=os.path.join('mlindex', 'models'),
                        help='The model tree to publish (default: mlindex/models).')
    parser.add_argument('--base', default='main',
                        help='Hub branch or revision to compare against and commit on top of '
                             '(default: main).')
    parser.add_argument('--apply', action='store_true',
                        help='Make the commit. Needs a hub token with write access.')
    parser.add_argument('--tag', default=None,
                        help='With --apply: tag the new commit. Refused if the tag exists.')
    parser.add_argument('--model-card', default=None, metavar='PATH',
                        help='Replace the hub README.md (the model card) with this file, in the '
                             'same commit as the model tree.')
    parser.add_argument('--message', default=None,
                        help='Commit message (default: names the operation counts).')
    parser.add_argument('--verify', default=None, metavar='REVISION',
                        help='Instead of planning, check that REVISION holds exactly the local '
                             'model tree. Exits non-zero on any difference.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete
    from huggingface_hub import HfApi

    api = HfApi()
    models_dir = Path(args.models_dir)
    local = local_tree(models_dir)
    if not local:
        raise SystemExit(f'No model files under {models_dir}/{MODEL_DIR_GLOB}. Is git lfs pulled?')
    print(f'local model tree: {len(local)} files under {models_dir}')

    if args.verify:
        hub, commit = hub_tree(api, args.repo_id, args.verify)
        problems = verify(local, hub)
        print(f'{args.repo_id}@{args.verify} (commit {commit[:10]}): {len(hub)} files')
        for problem in problems[:20]:
            print(f'  {problem}')
        if problems:
            raise SystemExit(f'{len(problems)} differences. Do NOT set model_revision to this.')
        print('identical model tree. model_revision may now name this revision.')
        return 0

    hub, base_commit = hub_tree(api, args.repo_id, args.base)
    plan = plan_sync(local, hub)
    print(f'plan against {args.repo_id}@{args.base} (commit {base_commit[:10]}):')
    _summarise(plan, local, hub)
    card_changed = False
    if args.model_card:
        card_changed = model_card_changes(Path(args.model_card).read_bytes(), hub)
        print(f'  model card: {"replaced from " + args.model_card if card_changed else "unchanged"}')

    if not args.apply:
        print('\ndry run: nothing was written. Re-run with --apply --tag <name> to publish.')
        return 0
    if not (plan['copy'] or plan['upload'] or plan['delete'] or card_changed):
        raise SystemExit('Nothing to change; refusing to make an empty commit.')
    if args.tag:
        existing = {tag.name for tag in api.list_repo_refs(args.repo_id).tags}
        if args.tag in existing:
            raise SystemExit(f'Tag {args.tag} already exists on {args.repo_id}; refusing to move it.')

    operations = ([CommitOperationCopy(src_path_in_repo=source, path_in_repo=destination)
                   for source, destination in plan['copy']]
                  + [CommitOperationAdd(path_in_repo=path,
                                        path_or_fileobj=str(models_dir / Path(path)))
                     for path in plan['upload']]
                  + [CommitOperationDelete(path_in_repo=path) for path in plan['delete']]
                  + ([CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=args.model_card)]
                     if card_changed else []))
    message = args.message or (f'Model tree: {len(plan["copy"])} copied, '
                               f'{len(plan["upload"])} uploaded, {len(plan["delete"])} deleted')
    info = api.create_commit(args.repo_id, operations=operations, commit_message=message,
                             revision=args.base, parent_commit=base_commit)
    print(f'committed {info.oid} to {args.repo_id}@{args.base}')
    if args.tag:
        api.create_tag(args.repo_id, tag=args.tag, revision=info.oid)
        print(f'tagged {args.tag}. Now run --verify {args.tag} before changing model_revision.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
