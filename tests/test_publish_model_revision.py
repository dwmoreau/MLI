"""The hub publisher's plan, which decides what a tagged model release deletes.

Planned against synthetic trees: the operations that touch the hub need a write token and are not
run here. What matters and can be pinned is that the plan is right -- a rename costs no upload
when the hub can copy the content, a regular file that moved is re-uploaded because the hub cannot
copy those, nothing outside the lattice-system directories is ever deleted, and verification
notices every kind of difference.
"""
from mlindex.scripts.publish_model_revision import local_tree
from mlindex.scripts.publish_model_revision import plan_sync
from mlindex.scripts.publish_model_revision import verify


def _local(**files):
    return {path: (f'sha-{content}', f'git-{content}') for path, content in files.items()}


def test_identical_trees_plan_nothing_but_keeps():
    local = _local(**{'cubic_1/abnn/cF_0/w.onnx': 'W'})
    hub = {'cubic_1/abnn/cF_0/w.onnx': ('lfs', 'sha-W'), 'README.md': ('git', 'git-R')}
    plan = plan_sync(local, hub)
    assert plan == {'keep': ['cubic_1/abnn/cF_0/w.onnx'], 'copy': [], 'upload': [], 'delete': []}


def test_a_renamed_lfs_file_is_copied_and_its_old_path_deleted():
    local = _local(**{'cubic_1/abnn/cF_0/cF_0_abnn_weights.onnx': 'W'})
    hub = {'cubic_1/integral_filter/cF_0/cF_0_pitf_weights.onnx': ('lfs', 'sha-W')}
    plan = plan_sync(local, hub)
    assert plan['copy'] == [('cubic_1/integral_filter/cF_0/cF_0_pitf_weights.onnx',
                             'cubic_1/abnn/cF_0/cF_0_abnn_weights.onnx')]
    assert plan['upload'] == []
    assert plan['delete'] == ['cubic_1/integral_filter/cF_0/cF_0_pitf_weights.onnx']


def test_a_renamed_regular_file_is_uploaded_because_the_hub_cannot_copy_it():
    local = _local(**{'cubic_1/abnn/cF_0/cF_0_abnn_params.csv': 'P'})
    hub = {'cubic_1/integral_filter/cF_0/cF_0_pitf_params.csv': ('git', 'git-P')}
    plan = plan_sync(local, hub)
    assert plan['copy'] == []
    assert plan['upload'] == ['cubic_1/abnn/cF_0/cF_0_abnn_params.csv']
    assert plan['delete'] == ['cubic_1/integral_filter/cF_0/cF_0_pitf_params.csv']


def test_a_file_the_repository_dropped_is_deleted_and_top_level_files_are_not():
    local = _local(**{'cubic_1/abnn/cF_0/w.onnx': 'W'})
    hub = {'cubic_1/abnn/cF_0/w.onnx': ('lfs', 'sha-W'),
           'cubic_1/integral_filter/cF_0/cF_0_calibration_weights.onnx': ('lfs', 'sha-C'),
           'README.md': ('git', 'git-R'),
           '.gitattributes': ('git', 'git-G'),
           'notes/other.txt': ('git', 'git-N')}
    plan = plan_sync(local, hub)
    assert plan['delete'] == ['cubic_1/integral_filter/cF_0/cF_0_calibration_weights.onnx']


def test_changed_content_at_the_same_path_is_uploaded():
    local = _local(**{'cubic_1/abnn/cF_0/w.onnx': 'NEW'})
    hub = {'cubic_1/abnn/cF_0/w.onnx': ('lfs', 'sha-OLD')}
    assert plan_sync(local, hub)['upload'] == ['cubic_1/abnn/cF_0/w.onnx']


def test_verify_reports_missing_changed_and_extra_model_paths_only():
    local = _local(**{'cubic_1/a.npy': 'A', 'cubic_1/b.npy': 'B', 'cubic_1/c.npy': 'C'})
    hub = {'cubic_1/a.npy': ('lfs', 'sha-A'),
           'cubic_1/b.npy': ('lfs', 'sha-CHANGED'),
           'cubic_1/stale.npy': ('lfs', 'sha-S'),
           'README.md': ('git', 'git-R')}
    problems = verify(local, hub)
    assert problems == ['different content on the hub: cubic_1/b.npy',
                        'missing on the hub: cubic_1/c.npy',
                        'on the hub but not in the repository: cubic_1/stale.npy']
    assert verify(local, {**hub, 'cubic_1/b.npy': ('lfs', 'sha-B'),
                          'cubic_1/c.npy': ('git', 'git-C')}) == [
        'on the hub but not in the repository: cubic_1/stale.npy']


def test_the_model_card_is_replaced_only_when_its_content_differs():
    from mlindex.scripts.publish_model_revision import git_blob_id
    from mlindex.scripts.publish_model_revision import model_card_changes

    card = b'# MLINDEX models\n'
    assert model_card_changes(card, {'README.md': ('git', git_blob_id(card))}) is False
    assert model_card_changes(card, {'README.md': ('git', git_blob_id(b'old card'))}) is True
    assert model_card_changes(card, {}) is True


def test_local_tree_uses_both_digests_skips_ds_store_and_other_directories(tmp_path):
    import hashlib

    (tmp_path / 'cubic_1' / 'abnn').mkdir(parents=True)
    (tmp_path / 'cubic_1' / 'abnn' / 'w.onnx').write_bytes(b'hello')
    (tmp_path / 'cubic_1' / '.DS_Store').write_bytes(b'x')
    (tmp_path / 'fom_prior').mkdir()
    (tmp_path / 'fom_prior' / 'model.bin').write_bytes(b'y')

    tree = local_tree(tmp_path)

    assert list(tree) == ['cubic_1/abnn/w.onnx']
    sha256, git_oid = tree['cubic_1/abnn/w.onnx']
    assert sha256 == hashlib.sha256(b'hello').hexdigest()
    # git's blob id for "hello": the value `git hash-object` prints
    assert git_oid == 'b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0'
