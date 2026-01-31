from pathlib import Path
from ez_openmmlab.utils.path import get_unique_dir


def test_get_unique_dir_returns_base_if_not_exists(tmp_path):
    """Verifies that the base path is returned if it doesn't exist."""
    base = tmp_path / "new_dir"
    unique = get_unique_dir(base)
    assert unique == base


def test_get_unique_dir_increments_if_exists(tmp_path):
    """Verifies that a numbered suffix is added if the path exists."""
    base = tmp_path / "preds"
    base.mkdir()
    
    unique = get_unique_dir(base)
    assert unique == tmp_path / "preds_1"


def test_get_unique_dir_skips_existing_increments(tmp_path):
    """Verifies that it finds the first available increment."""
    base = tmp_path / "results"
    base.mkdir()
    (tmp_path / "results_1").mkdir()
    (tmp_path / "results_2").mkdir()
    
    unique = get_unique_dir(base)
    assert unique == tmp_path / "results_3"
