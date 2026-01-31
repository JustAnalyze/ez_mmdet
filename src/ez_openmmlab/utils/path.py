from pathlib import Path
from typing import Union


def get_unique_dir(base_dir: Union[str, Path]) -> Path:
    """Returns a unique directory path by appending a numbered suffix if it already exists.

    Example:
        If 'runs/preds' exists, returns 'runs/preds_1'
        If 'runs/preds_1' exists, returns 'runs/preds_2'

    Args:
        base_dir: The desired base directory path.

    Returns:
        A Path object representing a unique directory.
    """
    path = Path(base_dir)
    if not path.exists():
        return path

    parent = path.parent
    stem = path.name
    
    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}"
        if not new_path.exists():
            return new_path
        counter += 1
