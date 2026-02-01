import os
from pathlib import Path
from typing import List

DATASETS_DIR = Path("./datasets")

def get_dataset_names() -> List[str]:
    """
    Reads dataset names from a file (one per line).
    - Strips whitespace
    - Skips blank lines
    - Skips comment lines starting with '#'
    - De-duplicates while preserving order
    """
    seen = set()
    names: List[str] = []

    base_dir = Path(__file__).resolve().parent
    file_path = base_dir / "datasets" / "datasets.txt"

    if not file_path.exists():
        return []

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line not in seen:
            seen.add(line)
            names.append(line)

    return names