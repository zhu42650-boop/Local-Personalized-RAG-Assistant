import os
import shutil
from typing import Iterable, List


def ensure_category_dirs(kb_dir: str) -> None:
    os.makedirs(os.path.join(kb_dir, "note"), exist_ok=True)
    os.makedirs(os.path.join(kb_dir, "paper"), exist_ok=True)


def add_files_to_category(kb_dir: str, category: str, files: Iterable[str]) -> List[str]:
    if category not in {"note", "paper"}:
        raise ValueError("category must be 'note' or 'paper'")
    ensure_category_dirs(kb_dir)
    target_dir = os.path.join(kb_dir, category)
    saved = []
    for file_path in files:
        if not os.path.isfile(file_path):
            continue
        basename = os.path.basename(file_path)
        dest = os.path.join(target_dir, basename)
        shutil.copy2(file_path, dest)
        saved.append(dest)
    return saved
