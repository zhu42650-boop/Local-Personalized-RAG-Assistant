import os
from typing import Dict

from config.loader import Settings


def ensure_dirs(paths: Dict[str, str]) -> None:
    for _, p in paths.items():
        os.makedirs(p, exist_ok=True)
    kb_dir = paths.get("knowledge_base_dir")
    if kb_dir:
        os.makedirs(os.path.join(kb_dir, "note"), exist_ok=True)
        os.makedirs(os.path.join(kb_dir, "paper"), exist_ok=True)


def check_embedding_model(settings: Settings) -> None:
    # Minimal check: ensure sentence-transformers is importable and model string is set.
    model_name = settings.get("embedding.model_name")
    if not model_name:
        raise ValueError("embedding.model_name is required")

    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except Exception as exc:  # pragma: no cover - import error should surface clearly
        raise RuntimeError("sentence-transformers is required for local embeddings") from exc
