import os
from dataclasses import dataclass
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class Settings:
    raw: Dict[str, Any]

    def get(self, path: str, default: Any = None) -> Any:
        current: Any = self.raw
        for key in path.split("."):
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current


def _validate_required(settings: Dict[str, Any]) -> None:
    required_paths = [
        "app.name",
        "paths.knowledge_base_dir",
        "paths.vector_store_dir",
        "embedding.model_name",
        "embedding.device",
        "chunk.size",
        "chunk.overlap",
        "retriever.top_k",
        "llm.api_base",
        "llm.api_key",
        "llm.model",
    ]

    missing = []
    for path in required_paths:
        cur = settings
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                missing.append(path)
                cur = None
                break
            cur = cur[key]
        if cur in (None, ""):
            missing.append(path)

    if missing:
        missing_sorted = sorted(set(missing))
        raise ValueError(f"Missing required config fields: {', '.join(missing_sorted)}")


def _resolve_path(base_dir: str, path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(base_dir, path_value))


def load_settings(path: str) -> Settings:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    _validate_required(raw)
    return Settings(raw=raw)


def resolve_paths(settings: Settings, config_path: str) -> Dict[str, str]:
    base_dir = os.path.dirname(os.path.abspath(config_path))
    kb_dir = settings.get("paths.knowledge_base_dir")
    vs_dir = settings.get("paths.vector_store_dir")
    if kb_dir is None or vs_dir is None:
        raise ValueError("paths.knowledge_base_dir and paths.vector_store_dir are required")
    return {
        "knowledge_base_dir": _resolve_path(base_dir, kb_dir),
        "vector_store_dir": _resolve_path(base_dir, vs_dir),
    }
