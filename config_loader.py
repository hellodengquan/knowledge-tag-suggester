import json
import os
import copy
from typing import Any, Dict, Optional, Tuple


_DEFAULT_CONFIG_PATHS = [
    "tag_suggester_config.json",
    os.path.expanduser("~/.tag_suggester_config.json"),
    os.environ.get("TAG_SUGGESTER_CONFIG", ""),
]


class AppConfig:
    """加载并提供知识标签推荐器的统一配置。"""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    @classmethod
    def load(cls, path: Optional[str] = None) -> "AppConfig":
        if path:
            if not os.path.exists(path):
                raise FileNotFoundError(f"配置文件不存在: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(data)

        for candidate in _DEFAULT_CONFIG_PATHS:
            if candidate and os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as f:
                    return cls(json.load(f))

        return cls(_BUILTIN_DEFAULTS)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def get_range(self, dotted_key: str) -> Tuple[Optional[float], Optional[float]]:
        spec = self.get(dotted_key)
        if not isinstance(spec, dict):
            return None, None
        return spec.get("min"), spec.get("max")

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)


_BUILTIN_DEFAULTS = {
    "model": {
        "default_model_path": "tag_model.pkl",
        "default_top_k": 5,
        "default_threshold": 0.1,
        "max_features": 5000,
        "default_fields": {
            "summary_field": "summary",
            "tags_field": "tags",
        },
    },
    "suggest": {
        "default_top_k": 5,
        "default_threshold": 0.1,
        "explain_words": 3,
    },
    "validation": {
        "top_k": {"min": 1, "max": 100},
        "threshold": {"min": 0.0, "max": 1.0},
        "max_features": {"min": 1, "max": 100000},
        "explain_words": {"min": 1, "max": 20},
        "limit": {"min": 1, "max": 10000},
        "offset": {"min": 0},
    },
    "feedback": {
        "default_backend": "json",
        "storage": {
            "json": "feedback_data.json",
            "sqlite": "feedback_data.db",
        },
    },
    "migrate": {
        "default_db_path": "feedback_data.db",
    },
    "cli": {
        "prog_name": "tag-suggester",
        "description": "知识标签推荐器 — 基于 TF-IDF + 多标签分类的文档标签推荐工具",
    },
}
