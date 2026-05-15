import json
from pathlib import Path
from typing import Dict, Tuple


DEFAULT_OBJECT_NAME_ALIASES: Dict[str, str] = {
    "cell phone": "cell phone",
    "mobile phone": "cell phone",
    "phone": "cell phone",
    "smartphone": "cell phone",
    "book": "book",
    "textbook": "book",
    "notebook": "notebook",
    "exercise book": "notebook",
    "laptop": "laptop",
    "computer": "laptop",
    "pen": "pen",
    "pencil": "pen",
}


DEFAULT_OBJECT_ACTION_PRIORS: Dict[str, Dict[str, float]] = {
    "cell phone": {"phone": 1.0, "head_down": 0.30, "write": 0.25},
    "book": {"read": 1.0, "note": 0.55, "head_down": 0.35},
    "notebook": {"note": 1.0, "write": 0.85, "read": 0.40},
    "laptop": {"laptop_related": 1.0, "read": 0.70, "note": 0.60},
    "pen": {"write": 0.90, "note": 0.65},
}


DEFAULT_AMBIGUITY_PAIRS = [
    ("head_down", "phone"),
    ("read", "note"),
    ("write", "phone"),
    ("laptop_related", "read"),
    ("laptop_related", "note"),
]


def _normalize_key(value: str) -> str:
    return str(value or "").strip().lower()


def load_object_evidence_config(config_path: Path = Path()) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    aliases = dict(DEFAULT_OBJECT_NAME_ALIASES)
    priors: Dict[str, Dict[str, float]] = {
        key: {k: float(v) for k, v in value.items()} for key, value in DEFAULT_OBJECT_ACTION_PRIORS.items()
    }
    if config_path and config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                alias_data = data.get("object_name_aliases", {})
                prior_data = data.get("object_action_priors", {})
                if isinstance(alias_data, dict):
                    for k, v in alias_data.items():
                        aliases[_normalize_key(str(k))] = _normalize_key(str(v))
                if isinstance(prior_data, dict):
                    merged: Dict[str, Dict[str, float]] = {}
                    for obj_name, obj_map in prior_data.items():
                        if not isinstance(obj_map, dict):
                            continue
                        m: Dict[str, float] = {}
                        for action_name, score in obj_map.items():
                            try:
                                m[_normalize_key(str(action_name))] = float(score)
                            except Exception:
                                continue
                        if m:
                            merged[_normalize_key(str(obj_name))] = m
                    if merged:
                        priors.update(merged)
        except Exception:
            pass
    return aliases, priors


def normalize_object_name(name: str, aliases: Dict[str, str]) -> str:
    raw = _normalize_key(name)
    return aliases.get(raw, raw)

