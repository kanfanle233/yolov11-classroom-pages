from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SEMANTIC_FIELDS = [
    "behavior_code",
    "behavior_label_zh",
    "behavior_label_en",
    "semantic_id",
    "semantic_label_zh",
    "semantic_label_en",
    "taxonomy_version",
]

REQUIRED_ACTION_FIELDS = [
    "track_id",
    "action",
    "behavior_code",
    "semantic_id",
    "semantic_label_zh",
    "semantic_label_en",
    "conf",
    "start_time",
    "end_time",
    "source",
    "taxonomy_version",
]

OBJECT_ACTION_PRIORS = {
    "phone": {"turn_head": 0.70},
    "cell phone": {"turn_head": 0.70},
    "mobile phone": {"turn_head": 0.70},
    "book": {"read": 0.90, "write": 0.45},
    "textbook": {"read": 0.90, "write": 0.45},
    "notebook": {"write": 0.85, "read": 0.35},
    "pen": {"write": 0.90},
    "pencil": {"write": 0.90},
    "laptop": {"read": 0.55, "write": 0.55},
}

OBJECT_ALIASES = {
    "cell phone": "phone",
    "mobile phone": "phone",
    "book": "book",
    "textbook": "book",
    "notebook": "notebook",
    "pen": "pen",
    "pencil": "pen",
    "laptop": "laptop",
}


def resolve_repo_root(anchor: Path) -> Path:
    for candidate in [anchor.resolve()] + list(anchor.resolve().parents):
        if (candidate / "data").exists() and (candidate / "scripts").exists():
            return candidate
    raise RuntimeError(f"Cannot resolve repo root from: {anchor}")


def resolve_path(repo_root: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (repo_root / p).resolve()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mapping_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Taxonomy file is empty: {path}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("Taxonomy is not JSON and PyYAML is unavailable.") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Taxonomy root must be an object.")
    return data


def normalize_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    return " ".join(token.split())


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def bbox_iou(a: List[float], b: List[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


class Taxonomy:
    def __init__(self, raw: Dict[str, Any]) -> None:
        classes = raw.get("classes")
        if not isinstance(classes, list) or not classes:
            raise ValueError("taxonomy.classes must be a non-empty list")
        self.version = str(raw.get("taxonomy_version", "smart_classroom_semantics_v1")).strip()
        self.by_code: Dict[str, Dict[str, str]] = {}
        self.by_semantic: Dict[str, Dict[str, str]] = {}
        self.by_alias: Dict[str, Dict[str, str]] = {}

        for item in classes:
            if not isinstance(item, dict):
                continue
            code = normalize_token(item.get("behavior_code"))
            semantic_id = normalize_token(item.get("semantic_id"))
            label_zh = str(item.get("label_zh", "")).strip()
            label_en = str(item.get("label_en", "")).strip()
            semantic_label_zh = str(item.get("semantic_label_zh", label_zh)).strip()
            semantic_label_en = str(item.get("semantic_label_en", semantic_id)).strip()
            if not code or not semantic_id or not label_zh or not label_en:
                raise ValueError(f"Invalid taxonomy class entry: {item}")

            entry = {
                "behavior_code": code,
                "behavior_label_zh": label_zh,
                "behavior_label_en": label_en,
                "semantic_id": semantic_id,
                "semantic_label_zh": semantic_label_zh,
                "semantic_label_en": semantic_label_en,
                "taxonomy_version": self.version,
            }
            self.by_code[code] = entry
            self.by_semantic[semantic_id] = entry

            aliases = item.get("aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            alias_tokens = {
                code,
                semantic_id,
                normalize_token(label_zh),
                normalize_token(label_en),
                normalize_token(semantic_label_zh),
                normalize_token(semantic_label_en),
            }
            for alias in aliases:
                alias_tokens.add(normalize_token(alias))
            for alias in alias_tokens:
                if alias:
                    self.by_alias[alias] = entry

        fallback_code = normalize_token(raw.get("fallback_behavior_code", "tt"))
        self.fallback = self.by_code.get(fallback_code, next(iter(self.by_code.values())))

    @property
    def valid_codes(self) -> set[str]:
        return set(self.by_code.keys())

    def resolve(self, hints: Iterable[Any]) -> Tuple[Dict[str, str], str]:
        for hint in hints:
            token = normalize_token(hint)
            if not token:
                continue
            if token in self.by_code:
                return self.by_code[token], "behavior_code"
            if token in self.by_semantic:
                return self.by_semantic[token], "semantic_id"
            if token in self.by_alias:
                return self.by_alias[token], "alias"
        return self.fallback, "fallback"

    def apply(self, row: Dict[str, Any], entry: Dict[str, str]) -> Dict[str, Any]:
        for key in SEMANTIC_FIELDS:
            row[key] = entry[key]
        return row


def load_taxonomy(path: Path) -> Taxonomy:
    return Taxonomy(load_mapping_file(path))


def normalize_object_name(name: Any) -> str:
    raw = normalize_token(name)
    return OBJECT_ALIASES.get(raw, raw)


def object_support_actions(name: Any) -> Dict[str, float]:
    norm = normalize_object_name(name)
    return dict(OBJECT_ACTION_PRIORS.get(norm, {}))


def semantic_coverage(rows: Iterable[Dict[str, Any]]) -> Tuple[int, int, List[Dict[str, Any]]]:
    total = 0
    invalid: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        total += 1
        missing = [key for key in REQUIRED_ACTION_FIELDS if not str(row.get(key, "")).strip()]
        if missing:
            invalid.append({"index": idx, "missing_fields": missing, "row": row})
    return total, total - len(invalid), invalid
