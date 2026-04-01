import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple


JsonDict = Dict[str, Any]
Validator = Callable[[JsonDict], Tuple[bool, str]]

SCHEMA_VERSION = "2026-04-01"
VERIFIED_LABELS = {"match", "uncertain", "mismatch"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _in_01(value: Any) -> bool:
    return _is_number(value) and 0.0 <= float(value) <= 1.0


def _pick(row: JsonDict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return default


def validate_event_query_record(row: JsonDict) -> Tuple[bool, str]:
    query_id = _pick(row, "query_id", "event_id")
    t_center = _pick(row, "t_center", "timestamp")

    required = {
        "query_id": query_id,
        "event_type": row.get("event_type"),
        "query_text": row.get("query_text"),
        "t_center": t_center,
        "start": row.get("start"),
        "end": row.get("end"),
        "confidence": row.get("confidence"),
        "source": row.get("source"),
    }
    for key, val in required.items():
        if val is None:
            return False, f"missing key: {key}"

    if not isinstance(query_id, str) or not query_id.strip():
        return False, "query_id must be non-empty string"
    if not isinstance(row["event_type"], str) or not row["event_type"].strip():
        return False, "event_type must be non-empty string"
    if not isinstance(row["query_text"], str) or not row["query_text"].strip():
        return False, "query_text must be non-empty string"
    if not _is_number(t_center):
        return False, "t_center must be number"
    if not _is_number(row["start"]) or not _is_number(row["end"]):
        return False, "start/end must be number"
    if float(row["end"]) < float(row["start"]):
        return False, "end must be >= start"
    if not _in_01(row["confidence"]):
        return False, "confidence must be in [0,1]"
    if not isinstance(row["source"], str) or not row["source"].strip():
        return False, "source must be non-empty string"
    if "trigger_words" in row:
        if not isinstance(row["trigger_words"], list):
            return False, "trigger_words must be list when present"
        for i, w in enumerate(row["trigger_words"]):
            if not isinstance(w, str):
                return False, f"trigger_words[{i}] must be string"
    return True, ""


def _validate_pose_uq_flat(row: JsonDict) -> Tuple[bool, str]:
    required = [
        "frame_idx",
        "track_id",
        "uq_score",
        "uq_source",
        "visible_kpt_ratio",
        "motion_stability",
        "bbox_stability",
    ]
    for key in required:
        if key not in row:
            return False, f"missing key: {key}"
    if not isinstance(row["frame_idx"], int):
        return False, "frame_idx must be int"
    if not isinstance(row["track_id"], int):
        return False, "track_id must be int"
    if not _in_01(row["uq_score"]):
        return False, "uq_score must be in [0,1]"
    if not isinstance(row["uq_source"], list):
        return False, "uq_source must be list"
    for i, src in enumerate(row["uq_source"]):
        if not isinstance(src, str):
            return False, f"uq_source[{i}] must be string"
    for key in ("visible_kpt_ratio", "motion_stability", "bbox_stability"):
        if not _in_01(row[key]):
            return False, f"{key} must be in [0,1]"
    return True, ""


def _validate_pose_uq_legacy(row: JsonDict) -> Tuple[bool, str]:
    if "frame" not in row or "persons" not in row:
        return False, "legacy pose UQ requires frame/persons"
    if not isinstance(row["frame"], int):
        return False, "legacy frame must be int"
    if not isinstance(row["persons"], list):
        return False, "legacy persons must be list"
    for idx, person in enumerate(row["persons"]):
        if not isinstance(person, dict):
            return False, f"legacy persons[{idx}] must be object"
        if not isinstance(person.get("track_id"), int):
            return False, f"legacy persons[{idx}].track_id must be int"
        uq_track = person.get("uq_track")
        if uq_track is None or (not _in_01(uq_track)):
            return False, f"legacy persons[{idx}].uq_track must be in [0,1]"
    return True, ""


def validate_pose_uq_record(row: JsonDict) -> Tuple[bool, str]:
    if "frame_idx" in row and "track_id" in row:
        return _validate_pose_uq_flat(row)
    return _validate_pose_uq_legacy(row)


def validate_align_record(row: JsonDict) -> Tuple[bool, str]:
    query_id = _pick(row, "query_id", "event_id")
    if query_id is None:
        return False, "missing key: query_id"
    if not isinstance(query_id, str) or not query_id.strip():
        return False, "query_id must be non-empty string"
    if not isinstance(row.get("event_type"), str):
        return False, "event_type must be string"
    if not isinstance(row.get("query_text"), str):
        return False, "query_text must be string"

    window = row.get("window")
    if not isinstance(window, dict):
        return False, "window must be object"
    for key in ("start", "end", "center", "size"):
        if not _is_number(window.get(key)):
            return False, f"window.{key} must be number"
    if float(window["end"]) < float(window["start"]):
        return False, "window.end must be >= window.start"

    for key in ("motion_basis", "uq_basis"):
        if key in row and not _in_01(row[key]):
            return False, f"{key} must be in [0,1]"

    candidates = row.get("candidates", [])
    if not isinstance(candidates, list):
        return False, "candidates must be list"
    for i, c in enumerate(candidates):
        if not isinstance(c, dict):
            return False, f"candidates[{i}] must be object"
        if not isinstance(c.get("track_id"), int):
            return False, f"candidates[{i}].track_id must be int"
        if not isinstance(c.get("action"), str):
            return False, f"candidates[{i}].action must be string"
        for key in ("overlap", "action_confidence", "uq_score"):
            if not _in_01(c.get(key, 0.0)):
                return False, f"candidates[{i}].{key} must be in [0,1]"
    return True, ""


def validate_verified_event_record(row: JsonDict) -> Tuple[bool, str]:
    query_id = _pick(row, "query_id", "event_id")
    match_label = _pick(row, "match_label", "label")
    reliability_score = _pick(row, "reliability_score", "reliability")

    if query_id is None:
        return False, "missing key: query_id"
    if not isinstance(query_id, str) or not query_id.strip():
        return False, "query_id must be non-empty string"
    if not isinstance(row.get("track_id"), int):
        return False, "track_id must be int"
    if not isinstance(row.get("event_type"), str) or not row["event_type"].strip():
        return False, "event_type must be non-empty string"

    window = row.get("window")
    if not isinstance(window, dict):
        return False, "window must be object"
    for key in ("start", "end"):
        if not _is_number(window.get(key)):
            return False, f"window.{key} must be number"
    if float(window["end"]) < float(window["start"]):
        return False, "window.end must be >= window.start"

    if not _in_01(reliability_score):
        return False, "reliability_score must be in [0,1]"
    if match_label not in VERIFIED_LABELS:
        return False, f"match_label must be one of: {sorted(VERIFIED_LABELS)}"

    evidence = row.get("evidence")
    if not isinstance(evidence, dict):
        return False, "evidence must be object"
    for key in ("visual_score", "text_score", "uq_score"):
        if not _in_01(evidence.get(key)):
            return False, f"evidence.{key} must be in [0,1]"

    if "p_match" in row and not _in_01(row["p_match"]):
        return False, "p_match must be in [0,1]"
    if "p_mismatch" in row and not _in_01(row["p_mismatch"]):
        return False, "p_mismatch must be in [0,1]"
    return True, ""


def iter_jsonl(path: Path) -> Iterable[Tuple[int, JsonDict]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                yield line_no, data


def validate_jsonl_file(path: Path, validator: Validator) -> Tuple[bool, int, List[str]]:
    errors: List[str] = []
    count = 0
    if not path.exists():
        return False, 0, [f"file not found: {path}"]
    try:
        for line_no, row in iter_jsonl(path):
            count += 1
            ok, msg = validator(row)
            if not ok:
                errors.append(f"{path.name}:{line_no}: {msg}")
    except json.JSONDecodeError as exc:
        errors.append(f"{path.name}: json decode error: {exc}")
    except Exception as exc:
        errors.append(f"{path.name}: unexpected error: {exc}")
    return len(errors) == 0, count, errors


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

