import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


JsonDict = Dict[str, Any]
Validator = Callable[[JsonDict], Tuple[bool, str]]
JsonValidator = Callable[[Any], Tuple[bool, str]]

SCHEMA_VERSION = "2026-04-01"
ARTIFACT_VERSION = "formal_verifier_contracts@2026-04-01"

VERIFIED_LABELS = {"match", "uncertain", "mismatch"}
SAMPLE_TYPES = {"positive", "temporal_shift", "semantic_mismatch"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _in_01(value: Any) -> bool:
    return _is_number(value) and 0.0 <= float(value) <= 1.0


def _is_non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _schema_version_ok(value: Any) -> bool:
    if not _is_non_empty_str(value):
        return False
    v = str(value).strip()
    # Version rule: use the frozen date prefix, optional suffixes for patching.
    return v == SCHEMA_VERSION or v.startswith(SCHEMA_VERSION + "+")


def _require_keys(obj: JsonDict, keys: Sequence[str]) -> Tuple[bool, str]:
    for key in keys:
        if key not in obj:
            return False, f"missing key: {key}"
    return True, ""


def _pick(obj: JsonDict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in obj:
            return obj[key]
    return default


def _as_event_id(row: JsonDict) -> Any:
    return _pick(row, "event_id", "query_id")


def validate_event_query_record(row: JsonDict) -> Tuple[bool, str]:
    required = [
        "event_id",
        "schema_version",
        "query_text",
        "event_type",
        "trigger_words",
        "timestamp",
        "start",
        "end",
        "confidence",
        "source",
    ]
    # Compatibility alias (reader side only): allow query_id + t_center.
    if "event_id" not in row and "query_id" in row:
        row = dict(row)
        row["event_id"] = row["query_id"]
    if "timestamp" not in row and "t_center" in row:
        row = dict(row)
        row["timestamp"] = row["t_center"]

    ok, msg = _require_keys(row, required)
    if not ok:
        return False, msg
    if not _is_non_empty_str(row["event_id"]):
        return False, "event_id must be non-empty string"
    if not _schema_version_ok(row["schema_version"]):
        return False, f"schema_version must be '{SCHEMA_VERSION}' or '{SCHEMA_VERSION}+*'"
    if not _is_non_empty_str(row["query_text"]):
        return False, "query_text must be non-empty string"
    if not _is_non_empty_str(row["event_type"]):
        return False, "event_type must be non-empty string"
    if not isinstance(row["trigger_words"], list):
        return False, "trigger_words must be list"
    for idx, word in enumerate(row["trigger_words"]):
        if not isinstance(word, str):
            return False, f"trigger_words[{idx}] must be string"
    if not _is_number(row["timestamp"]):
        return False, "timestamp must be number"
    if not _is_number(row["start"]) or not _is_number(row["end"]):
        return False, "start/end must be numbers"
    if float(row["end"]) < float(row["start"]):
        return False, "end must be >= start"
    if not _in_01(row["confidence"]):
        return False, "confidence must be in [0,1]"
    if not _is_non_empty_str(row["source"]):
        return False, "source must be non-empty string"
    return True, ""


def validate_pose_uq_record(row: JsonDict) -> Tuple[bool, str]:
    required = ["frame", "t", "persons", "uq_frame"]
    ok, msg = _require_keys(row, required)
    if not ok:
        return False, msg
    if not isinstance(row["frame"], int):
        return False, "frame must be int"
    if not _is_number(row["t"]):
        return False, "t must be number"
    if not _in_01(row["uq_frame"]):
        return False, "uq_frame must be in [0,1]"
    if not isinstance(row["persons"], list):
        return False, "persons must be list"
    for idx, person in enumerate(row["persons"]):
        if not isinstance(person, dict):
            return False, f"persons[{idx}] must be object"
        p_required = ["track_id", "uq_track", "uq_conf", "uq_motion", "uq_kpt"]
        p_ok, p_msg = _require_keys(person, p_required)
        if not p_ok:
            return False, f"persons[{idx}] {p_msg}"
        if not isinstance(person["track_id"], int):
            return False, f"persons[{idx}].track_id must be int"
        for key in ("uq_track", "uq_conf", "uq_motion", "uq_kpt"):
            if not _in_01(person[key]):
                return False, f"persons[{idx}].{key} must be in [0,1]"
        if "log_sigma2" in person and not _is_number(person["log_sigma2"]):
            return False, f"persons[{idx}].log_sigma2 must be number when present"
    return True, ""


def _validate_align_candidate(cand: JsonDict, idx: int) -> Tuple[bool, str]:
    c_required = ["track_id", "action", "start_time", "end_time", "overlap", "action_confidence", "uq_track"]
    ok, msg = _require_keys(cand, c_required)
    if not ok:
        return False, f"candidates[{idx}] {msg}"
    if not isinstance(cand["track_id"], int):
        return False, f"candidates[{idx}].track_id must be int"
    if not _is_non_empty_str(cand["action"]):
        return False, f"candidates[{idx}].action must be non-empty string"
    if not _is_number(cand["start_time"]) or not _is_number(cand["end_time"]):
        return False, f"candidates[{idx}].start_time/end_time must be numbers"
    if float(cand["end_time"]) < float(cand["start_time"]):
        return False, f"candidates[{idx}].end_time must be >= start_time"
    for key in ("overlap", "action_confidence", "uq_track"):
        if not _in_01(cand[key]):
            return False, f"candidates[{idx}].{key} must be in [0,1]"
    return True, ""


def validate_align_record(row: JsonDict) -> Tuple[bool, str]:
    required = [
        "event_id",
        "query_text",
        "event_type",
        "window_center",
        "window_start",
        "window_end",
        "window_size",
        "basis_motion",
        "basis_uq",
        "candidates",
    ]
    # Compatibility aliases.
    if "event_id" not in row and "query_id" in row:
        row = dict(row)
        row["event_id"] = row["query_id"]
    if "window_start" not in row and isinstance(row.get("window"), dict):
        row = dict(row)
        win = row["window"]
        row["window_start"] = win.get("start")
        row["window_end"] = win.get("end")
        row["window_center"] = win.get("center")
        row["window_size"] = win.get("size")
    if "basis_motion" not in row and "motion_basis" in row:
        row = dict(row)
        row["basis_motion"] = row.get("motion_basis")
    if "basis_uq" not in row and "uq_basis" in row:
        row = dict(row)
        row["basis_uq"] = row.get("uq_basis")

    ok, msg = _require_keys(row, required)
    if not ok:
        return False, msg
    if not _is_non_empty_str(row["event_id"]):
        return False, "event_id must be non-empty string"
    if not _is_non_empty_str(row["query_text"]):
        return False, "query_text must be non-empty string"
    if not _is_non_empty_str(row["event_type"]):
        return False, "event_type must be non-empty string"
    for key in ("window_center", "window_start", "window_end", "window_size"):
        if not _is_number(row[key]):
            return False, f"{key} must be number"
    if float(row["window_end"]) < float(row["window_start"]):
        return False, "window_end must be >= window_start"
    if float(row["window_size"]) <= 0:
        return False, "window_size must be > 0"
    if not _in_01(row["basis_motion"]):
        return False, "basis_motion must be in [0,1]"
    if not _in_01(row["basis_uq"]):
        return False, "basis_uq must be in [0,1]"
    if not isinstance(row["candidates"], list):
        return False, "candidates must be list"
    for idx, cand in enumerate(row["candidates"]):
        if not isinstance(cand, dict):
            return False, f"candidates[{idx}] must be object"
        c_ok, c_msg = _validate_align_candidate(cand, idx)
        if not c_ok:
            return False, c_msg
    return True, ""


def validate_verifier_sample_record(row: JsonDict) -> Tuple[bool, str]:
    required = [
        "sample_id",
        "event_id",
        "sample_type",
        "query_text",
        "event_type",
        "track_id",
        "clip_start",
        "clip_end",
        "target_label",
        "negative_kind",
        "provenance",
    ]
    # Compatibility aliases from current generator.
    if "event_id" not in row and "query_id" in row:
        row = dict(row)
        row["event_id"] = row["query_id"]
    if "clip_start" not in row:
        # approximate fallback for old overlap-only records
        row = dict(row)
        row["clip_start"] = float(row.get("window_start", 0.0))
        row["clip_end"] = float(row.get("window_end", row["clip_start"] + 0.2))
    if "target_label" not in row and "target" in row:
        row = dict(row)
        row["target_label"] = "match" if int(row.get("target", 0)) == 1 else "mismatch"
    if "negative_kind" not in row:
        row = dict(row)
        st = str(row.get("sample_type", ""))
        row["negative_kind"] = "" if st == "positive" else st
    if "provenance" not in row:
        row = dict(row)
        row["provenance"] = {"source": "converted_legacy_sample"}

    ok, msg = _require_keys(row, required)
    if not ok:
        return False, msg
    if not _is_non_empty_str(row["sample_id"]):
        return False, "sample_id must be non-empty string"
    if not _is_non_empty_str(row["event_id"]):
        return False, "event_id must be non-empty string"
    if row["sample_type"] not in SAMPLE_TYPES:
        return False, f"sample_type must be one of {sorted(SAMPLE_TYPES)}"
    if not _is_non_empty_str(row["query_text"]):
        return False, "query_text must be non-empty string"
    if not _is_non_empty_str(row["event_type"]):
        return False, "event_type must be non-empty string"
    if not isinstance(row["track_id"], int):
        return False, "track_id must be int"
    if not _is_number(row["clip_start"]) or not _is_number(row["clip_end"]):
        return False, "clip_start/clip_end must be numbers"
    if float(row["clip_end"]) < float(row["clip_start"]):
        return False, "clip_end must be >= clip_start"
    if row["target_label"] not in {"match", "mismatch"}:
        return False, "target_label must be 'match' or 'mismatch'"
    if not isinstance(row["negative_kind"], str):
        return False, "negative_kind must be string"
    if not isinstance(row["provenance"], dict):
        return False, "provenance must be object"
    return True, ""


def validate_verified_event_record(row: JsonDict) -> Tuple[bool, str]:
    required = [
        "event_id",
        "track_id",
        "event_type",
        "query_text",
        "query_time",
        "window_start",
        "window_end",
        "p_match",
        "p_mismatch",
        "reliability_score",
        "uncertainty",
        "label",
        "threshold_source",
        "model_version",
    ]
    # Compatibility aliases from current inference output.
    if "event_id" not in row and "query_id" in row:
        row = dict(row)
        row["event_id"] = row["query_id"]
    if "query_time" not in row:
        row = dict(row)
        row["query_time"] = _pick(row, "timestamp", "t_center", default=row.get("window", {}).get("center", 0.0))
    if "window_start" not in row and isinstance(row.get("window"), dict):
        row = dict(row)
        row["window_start"] = row["window"].get("start")
        row["window_end"] = row["window"].get("end")
    if "label" not in row:
        row = dict(row)
        row["label"] = _pick(row, "match_label", default="mismatch")
    if "uncertainty" not in row:
        row = dict(row)
        row["uncertainty"] = 1.0 - float(_pick(row, "reliability_score", default=0.0))
    if "threshold_source" not in row:
        row = dict(row)
        row["threshold_source"] = "runtime_config"
    if "model_version" not in row:
        row = dict(row)
        row["model_version"] = "verifier_v2"

    ok, msg = _require_keys(row, required)
    if not ok:
        return False, msg
    if not _is_non_empty_str(row["event_id"]):
        return False, "event_id must be non-empty string"
    if not isinstance(row["track_id"], int):
        return False, "track_id must be int"
    if not _is_non_empty_str(row["event_type"]):
        return False, "event_type must be non-empty string"
    if not _is_non_empty_str(row["query_text"]):
        return False, "query_text must be non-empty string"
    if not _is_number(row["query_time"]):
        return False, "query_time must be number"
    if not _is_number(row["window_start"]) or not _is_number(row["window_end"]):
        return False, "window_start/window_end must be numbers"
    if float(row["window_end"]) < float(row["window_start"]):
        return False, "window_end must be >= window_start"
    if not _in_01(row["p_match"]) or not _in_01(row["p_mismatch"]):
        return False, "p_match/p_mismatch must be in [0,1]"
    if abs(float(row["p_match"]) + float(row["p_mismatch"]) - 1.0) > 0.05:
        return False, "p_match + p_mismatch must be close to 1.0"
    if not _in_01(row["reliability_score"]):
        return False, "reliability_score must be in [0,1]"
    if not _in_01(row["uncertainty"]):
        return False, "uncertainty must be in [0,1]"
    if row["label"] not in VERIFIED_LABELS:
        return False, f"label must be one of {sorted(VERIFIED_LABELS)}"
    if not _is_non_empty_str(row["threshold_source"]):
        return False, "threshold_source must be non-empty string"
    if not _is_non_empty_str(row["model_version"]):
        return False, "model_version must be non-empty string"
    return True, ""


def validate_verifier_eval_report(obj: Any) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "report must be object"
    required = ["split", "counts", "metrics", "config", "artifact_version"]
    ok, msg = _require_keys(obj, required)
    if not ok:
        return False, msg
    if not _is_non_empty_str(obj["split"]):
        return False, "split must be non-empty string"
    for key in ("counts", "metrics", "config"):
        if not isinstance(obj[key], dict):
            return False, f"{key} must be object"
    if not _is_non_empty_str(obj["artifact_version"]):
        return False, "artifact_version must be non-empty string"
    return True, ""


def validate_verifier_calibration_report(obj: Any) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "report must be object"
    required = ["split", "ece", "brier", "temperature", "bin_stats", "artifact_version"]
    ok, msg = _require_keys(obj, required)
    if not ok:
        return False, msg
    if not _is_non_empty_str(obj["split"]):
        return False, "split must be non-empty string"
    if not _is_number(obj["ece"]) or float(obj["ece"]) < 0.0:
        return False, "ece must be >= 0"
    if not _is_number(obj["brier"]) or float(obj["brier"]) < 0.0:
        return False, "brier must be >= 0"
    if not _is_number(obj["temperature"]) or float(obj["temperature"]) <= 0.0:
        return False, "temperature must be > 0"
    if not isinstance(obj["bin_stats"], list):
        return False, "bin_stats must be list"
    for idx, bin_row in enumerate(obj["bin_stats"]):
        if not isinstance(bin_row, dict):
            return False, f"bin_stats[{idx}] must be object"
    if not _is_non_empty_str(obj["artifact_version"]):
        return False, "artifact_version must be non-empty string"
    return True, ""


def validate_pipeline_manifest(obj: Any) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "manifest must be object"
    required = ["case_id", "video_id", "schema_version", "artifacts", "config_snapshot"]
    ok, msg = _require_keys(obj, required)
    if not ok:
        return False, msg
    if not _is_non_empty_str(obj["case_id"]):
        return False, "case_id must be non-empty string"
    if not _is_non_empty_str(obj["video_id"]):
        return False, "video_id must be non-empty string"
    if not _schema_version_ok(obj["schema_version"]):
        return False, f"schema_version must be '{SCHEMA_VERSION}' or '{SCHEMA_VERSION}+*'"
    if not isinstance(obj["artifacts"], dict):
        return False, "artifacts must be object"
    if not isinstance(obj["config_snapshot"], dict):
        return False, "config_snapshot must be object"
    return True, ""


def validate_align_file(obj: Any) -> Tuple[bool, str]:
    if not isinstance(obj, list):
        return False, "align_multimodal must be a JSON list"
    for idx, row in enumerate(obj):
        if not isinstance(row, dict):
            return False, f"align[{idx}] must be object"
        ok, msg = validate_align_record(row)
        if not ok:
            return False, f"align[{idx}] {msg}"
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


def validate_json_file(path: Path, validator: JsonValidator) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not path.exists():
        return False, [f"file not found: {path}"]
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        ok, msg = validator(data)
        if not ok:
            errors.append(f"{path.name}: {msg}")
    except json.JSONDecodeError as exc:
        errors.append(f"{path.name}: json decode error: {exc}")
    except Exception as exc:
        errors.append(f"{path.name}: unexpected error: {exc}")
    return len(errors) == 0, errors


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
