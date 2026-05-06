"""
Canonical event evidence schema v1.

Defines canonical field names and normalization from raw pipeline output
to the unified evidence structure used by the paper evidence API.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


# ── Canonical field names (the ONE true name per concept) ──────────
CANONICAL_EVENT_ID = "event_id"
CANONICAL_QUERY_TEXT = "query_text"
CANONICAL_QUERY_SOURCE = "query_source"          # "asr" | "visual_fallback"
CANONICAL_QUERY_TIME = "query_time"
CANONICAL_WINDOW_START = "window_start"
CANONICAL_WINDOW_END = "window_end"
CANONICAL_TRACK_ID = "track_id"
CANONICAL_STUDENT_ID = "student_id"
CANONICAL_LABEL = "label"                         # "match" | "uncertain" | "mismatch"
CANONICAL_P_MATCH = "p_match"
CANONICAL_P_MISMATCH = "p_mismatch"
CANONICAL_RELIABILITY = "reliability_score"
CANONICAL_UNCERTAINTY = "uncertainty"
CANONICAL_VISUAL_SCORE = "visual_score"
CANONICAL_TEXT_SCORE = "text_score"
CANONICAL_UQ_SCORE = "uq_score"
CANONICAL_ACTION_CONFIDENCE = "action_confidence"
CANONICAL_OVERLAP = "overlap"

# Aliases that appear in raw pipeline output -> canonical name
_FIELD_ALIASES: Dict[str, str] = {
    "query_id": CANONICAL_EVENT_ID,
    "timestamp": CANONICAL_QUERY_TIME,
    "t_center": CANONICAL_QUERY_TIME,
    "match_label": CANONICAL_LABEL,
    "verification_status": CANONICAL_LABEL,
    "reliability": CANONICAL_RELIABILITY,
    "reliability_final": CANONICAL_RELIABILITY,
    "c_visual": CANONICAL_VISUAL_SCORE,
    "cv": CANONICAL_VISUAL_SCORE,
    "c_text": CANONICAL_TEXT_SCORE,
    "ct": CANONICAL_TEXT_SCORE,
    "uq_track": CANONICAL_UQ_SCORE,
    "uncertainty_score": CANONICAL_UNCERTAINTY,
}

# Canonical names for all aliases of a concept, for lookup
_CONCEPT_KEYS: Dict[str, List[str]] = {
    CANONICAL_EVENT_ID: ["event_id", "query_id"],
    CANONICAL_QUERY_TIME: ["query_time", "timestamp", "t_center"],
    CANONICAL_LABEL: ["label", "match_label", "verification_status"],
    CANONICAL_RELIABILITY: ["reliability_score", "reliability", "reliability_final"],
    CANONICAL_VISUAL_SCORE: ["visual_score", "c_visual", "cv"],
    CANONICAL_TEXT_SCORE: ["text_score", "c_text", "ct"],
    CANONICAL_UQ_SCORE: ["uq_score", "uq_track", "uq_track_score"],
    CANONICAL_UNCERTAINTY: ["uncertainty", "uncertainty_score"],
    CANONICAL_WINDOW_START: ["window_start", "window.start"],
    CANONICAL_WINDOW_END: ["window_end", "window.end"],
}


def _pick(row: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first key present in row, or default."""
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ── TypedDicts ────────────────────────────────────────────────────

class EvidenceQuery(TypedDict, total=False):
    event_id: str
    query_text: str
    query_source: str          # "asr" | "visual_fallback"
    is_visual_fallback: bool
    query_time: float
    window_start: float
    window_end: float
    confidence: float
    event_type: str


class EvidenceCandidate(TypedDict, total=False):
    track_id: int
    student_id: str
    action: str
    semantic_id: str
    semantic_label_zh: str
    semantic_label_en: str
    behavior_code: str
    behavior_label_zh: str
    behavior_label_en: str
    start_time: float
    end_time: float
    overlap: float
    action_confidence: float
    uq_track: float
    is_selected: bool
    rank: int


class EvidenceSelected(TypedDict, total=False):
    track_id: int
    student_id: str
    label: str
    p_match: float
    p_mismatch: float
    reliability_score: float
    uncertainty: float
    visual_score: float
    text_score: float
    uq_score: float
    selected_candidate_rank: int
    candidate_count: int


class EvidenceMedia(TypedDict, total=False):
    video_url: str
    start_sec: float
    end_sec: float


class EventEvidenceV1(TypedDict, total=False):
    case_id: str
    event_id: str
    query: EvidenceQuery
    selected: EvidenceSelected
    align_candidates: List[EvidenceCandidate]
    media: EvidenceMedia
    source_files: Dict[str, str]


# ── Normalization helpers ─────────────────────────────────────────

def normalize_verified_event(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw verified_events row to canonical field names."""
    ws = _safe_float(_pick(raw, "window_start", "window.start"), 0.0)
    we = _safe_float(_pick(raw, "window_end", "window.end"), 0.0)
    qt = _safe_float(_pick(raw, "query_time", "timestamp", "t_center"), (ws + we) / 2.0)
    evidence = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}

    return {
        CANONICAL_EVENT_ID: str(_pick(raw, "event_id", "query_id", default="")),
        CANONICAL_QUERY_TEXT: str(raw.get("query_text", "")),
        CANONICAL_QUERY_SOURCE: str(_pick(raw, "query_source", "source", "asr_source", default="unknown")),
        CANONICAL_QUERY_TIME: round(qt, 4),
        CANONICAL_WINDOW_START: round(ws, 4),
        CANONICAL_WINDOW_END: round(we, 4),
        CANONICAL_TRACK_ID: _safe_int(raw.get("track_id"), -1),
        CANONICAL_LABEL: str(_pick(raw, "label", "match_label", "verification_status", default="unverified")),
        CANONICAL_P_MATCH: _safe_float(raw.get("p_match"), 0.0),
        CANONICAL_P_MISMATCH: _safe_float(raw.get("p_mismatch"), 0.0),
        CANONICAL_RELIABILITY: _safe_float(_pick(raw, "reliability_score", "reliability"), 0.0),
        CANONICAL_UNCERTAINTY: _safe_float(raw.get("uncertainty"), 0.0),
        CANONICAL_VISUAL_SCORE: _safe_float(_pick(evidence, "visual_score", "c_visual", "cv"), 0.0),
        CANONICAL_TEXT_SCORE: _safe_float(_pick(evidence, "text_score", "c_text", "ct"), 0.0),
        CANONICAL_UQ_SCORE: _safe_float(_pick(evidence, "uq_score", "uq_track"), 0.0),
    }


def normalize_event_query(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw event_queries row to canonical field names."""
    return {
        CANONICAL_EVENT_ID: str(_pick(raw, "event_id", "query_id", default="")),
        CANONICAL_QUERY_TEXT: str(raw.get("query_text", "")),
        CANONICAL_QUERY_SOURCE: str(raw.get("source", "unknown")),
        "is_visual_fallback": str(raw.get("source", "")).lower() == "visual_fallback",
        CANONICAL_QUERY_TIME: _safe_float(_pick(raw, "timestamp", "t_center", "query_time"), 0.0),
        CANONICAL_WINDOW_START: _safe_float(_pick(raw, "start", "window_start"), 0.0),
        CANONICAL_WINDOW_END: _safe_float(_pick(raw, "end", "window_end"), 0.0),
        "confidence": _safe_float(raw.get("confidence"), 0.0),
        "event_type": str(raw.get("event_type", "")),
    }


def normalize_align_candidate(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw alignment candidate to canonical field names."""
    return {
        CANONICAL_TRACK_ID: _safe_int(raw.get("track_id"), -1),
        "action": str(raw.get("action", "")),
        "semantic_id": str(raw.get("semantic_id", "")),
        "semantic_label_zh": str(raw.get("semantic_label_zh", "")),
        "semantic_label_en": str(raw.get("semantic_label_en", "")),
        "behavior_code": str(raw.get("behavior_code", "")),
        "behavior_label_zh": str(raw.get("behavior_label_zh", "")),
        "behavior_label_en": str(raw.get("behavior_label_en", "")),
        "start_time": _safe_float(raw.get("start_time"), 0.0),
        "end_time": _safe_float(raw.get("end_time"), 0.0),
        CANONICAL_OVERLAP: _safe_float(raw.get("overlap"), 0.0),
        CANONICAL_ACTION_CONFIDENCE: _safe_float(raw.get("action_confidence"), 0.0),
        CANONICAL_UQ_SCORE: _safe_float(_pick(raw, "uq_track", "uq_score"), 0.0),
    }
