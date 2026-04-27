import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts.schemas import (
    SCHEMA_VERSION,
    validate_jsonl_file,
    validate_verified_event_record,
    write_jsonl,
)
from verifier.model import VerifierMLP, VerifierRuntimeConfig, build_feature_vector


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _load_uq_index(path: Optional[Path]) -> Dict[int, float]:
    if path is None or (not path.exists()):
        return {}
    rows = _load_jsonl(path)
    by_track: Dict[int, List[float]] = {}
    for row in rows:
        persons = row.get("persons")
        if isinstance(persons, list):
            for person in persons:
                if not isinstance(person, dict):
                    continue
                tid = person.get("track_id")
                uq = person.get("uq_track", person.get("uq_score"))
                if isinstance(tid, int) and isinstance(uq, (int, float)):
                    by_track.setdefault(tid, []).append(float(uq))
            continue
        tid = row.get("track_id")
        uq = row.get("uq_score", row.get("uq_track"))
        if isinstance(tid, int) and isinstance(uq, (int, float)):
            by_track.setdefault(tid, []).append(float(uq))
    return {tid: (sum(vals) / len(vals) if vals else 0.5) for tid, vals in by_track.items()}


def _load_model(model_path: Optional[Path]) -> Tuple[Optional[VerifierMLP], VerifierRuntimeConfig]:
    runtime_cfg = VerifierRuntimeConfig()
    if model_path is None or (not model_path.exists()):
        return None, runtime_cfg
    ckpt = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return None, runtime_cfg

    in_dim = int(ckpt.get("in_dim", 4))
    hidden_dim = int(ckpt.get("hidden_dim", 16))
    state_dict = ckpt.get("state_dict")
    if isinstance(ckpt.get("runtime_config"), dict):
        runtime_cfg = VerifierRuntimeConfig.from_dict(ckpt["runtime_config"])
    if not isinstance(state_dict, dict):
        return None, runtime_cfg

    model = VerifierMLP(in_dim=in_dim, hidden_dim=hidden_dim)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, runtime_cfg


def _predict_one(
    *,
    model: Optional[VerifierMLP],
    runtime_cfg: VerifierRuntimeConfig,
    event_type: str,
    query_text: str,
    cand: Dict[str, Any],
    uq_default: float,
) -> Dict[str, Any]:
    overlap = _safe_float(cand.get("overlap", 0.0), 0.0)
    action_conf = _safe_float(cand.get("action_confidence", cand.get("confidence", 0.0)), 0.0)
    uq = _safe_float(cand.get("uq_score", cand.get("uq_track", uq_default)), uq_default)
    action_label = str(cand.get("semantic_id", cand.get("action", ""))).strip().lower()
    raw_action = str(cand.get("action", "")).strip().lower()
    behavior_code = str(cand.get("behavior_code", "")).strip().lower()
    behavior_label_zh = str(cand.get("behavior_label_zh", "")).strip()
    behavior_label_en = str(cand.get("behavior_label_en", "")).strip()
    semantic_label_zh = str(cand.get("semantic_label_zh", "")).strip()
    semantic_label_en = str(cand.get("semantic_label_en", "")).strip()
    taxonomy_version = str(cand.get("taxonomy_version", "")).strip()

    feat = build_feature_vector(
        event_type=event_type,
        query_text=query_text,
        action_label=action_label,
        overlap=overlap,
        action_confidence=action_conf,
        uq_score=uq,
    )
    text_score = feat[2]
    visual_score = _clamp01(0.65 * feat[0] + 0.35 * feat[1])

    if model is not None:
        x = torch.tensor([feat], dtype=torch.float32)
        with torch.no_grad():
            logits = model(x).squeeze(0)
        p_match = float(torch.sigmoid(logits / max(1e-6, runtime_cfg.temperature)).item())
    else:
        # Fallback heuristic when no trained model is provided.
        p_match = _clamp01(0.55 * visual_score + 0.45 * text_score)

    p_mismatch = _clamp01(1.0 - p_match)
    reliability = _clamp01(p_match * (1.0 - runtime_cfg.uq_gate * _clamp01(uq)))

    if reliability >= runtime_cfg.match_threshold:
        label = "match"
    elif reliability >= runtime_cfg.uncertain_threshold:
        label = "uncertain"
    else:
        label = "mismatch"

    return {
        "p_match": p_match,
        "p_mismatch": p_mismatch,
        "match_label": label,
        "reliability_score": reliability,
        "uncertainty": _clamp01(1.0 - reliability),
        "visual_score": visual_score,
        "text_score": text_score,
        "uq_score": _clamp01(uq),
        "action": action_label,
        "raw_action": raw_action,
        "behavior_code": behavior_code,
        "behavior_label_zh": behavior_label_zh,
        "behavior_label_en": behavior_label_en,
        "semantic_id": action_label,
        "semantic_label_zh": semantic_label_zh,
        "semantic_label_en": semantic_label_en,
        "taxonomy_version": taxonomy_version,
        "runtime_config": runtime_cfg.to_dict(),
    }


def infer_verified_rows(
    *,
    event_queries_path: Path,
    aligned_path: Path,
    pose_uq_path: Optional[Path],
    model_path: Optional[Path],
    keep_all_candidates: bool = False,
) -> List[Dict[str, Any]]:
    queries = _load_jsonl(event_queries_path)
    q_index = {str(q.get("query_id", q.get("event_id", ""))): q for q in queries}
    aligned_obj = _load_json(aligned_path)
    aligned = aligned_obj if isinstance(aligned_obj, list) else []
    uq_index = _load_uq_index(pose_uq_path)
    model, runtime_cfg = _load_model(model_path)
    threshold_source = "model_runtime_config" if model is not None else "heuristic_default"

    rows: List[Dict[str, Any]] = []
    for block in aligned:
        if not isinstance(block, dict):
            continue
        query_id = str(block.get("query_id", block.get("event_id", "")))
        query = q_index.get(query_id, {})
        event_type = str(query.get("event_type", block.get("event_type", "unknown")))
        query_text = str(query.get("query_text", block.get("query_text", "")))
        window = block.get("window", {})
        if not isinstance(window, dict):
            window = {
                "start": _safe_float(block.get("window_start", 0.0), 0.0),
                "end": _safe_float(block.get("window_end", 0.0), 0.0),
            }
        w_start = _safe_float(window.get("start", 0.0), 0.0)
        w_end = _safe_float(window.get("end", max(w_start, 0.0)), max(w_start, 0.0))
        if w_end < w_start:
            w_start, w_end = w_end, w_start

        candidates = block.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        scored: List[Tuple[Dict[str, Any], Dict[str, float]]] = []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            tid = int(cand.get("track_id", -1))
            uq_default = float(uq_index.get(tid, 0.5))
            score = _predict_one(
                model=model,
                runtime_cfg=runtime_cfg,
                event_type=event_type,
                query_text=query_text,
                cand=cand,
                uq_default=uq_default,
            )
            scored.append((cand, score))
        scored.sort(key=lambda x: x[1]["reliability_score"], reverse=True)

        if not scored:
            row = {
                "schema_version": SCHEMA_VERSION,
                "query_id": query_id,
                "track_id": -1,
                "event_type": event_type,
                "query_text": query_text,
                "window": {"start": w_start, "end": w_end},
                "match_label": "mismatch",
                "reliability_score": 0.0,
                "uncertainty": 1.0,
                "p_match": 0.0,
                "p_mismatch": 1.0,
                "action": "",
                "raw_action": "",
                "behavior_code": "",
                "behavior_label_zh": "",
                "behavior_label_en": "",
                "semantic_id": "",
                "semantic_label_zh": "",
                "semantic_label_en": "",
                "taxonomy_version": "",
                "threshold_source": threshold_source,
                "runtime_config": runtime_cfg.to_dict(),
                "evidence": {"visual_score": 0.0, "text_score": 0.0, "uq_score": 1.0},
            }
            rows.append(row)
            continue

        selected = scored if keep_all_candidates else [scored[0]]
        for cand, score in selected:
            row = {
                "schema_version": SCHEMA_VERSION,
                "query_id": query_id,
                "track_id": int(cand.get("track_id", -1)),
                "event_type": event_type,
                "query_text": query_text,
                "window": {"start": w_start, "end": w_end},
                "match_label": str(score["match_label"]),
                "reliability_score": float(score["reliability_score"]),
                "uncertainty": float(score["uncertainty"]),
                "p_match": float(score["p_match"]),
                "p_mismatch": float(score["p_mismatch"]),
                "action": str(score.get("action", "")),
                "raw_action": str(score.get("raw_action", "")),
                "behavior_code": str(score.get("behavior_code", "")),
                "behavior_label_zh": str(score.get("behavior_label_zh", "")),
                "behavior_label_en": str(score.get("behavior_label_en", "")),
                "semantic_id": str(score.get("semantic_id", "")),
                "semantic_label_zh": str(score.get("semantic_label_zh", "")),
                "semantic_label_en": str(score.get("semantic_label_en", "")),
                "taxonomy_version": str(score.get("taxonomy_version", "")),
                "threshold_source": threshold_source,
                "runtime_config": score["runtime_config"],
                "evidence": {
                    "visual_score": float(score["visual_score"]),
                    "text_score": float(score["text_score"]),
                    "uq_score": float(score["uq_score"]),
                },
            }
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifier inference -> verified_events.jsonl")
    parser.add_argument("--event_queries", required=True, type=str)
    parser.add_argument("--aligned", required=True, type=str)
    parser.add_argument("--pose_uq", default="", type=str)
    parser.add_argument("--model", default="", type=str, help="verifier.pt")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--keep_all_candidates", type=int, default=0)
    parser.add_argument("--validate", type=int, default=1)
    args = parser.parse_args()

    event_queries = Path(args.event_queries).resolve()
    aligned = Path(args.aligned).resolve()
    pose_uq = Path(args.pose_uq).resolve() if args.pose_uq else None
    model = Path(args.model).resolve() if args.model else None
    out = Path(args.out).resolve()

    rows = infer_verified_rows(
        event_queries_path=event_queries,
        aligned_path=aligned,
        pose_uq_path=pose_uq,
        model_path=model,
        keep_all_candidates=bool(int(args.keep_all_candidates)),
    )
    write_jsonl(out, rows)

    if int(args.validate) == 1:
        ok, _, errors = validate_jsonl_file(out, validate_verified_event_record)
        if not ok:
            first_error = errors[0] if errors else "unknown schema error"
            raise ValueError(f"invalid verified events schema: {first_error}")

    print(f"[DONE] verified events: {out} ({len(rows)})")


if __name__ == "__main__":
    main()
