import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from contracts.schemas import write_jsonl
from verifier.model import action_match_score
from verifier.text_similarity import TextSimilarityScorer


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
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
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
                rows.append(obj)
    return rows


def _normalize_aligned(aligned_obj: Any) -> List[Dict[str, Any]]:
    if isinstance(aligned_obj, list):
        return [x for x in aligned_obj if isinstance(x, dict)]
    return []


def _compute_text_score(
    *,
    text_scorer: Optional[TextSimilarityScorer],
    event_type: str,
    query_text: str,
    action_label: str,
) -> float:
    if text_scorer is None:
        return _clamp01(action_match_score(event_type, query_text, action_label))
    score, _ = text_scorer.score(event_type=event_type, query_text=query_text, action_label=action_label)
    return _clamp01(score)


def build_training_samples(
    *,
    event_queries_path: Path,
    aligned_path: Path,
    actions_path: Path,
    max_semantic_neg_per_query: int = 2,
    text_score_mode: str = "rule",
    embedding_cache_path: str = "output/cache/text_embeddings.pkl",
    embedding_backend: str = "auto",
    embedding_model: str = "",
    embedding_dim: int = 128,
    hybrid_alpha: float = 0.5,
) -> List[Dict[str, Any]]:
    queries = _load_jsonl(event_queries_path)
    query_by_id = {str(q.get("query_id", q.get("event_id", ""))): q for q in queries}
    aligned = _normalize_aligned(_load_json(aligned_path))
    actions = _load_jsonl(actions_path)

    samples: List[Dict[str, Any]] = []
    sid = 0
    text_scorer = TextSimilarityScorer(
        mode=text_score_mode,
        cache_path=embedding_cache_path,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_dim=int(embedding_dim),
        hybrid_alpha=float(hybrid_alpha),
    )
    try:
        for row in aligned:
            query_id = str(row.get("query_id", row.get("event_id", "")))
            q = query_by_id.get(query_id, {})
            event_type = str(q.get("event_type", row.get("event_type", "unknown")))
            query_text = str(q.get("query_text", row.get("query_text", "")))
            candidates = row.get("candidates", [])
            if not isinstance(candidates, list):
                candidates = []
            if not candidates:
                continue

            # Score candidates by semantic*overlap to pick one positive anchor.
            ranked = []
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                action = str(c.get("action", ""))
                overlap = _safe_float(c.get("overlap", 0.0), 0.0)
                conf = _safe_float(c.get("action_confidence", c.get("confidence", c.get("conf", 0.0))), 0.0)
                uq = _safe_float(c.get("uq_score", c.get("uq_track", 0.5)), 0.5)
                text_s = _compute_text_score(
                    text_scorer=text_scorer,
                    event_type=event_type,
                    query_text=query_text,
                    action_label=action,
                )
                ranked.append((text_s * (0.5 + 0.5 * overlap), text_s, overlap, conf, uq, c))
            ranked.sort(key=lambda x: x[0], reverse=True)
            best = ranked[0]
            _, text_s, overlap, conf, uq, cand = best
            track_id = int(cand.get("track_id", -1))
            action = str(cand.get("action", ""))

            # Positive sample
            samples.append(
                {
                    "sample_id": f"s_{sid:07d}",
                    "query_id": query_id,
                    "event_type": event_type,
                    "query_text": query_text,
                    "track_id": track_id,
                    "action_label": action,
                    "overlap": _clamp01(overlap),
                    "action_confidence": _clamp01(conf),
                    "uq_score": _clamp01(uq),
                    "text_score": _clamp01(text_s),
                    "sample_type": "positive",
                    "target": 1,
                }
            )
            sid += 1

            # Temporal negative: same action but shifted/weak overlap.
            temporal_overlap = _clamp01(max(0.0, overlap - 0.6))
            samples.append(
                {
                    "sample_id": f"s_{sid:07d}",
                    "query_id": query_id,
                    "event_type": event_type,
                    "query_text": query_text,
                    "track_id": track_id,
                    "action_label": action,
                    "overlap": temporal_overlap,
                    "action_confidence": _clamp01(conf),
                    "uq_score": _clamp01(min(1.0, uq + 0.15)),
                    "text_score": _clamp01(text_s),
                    "sample_type": "temporal_shift",
                    "target": 0,
                }
            )
            sid += 1

            # Semantic negatives: keep overlap high but mismatch query semantics.
            sem_neg_count = 0
            for _, neg_text_s, neg_overlap, neg_conf, neg_uq, neg_c in ranked[1:]:
                if sem_neg_count >= max_semantic_neg_per_query:
                    break
                if neg_text_s > 0.45:
                    continue
                samples.append(
                    {
                        "sample_id": f"s_{sid:07d}",
                        "query_id": query_id,
                        "event_type": event_type,
                        "query_text": query_text,
                        "track_id": int(neg_c.get("track_id", -1)),
                        "action_label": str(neg_c.get("action", "")),
                        "overlap": _clamp01(neg_overlap),
                        "action_confidence": _clamp01(neg_conf),
                        "uq_score": _clamp01(neg_uq),
                        "text_score": _clamp01(neg_text_s),
                        "sample_type": "semantic_mismatch",
                        "target": 0,
                    }
                )
                sid += 1
                sem_neg_count += 1

        # Fallback when aligned is empty: weak pseudo samples from actions.
        if not samples:
            for q in queries[:50]:
                qid = str(q.get("query_id", q.get("event_id", "")))
                et = str(q.get("event_type", "unknown"))
                qt = str(q.get("query_text", ""))
                for a in actions[:100]:
                    tid = a.get("track_id")
                    if not isinstance(tid, int):
                        continue
                    action = str(a.get("action", ""))
                    conf = _safe_float(a.get("conf", a.get("confidence", 0.0)), 0.0)
                    text_s = _compute_text_score(
                        text_scorer=text_scorer,
                        event_type=et,
                        query_text=qt,
                        action_label=action,
                    )
                    target = 1 if text_s >= 0.75 else 0
                    stype = "positive" if target == 1 else "semantic_mismatch"
                    samples.append(
                        {
                            "sample_id": f"s_{sid:07d}",
                            "query_id": qid,
                            "event_type": et,
                            "query_text": qt,
                            "track_id": int(tid),
                            "action_label": action,
                            "overlap": 0.5 if target == 1 else 0.2,
                            "action_confidence": _clamp01(conf),
                            "uq_score": 0.5,
                            "text_score": _clamp01(text_s),
                            "sample_type": stype,
                            "target": int(target),
                        }
                    )
                    sid += 1
        return samples
    finally:
        text_scorer.close()


def convert_to_contract_samples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        event_id = str(row.get("event_id", row.get("query_id", "")))
        target = int(row.get("target", 0))
        sample_type = str(row.get("sample_type", "semantic_mismatch"))
        converted.append(
            {
                "sample_id": str(row.get("sample_id", f"s_{index:07d}")),
                "event_id": event_id,
                "sample_type": sample_type,
                "query_text": str(row.get("query_text", "")),
                "event_type": str(row.get("event_type", "unknown")),
                "track_id": int(row.get("track_id", -1)),
                "clip_start": float(row.get("clip_start", row.get("window_start", 0.0))),
                "clip_end": float(row.get("clip_end", row.get("window_end", row.get("clip_start", 0.0)))),
                "target_label": "match" if target == 1 else "mismatch",
                "negative_kind": "" if sample_type == "positive" else sample_type,
                "provenance": {
                    "source": "verifier.dataset",
                    "legacy_fields": {
                        "overlap": float(row.get("overlap", 0.0)),
                        "action_confidence": float(row.get("action_confidence", 0.0)),
                        "uq_score": float(row.get("uq_score", 0.0)),
                        "text_score": float(row.get("text_score", 0.0)),
                    },
                },
            }
        )
    return converted


def save_training_samples(path: Path, rows: List[Dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = min(a1, b1) - max(a0, b0)
    if inter <= 0:
        return 0.0
    denom = max(1e-6, min(a1 - a0, b1 - b0))
    return inter / denom


def _normalize_actions(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        tid = row.get("track_id")
        if not isinstance(tid, int):
            continue
        st = _safe_float(row.get("start_time", row.get("start", row.get("t", 0.0))), 0.0)
        ed = _safe_float(row.get("end_time", row.get("end", st + 0.2)), st + 0.2)
        if ed < st:
            st, ed = ed, st
        if ed <= st:
            ed = st + 0.2
        conf = _safe_float(row.get("action_confidence", row.get("confidence", row.get("conf", 0.5))), 0.5)
        out.append(
            {
                "track_id": int(tid),
                "action": str(row.get("action", row.get("label", ""))).strip().lower(),
                "start_time": st,
                "end_time": ed,
                "action_confidence": _clamp01(conf),
            }
        )
    out.sort(key=lambda x: (x["start_time"], x["track_id"]))
    return out


def _query_timestamp(query: Dict[str, Any]) -> float:
    return _safe_float(query.get("timestamp", query.get("t_center", query.get("start", 0.0))), 0.0)


def _query_window(query: Dict[str, Any]) -> Tuple[float, float]:
    center = _query_timestamp(query)
    st = _safe_float(query.get("start", center - 0.8), center - 0.8)
    ed = _safe_float(query.get("end", center + 0.8), center + 0.8)
    if ed < st:
        st, ed = ed, st
    if ed <= st:
        ed = st + 0.2
    return st, ed


def _pick_positive_candidate(
    *,
    query: Dict[str, Any],
    aligned_row: Dict[str, Any],
    actions: Sequence[Dict[str, Any]],
    text_scorer: Optional[TextSimilarityScorer] = None,
) -> Dict[str, Any]:
    event_type = str(query.get("event_type", aligned_row.get("event_type", "unknown")))
    query_text = str(query.get("query_text", aligned_row.get("query_text", "")))
    query_start, query_end = _query_window(query)
    candidates = aligned_row.get("candidates", [])
    scored: List[Tuple[float, Dict[str, Any]]] = []
    if isinstance(candidates, list):
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            action = str(cand.get("action", "")).strip().lower()
            if not action:
                continue
            overlap = _clamp01(_safe_float(cand.get("overlap", 0.0), 0.0))
            conf = _clamp01(_safe_float(cand.get("action_confidence", cand.get("confidence", 0.5)), 0.5))
            uq_score = _clamp01(_safe_float(cand.get("uq_score", cand.get("uq_track", 0.5)), 0.5))
            text_score = _compute_text_score(
                text_scorer=text_scorer,
                event_type=event_type,
                query_text=query_text,
                action_label=action,
            )
            score = 0.60 * text_score + 0.25 * overlap + 0.15 * conf
            scored.append(
                (
                    score,
                    {
                        "track_id": int(cand.get("track_id", -1)),
                        "action": action,
                        "overlap": overlap,
                        "action_confidence": conf,
                        "uq_score": uq_score,
                        "text_score": text_score,
                        "start_time": _safe_float(cand.get("start_time", query_start), query_start),
                        "end_time": _safe_float(cand.get("end_time", query_end), query_end),
                    },
                )
            )
    if not scored:
        for action_row in actions:
            action = str(action_row.get("action", "")).strip().lower()
            if not action:
                continue
            overlap = _clamp01(
                _interval_overlap(
                    query_start,
                    query_end,
                    _safe_float(action_row.get("start_time", 0.0), 0.0),
                    _safe_float(action_row.get("end_time", 0.0), 0.0),
                )
            )
            if overlap <= 0:
                continue
            conf = _clamp01(_safe_float(action_row.get("action_confidence", 0.5), 0.5))
            uq_score = 0.5
            text_score = _compute_text_score(
                text_scorer=text_scorer,
                event_type=event_type,
                query_text=query_text,
                action_label=action,
            )
            score = 0.60 * text_score + 0.25 * overlap + 0.15 * conf
            scored.append(
                (
                    score,
                    {
                        "track_id": int(action_row.get("track_id", -1)),
                        "action": action,
                        "overlap": overlap,
                        "action_confidence": conf,
                        "uq_score": uq_score,
                        "text_score": text_score,
                        "start_time": _safe_float(action_row.get("start_time", query_start), query_start),
                        "end_time": _safe_float(action_row.get("end_time", query_end), query_end),
                    },
                )
            )
    if not scored:
        return {
            "track_id": -1,
            "action": "",
            "overlap": 0.0,
            "action_confidence": 0.5,
            "uq_score": 0.5,
            "text_score": 0.0,
            "start_time": query_start,
            "end_time": query_end,
        }
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _build_temporal_shift_negative(
    *,
    query: Dict[str, Any],
    positive: Dict[str, Any],
    actions: Sequence[Dict[str, Any]],
    shift_seconds: Sequence[float],
    text_scorer: Optional[TextSimilarityScorer] = None,
) -> Dict[str, Any]:
    event_type = str(query.get("event_type", "unknown"))
    query_text = str(query.get("query_text", ""))
    query_start, query_end = _query_window(query)
    best: Dict[str, Any] = {}
    best_rank = -1.0
    for shift in shift_seconds:
        for sign in (-1.0, 1.0):
            s = float(sign * shift)
            shifted_start = query_start + s
            shifted_end = query_end + s
            for act in actions:
                action = str(act.get("action", "")).strip().lower()
                if not action:
                    continue
                overlap_shifted = _clamp01(
                    _interval_overlap(
                        shifted_start,
                        shifted_end,
                        _safe_float(act.get("start_time", 0.0), 0.0),
                        _safe_float(act.get("end_time", 0.0), 0.0),
                    )
                )
                if overlap_shifted <= 0:
                    continue
                overlap_original = _clamp01(
                    _interval_overlap(
                        query_start,
                        query_end,
                        _safe_float(act.get("start_time", 0.0), 0.0),
                        _safe_float(act.get("end_time", 0.0), 0.0),
                    )
                )
                if overlap_original > 0.65:
                    continue
                text_score = _compute_text_score(
                    text_scorer=text_scorer,
                    event_type=event_type,
                    query_text=query_text,
                    action_label=action,
                )
                conf = _clamp01(_safe_float(act.get("action_confidence", 0.5), 0.5))
                uq = _clamp01(min(1.0, _safe_float(positive.get("uq_score", 0.5), 0.5) + 0.10))
                # Hard temporal negative: shifted overlap high but original overlap low and semantic weak.
                rank = overlap_shifted * (1.0 - overlap_original) * (1.0 - text_score + 1e-4)
                if rank > best_rank:
                    best_rank = rank
                    best = {
                        "track_id": int(act.get("track_id", -1)),
                        "action": action,
                        "overlap": overlap_original,
                        "action_confidence": conf,
                        "uq_score": uq,
                        "text_score": text_score,
                    }
    if best_rank >= 0:
        return best
    # Fallback: preserve action/conf and inject temporal misalignment by reducing overlap.
    return {
        "track_id": int(positive.get("track_id", -1)),
        "action": str(positive.get("action", "")),
        "overlap": _clamp01(max(0.0, _safe_float(positive.get("overlap", 0.0), 0.0) - 0.65)),
        "action_confidence": _clamp01(_safe_float(positive.get("action_confidence", 0.5), 0.5)),
        "uq_score": _clamp01(min(1.0, _safe_float(positive.get("uq_score", 0.5), 0.5) + 0.15)),
        "text_score": _clamp01(_safe_float(positive.get("text_score", 0.0), 0.0)),
    }


def _build_semantic_mismatch_negative(
    *,
    query: Dict[str, Any],
    aligned_row: Dict[str, Any],
    positive: Dict[str, Any],
    actions: Sequence[Dict[str, Any]],
    text_scorer: Optional[TextSimilarityScorer] = None,
) -> Dict[str, Any]:
    event_type = str(query.get("event_type", aligned_row.get("event_type", "unknown")))
    query_text = str(query.get("query_text", aligned_row.get("query_text", "")))
    pos_overlap = _clamp01(_safe_float(positive.get("overlap", 0.0), 0.0))
    best: Dict[str, Any] = {}
    best_rank = -1.0

    candidates = aligned_row.get("candidates", [])
    if isinstance(candidates, list):
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            action = str(cand.get("action", "")).strip().lower()
            if not action:
                continue
            overlap = _clamp01(_safe_float(cand.get("overlap", 0.0), 0.0))
            text_score = _compute_text_score(
                text_scorer=text_scorer,
                event_type=event_type,
                query_text=query_text,
                action_label=action,
            )
            if overlap < max(0.3, pos_overlap * 0.5):
                continue
            rank = overlap * (1.0 - text_score + 1e-4)
            if rank > best_rank:
                best_rank = rank
                best = {
                    "track_id": int(cand.get("track_id", -1)),
                    "action": action,
                    "overlap": overlap,
                    "action_confidence": _clamp01(_safe_float(cand.get("action_confidence", 0.5), 0.5)),
                    "uq_score": _clamp01(_safe_float(cand.get("uq_score", cand.get("uq_track", 0.5)), 0.5)),
                    "text_score": text_score,
                }

    if best_rank >= 0 and best.get("text_score", 1.0) <= 0.5:
        return best

    # Fallback: replace action label with a semantically conflicting one while preserving time evidence.
    action_pool: List[str] = []
    for act in actions:
        a = str(act.get("action", "")).strip().lower()
        if a:
            action_pool.append(a)
    if not action_pool:
        action_pool = ["unknown_action"]
    conflict_action = action_pool[0]
    conflict_score = 1.0
    for a in action_pool:
        ts = _compute_text_score(
            text_scorer=text_scorer,
            event_type=event_type,
            query_text=query_text,
            action_label=a,
        )
        if ts < conflict_score:
            conflict_score = ts
            conflict_action = a
    return {
        "track_id": int(positive.get("track_id", -1)),
        "action": conflict_action,
        "overlap": _clamp01(_safe_float(positive.get("overlap", 0.0), 0.0)),
        "action_confidence": _clamp01(_safe_float(positive.get("action_confidence", 0.5), 0.5)),
        "uq_score": _clamp01(min(1.0, _safe_float(positive.get("uq_score", 0.5), 0.5) + 0.05)),
        "text_score": _clamp01(conflict_score),
    }


def _assign_split(query_id: str, train_ratio: float) -> str:
    ratio = max(0.05, min(0.95, float(train_ratio)))
    threshold = int(round(ratio * 100))
    h = sum(ord(ch) for ch in query_id) % 100
    return "train" if h < threshold else "eval"


def build_negative_sampling_samples(
    *,
    event_queries_path: Path,
    aligned_path: Path,
    actions_path: Path,
    train_ratio: float = 0.8,
    shift_seconds: Sequence[float] = (1.0, 2.0),
    text_score_mode: str = "rule",
    embedding_cache_path: str = "output/cache/text_embeddings.pkl",
    embedding_backend: str = "auto",
    embedding_model: str = "",
    embedding_dim: int = 128,
    hybrid_alpha: float = 0.5,
) -> List[Dict[str, Any]]:
    queries = _load_jsonl(event_queries_path)
    query_by_id = {str(q.get("query_id", q.get("event_id", ""))): q for q in queries}
    aligned = _normalize_aligned(_load_json(aligned_path))
    actions = _normalize_actions(_load_jsonl(actions_path))

    rows: List[Dict[str, Any]] = []
    sample_index = 0
    text_scorer = TextSimilarityScorer(
        mode=text_score_mode,
        cache_path=embedding_cache_path,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_dim=int(embedding_dim),
        hybrid_alpha=float(hybrid_alpha),
    )
    try:
        for aligned_row in aligned:
            query_id = str(aligned_row.get("query_id", aligned_row.get("event_id", ""))).strip()
            if not query_id:
                continue
            query = query_by_id.get(query_id, aligned_row)
            event_type = str(query.get("event_type", aligned_row.get("event_type", "unknown")))
            query_text = str(query.get("query_text", aligned_row.get("query_text", "")))
            split = _assign_split(query_id, train_ratio)

            positive = _pick_positive_candidate(
                query=query,
                aligned_row=aligned_row,
                actions=actions,
                text_scorer=text_scorer,
            )
            temporal_neg = _build_temporal_shift_negative(
                query=query,
                positive=positive,
                actions=actions,
                shift_seconds=shift_seconds,
                text_scorer=text_scorer,
            )
            semantic_neg = _build_semantic_mismatch_negative(
                query=query,
                aligned_row=aligned_row,
                positive=positive,
                actions=actions,
                text_scorer=text_scorer,
            )

            for neg_type, label, cand in (
                ("none", 1, positive),
                ("temporal_shift", 0, temporal_neg),
                ("semantic_mismatch", 0, semantic_neg),
            ):
                candidate_id = f"{query_id}__{sample_index:06d}__{neg_type}"
                uq = _clamp01(_safe_float(cand.get("uq_score", 0.5), 0.5))
                rows.append(
                    {
                        "query_id": query_id,
                        "candidate_id": candidate_id,
                        "overlap": _clamp01(_safe_float(cand.get("overlap", 0.0), 0.0)),
                        "action_confidence": _clamp01(_safe_float(cand.get("action_confidence", 0.5), 0.5)),
                        "text_score": _clamp01(_safe_float(cand.get("text_score", 0.0), 0.0)),
                        "uq_score": uq,
                        "stability_score": _clamp01(1.0 - uq),
                        "label": int(label),
                        "negative_type": neg_type,
                        "split": split,
                        "event_type": event_type,
                        "query_text": query_text,
                        "action_label": str(cand.get("action", "")),
                        "track_id": int(cand.get("track_id", -1)),
                    }
                )
                sample_index += 1

        if rows:
            train_count = sum(1 for r in rows if str(r.get("split", "")) == "train")
            eval_count = sum(1 for r in rows if str(r.get("split", "")) == "eval")
            if train_count == 0 and rows:
                first_query = str(rows[0].get("query_id", ""))
                for r in rows:
                    if str(r.get("query_id", "")) == first_query:
                        r["split"] = "train"
            if eval_count == 0 and rows:
                last_query = str(rows[-1].get("query_id", ""))
                for r in rows:
                    if str(r.get("query_id", "")) == last_query:
                        r["split"] = "eval"
            return rows

        # Fallback path when align is unavailable: synthesize from query x actions pairs.
        for query in queries:
            query_id = str(query.get("query_id", query.get("event_id", ""))).strip()
            if not query_id:
                continue
            split = _assign_split(query_id, train_ratio)
            event_type = str(query.get("event_type", "unknown"))
            query_text = str(query.get("query_text", ""))
            picked = actions[:3] if actions else []
            for i, act in enumerate(picked):
                action = str(act.get("action", "")).strip().lower()
                if not action:
                    continue
                text_score = _compute_text_score(
                    text_scorer=text_scorer,
                    event_type=event_type,
                    query_text=query_text,
                    action_label=action,
                )
                label = 1 if i == 0 else 0
                neg_type = "none" if i == 0 else ("temporal_shift" if i == 1 else "semantic_mismatch")
                uq = 0.5 if i == 0 else 0.6
                rows.append(
                    {
                        "query_id": query_id,
                        "candidate_id": f"{query_id}__fb_{i}",
                        "overlap": 0.6 if i == 0 else 0.2,
                        "action_confidence": _clamp01(_safe_float(act.get("action_confidence", 0.5), 0.5)),
                        "text_score": text_score if i != 2 else _clamp01(min(text_score, 0.3)),
                        "uq_score": uq,
                        "stability_score": _clamp01(1.0 - uq),
                        "label": int(label),
                        "negative_type": neg_type,
                        "split": split,
                        "event_type": event_type,
                        "query_text": query_text,
                        "action_label": action,
                        "track_id": int(act.get("track_id", -1)),
                    }
                )
        return rows
    finally:
        text_scorer.close()


def select_samples_for_setting(
    rows: Sequence[Dict[str, Any]],
    *,
    setting: str,
    train_split: str = "train",
) -> List[Dict[str, Any]]:
    setting = str(setting).strip().lower()
    allowed_neg: Dict[str, List[str]] = {
        "positive_only": [],
        "positive_plus_temporal_shift": ["temporal_shift"],
        "positive_plus_semantic_mismatch": ["semantic_mismatch"],
        "positive_plus_both": ["temporal_shift", "semantic_mismatch"],
    }
    if setting not in allowed_neg:
        raise ValueError(f"unsupported setting: {setting}")
    keep_neg = set(allowed_neg[setting])

    selected: List[Dict[str, Any]] = []
    for row in rows:
        split = str(row.get("split", "train"))
        neg_type = str(row.get("negative_type", "none"))
        if split != train_split:
            continue
        if int(row.get("label", 0)) == 1:
            selected.append(dict(row))
            continue
        if neg_type in keep_neg:
            selected.append(dict(row))
    return selected

