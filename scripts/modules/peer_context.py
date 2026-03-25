"""Peer-aware spatial context utilities."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


NEGATIVE_ACTIONS = {"distract", "phone", "doze", "sleep", "playing_phone"}
POSITIVE_ACTIONS = {"listen", "listening", "note", "writing", "read", "reading", "reading_writing"}


def _bbox_center(bbox: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _action_label(action: Mapping[str, Any]) -> str:
    raw = action.get("action", action.get("label", "unknown"))
    return str(raw).strip().lower()


def _frame_range(action: Mapping[str, Any]) -> Tuple[int, int]:
    sf = action.get("start_frame", action.get("frame"))
    ef = action.get("end_frame", action.get("frame"))
    try:
        sf_i = int(float(sf))
    except Exception:
        sf_i = -1
    try:
        ef_i = int(float(ef))
    except Exception:
        ef_i = sf_i
    if ef_i < sf_i:
        sf_i, ef_i = ef_i, sf_i
    return sf_i, ef_i


def _find_action_at_frame(actions: Sequence[Mapping[str, Any]], frame_idx: int) -> Optional[str]:
    for item in actions:
        sf, ef = _frame_range(item)
        if sf <= frame_idx <= ef:
            return _action_label(item)
    if actions:
        return _action_label(actions[-1])
    return None


def _safe_track_id(person: Mapping[str, Any], default: int) -> int:
    tid = person.get("track_id", person.get("id", default))
    try:
        return int(tid)
    except Exception:
        return int(default)


def build_spatial_neighbor_index(
    pose_tracks_file: Union[str, Path],
    peer_radius: float = 0.15,
) -> Dict[int, Dict[int, List[int]]]:
    """
    Build frame-level spatial neighbors:
      {track_id: {frame_idx: [neighbor_track_id, ...]}}
    """
    path = Path(pose_tracks_file)
    index: Dict[int, Dict[int, List[int]]] = defaultdict(dict)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            frame_idx = int(rec.get("frame", 0))
            persons = rec.get("persons", [])
            if not persons or len(persons) < 2:
                continue

            centers: List[Tuple[int, float, float]] = []
            x_all: List[float] = []
            y_all: List[float] = []
            for i, p in enumerate(persons):
                bbox = p.get("bbox")
                if not isinstance(bbox, list) or len(bbox) < 4:
                    continue
                tid = _safe_track_id(p, i)
                cx, cy = _bbox_center(bbox)
                centers.append((tid, cx, cy))
                x_all.extend([float(bbox[0]), float(bbox[2])])
                y_all.extend([float(bbox[1]), float(bbox[3])])

            if len(centers) < 2:
                continue

            # Use in-frame spread to normalize distance threshold.
            dx = (max(x_all) - min(x_all)) if x_all else 1.0
            dy = (max(y_all) - min(y_all)) if y_all else 1.0
            scene_diag = max(1.0, math.hypot(dx, dy))
            dist_th = float(peer_radius) * scene_diag

            neigh_map: Dict[int, List[int]] = defaultdict(list)
            for i in range(len(centers)):
                tid_i, xi, yi = centers[i]
                for j in range(i + 1, len(centers)):
                    tid_j, xj, yj = centers[j]
                    if math.hypot(xi - xj, yi - yj) <= dist_th:
                        neigh_map[tid_i].append(tid_j)
                        neigh_map[tid_j].append(tid_i)

            for tid, nbs in neigh_map.items():
                index[tid][frame_idx] = sorted(set(nbs))

    return {int(k): v for k, v in index.items()}


def extract_peer_features(
    target_id: int,
    frame_idx: int,
    neighbors_dict: Mapping[int, Mapping[int, Sequence[int]]],
    target_actions: Sequence[Mapping[str, Any]],
    neighbor_actions: Mapping[int, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """
    Build peer context features for one target at one frame.
    """
    target_id = int(target_id)
    frame_idx = int(frame_idx)

    frame_neighbors = list(neighbors_dict.get(target_id, {}).get(frame_idx, []))
    frame_neighbors = [int(x) for x in frame_neighbors]

    labels: List[str] = []
    for nb in frame_neighbors:
        acts = neighbor_actions.get(nb, [])
        label = _find_action_at_frame(acts, frame_idx)
        if label:
            labels.append(label)

    cnt = Counter(labels)
    total = max(1, len(labels))
    dominant = cnt.most_common(1)[0][0] if cnt else "unknown"

    target_label = _find_action_at_frame(target_actions, frame_idx)
    same_cnt = int(cnt.get(target_label, 0)) if target_label else 0
    agreement = (same_cnt / len(labels)) if labels else 0.0

    listen_ratio = (cnt.get("listen", 0) + cnt.get("listening", 0)) / total
    note_ratio = (cnt.get("note", 0) + cnt.get("writing", 0)) / total
    distract_ratio = (cnt.get("distract", 0) + cnt.get("phone", 0) + cnt.get("doze", 0)) / total

    return {
        "target_action": target_label or "unknown",
        "neighbor_count": len(frame_neighbors),
        "neighbor_ids": frame_neighbors,
        "dominant_peer_action": dominant,
        "peer_agreement_score": float(round(agreement, 4)),
        "listen_ratio": float(round(listen_ratio, 4)),
        "note_ratio": float(round(note_ratio, 4)),
        "distract_ratio": float(round(distract_ratio, 4)),
        "distribution": {k: float(round(v / total, 4)) for k, v in cnt.items()},
    }


def apply_peer_correction(
    action_confidence: Mapping[str, float],
    peer_features: Mapping[str, Any],
    min_neighbors: int = 2,
) -> Tuple[Dict[str, float], bool]:
    """
    Rule-based confidence correction from peer context.
    """
    scores: Dict[str, float] = {str(k): float(v) for k, v in action_confidence.items()}
    if not scores:
        return {}, False

    neighbor_count = int(peer_features.get("neighbor_count", 0))
    dominant = str(peer_features.get("dominant_peer_action", "unknown"))
    agreement = float(peer_features.get("peer_agreement_score", 0.0))

    if neighbor_count < min_neighbors or dominant == "unknown":
        return scores, False

    changed = False

    # If peers strongly stay in positive learning actions, down-weight negative actions.
    if dominant in POSITIVE_ACTIONS and agreement < 0.4:
        decay = 0.75
        boost = 0.15
        for k in list(scores.keys()):
            lk = k.lower().strip()
            if lk in NEGATIVE_ACTIONS:
                scores[k] *= decay
                changed = True
        if dominant in scores:
            scores[dominant] = scores.get(dominant, 0.0) + boost
            changed = True

    # If peers are mostly distracted, slightly down-weight "listen/note" overconfidence.
    if dominant in NEGATIVE_ACTIONS and agreement < 0.3:
        for k in list(scores.keys()):
            lk = k.lower().strip()
            if lk in {"listen", "listening", "note", "writing"}:
                scores[k] *= 0.9
                changed = True
        if dominant in scores:
            scores[dominant] = scores.get(dominant, 0.0) + 0.08
            changed = True

    total = sum(max(0.0, v) for v in scores.values())
    if total > 1e-9:
        scores = {k: max(0.0, v) / total for k, v in scores.items()}

    return scores, changed
