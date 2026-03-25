"""Peer-aware context features and lightweight correction logic."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn as nn


CANONICAL_ACTIONS = (
    "listen",
    "note",
    "distract",
    "phone",
    "doze",
    "chat",
    "raise_hand",
    "stand",
    "read",
)

ACTION_TO_CODE = {
    "listen": 0,
    "distract": 1,
    "phone": 2,
    "doze": 3,
    "chat": 4,
    "note": 5,
    "raise_hand": 6,
    "stand": 7,
    "read": 8,
}

_ALIASES = {
    "listening": "listen",
    "writing": "note",
    "reading_writing": "note",
    "playing_phone": "phone",
    "sleeping": "doze",
    "chatting": "chat",
    "raising_hand": "raise_hand",
    "raise": "raise_hand",
    "standing": "stand",
    "individual_work": "note",
}


def _norm_action(x: Any) -> str:
    a = str(x or "").lower().strip()
    return _ALIASES.get(a, a)


def _center_from_bbox(b: Sequence[float]) -> tuple[float, float]:
    return (float(b[0]) + float(b[2])) * 0.5, (float(b[1]) + float(b[3])) * 0.5


def build_spatial_neighbor_index(pose_tracks: List[Dict[str, Any]], radius: float = 0.15) -> Dict[int, Dict[int, List[int]]]:
    """
    Build spatio-temporal neighbor index:
      {track_id: {frame: [neighbor_ids]}}

    Radius uses normalized coordinates (bbox center / image size estimated from frame boxes).
    """
    out: Dict[int, Dict[int, List[int]]] = defaultdict(dict)
    for rec in pose_tracks:
        frame = int(rec.get("frame", -1))
        persons = rec.get("persons", []) or []
        if frame < 0 or len(persons) <= 1:
            continue

        boxes = [p.get("bbox", [0, 0, 0, 0]) for p in persons]
        max_x = max(float(b[2]) for b in boxes) + 1e-6
        max_y = max(float(b[3]) for b in boxes) + 1e-6
        centers = []
        ids = []
        for p in persons:
            tid = p.get("track_id")
            if tid is None:
                continue
            cx, cy = _center_from_bbox(p.get("bbox", [0, 0, 0, 0]))
            centers.append((cx / max_x, cy / max_y))
            ids.append(int(tid))

        n = len(ids)
        if n <= 1:
            continue
        for i in range(n):
            nei = []
            for j in range(n):
                if i == j:
                    continue
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                if (dx * dx + dy * dy) ** 0.5 <= radius:
                    nei.append(ids[j])
            out[ids[i]][frame] = nei
    return {k: dict(v) for k, v in out.items()}


def extract_peer_features(
    target_id: int,
    neighbors: Iterable[int],
    actions: List[Dict[str, Any]],
    action_space: Sequence[str] = CANONICAL_ACTIONS,
) -> Dict[str, Any]:
    """Aggregate neighbor action distribution vector within a shared time range."""
    neighbors_set: Set[int] = {int(x) for x in neighbors if x is not None}
    counts = {a: 0.0 for a in action_space}
    total = 0.0
    for item in actions:
        tid = item.get("track_id")
        if tid is None or int(tid) not in neighbors_set:
            continue
        act = _norm_action(item.get("action", item.get("label", "")))
        if act not in counts:
            continue
        conf = float(item.get("confidence", item.get("conf", 1.0)) or 1.0)
        counts[act] += max(conf, 0.0)
        total += max(conf, 0.0)

    if total <= 1e-9:
        vec = [0.0 for _ in action_space]
        dominant = "none"
        agreement = 0.0
    else:
        vec = [counts[a] / total for a in action_space]
        dominant = action_space[int(np.argmax(vec))]
        agreement = float(np.max(vec))

    return {
        "target_id": int(target_id),
        "neighbor_count": int(len(neighbors_set)),
        "peer_action_dist": {a: float(v) for a, v in zip(action_space, vec)},
        "dominant_peer_action": dominant,
        "peer_agreement_score": float(agreement),
    }


class PeerAwareClassifier(nn.Module):
    """Lightweight fusion head for self action + peer context correction."""

    def __init__(self, self_dim: int = 9, peer_dim: int = 9, hidden_dim: int = 64, out_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self_dim + peer_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, self_action_emb: torch.Tensor, peer_context_vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self_action_emb, peer_context_vec], dim=-1)
        return self.net(x)


def apply_peer_correction(
    person_actions: List[Dict[str, Any]],
    peer_features: Dict[str, Any],
    threshold: float = 0.55,
) -> Dict[str, Any]:
    """
    Rule + confidence correction.
    If self action is distract-like but peers are strongly note/listen-like, reduce distract confidence.
    """
    dominant = peer_features.get("dominant_peer_action", "none")
    agree = float(peer_features.get("peer_agreement_score", 0.0))
    corrected = []
    changed = False

    for a in person_actions:
        b = dict(a)
        act = _norm_action(b.get("action", b.get("label", "")))
        conf = float(b.get("confidence", b.get("conf", 1.0)) or 1.0)
        correction_applied = False
        corrected_action = act

        if agree >= threshold and dominant in {"note", "listen", "read"} and act in {"distract", "phone", "doze"}:
            conf = max(0.05, conf - 0.25)
            corrected_action = dominant
            correction_applied = True
            changed = True

        b["confidence"] = float(conf)
        b["corrected_action"] = corrected_action
        b["corrected_action_code"] = int(ACTION_TO_CODE.get(corrected_action, ACTION_TO_CODE.get(act, 0)))
        b["peer_correction_applied"] = bool(correction_applied)
        corrected.append(b)

    return {
        "actions": corrected,
        "changed": changed,
    }
