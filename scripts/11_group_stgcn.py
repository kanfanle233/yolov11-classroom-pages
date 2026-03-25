import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.igformer import IGFormerEncoder, InteractionClassifier
from models.interaction_graph import DSIGBuilder, DualGraphFusion, SDIGBuilder


class GraphConv(nn.Module):
    """Legacy light GCN layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x_out = torch.einsum("nvu,nctu->nctv", a, x)
        return self.conv(x_out)


class ClassroomSTGCN(nn.Module):
    """Legacy ST-GCN fallback model."""

    def __init__(self, in_channels: int = 17, num_classes: int = 3):
        super().__init__()
        self.gcn1 = GraphConv(in_channels, 64)
        self.gcn2 = GraphConv(64, 128)
        self.tcn1 = nn.Conv2d(64, 64, kernel_size=(9, 1), padding=(4, 0))
        self.tcn2 = nn.Conv2d(128, 128, kernel_size=(9, 1), padding=(4, 0))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.tcn1(self.gcn1(x, a)))
        x = F.relu(self.tcn2(self.gcn2(x, a)))
        x = F.avg_pool2d(x, x.size()[2:])
        x = self.dropout(x.view(x.size(0), -1))
        return self.fc(x)


class IGFormerNet(nn.Module):
    """IGFormer encoder + classifier."""

    def __init__(self, in_dim: int = 17, d_model: int = 128, n_layers: int = 4, max_time: int = 256, max_nodes: int = 64):
        super().__init__()
        self.encoder = IGFormerEncoder(
            in_dim=in_dim, d_model=d_model, n_heads=4, n_layers=n_layers, max_time=max_time, max_nodes=max_nodes
        )
        self.classifier = InteractionClassifier(d_model=d_model, hidden_dim=256, num_classes=6)

    def forward(self, x: torch.Tensor, a_fused: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, a_fused)
        return self.classifier(z)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba) + 1e-6
    nbc = np.linalg.norm(bc) + 1e-6
    cosv = np.clip(float(np.dot(ba, bc) / (nba * nbc)), -1.0, 1.0)
    return math.degrees(math.acos(cosv))


def _kp_xyc(k: Any) -> Tuple[float, float, float]:
    if isinstance(k, dict):
        return float(k.get("x", 0.0)), float(k.get("y", 0.0)), float(k.get("c", 0.0) or 0.0)
    if isinstance(k, (list, tuple)) and len(k) >= 2:
        c = float(k[2]) if len(k) > 2 and k[2] is not None else 0.0
        return float(k[0]), float(k[1]), c
    return 0.0, 0.0, 0.0


def pose_angle_features(keypoints: List[Any]) -> List[float]:
    pts = np.zeros((17, 3), dtype=np.float32)
    for i in range(min(17, len(keypoints))):
        x, y, c = _kp_xyc(keypoints[i])
        pts[i] = [x, y, c]

    xy = pts[:, :2]
    # COCO: 5 LS, 6 RS, 7 LE, 8 RE, 9 LW, 10 RW, 11 LH, 12 RH
    feats = []
    try:
        feats.append(_angle(xy[5], xy[7], xy[9]) / 180.0)
    except Exception:
        feats.append(0.0)
    try:
        feats.append(_angle(xy[6], xy[8], xy[10]) / 180.0)
    except Exception:
        feats.append(0.0)
    # shoulder slope
    sh = xy[6] - xy[5]
    feats.append((math.atan2(float(sh[1]), float(sh[0] + 1e-6)) / math.pi + 1.0) * 0.5)
    # hip slope
    hp = xy[12] - xy[11]
    feats.append((math.atan2(float(hp[1]), float(hp[0] + 1e-6)) / math.pi + 1.0) * 0.5)
    # torso lean
    mid_sh = (xy[5] + xy[6]) * 0.5
    mid_hp = (xy[11] + xy[12]) * 0.5
    torso = mid_hp - mid_sh
    feats.append((math.atan2(float(torso[1]), float(torso[0] + 1e-6)) / math.pi + 1.0) * 0.5)
    return [float(np.clip(x, 0.0, 1.0)) for x in feats[:5]]


def action_lookup(actions: List[dict], fps: float) -> Dict[int, Dict[int, int]]:
    """frame -> {track_id: action_code}"""
    out: Dict[int, Dict[int, int]] = defaultdict(dict)
    for a in actions:
        tid = a.get("track_id")
        if tid is None:
            continue
        code = a.get("action_code")
        if code is None:
            # string fallback
            action = str(a.get("action", "")).lower().strip()
            code = {
                "listen": 0,
                "distract": 1,
                "phone": 2,
                "doze": 3,
                "chat": 4,
                "note": 5,
                "raise_hand": 6,
                "stand": 7,
                "read": 8,
            }.get(action, 0)

        sf = a.get("start_frame")
        ef = a.get("end_frame")
        if sf is None or ef is None:
            st = float(a.get("start_time", a.get("time", 0.0)) or 0.0)
            ed = float(a.get("end_time", st + 0.2) or (st + 0.2))
            sf = int(st * fps)
            ef = int(ed * fps)
        sf = int(sf)
        ef = int(max(sf, ef))
        for f in range(sf, ef + 1):
            out[f][int(tid)] = int(code)
    return out


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return rows


def build_graph_from_frame(
    persons: List[dict],
    frame_action_map: Dict[int, int],
    img_width: int,
    img_height: int,
    max_nodes: int,
    dsig_builder: DSIGBuilder,
    sdig_builder: SDIGBuilder,
    fusion: DualGraphFusion,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], float, float]:
    """Construct per-frame feature + DSIG + SDIG + fused graph."""
    feat = np.zeros((17, max_nodes), dtype=np.float32)
    valid = persons[: max_nodes]
    track_ids: List[int] = []

    for i, p in enumerate(valid):
        bbox = p.get("bbox", [0, 0, 0, 0])
        cx = ((float(bbox[0]) + float(bbox[2])) * 0.5) / max(float(img_width), 1.0)
        cy = ((float(bbox[1]) + float(bbox[3])) * 0.5) / max(float(img_height), 1.0)
        conf = float(p.get("conf", 1.0) or 0.0)
        tid = int(p.get("track_id", p.get("id", i)))
        track_ids.append(tid)
        action_code = int(frame_action_map.get(tid, 0))
        onehot = [0.0] * 9
        if 0 <= action_code < 9:
            onehot[action_code] = 1.0
        pose5 = pose_angle_features(p.get("keypoints", []))
        vec17 = [cx, cy, conf] + onehot + pose5
        feat[:, i] = np.asarray(vec17, dtype=np.float32)

    a_dsig, _, adaptive_th = dsig_builder.build(valid, img_width, img_height, max_nodes=max_nodes)
    a_sdig, semantic_avg = sdig_builder.build(valid, frame_action_map, max_nodes=max_nodes)
    a_fused = fusion.fuse_numpy(a_dsig, a_sdig)
    return feat, a_dsig, a_sdig, a_fused, track_ids, semantic_avg, adaptive_th


def interaction_pairs(a_fused: np.ndarray, track_ids: List[int], threshold: float = 0.35) -> List[Dict[str, Any]]:
    n = len(track_ids)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(a_fused[i, j])
            if s < threshold:
                continue
            t = "pair_chat" if s >= 0.75 else "group_discuss"
            pairs.append({"id_a": int(track_ids[i]), "id_b": int(track_ids[j]), "type": t, "score": float(f"{s:.3f}")})
    pairs.sort(key=lambda x: x["score"], reverse=True)
    return pairs[:20]


def heuristic_label(graph_density: float, semantic_sim: float, n_people: int) -> Tuple[str, float]:
    if n_people <= 1:
        return "individual_work", 0.60
    if graph_density > 0.55 and semantic_sim > 0.55:
        return "group_discuss", 0.78
    if graph_density > 0.35:
        return "pair_chat", 0.65
    if semantic_sim > 0.45:
        return "lecture", 0.60
    return "transition", 0.50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", required=True, help="pose_tracks_smooth.jsonl")
    parser.add_argument("--actions", default="", help="actions.jsonl for SDIG semantic action features")
    parser.add_argument("--out", required=True, help="group_events.jsonl")
    parser.add_argument("--model_weight", default="", help="Path to trained model weights")
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--max_nodes", type=int, default=50)
    parser.add_argument("--interaction_model", choices=["igformer", "legacy"], default="igformer")
    parser.add_argument("--legacy_stgcn", action="store_true", help="force use legacy ST-GCN model")
    parser.add_argument("--dsig_k", type=int, default=3)
    parser.add_argument("--sdig_threshold", type=float, default=0.35)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    pose_path = Path(args.pose)
    out_path = Path(args.out)
    actions_path = Path(args.actions) if args.actions else None
    if not pose_path.is_absolute():
        pose_path = (base_dir / pose_path).resolve()
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()
    if actions_path and not actions_path.is_absolute():
        actions_path = (base_dir / actions_path).resolve()

    pose_rows = load_jsonl(pose_path)
    if not pose_rows:
        raise FileNotFoundError(f"No pose records: {pose_path}")
    data_by_frame = {int(r.get("frame", i)): r.get("persons", []) for i, r in enumerate(pose_rows)}
    max_frame = max(data_by_frame.keys())

    action_rows = load_jsonl(actions_path) if actions_path and actions_path.exists() else []
    act_by_frame = action_lookup(action_rows, fps=args.fps) if action_rows else defaultdict(dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_legacy = bool(args.legacy_stgcn or args.interaction_model == "legacy")
    if use_legacy:
        model = ClassroomSTGCN(in_channels=17, num_classes=3).to(device)
        label_map = {0: "lecture", 1: "group_discuss", 2: "break"}
    else:
        model = IGFormerNet(in_dim=17, d_model=128, n_layers=4, max_time=args.window_size, max_nodes=args.max_nodes).to(device)
        label_map = {i: k for i, k in enumerate(InteractionClassifier.LABELS)}

    has_weights = False
    if args.model_weight:
        w = Path(args.model_weight)
        if not w.is_absolute():
            w = (base_dir / w).resolve()
        if w.exists():
            sd = torch.load(w, map_location=device)
            try:
                model.load_state_dict(sd, strict=False)
            except Exception:
                if isinstance(sd, dict) and "state_dict" in sd:
                    model.load_state_dict(sd["state_dict"], strict=False)
            has_weights = True
    model.eval()

    dsig_builder = DSIGBuilder(dist_thres=0.15, k=args.dsig_k)
    sdig_builder = SDIGBuilder(action_dim=9, threshold=args.sdig_threshold)
    fusion = DualGraphFusion(alpha_init=0.5)

    events = []
    step = max(1, args.window_size // 2)

    for start_f in range(0, max_frame + 1, step):
        end_f = start_f + args.window_size
        feat_list = []
        dsig_list = []
        sdig_list = []
        fused_list = []
        semantic_vals = []
        adaptive_vals = []
        mid_track_ids: List[int] = []
        for f_idx in range(start_f, end_f):
            persons = data_by_frame.get(f_idx, [])
            frame_action = act_by_frame.get(f_idx, {})
            feat, a_d, a_s, a_f, track_ids, sem_avg, adp_th = build_graph_from_frame(
                persons=persons,
                frame_action_map=frame_action,
                img_width=args.width,
                img_height=args.height,
                max_nodes=args.max_nodes,
                dsig_builder=dsig_builder,
                sdig_builder=sdig_builder,
                fusion=fusion,
            )
            feat_list.append(feat)
            dsig_list.append(a_d)
            sdig_list.append(a_s)
            fused_list.append(a_f)
            semantic_vals.append(sem_avg)
            adaptive_vals.append(adp_th)
            if f_idx == start_f + args.window_size // 2:
                mid_track_ids = track_ids

        tx = np.stack(feat_list, axis=0)  # (T, C, V)
        avg_fused = np.mean(np.stack(fused_list, axis=0), axis=0).astype(np.float32)
        avg_sdig = np.mean(np.stack(sdig_list, axis=0), axis=0).astype(np.float32)

        n_people = len(mid_track_ids)
        density = 0.0
        if n_people > 1:
            sub = avg_fused[:n_people, :n_people].copy()
            np.fill_diagonal(sub, 0.0)
            density = float((sub > args.sdig_threshold).sum()) / float(n_people * (n_people - 1))
        semantic_similarity_avg = float(np.mean(semantic_vals)) if semantic_vals else 0.0
        pairs = interaction_pairs(avg_fused, mid_track_ids, threshold=args.sdig_threshold)

        label = "transition"
        conf = 0.45
        if has_weights:
            with torch.no_grad():
                if use_legacy:
                    input_x = torch.tensor(np.transpose(tx, (1, 0, 2)), dtype=torch.float32).unsqueeze(0).to(device)
                    input_a = torch.tensor(avg_fused, dtype=torch.float32).unsqueeze(0).to(device)
                    logits = model(input_x, input_a)
                else:
                    x = torch.tensor(np.transpose(tx, (0, 2, 1)), dtype=torch.float32).unsqueeze(0).to(device)  # (1,T,V,C)
                    a = torch.tensor(avg_fused, dtype=torch.float32).unsqueeze(0).to(device)
                    logits = model(x, a)
                prob = F.softmax(logits, dim=1)
                c, idx = torch.max(prob, dim=1)
                label = label_map.get(int(idx.item()), "transition")
                conf = float(c.item())
        else:
            label, conf = heuristic_label(density, semantic_similarity_avg, n_people)

        if n_people > 0:
            events.append(
                {
                    "start_frame": int(start_f),
                    "end_frame": int(end_f),
                    "start_time": float(start_f / max(args.fps, 1e-6)),
                    "end_time": float(end_f / max(args.fps, 1e-6)),
                    "group_event": label,
                    "confidence": float(f"{conf:.3f}"),
                    "interaction_pairs": pairs,
                    "graph_density": float(f"{density:.4f}"),
                    "semantic_similarity_avg": float(f"{semantic_similarity_avg:.4f}"),
                    "fusion_alpha": float(f"{fusion.alpha.detach().cpu().item():.4f}"),
                    "adaptive_dist_thres": float(f"{np.mean(adaptive_vals):.4f}" if adaptive_vals else "0.1500"),
                    "interaction_model": "legacy" if use_legacy else "igformer",
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"[Done] Group interactions -> {out_path} (segments={len(events)})")


if __name__ == "__main__":
    main()

