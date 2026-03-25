import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.igformer import IGFormerEncoder, InteractionClassifier
from models.interaction_graph import DSIGBuilder, DualGraphFusion, SDIGBuilder


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x_out = torch.einsum("nvu,nctu->nctv", a, x)
        return self.conv(x_out)


class ClassroomSTGCN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 3):
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
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


def _to_xyc(kp: object) -> Tuple[float, float, float]:
    if isinstance(kp, dict):
        return float(kp.get("x", 0.0)), float(kp.get("y", 0.0)), float(kp.get("c", 0.0) or 0.0)
    if isinstance(kp, (list, tuple)) and len(kp) >= 2:
        c = float(kp[2]) if len(kp) > 2 and kp[2] is not None else 0.0
        return float(kp[0]), float(kp[1]), c
    return 0.0, 0.0, 0.0


def pose_angle_features(person: dict) -> np.ndarray:
    """
    Return 5 lightweight angle-like features from 17-point skeleton.
    """
    kpts = person.get("keypoints", []) or []
    pts = []
    for i in range(min(17, len(kpts))):
        x, y, c = _to_xyc(kpts[i])
        pts.append((x, y, c))
    if len(pts) < 17:
        pts += [(0.0, 0.0, 0.0)] * (17 - len(pts))

    def vec(a: int, b: int) -> np.ndarray:
        ax, ay, ac = pts[a]
        bx, by, bc = pts[b]
        if ac < 0.05 or bc < 0.05:
            return np.zeros((2,), dtype=np.float32)
        return np.array([bx - ax, by - ay], dtype=np.float32)

    def cos_angle(v1: np.ndarray, v2: np.ndarray) -> float:
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        return float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))

    # COCO index: shoulders(5,6), elbows(7,8), wrists(9,10), hips(11,12), knees(13,14)
    left_arm = cos_angle(vec(5, 7), vec(7, 9))
    right_arm = cos_angle(vec(6, 8), vec(8, 10))
    left_leg = cos_angle(vec(11, 13), vec(13, 15))
    right_leg = cos_angle(vec(12, 14), vec(14, 16))
    torso = cos_angle(vec(5, 11), vec(6, 12))
    return np.array([left_arm, right_arm, left_leg, right_leg, torso], dtype=np.float32)


def load_pose_by_frame(pose_path: Path) -> Tuple[Dict[int, List[dict]], int]:
    data_by_frame: Dict[int, List[dict]] = defaultdict(list)
    max_frame = 0
    with open(pose_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            frame = int(d.get("frame", 0))
            data_by_frame[frame] = d.get("persons", [])
            max_frame = max(max_frame, frame)
    return data_by_frame, max_frame


def load_actions_by_frame(actions_path: Optional[Path], max_frame: int, fps: float) -> Dict[int, Dict[int, int]]:
    frame_actions: Dict[int, Dict[int, int]] = defaultdict(dict)
    if actions_path is None or (not actions_path.exists()):
        return frame_actions

    with open(actions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                a = json.loads(line)
            except Exception:
                continue

            tid = a.get("track_id", a.get("id"))
            code = a.get("action_code", a.get("code", -1))
            try:
                tid = int(tid)
                code = int(code)
            except Exception:
                continue

            sf = a.get("start_frame")
            ef = a.get("end_frame")
            if sf is None:
                st = a.get("start_time", a.get("time", a.get("t")))
                if st is not None:
                    sf = int(float(st) * fps)
            if ef is None:
                et = a.get("end_time", a.get("time", a.get("t")))
                if et is not None:
                    ef = int(float(et) * fps)
            if sf is None:
                sf = int(a.get("frame", 0))
            if ef is None:
                ef = int(sf)
            sf = max(0, int(sf))
            ef = min(max_frame, max(sf, int(ef)))

            for fr in range(sf, ef + 1):
                frame_actions[fr][tid] = code

    return frame_actions


def node_features_17d(persons: Sequence[dict], action_map: Dict[int, int], max_nodes: int, img_w: float, img_h: float) -> np.ndarray:
    feat = np.zeros((17, max_nodes), dtype=np.float32)
    valid = min(len(persons), max_nodes)
    for i in range(valid):
        p = persons[i]
        bbox = p.get("bbox", [0, 0, 0, 0])
        cx = ((float(bbox[0]) + float(bbox[2])) * 0.5) / max(1.0, img_w)
        cy = ((float(bbox[1]) + float(bbox[3])) * 0.5) / max(1.0, img_h)
        conf = float(p.get("conf", 1.0))

        tid = int(p.get("track_id", p.get("id", i)))
        a_code = int(action_map.get(tid, -1))
        onehot = np.zeros((9,), dtype=np.float32)
        if 0 <= a_code < 9:
            onehot[a_code] = 1.0

        angles = pose_angle_features(p)
        feat[:, i] = np.concatenate([np.array([cx, cy, conf], dtype=np.float32), onehot, angles], axis=0)
    return feat


def build_graph_from_frame(
    persons: Sequence[dict],
    action_map: Dict[int, int],
    dsig_builder: DSIGBuilder,
    sdig_builder: SDIGBuilder,
    fusion: DualGraphFusion,
    max_nodes: int = 50,
    img_w: float = 1920.0,
    img_h: float = 1080.0,
) -> Tuple[np.ndarray, np.ndarray, float, float, List[dict]]:
    """
    Returns:
      features_17d: (17, V)
      A_fused: (V, V)
      graph_density: float
      semantic_similarity_avg: float
      interaction_pairs: [{id_a, id_b, type, score}]
    """
    features = node_features_17d(persons, action_map, max_nodes=max_nodes, img_w=img_w, img_h=img_h)
    a_dsig, _, _ = dsig_builder.build(persons, img_w=img_w, img_h=img_h, max_nodes=max_nodes)
    a_sdig, sem_avg = sdig_builder.build(persons, action_map=action_map, max_nodes=max_nodes)
    a_fused = fusion.fuse_numpy(a_dsig, a_sdig).astype(np.float32)

    valid = min(len(persons), max_nodes)
    pairs = []
    if valid > 1:
        threshold = 0.30
        edge_cnt = 0
        for i in range(valid):
            for j in range(i + 1, valid):
                s = float(a_fused[i, j])
                if s >= threshold:
                    edge_cnt += 1
                    id_a = int(persons[i].get("track_id", persons[i].get("id", i)))
                    id_b = int(persons[j].get("track_id", persons[j].get("id", j)))
                    inter_type = "semantic" if float(a_sdig[i, j]) >= float(a_dsig[i, j]) else "proximity"
                    pairs.append(
                        {
                            "id_a": id_a,
                            "id_b": id_b,
                            "type": inter_type,
                            "score": float(round(s, 4)),
                        }
                    )
        density = edge_cnt / max(1.0, valid * (valid - 1) * 0.5)
    else:
        density = 0.0
    return features, a_fused, float(density), float(sem_avg), pairs


def load_igformer_weights(model_weight: str, encoder: IGFormerEncoder, classifier: InteractionClassifier, device: str) -> bool:
    path = Path(model_weight)
    if not model_weight or (not path.exists()):
        return False
    try:
        ckpt = torch.load(path, map_location=device)
        if isinstance(ckpt, dict) and "encoder" in ckpt and "classifier" in ckpt:
            encoder.load_state_dict(ckpt["encoder"], strict=False)
            classifier.load_state_dict(ckpt["classifier"], strict=False)
        elif isinstance(ckpt, dict):
            enc_sd = {k.replace("encoder.", "", 1): v for k, v in ckpt.items() if k.startswith("encoder.")}
            cls_sd = {k.replace("classifier.", "", 1): v for k, v in ckpt.items() if k.startswith("classifier.")}
            if enc_sd:
                encoder.load_state_dict(enc_sd, strict=False)
            if cls_sd:
                classifier.load_state_dict(cls_sd, strict=False)
            if not enc_sd and not cls_sd:
                # fallback: maybe this is direct legacy checkpoint
                return False
        encoder.eval()
        classifier.eval()
        return True
    except Exception as e:
        print(f"[WARN] Failed to load IGFormer weights: {e}")
        return False


def heuristic_legacy_label(avg_density: float, num_people: int) -> Tuple[str, float]:
    if num_people <= 1:
        return "empty", 1.0
    if avg_density > 0.6:
        return "discussion", 0.85
    if avg_density > 0.1:
        return "lecture", 0.60
    return "individual_work", 0.50


def heuristic_igformer_label(avg_density: float, sem_avg: float, num_people: int) -> Tuple[str, float]:
    if num_people <= 1:
        return "individual_work", 0.60
    if avg_density > 0.65 and sem_avg > 0.58:
        return "group_discuss", 0.84
    if avg_density > 0.45:
        return "pair_chat", 0.73
    if avg_density > 0.22:
        return "lecture", 0.66
    if avg_density < 0.08:
        return "break", 0.56
    return "transition", 0.52


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", required=True, help="pose_tracks_smooth.jsonl")
    parser.add_argument("--out", required=True, help="group_events.jsonl")
    parser.add_argument("--actions", default="", help="actions.jsonl for SDIG semantic graph")
    parser.add_argument("--model_weight", default="", help="model weight path")
    parser.add_argument("--window_size", type=int, default=50, help="frame window size")
    parser.add_argument("--fps", type=float, default=25.0, help="fps for output times")
    parser.add_argument("--interaction_model", type=str, default="legacy", choices=["legacy", "igformer"])
    parser.add_argument("--legacy_stgcn", action="store_true", help="force legacy ST-GCN")
    parser.add_argument("--dsig_k", type=int, default=3, help="KNN k for DSIG")
    parser.add_argument("--sdig_threshold", type=float, default=0.7, help="semantic threshold for SDIG")
    parser.add_argument("--dist_thres", type=float, default=0.15, help="base distance threshold for DSIG")
    args = parser.parse_args()

    pose_path = Path(args.pose)
    out_path = Path(args.out)
    actions_path = Path(args.actions) if args.actions else None

    data_by_frame, max_frame = load_pose_by_frame(pose_path)
    print(f"[Info] Loaded pose data to frame={max_frame}")
    frame_actions = load_actions_by_frame(actions_path, max_frame=max_frame, fps=float(args.fps))
    if actions_path and actions_path.exists():
        print(f"[Info] Loaded action semantics from {actions_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_legacy = bool(args.legacy_stgcn) or (args.interaction_model == "legacy")

    legacy_model = ClassroomSTGCN(num_classes=3).to(device)
    has_legacy_weights = False
    if use_legacy and args.model_weight and Path(args.model_weight).exists():
        try:
            legacy_model.load_state_dict(torch.load(args.model_weight, map_location=device), strict=False)
            legacy_model.eval()
            has_legacy_weights = True
            print(f"[Model] Legacy ST-GCN weights loaded from {args.model_weight}")
        except Exception as e:
            print(f"[WARN] Failed loading legacy weights: {e}")

    ig_encoder = IGFormerEncoder(in_dim=17, d_model=128, n_heads=4, n_layers=4, max_time=max(64, args.window_size), max_nodes=64).to(device)
    ig_classifier = InteractionClassifier(d_model=128, num_classes=6).to(device)
    has_ig_weights = (not use_legacy) and load_igformer_weights(args.model_weight, ig_encoder, ig_classifier, device=device)
    if (not use_legacy) and has_ig_weights:
        print(f"[Model] IGFormer weights loaded from {args.model_weight}")
    elif not use_legacy:
        print("[Model] IGFormer running in heuristic mode (no trained weights).")

    dsig_builder = DSIGBuilder(dist_thres=float(args.dist_thres), k=int(args.dsig_k), adaptive_scale=1.0)
    sdig_builder = SDIGBuilder(action_dim=9, threshold=float(args.sdig_threshold))
    fusion = DualGraphFusion(alpha_init=0.5)

    window_step = max(1, int(args.window_size // 2))
    events = []

    for start_f in range(0, max_frame + 1, window_step):
        end_f = start_f + int(args.window_size)
        clip_feat = []
        clip_adj = []
        clip_density = []
        clip_sem = []
        clip_pair_frames = []
        mid_people_count = 0

        for fr in range(start_f, end_f):
            persons = data_by_frame.get(fr, [])
            action_map = frame_actions.get(fr, {})
            feat, adj, density, sem_avg, pairs = build_graph_from_frame(
                persons=persons,
                action_map=action_map,
                dsig_builder=dsig_builder,
                sdig_builder=sdig_builder,
                fusion=fusion,
                max_nodes=50,
            )
            clip_feat.append(feat)
            clip_adj.append(adj)
            clip_density.append(density)
            clip_sem.append(sem_avg)
            clip_pair_frames.append(pairs)

        if not clip_feat:
            continue

        mid_idx = len(clip_feat) // 2
        mid_people_count = min(len(data_by_frame.get(start_f + mid_idx, [])), 50)
        avg_density = float(np.mean(clip_density))
        avg_sem = float(np.mean(clip_sem))
        interaction_pairs = clip_pair_frames[mid_idx]

        label = "unknown"
        confidence = 0.0

        if use_legacy:
            # Legacy input: (N, C, T, V), C=3
            tx = np.stack([f[:3, :] for f in clip_feat], axis=0)  # (T,3,V)
            tx = np.transpose(tx, (1, 0, 2))  # (3,T,V)
            input_x = torch.tensor(tx, dtype=torch.float32).unsqueeze(0).to(device)
            input_a = torch.tensor(clip_adj[mid_idx], dtype=torch.float32).unsqueeze(0).to(device)

            if has_legacy_weights:
                with torch.no_grad():
                    logits = legacy_model(input_x, input_a)
                    probs = F.softmax(logits, dim=1)
                    conf, pred_idx = torch.max(probs, 1)
                mapping = {0: "lecture", 1: "discussion", 2: "break"}
                label = mapping.get(int(pred_idx.item()), "unknown")
                confidence = float(conf.item())
            else:
                label, confidence = heuristic_legacy_label(avg_density, mid_people_count)
        else:
            # IGFormer input: (B,T,V,C)
            tx = np.stack(clip_feat, axis=0)  # (T,17,V)
            tx = np.transpose(tx, (0, 2, 1))  # (T,V,17)
            input_x = torch.tensor(tx, dtype=torch.float32).unsqueeze(0).to(device)
            input_a = torch.tensor(clip_adj[mid_idx], dtype=torch.float32).unsqueeze(0).to(device)

            if has_ig_weights:
                with torch.no_grad():
                    z = ig_encoder(input_x, input_a)
                    logits = ig_classifier(z)
                    probs = F.softmax(logits, dim=1)
                    conf, pred_idx = torch.max(probs, 1)
                labels = InteractionClassifier.LABELS
                label = labels[int(pred_idx.item())] if int(pred_idx.item()) < len(labels) else "transition"
                confidence = float(conf.item())
            else:
                label, confidence = heuristic_igformer_label(avg_density, avg_sem, mid_people_count)

        if label != "empty":
            events.append(
                {
                    "start_frame": int(start_f),
                    "end_frame": int(end_f),
                    "start_time": float(start_f / float(args.fps)),
                    "end_time": float(end_f / float(args.fps)),
                    "group_event": label,
                    "confidence": float(round(confidence, 4)),
                    "graph_density": float(round(avg_density, 4)),
                    "semantic_similarity_avg": float(round(avg_sem, 4)),
                    "interaction_pairs": interaction_pairs,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"[Done] Detected {len(events)} group interaction segments.")
    print(f"[Done] Output: {out_path}")


if __name__ == "__main__":
    main()

