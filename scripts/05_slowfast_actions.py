import os
import sys
import json
import cv2
import torch
import argparse
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List
import torchvision

# === 鍏煎鎬цˉ涓?START ===
# PyTorchVideo 渚濊禆鏃х増 torchvision API锛岃繖閲屾墜鍔ㄨ繘琛屾槧灏勪慨澶?
try:
    import torchvision.transforms.functional_tensor as F_t
except ImportError:
    from torchvision.transforms import functional as F_t

    sys.modules["torchvision.transforms.functional_tensor"] = F_t
# === 鍏煎鎬цˉ涓?END ===

# PyTorchVideo & TorchVision
from pytorchvideo.models.hub import slowfast_r50
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

# === 閰嶇疆鍖?===
CLASS_LABELS = {
    0: "listen",
    1: "distract",
    2: "phone",
    3: "doze",
    4: "chat",
    5: "note",
    6: "raise_hand",
    7: "stand",
    8: "read"
}

# 鍔ㄤ綔閲囨牱鍙傛暟
CLIP_DURATION = 32  # 杈撳叆妯″瀷鐨勫抚鏁?(SlowFast 榛樿 32)
SAMPLING_RATE = 2  # 璺冲抚閲囨牱 (瑕嗙洊 64 甯х殑鐗╃悊鏃堕棿)
INFERENCE_STRIDE = 30  # 姣忛殧澶氬皯甯у仛涓€娆℃帹鐞?(绾?1.2 绉掍竴娆?
CROP_SIZE = 256  # 杈撳叆妯″瀷鐨勫浘鍍忓ぇ灏?


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class ActionRecognizer:
    def __init__(self, model_path=None, device="cuda"):
        self.device = device
        print(f"[Model] Loading SlowFast R50 (device={device})...")

        # 1. 鍔犺浇棰勮缁冪殑 Backbone (Kinetics-400)
        self.model = slowfast_r50(pretrained=True)

        # 2. [鍏抽敭淇敼] 寮哄埗淇敼鍏ㄨ繛鎺ュ眰浠ュ尮閰嶆暀瀹よ涓虹被鍒暟 (9绫?
        # 鑾峰彇杈撳叆鐗瑰緛缁村害 (閫氬父鏄?2304)
        in_features = self.model.blocks[-1].proj.in_features
        self.model.blocks[-1].proj = torch.nn.Linear(in_features, len(CLASS_LABELS))

        # 3. 鍔犺浇鏉冮噸 (濡傛灉鏈?
        if model_path and os.path.exists(model_path):
            print(f"[Model] Loading custom weights from {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            self.model.load_state_dict(state_dict, strict=False)
        else:
            print("[Model] Warning: No custom weights found. Using random head for demo purposes.")

        self.model = self.model.eval().to(device)

        # === [淇] 鐗瑰緛鎻愬彇 Hook ===
        # 鍘熸潵鐨?blocks[-1].pool 鏄?None锛屽鑷存姤閿欍€?
        # 鐜板湪鐨勭瓥鐣ワ細Hook 鍒板叏杩炴帴灞?(proj)锛屾埅鍙栧畠鐨?杈撳叆" (Input)
        self.current_embedding = None

        def hook_fn(module, input, output):
            # input 鏄竴涓?tuple (tensor, )
            # tensor shape: (Batch, 2304) -> 杩欏氨鏄垜浠鐨?Embedding
            self.current_embedding = input[0].detach().cpu().numpy()

        # 娉ㄥ唽 Hook 鍒?blocks[-1].proj (鍏ㄨ繛鎺ュ垎绫诲眰)
        self.model.blocks[-1].proj.register_forward_hook(hook_fn)

        # 4. 瀹氫箟棰勫鐞?Pipeline
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]

        self.transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(CLIP_DURATION),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=side_size),
                    CenterCropVideo(CROP_SIZE),
                ]
            ),
        )

    def infer(self, frame_buffer: List[np.ndarray]):
        """
        frame_buffer: list of BGR images (H, W, 3)
        """
        # 1. 杞崲鏍煎紡
        frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frame_buffer]
        tensor = torch.from_numpy(np.array(frames)).float()
        tensor = tensor.permute(3, 0, 1, 2)  # (C, T, H, W)

        # 2. 搴旂敤棰勫鐞?
        input_data = {"video": tensor}
        input_data = self.transform(input_data)
        video_data = input_data["video"]

        # 3. 鏋勯€?SlowFast 杈撳叆
        inputs = [i.to(self.device)[None, ...] for i in self.model_pack_pathways(video_data)]

        # 4. 鎺ㄧ悊 (Hook 浼氬湪杩欓噷鑷姩瑙﹀彂锛屽～鍏?self.current_embedding)
        with torch.no_grad():
            preds = self.model(inputs)
            probs = torch.nn.functional.softmax(preds, dim=1)

        # 5. 鑾峰彇缁撴灉
        top_val, top_idx = torch.max(probs, dim=1)
        idx = top_idx.item()
        conf = top_val.item()

        return idx, conf

    def model_pack_pathways(self, video_tensor):
        """
        鏋勫缓 SlowFast 鍙屾祦杈撳叆
        """
        fast_pathway = video_tensor
        slow_pathway = torch.index_select(
            video_tensor,
            1,
            torch.linspace(
                0, video_tensor.shape[1] - 1, video_tensor.shape[1] // 4
            ).long(),
        )
        return [slow_pathway, fast_pathway]


def crop_person(frame, bbox, padding=0.2):
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    w_box = x2 - x1
    h_box = y2 - y1
    cx, cy = x1 + w_box // 2, y1 + h_box // 2

    size = max(w_box, h_box) * (1 + padding)
    half_size = int(size / 2)

    x1_new = max(0, cx - half_size)
    y1_new = max(0, cy - half_size)
    x2_new = min(w_img, cx + half_size)
    y2_new = min(h_img, cy + half_size)

    crop = frame[y1_new:y2_new, x1_new:x2_new]

    if crop.size == 0:
        return cv2.resize(frame, (CROP_SIZE, CROP_SIZE))

    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE))


def load_tracks_by_frame(path):
    tracks = defaultdict(list)
    if not os.path.exists(path):
        return tracks
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                tracks[d["frame"]] = d["persons"]
            except:
                continue
    return tracks


def main():
    base_dir = Path(__file__).resolve().parents[1]

    global CLIP_DURATION, SAMPLING_RATE, INFERENCE_STRIDE, CROP_SIZE

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--pose", type=str, required=True, help="pose_tracks_smooth.jsonl")
    parser.add_argument("--out", type=str, required=True, help="actions.jsonl")
    parser.add_argument("--model_weight", type=str, default="", help="path to custom .pth")
    parser.add_argument("--emb_out", type=str, default="", help="output path for embeddings.pkl")
    parser.add_argument("--device", type=str, default="", help="device: 0/cuda/cpu; empty=auto")
    parser.add_argument("--stride", type=int, default=INFERENCE_STRIDE, help="frames between inferences")
    parser.add_argument("--clip_duration", type=int, default=CLIP_DURATION, help="frames per clip")
    parser.add_argument("--sampling_rate", type=int, default=SAMPLING_RATE, help="temporal sampling rate")
    parser.add_argument("--crop_size", type=int, default=CROP_SIZE, help="crop size for model input")
    parser.add_argument("--save_keyframes", type=int, default=0, help="1=save sampled keyframes for MLLM")
    parser.add_argument("--keyframe_dir", type=str, default="", help="keyframe directory (default: <out_dir>/keyframes)")

    args = parser.parse_args()

    CLIP_DURATION = int(args.clip_duration)
    SAMPLING_RATE = int(args.sampling_rate)
    INFERENCE_STRIDE = int(args.stride)
    CROP_SIZE = int(args.crop_size)

    video_path = Path(args.video)
    if not video_path.is_absolute(): video_path = base_dir / video_path

    pose_path = Path(args.pose)
    if not pose_path.is_absolute(): pose_path = base_dir / pose_path

    out_path = Path(args.out)
    if not out_path.is_absolute(): out_path = base_dir / out_path

    emb_out_path = Path(args.emb_out) if args.emb_out else out_path.parent / "embeddings.pkl"

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not pose_path.exists():
        raise FileNotFoundError(f"Pose tracks not found: {pose_path}")

    device = args.device if args.device else get_device()

    # 1. 鍔犺浇妯″瀷
    recognizer = ActionRecognizer(model_path=args.model_weight, device=device)

    # 2. 鍔犺浇 Track 鏁版嵁
    print(f"[Data] Loading tracks from {pose_path}...")
    frame_tracks = load_tracks_by_frame(pose_path)
    if not frame_tracks:
        print("[Error] No tracks loaded. Please check pose_tracks_smooth.jsonl.")
        return

    # 3. 鍑嗗瑙嗛璇诲彇
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Buffer
    person_buffers = defaultdict(lambda: deque(maxlen=CLIP_DURATION * SAMPLING_RATE))
    last_seen = {}

    actions_out = []
    track_embeddings = defaultdict(list)

    keyframe_dir = Path(args.keyframe_dir).resolve() if args.keyframe_dir else (out_path.parent / "keyframes")
    if int(args.save_keyframes) == 1:
        keyframe_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    print(f"[Info] Starting inference on {video_path.name}...")

    while True:
        ok, frame = cap.read()
        if not ok: break

        persons = frame_tracks.get(frame_idx, [])

        for p in persons:
            tid = p["track_id"]
            bbox = p["bbox"]
            last_seen[tid] = frame_idx

            crop = crop_person(frame, bbox)
            person_buffers[tid].append(crop)

            if len(person_buffers[tid]) >= CLIP_DURATION and frame_idx % INFERENCE_STRIDE == 0:
                # 閲囨牱
                raw_buffer = list(person_buffers[tid])
                indices = np.linspace(0, len(raw_buffer) - 1, CLIP_DURATION).astype(int)
                input_frames = [raw_buffer[i] for i in indices]

                # === 鎺ㄧ悊 ===
                cls_idx, conf = recognizer.infer(input_frames)

                # [鏂板] 鏀堕泦鐗瑰緛
                if recognizer.current_embedding is not None:
                    emb_vec = recognizer.current_embedding[0].astype(np.float32).reshape(-1).tolist()
                    frame_range = [int(frame_idx), int(frame_idx + INFERENCE_STRIDE)]
                    # Timestamp-indexed format expected by downstream CCA:
                    # {track_id: [([start_frame, end_frame], embedding_vector), ...]}
                    track_embeddings[tid].append((frame_range, emb_vec))

                label = CLASS_LABELS.get(cls_idx, "unknown")
                timestamp = frame_idx / fps
                duration = INFERENCE_STRIDE / fps
                if int(args.save_keyframes) == 1:
                    kf = keyframe_dir / f"tid{int(tid):03d}_f{int(frame_idx):06d}.jpg"
                    cv2.imwrite(str(kf), frame)

                actions_out.append({
                    "track_id": tid,
                    "action": label,
                    "action_code": cls_idx,
                    "conf": float(f"{conf:.4f}"),
                    "start_time": float(f"{timestamp:.4f}"),
                    "end_time": float(f"{timestamp + duration:.4f}"),
                    "start_frame": frame_idx,
                    "end_frame": frame_idx + INFERENCE_STRIDE,
                    "frame": frame_idx,
                    "t": timestamp
                })

                if len(actions_out) % 10 == 0:
                    print(f"\r[Process] Frame {frame_idx}/{total_frames} | Detected: ID {tid} -> {label} ({conf:.2f})",
                          end="")

        # 鍐呭瓨浼樺寲
        if frame_idx % 100 == 0:
            stale_ids = [tid for tid, last_f in last_seen.items() if frame_idx - last_f > 100]
            for tid in stale_ids:
                if tid in person_buffers: del person_buffers[tid]
                del last_seen[tid]

        frame_idx += 1

    cap.release()

    # 1. 淇濆瓨 Action JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for a in actions_out:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    # 2. 淇濆瓨 Embeddings
    emb_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(emb_out_path, "wb") as f:
        pickle.dump(dict(track_embeddings), f)

    print(f"\n[Done] Saved {len(actions_out)} actions to {out_path}")
    print(f"[Done] Saved embeddings to {emb_out_path} (IDs: {len(track_embeddings)})")
    if int(args.save_keyframes) == 1:
        print(f"[Done] Saved keyframes to {keyframe_dir}")


if __name__ == "__main__":
    main()
