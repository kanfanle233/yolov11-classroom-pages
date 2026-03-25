import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def interpolate_occluded_keypoints(xy: np.ndarray, kc: np.ndarray, conf_th: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight interpolation for low-confidence keypoints.
    Use neighbor-index average when both neighbors are confident.
    """
    xy = xy.copy()
    kc = kc.copy()
    n = xy.shape[0]
    for i in range(n):
        c = float(kc[i])
        if c >= conf_th:
            continue
        left = i - 1
        right = i + 1
        if left >= 0 and right < n and kc[left] >= conf_th and kc[right] >= conf_th:
            xy[i] = 0.5 * (xy[left] + xy[right])
            kc[i] = 0.5 * float(kc[left] + kc[right]) * 0.6
    return xy, kc


def main():
    base_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/videos/demo1.mp4", help="input video path")
    parser.add_argument("--out", type=str, default="output/pose_keypoints_v2.jsonl", help="output jsonl path")
    parser.add_argument("--model", type=str, default="yolo11s-pose.pt", help="model path")
    parser.add_argument("--conf", type=float, default=0.45, help="person conf threshold")
    parser.add_argument(
        "--interpolate_occluded",
        action="store_true",
        help="enable keypoint interpolation for low-confidence points",
    )
    args = parser.parse_args()

    video_path = (base_dir / args.video).resolve()
    jsonl_path = (base_dir / args.out).resolve()
    model_path = (base_dir / args.model).resolve()

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    conf_thres = float(args.conf)

    with open(str(jsonl_path), "w", encoding="utf-8") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t = frame_idx / fps
            r = model.predict(frame, verbose=False, conf=conf_thres)[0]
            persons = []

            if r.boxes is not None and r.keypoints is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                xy = r.keypoints.xy.cpu().numpy()
                kc = r.keypoints.conf.cpu().numpy()

                n = min(len(boxes), len(xy))
                for pid in range(n):
                    p_xy = xy[pid]
                    p_kc = kc[pid] if kc is not None else np.ones((p_xy.shape[0],), dtype=np.float32)

                    if args.interpolate_occluded:
                        p_xy, p_kc = interpolate_occluded_keypoints(p_xy, p_kc, conf_th=0.25)

                    visible = float(np.mean((p_kc >= 0.25).astype(np.float32)))
                    occlusion_score = float(round(1.0 - visible, 4))

                    kpts = []
                    for k in range(p_xy.shape[0]):
                        kpts.append(
                            {
                                "x": float(p_xy[k, 0]),
                                "y": float(p_xy[k, 1]),
                                "c": float(p_kc[k]),
                            }
                        )

                    persons.append(
                        {
                            "person_idx": int(pid),
                            "conf": float(scores[pid]),
                            "bbox": [float(v) for v in boxes[pid].tolist()],
                            "keypoints": kpts,
                            "occlusion_score": occlusion_score,
                        }
                    )

            f.write(json.dumps({"frame": frame_idx, "t": float(t), "persons": persons}, ensure_ascii=False) + "\n")
            frame_idx += 1
            if frame_idx % 300 == 0:
                print(f"[INFO] exported {frame_idx} frames")

    cap.release()
    print("[DONE] JSONL:", jsonl_path)


if __name__ == "__main__":
    main()
