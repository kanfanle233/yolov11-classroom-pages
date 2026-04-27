import json
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO


def interpolate_occluded_keypoints(kpts, conf_thres=0.2):
    visible = [p for p in kpts if p["c"] is not None and p["c"] >= conf_thres]
    if not visible:
        return kpts
    mx = sum(p["x"] for p in visible) / len(visible)
    my = sum(p["y"] for p in visible) / len(visible)
    out = []
    for p in kpts:
        if p["c"] is None or p["c"] < conf_thres:
            out.append({"x": float(mx), "y": float(my), "c": p["c"]})
        else:
            out.append(p)
    return out


def main():
    base_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/videos/demo1.mp4", help="input video path (relative to project root)")
    parser.add_argument("--out", type=str, default="output/pose_keypoints_v2.jsonl", help="output jsonl path (relative to project root)")
    parser.add_argument("--model", type=str, default="yolo11x-pose.pt", help="model path (relative to project root)")
    parser.add_argument("--conf", type=float, default=0.45, help="person conf threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="inference image size")
    parser.add_argument("--device", type=str, default="", help="device: 0/cuda/cpu; empty=auto")
    parser.add_argument("--half", type=int, default=0, help="1=fp16 (cuda only), 0=fp32")
    parser.add_argument("--interpolate_occluded", type=int, default=0, help="1=interpolate low-conf keypoints")
    parser.add_argument("--occlusion_conf_thres", type=float, default=0.2, help="low-conf threshold for occlusion")
    args = parser.parse_args()

    video_path = (base_dir / args.video).resolve()
    jsonl_path = (base_dir / args.out).resolve()
    model_arg = args.model

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    print("[CHECK] video exists?", video_path.exists(), video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    model_path = Path(model_arg)
    if model_path.is_absolute():
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model_ref = str(model_path)
    else:
        local_model = (base_dir / model_arg).resolve()
        model_ref = str(local_model) if local_model.exists() else model_arg
    print("[CHECK] model ref:", model_ref)

    model = YOLO(model_ref)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0

    conf_thres = float(args.conf)
    device = args.device if args.device else None
    use_half = bool(int(args.half))

    with open(str(jsonl_path), "w", encoding="utf-8") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t = frame_idx / fps
            r = model.predict(
                frame,
                verbose=False,
                conf=conf_thres,
                imgsz=args.imgsz,
                device=device,
                half=use_half,
            )[0]

            persons = []
            if r.boxes is not None and r.keypoints is not None:
                boxes = r.boxes.xyxy.cpu().numpy()   # N,4
                scores = r.boxes.conf.cpu().numpy()  # N
                xy = r.keypoints.xy.cpu().numpy()    # N,K,2
                kc = r.keypoints.conf.cpu().numpy()  # N,K

                n = min(len(boxes), len(xy))
                for pid in range(n):
                    kpts = []
                    for k in range(xy.shape[1]):
                        kpts.append({
                            "x": float(xy[pid, k, 0]),
                            "y": float(xy[pid, k, 1]),
                            "c": float(kc[pid, k]) if kc is not None else None
                        })

                    if int(args.interpolate_occluded) == 1:
                        kpts = interpolate_occluded_keypoints(kpts, conf_thres=float(args.occlusion_conf_thres))
                    vis = [1 for p in kpts if p["c"] is not None and p["c"] >= float(args.occlusion_conf_thres)]
                    occlusion_score = 1.0 - (len(vis) / max(1, len(kpts)))

                    persons.append({
                        "person_idx": int(pid),
                        "conf": float(scores[pid]),
                        "bbox": [float(v) for v in boxes[pid].tolist()],  # [x1,y1,x2,y2]
                        "keypoints": kpts,
                        "occlusion_score": float(round(occlusion_score, 4)),
                    })

            f.write(json.dumps({"frame": frame_idx, "t": float(t), "persons": persons}, ensure_ascii=False) + "\n")

            frame_idx += 1
            if frame_idx % 300 == 0:
                print(f"[INFO] exported {frame_idx} frames")

    cap.release()
    print("[DONE] JSONL:", jsonl_path)


if __name__ == "__main__":
    main()
