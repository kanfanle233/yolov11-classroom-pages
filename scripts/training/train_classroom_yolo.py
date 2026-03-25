import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.yolov11_classroom.glide_loss import GLIDELoss


def main():
    parser = argparse.ArgumentParser(description="Train enhanced classroom YOLO model")
    parser.add_argument("--data", type=str, default="data/custom_classroom_data/data.yaml", help="dataset yaml path")
    parser.add_argument(
        "--cfg",
        type=str,
        default="models/yolov11_classroom/classroom_yolo_config.yaml",
        help="enhanced YOLO config",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="classroom_yolo_enhanced")
    parser.add_argument("--pretrained", type=str, default="yolo11s-pose.pt")
    parser.add_argument("--freeze", type=int, default=10, help="freeze first N layers")
    parser.add_argument("--use_glide_loss", type=int, default=1, help="1=enable GLIDE loss placeholder")
    parser.add_argument("--out_model", type=str, default="models/classroom_yolo_enhanced.pt")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_yaml = (root / args.data).resolve()
    cfg_yaml = (root / args.cfg).resolve()
    pretrained = (root / args.pretrained).resolve()
    out_model = (root / args.out_model).resolve()

    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset yaml not found: {data_yaml}")
    if not cfg_yaml.exists():
        raise FileNotFoundError(f"cfg not found: {cfg_yaml}")

    # Instantiate GLIDE loss so the script has a concrete integration point
    # for future trainer extension. Ultralytics default trainer cannot directly
    # swap keypoint criterion without custom trainer subclassing.
    glide = GLIDELoss()
    if int(args.use_glide_loss) == 1:
        dummy_pred = torch.zeros((1, 17, 2))
        dummy_gt = torch.zeros((1, 17, 2))
        _ = glide(dummy_pred, dummy_gt)
        print("[INFO] GLIDE loss hook initialized (placeholder warmup).")

    model = YOLO(str(cfg_yaml))
    if pretrained.exists():
        model.load(str(pretrained))
        print(f"[INFO] Loaded pretrained weights: {pretrained}")

    print("[INFO] Starting training...")
    model.train(
        data=str(data_yaml),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=args.device,
        workers=int(args.workers),
        project=args.project,
        name=args.name,
        freeze=int(args.freeze),
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.exists():
        out_model.parent.mkdir(parents=True, exist_ok=True)
        out_model.write_bytes(best.read_bytes())
        print(f"[DONE] Exported enhanced model to: {out_model}")
    else:
        print(f"[WARN] best.pt not found at {best}. Check training logs.")


if __name__ == "__main__":
    main()
