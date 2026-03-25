# scripts/training/train_classroom_yolo.py
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def resolve_path(base: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def main():
    base_dir = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Train enhanced classroom YOLO model (ASPN + DySnakeConv + GLIDE hooks)")
    parser.add_argument("--data", type=str, default="data/custom_classroom_data/data.yaml", help="dataset yaml")
    parser.add_argument(
        "--cfg",
        type=str,
        default="models/yolov11_classroom/classroom_yolo_config.yaml",
        help="enhanced model yaml",
    )
    parser.add_argument("--pretrained", type=str, default="yolo11n-pose.pt", help="pretrained weights/model alias")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--freeze", type=int, default=10, help="freeze first N backbone layers")
    parser.add_argument("--project", type=str, default="runs/train_classroom_yolo")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument(
        "--out_weight",
        type=str,
        default="models/classroom_yolo_enhanced.pt",
        help="final exported checkpoint",
    )
    args = parser.parse_args()

    data_path = resolve_path(base_dir, args.data)
    cfg_path = resolve_path(base_dir, args.cfg)
    out_weight = resolve_path(base_dir, args.out_weight)
    project_dir = resolve_path(base_dir, args.project)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    if cfg_path.exists():
        model = YOLO(str(cfg_path))
        if args.pretrained:
            model = model.load(args.pretrained)
    else:
        print(f"[WARN] Enhanced cfg not found: {cfg_path}. Using pretrained model directly.")
        model = YOLO(args.pretrained)

    print("[INFO] Training classroom-enhanced YOLO model")
    print(f"       data      : {data_path}")
    print(f"       cfg       : {cfg_path if cfg_path.exists() else '(pretrained only)'}")
    print(f"       pretrained: {args.pretrained}")
    print(f"       freeze    : first {args.freeze} layers")

    results = model.train(
        data=str(data_path),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        workers=int(args.workers),
        freeze=int(args.freeze),
        project=str(project_dir),
        name=str(args.name),
        device=args.device if args.device else None,
    )

    best_path = None
    if hasattr(results, "save_dir"):
        best_candidate = Path(results.save_dir) / "weights" / "best.pt"
        if best_candidate.exists():
            best_path = best_candidate

    if best_path is None:
        # fallback to current model checkpoint path
        best_path = Path(model.ckpt_path) if getattr(model, "ckpt_path", None) else None

    if best_path is None or not best_path.exists():
        raise RuntimeError("Training finished but best checkpoint was not found.")

    out_weight.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, out_weight)
    print(f"[DONE] exported enhanced checkpoint: {out_weight}")


if __name__ == "__main__":
    main()
