from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _resolve_repo_root(anchor: Path) -> Path:
    for p in [anchor] + list(anchor.parents):
        if (p / "data").exists() and (p / "scripts").exists():
            return p
    raise RuntimeError(f"Cannot resolve repo root from: {anchor}")


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def build_command(args: argparse.Namespace, repo_root: Path) -> list[str]:
    variant = str(args.variant).lower()
    if variant not in {"s", "m"}:
        raise ValueError("--variant must be one of: s, m")

    model_default = "yolo11s.pt" if variant == "s" else "yolo11m.pt"
    model = args.model or model_default
    script = repo_root / "scripts" / "intelligence_class" / "training" / "03_train_case_yolo.py"
    if not script.exists():
        raise FileNotFoundError(f"Training script not found: {script}")

    cmd = [
        str(_resolve_path(repo_root, args.py)),
        str(script),
        "--data",
        str(args.data),
        "--model",
        str(model),
        "--epochs",
        str(int(args.epochs)),
        "--imgsz",
        str(int(args.imgsz)),
        "--batch",
        str(int(args.batch)),
        "--device",
        str(args.device),
        "--workers",
        str(int(args.workers)),
        "--project",
        str(args.project),
        "--name",
        str(args.name),
    ]
    return cmd


def expected_outputs(repo_root: Path, project: str, name: str) -> list[str]:
    proj = _resolve_path(repo_root, project)
    return [
        str((proj / name / "weights" / "best.pt").resolve()),
        str((proj / name / "results.csv").resolve()),
        str((proj / name / "confusion_matrix.png").resolve()),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Wrapper for wisdom-classroom train stage.")
    parser.add_argument("--variant", type=str, default="s")
    parser.add_argument("--data", type=str, default="data/processed/classroom_yolo/dataset.yaml")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="wisdom8_yolo11s_detect_v1")
    parser.add_argument("--py", type=str, default=sys.executable)
    parser.add_argument("--emit_only", type=int, default=1)
    parser.add_argument("--dry_run", type=int, default=0)
    parser.add_argument("--print_json", type=int, default=1)
    args = parser.parse_args()

    repo_root = _resolve_repo_root(Path(__file__).resolve())
    cmd = build_command(args, repo_root)
    outputs = expected_outputs(repo_root, args.project, args.name)

    payload = {
        "stage": f"train_wisdom_{args.variant}",
        "command": cmd,
        "outputs": outputs,
        "emit_only": int(args.emit_only),
        "dry_run": int(args.dry_run),
    }

    if int(args.print_json) == 1:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(" ".join(cmd))

    if int(args.emit_only) == 1 or int(args.dry_run) == 1:
        return

    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        raise RuntimeError(f"Train wrapper failed, exit={ret.returncode}")


if __name__ == "__main__":
    main()
