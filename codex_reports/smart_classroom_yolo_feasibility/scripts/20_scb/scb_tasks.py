from __future__ import annotations

import argparse
import json
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


def build_placeholder(task: str, repo_root: Path, args: argparse.Namespace) -> dict:
    if task == "clean":
        return {
            "task": "scb_clean",
            "command": [
                str(_resolve_path(repo_root, args.py)),
                "-c",
                "print('SCB clean placeholder stage. Implement cleaner as needed.')",
            ],
            "outputs": [str(_resolve_path(repo_root, args.clean_out))],
        }
    if task == "train":
        return {
            "task": "scb_train",
            "command": [
                str(_resolve_path(repo_root, args.py)),
                "-c",
                "print('SCB train placeholder stage. Implement train profile as needed.')",
            ],
            "outputs": [
                str(
                    (_resolve_path(repo_root, args.project) / args.name / "weights" / "best.pt").resolve()
                )
            ],
        }
    raise ValueError(f"Unknown task: {task}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SCB stage placeholders (reserved by design).")
    parser.add_argument("--task", choices=["clean", "train"], required=True)
    parser.add_argument("--clean_out", type=str, default="data/processed/scb_yolo_clean")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="scb_hrw_yolo11s_detect_v1")
    parser.add_argument("--py", type=str, default=sys.executable)
    parser.add_argument("--emit_only", type=int, default=1)
    parser.add_argument("--dry_run", type=int, default=0)
    parser.add_argument("--print_json", type=int, default=1)
    args = parser.parse_args()

    repo_root = _resolve_repo_root(Path(__file__).resolve())
    payload = build_placeholder(args.task, repo_root, args)
    payload["emit_only"] = int(args.emit_only)
    payload["dry_run"] = int(args.dry_run)

    if int(args.print_json) == 1:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(" ".join(payload["command"]))

    # Reserved stage: by default we only emit commands and keep no-op behavior.
    if int(args.emit_only) == 1 or int(args.dry_run) == 1:
        return

    # Intentionally no-op to keep this stage reserved unless explicitly implemented later.
    print(f"[INFO] Reserved SCB stage `{args.task}` completed as no-op.")


if __name__ == "__main__":
    main()
