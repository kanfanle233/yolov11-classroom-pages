#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

REQUIRED_JSON_FILES = [
    "timeline_viz.json",
    "timeline_chart.json",
    "student_projection.json",
    "transcript.jsonl",
    "pose_tracks_smooth.jsonl",
]
VIDEO_CANDIDATES = [
    "*_overlay.mp4",
    "pose_demo_out.mp4",
    "objects_demo_out.mp4",
]


def pick_video(case_dir: Path) -> Path | None:
    for pattern in VIDEO_CANDIDATES:
        matches = sorted(case_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_cases(demo_root: Path, views: List[str] | None) -> List[Path]:
    cases: List[Path] = []
    if not demo_root.exists():
        return cases

    view_dirs = [d for d in demo_root.iterdir() if d.is_dir()]
    if views:
        allowed = set(views)
        view_dirs = [d for d in view_dirs if d.name in allowed]

    for view_dir in view_dirs:
        for case_dir in sorted([d for d in view_dir.iterdir() if d.is_dir()]):
            if (case_dir / "timeline_viz.json").exists() or (case_dir / "timeline_chart.json").exists():
                cases.append(case_dir)
    return cases


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static demo package for GitHub Pages")
    parser.add_argument("--demo_root", required=True, help="Path to output/.../_demo_web")
    parser.add_argument("--docs_root", default="docs", help="Pages root folder")
    parser.add_argument("--views", default="", help="Comma-separated views to include")
    parser.add_argument("--case_ids", default="", help="Comma-separated video_id list to include (exact match)")
    parser.add_argument("--limit", type=int, default=10, help="Max cases to include")
    parser.add_argument("--clean", action="store_true", help="Clean docs/data and docs/assets/videos before copy")
    args = parser.parse_args()

    demo_root = Path(args.demo_root).resolve()
    docs_root = Path(args.docs_root).resolve()
    data_root = docs_root / "data"
    cases_root = data_root / "cases"
    videos_root = docs_root / "assets" / "videos"

    if args.clean:
        shutil.rmtree(cases_root, ignore_errors=True)
        shutil.rmtree(videos_root, ignore_errors=True)

    cases_root.mkdir(parents=True, exist_ok=True)
    videos_root.mkdir(parents=True, exist_ok=True)

    views = [v.strip() for v in args.views.split(",") if v.strip()] if args.views else None
    case_ids = [c.strip() for c in args.case_ids.split(",") if c.strip()] if args.case_ids else []

    case_dirs = find_cases(demo_root, views)
    if case_ids:
        allow = set(case_ids)
        case_dirs = [c for c in case_dirs if c.name in allow]
        # Keep user-specified order when possible
        rank = {cid: i for i, cid in enumerate(case_ids)}
        case_dirs = sorted(case_dirs, key=lambda c: rank.get(c.name, 10**9))
    if args.limit > 0:
        case_dirs = case_dirs[: args.limit]

    manifest_cases: List[Dict] = []

    for case_dir in case_dirs:
        video_id = case_dir.name
        view_name = case_dir.parent.name
        case_id = video_id.split("__")[-1]

        out_case_dir = cases_root / video_id
        out_case_dir.mkdir(parents=True, exist_ok=True)

        # copy json/jsonl assets
        copied: Dict[str, str] = {}
        for name in REQUIRED_JSON_FILES:
            src = case_dir / name
            dst = out_case_dir / name
            if copy_if_exists(src, dst):
                copied[name] = f"./data/cases/{video_id}/{name}"

        # timeline fallback alias
        if "timeline_viz.json" not in copied and "timeline_chart.json" in copied:
            src = out_case_dir / "timeline_chart.json"
            dst = out_case_dir / "timeline_viz.json"
            shutil.copy2(src, dst)
            copied["timeline_viz.json"] = f"./data/cases/{video_id}/timeline_viz.json"

        video_src = pick_video(case_dir)
        video_rel = None
        if video_src is not None:
            video_dst = videos_root / f"{video_id}.mp4"
            copy_if_exists(video_src, video_dst)
            video_rel = f"./assets/videos/{video_id}.mp4"

        manifest_cases.append(
            {
                "video_id": video_id,
                "view": view_name,
                "case_id": case_id,
                "paths": {
                    "timeline": copied.get("timeline_viz.json", None),
                    "projection": copied.get("student_projection.json", None),
                    "transcript": copied.get("transcript.jsonl", None),
                    "tracks": copied.get("pose_tracks_smooth.jsonl", None),
                    "stats": copied.get("timeline_chart_stats.json", None),
                    "video_original": video_rel,
                    "video_overlay": video_rel,
                    "video_pose": video_rel,
                    "video_objects": video_rel,
                },
            }
        )

    data_root.mkdir(parents=True, exist_ok=True)
    manifest = {"cases": manifest_cases}
    (data_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (data_root / "list_cases.json").write_text(json.dumps(manifest_cases, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] cases={len(manifest_cases)}")
    print(f"[DONE] manifest={data_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
