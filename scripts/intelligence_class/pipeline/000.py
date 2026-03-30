# -*- coding: utf-8 -*-
"""
Scan multi-view classroom dataset and build an auditable index.

Compatibility:
- Keep legacy key `videos` for `02_batch_run.py`.
- Add key `assets` for full audit (videos + case labeled images).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore


_THIS = Path(__file__).resolve()
for _p in [_THIS] + list(_THIS.parents):
    if (_p / "data").exists() and (_p / "scripts").exists():
        sys.path.insert(0, str(_p))
        break

from scripts.intelligence_class._utils.pathing import find_project_root


CN_DATASET = "\u667a\u6167\u8bfe\u5802\u5b66\u751f\u884c\u4e3a\u6570\u636e\u96c6"
CN_FRONT = "\u6b63\u65b9\u89c6\u89d2"
CN_REAR = "\u540e\u65b9\u89c6\u89d2"
CN_TOP1 = "\u659c\u4e0a\u65b9\u89c6\u89d21"
CN_TOP2 = "\u659c\u4e0a\u65b9\u89c6\u89d22"
CN_TEACHER = "\u6559\u5e08\u89c6\u89d2"
CN_CASE = "\u6848\u4f8b"
CN_CASE_FRONT_ANN = "\u6848\u4f8b/\u6b63\u65b9\u89c6\u89d2\u6807\u6ce8\u5e27"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

INDEX_COLUMNS = [
    "asset_type",
    "case_id",
    "video_id",
    "video_path",
    "rel_video_path",
    "file_name",
    "stem",
    "view_name",
    "view_code",
    "duration_sec",
    "resolution",
    "has_annotation",
    "annotation_source",
    "is_labeled_train_asset",
    "can_use_for_pseudo_label",
    "can_use_for_eval_only",
    "notes",
]


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_project_paths() -> Tuple[Path, Path, Path]:
    script_path = Path(__file__).resolve()
    project_root = find_project_root(script_path)
    default_dataset_root = project_root / "data" / CN_DATASET
    default_output_dir = project_root / "output"
    return project_root, default_dataset_root, default_output_dir


def normalize_token(text: str) -> str:
    return "".join(str(text).strip().lower().split())


def detect_view(folder_name: str) -> Tuple[str, str]:
    """
    Return (view_code, canonical_view_name).
    """
    raw = str(folder_name).strip()
    n = normalize_token(raw)

    if normalize_token(CN_FRONT) in n:
        return "front", CN_FRONT
    if normalize_token(CN_REAR) in n:
        return "rear", CN_REAR
    if normalize_token(CN_TEACHER) in n:
        return "teacher", CN_TEACHER
    if normalize_token(CN_TOP1) in n:
        return "top1", CN_TOP1
    if normalize_token(CN_TOP2) in n:
        return "top2", CN_TOP2
    if normalize_token(CN_CASE) in n:
        return "case", CN_CASE
    return "unknown", raw


def parse_views_arg(views: Optional[str]) -> Optional[Set[str]]:
    if not views:
        return None
    tokens = [normalize_token(x) for x in str(views).split(",") if str(x).strip()]
    return set(tokens) if tokens else None


def is_view_selected(view_code: str, view_name: str, selected: Optional[Set[str]]) -> bool:
    if not selected:
        return True
    aliases = {
        normalize_token(view_code),
        normalize_token(view_name),
    }
    return any(a in selected for a in aliases)


def safe_relative(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(path.resolve())


def extract_case_id(stem: str) -> str:
    tokens = re.split(r"[_\-\s]+", stem)
    for tok in tokens:
        if re.fullmatch(r"\d+", tok):
            try:
                return str(int(tok))
            except Exception:
                return tok

    m = re.search(r"(\d+)", stem)
    if m:
        try:
            return str(int(m.group(1)))
        except Exception:
            return m.group(1)

    return stem


def probe_video_metadata(video_path: Path) -> Tuple[Optional[float], str]:
    if cv2 is None:
        return None, ""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, ""

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        duration_sec: Optional[float] = None
        if fps > 0 and frame_count >= 0:
            duration_sec = round(frame_count / fps, 3)

        resolution = f"{width}x{height}" if width > 0 and height > 0 else ""
        return duration_sec, resolution
    finally:
        cap.release()


def probe_image_resolution(image_path: Path) -> str:
    if Image is not None:
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                if w > 0 and h > 0:
                    return f"{w}x{h}"
        except Exception:
            pass

    if cv2 is not None:
        try:
            arr = cv2.imread(str(image_path))
            if arr is not None and arr.shape[0] > 0 and arr.shape[1] > 0:
                h, w = int(arr.shape[0]), int(arr.shape[1])
                return f"{w}x{h}"
        except Exception:
            pass

    return ""


def note_for_video(view_code: str) -> str:
    if view_code == "front":
        return "Front-view unlabeled video; candidate for pseudo-labeling."
    if view_code in {"rear", "top1", "top2", "teacher"}:
        return "Non-front unlabeled video; mark as eval-only for now."
    if view_code == "case":
        return "Video asset under case directory."
    return "Unclassified view video asset."


def build_video_entry(
    file_path: Path,
    project_root: Path,
    view_name: str,
    view_code: str,
    duration_sec: Optional[float],
    resolution: str,
) -> Dict[str, Any]:
    stem = file_path.stem
    case_id = extract_case_id(stem)
    video_id = f"{view_code}__{stem}"
    can_pseudo = view_code == "front"
    can_eval_only = view_code in {"rear", "top1", "top2", "teacher"}

    return {
        "asset_type": "video",
        "case_id": case_id,
        "video_path": str(file_path.resolve()),
        "file_name": file_path.name,
        "stem": stem,
        "view_name": view_name,
        "duration_sec": duration_sec,
        "resolution": resolution,
        "has_annotation": False,
        "annotation_source": "",
        "is_labeled_train_asset": False,
        "can_use_for_pseudo_label": can_pseudo,
        "can_use_for_eval_only": can_eval_only,
        "notes": note_for_video(view_code),
        # legacy compatibility keys
        "view_display": view_name,
        "view_code": view_code,
        "view": view_code,
        "rel_video_path": safe_relative(file_path, project_root),
        "video_id": video_id,
    }


def build_case_image_entry(
    image_path: Path,
    annotation_path: Optional[Path],
    project_root: Path,
) -> Dict[str, Any]:
    stem = image_path.stem
    case_id = extract_case_id(stem)
    has_ann = annotation_path is not None and annotation_path.exists()

    return {
        "asset_type": "labeled_image" if has_ann else "image",
        "case_id": case_id,
        # keep schema field name for compatibility with downstream generic readers
        "video_path": str(image_path.resolve()),
        "file_name": image_path.name,
        "stem": stem,
        "view_name": CN_CASE_FRONT_ANN,
        "duration_sec": None,
        "resolution": probe_image_resolution(image_path),
        "has_annotation": bool(has_ann),
        "annotation_source": str(annotation_path.resolve()) if has_ann else "",
        "is_labeled_train_asset": bool(has_ann),
        "can_use_for_pseudo_label": False,
        "can_use_for_eval_only": False,
        "notes": "Case labeled frame asset." if has_ann else "Case image without same-name JSON.",
        # common keys
        "view_display": CN_CASE,
        "view_code": "case",
        "view": "case",
        "rel_video_path": safe_relative(image_path, project_root),
        "video_id": f"case__{stem}",
    }


def iter_video_files(folder: Path, recursive: bool) -> Iterable[Path]:
    iterator = folder.rglob("*") if recursive else folder.glob("*")
    for p in iterator:
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            yield p


def find_case_dir(dataset_root: Path) -> Optional[Path]:
    exact = dataset_root / CN_CASE
    if exact.exists() and exact.is_dir():
        return exact

    for child in sorted(dataset_root.iterdir()):
        if child.is_dir() and normalize_token(CN_CASE) in normalize_token(child.name):
            return child
    return None


def scan_videos(
    dataset_root: Path,
    project_root: Path,
    selected_views: Optional[Set[str]],
    recursive: bool,
    limit: int,
    include_case_videos: bool,
    probe_metadata: bool,
) -> List[Dict[str, Any]]:
    candidates: List[Tuple[str, str, Path]] = []

    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue

        view_code, canonical_name = detect_view(child.name)
        if view_code == "unknown":
            print(f"[WARN] Unknown view directory skipped: {child}")
            continue

        if view_code == "case" and not include_case_videos:
            continue

        if not is_view_selected(view_code, canonical_name, selected_views):
            continue

        print(f"[SCAN] Video dir: {child.name} ({view_code})")
        files = sorted(iter_video_files(child, recursive), key=lambda x: (x.stem, x.name))
        for fp in files:
            candidates.append((view_code, canonical_name, fp))

    candidates.sort(key=lambda x: (x[0], x[2].stem, x[2].name))
    if limit > 0:
        candidates = candidates[:limit]

    entries: List[Dict[str, Any]] = []
    total = len(candidates)
    for i, (view_code, view_name, file_path) in enumerate(candidates, start=1):
        duration_sec: Optional[float] = None
        resolution = ""
        if probe_metadata:
            duration_sec, resolution = probe_video_metadata(file_path)

        entries.append(
            build_video_entry(
                file_path=file_path,
                project_root=project_root,
                view_name=view_name,
                view_code=view_code,
                duration_sec=duration_sec,
                resolution=resolution,
            )
        )

        if i == 1 or i % 100 == 0 or i == total:
            print(f"[INFO] Video index progress: {i}/{total}")

    return entries


def scan_case_assets(
    dataset_root: Path,
    project_root: Path,
    recursive: bool,
    case_limit: int,
) -> List[Dict[str, Any]]:
    case_dir = find_case_dir(dataset_root)
    if case_dir is None:
        print("[WARN] Case directory not found; skip case asset scan.")
        return []

    print(f"[SCAN] Case asset dir: {case_dir}")
    iterator = case_dir.rglob("*") if recursive else case_dir.glob("*")
    image_files = [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    image_files.sort(key=lambda p: (str(p.parent), p.stem))
    if case_limit > 0:
        image_files = image_files[:case_limit]

    assets: List[Dict[str, Any]] = []
    total = len(image_files)
    for i, image_path in enumerate(image_files, start=1):
        ann_path = image_path.with_suffix(".json")
        assets.append(
            build_case_image_entry(
                image_path=image_path,
                annotation_path=ann_path if ann_path.exists() else None,
                project_root=project_root,
            )
        )
        if i == 1 or i % 1000 == 0 or i == total:
            print(f"[INFO] Case asset index progress: {i}/{total}")

    return assets


def summarize(
    videos: Sequence[Dict[str, Any]],
    assets: Sequence[Dict[str, Any]],
    dataset_root: Path,
) -> Dict[str, Any]:
    view_counts: Dict[str, int] = defaultdict(int)
    view_duration: Dict[str, float] = defaultdict(float)
    view_unknown_duration: Dict[str, int] = defaultdict(int)
    view_name_map: Dict[str, str] = {}

    for v in videos:
        code = str(v.get("view_code", "unknown"))
        view_counts[code] += 1
        view_name_map[code] = str(v.get("view_name") or v.get("view_display") or code)

        dur = v.get("duration_sec")
        if isinstance(dur, (int, float)):
            view_duration[code] += float(dur)
        else:
            view_unknown_duration[code] += 1

    labeled_train_asset_count = sum(1 for a in assets if bool(a.get("is_labeled_train_asset")))
    pseudo_label_candidate_count = sum(1 for a in assets if bool(a.get("can_use_for_pseudo_label")))
    eval_only_asset_count = sum(1 for a in assets if bool(a.get("can_use_for_eval_only")))

    per_view: Dict[str, Dict[str, Any]] = {}
    for code in sorted(view_counts.keys()):
        per_view[code] = {
            "view_name": view_name_map.get(code, code),
            "video_count": int(view_counts[code]),
            "total_duration_sec": round(float(view_duration.get(code, 0.0)), 3),
            "unknown_duration_videos": int(view_unknown_duration.get(code, 0)),
        }

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root.resolve()),
        "total_video_assets": len(videos),
        "total_assets": len(assets),
        "video_view_counts": {k: int(v) for k, v in sorted(view_counts.items())},
        "video_view_total_duration_sec": {k: round(float(v), 3) for k, v in sorted(view_duration.items())},
        "labeled_train_asset_count": int(labeled_train_asset_count),
        "pseudo_label_candidate_count": int(pseudo_label_candidate_count),
        "eval_only_asset_count": int(eval_only_asset_count),
        "per_view": per_view,
    }


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            row: Dict[str, Any] = {}
            for k in fieldnames:
                v = r.get(k, "")
                if v is None:
                    row[k] = ""
                elif isinstance(v, bool):
                    row[k] = "1" if v else "0"
                else:
                    row[k] = v
            writer.writerow(row)


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    rows: List[Dict[str, Any]] = [
        {"metric": "total_video_assets", "view_code": "", "view_name": "", "value": summary["total_video_assets"]},
        {"metric": "total_assets", "view_code": "", "view_name": "", "value": summary["total_assets"]},
        {
            "metric": "labeled_train_asset_count",
            "view_code": "",
            "view_name": "",
            "value": summary["labeled_train_asset_count"],
        },
        {
            "metric": "pseudo_label_candidate_count",
            "view_code": "",
            "view_name": "",
            "value": summary["pseudo_label_candidate_count"],
        },
        {"metric": "eval_only_asset_count", "view_code": "", "view_name": "", "value": summary["eval_only_asset_count"]},
    ]

    per_view = summary.get("per_view", {}) or {}
    for view_code, info in sorted(per_view.items(), key=lambda kv: kv[0]):
        rows.append(
            {
                "metric": "video_count",
                "view_code": view_code,
                "view_name": info.get("view_name", ""),
                "value": info.get("video_count", 0),
            }
        )
        rows.append(
            {
                "metric": "total_duration_sec",
                "view_code": view_code,
                "view_name": info.get("view_name", ""),
                "value": info.get("total_duration_sec", 0),
            }
        )
        rows.append(
            {
                "metric": "unknown_duration_videos",
                "view_code": view_code,
                "view_name": info.get("view_name", ""),
                "value": info.get("unknown_duration_videos", 0),
            }
        )

    write_csv_rows(path, rows, fieldnames=["metric", "view_code", "view_name", "value"])


def print_preview(videos: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 68)
    print("Dataset Scan Summary")
    print("=" * 68)

    per_view = summary.get("per_view", {}) or {}
    for view_code, info in sorted(per_view.items(), key=lambda kv: kv[0]):
        print(
            f"{view_code:<8}  "
            f"count={int(info.get('video_count', 0)):>5}  "
            f"duration_sec={float(info.get('total_duration_sec', 0.0)):>10.3f}  "
            f"unknown_dur={int(info.get('unknown_duration_videos', 0)):>4}"
        )

    print("-" * 68)
    print(f"total_video_assets          : {summary.get('total_video_assets', 0)}")
    print(f"labeled_train_asset_count   : {summary.get('labeled_train_asset_count', 0)}")
    print(f"pseudo_label_candidate_count: {summary.get('pseudo_label_candidate_count', 0)}")
    print(f"eval_only_asset_count       : {summary.get('eval_only_asset_count', 0)}")
    print("=" * 68)

    print("[Preview] first 10 video ids:")
    for i, item in enumerate(videos[:10], start=1):
        print(f" {i:>2}. {item.get('video_id')} -> {item.get('rel_video_path')}")
    if len(videos) > 10:
        print(" ...")


def main() -> None:
    project_root, default_dataset_root, default_output_dir = resolve_project_paths()

    parser = argparse.ArgumentParser(description="Scan dataset and generate auditable index.")
    parser.add_argument("--dataset_root", type=str, default=str(default_dataset_root), help="Dataset root directory.")
    parser.add_argument("--output_dir", type=str, default=str(default_output_dir), help="Output directory.")
    parser.add_argument("--views", type=str, default=None, help="View filters by code/name, comma-separated.")
    parser.add_argument("--recursive", type=str2bool, nargs="?", const=True, default=True, help="Recursive scan.")
    parser.add_argument("--limit", type=int, default=0, help="Max number of video records. 0 means unlimited.")
    parser.add_argument(
        "--include_case",
        action="store_true",
        default=False,
        help="Legacy flag: include case directory when scanning videos.",
    )
    parser.add_argument(
        "--scan_case_assets",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to scan case labeled image assets.",
    )
    parser.add_argument("--case_limit", type=int, default=0, help="Max number of case image records. 0 means unlimited.")
    parser.add_argument(
        "--probe_metadata",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Try reading duration and resolution for videos.",
    )
    parser.add_argument(
        "--write_csv",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Write dataset_index.csv and dataset_index_summary.csv.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    selected_views = parse_views_arg(args.views)

    if not dataset_root.exists():
        print(f"[ERROR] dataset_root not found: {dataset_root}")
        sys.exit(1)

    print(f"[INFO] project_root: {project_root}")
    print(f"[INFO] dataset_root: {dataset_root}")
    print(f"[INFO] output_dir  : {output_dir}")

    videos = scan_videos(
        dataset_root=dataset_root,
        project_root=project_root,
        selected_views=selected_views,
        recursive=bool(args.recursive),
        limit=int(args.limit),
        include_case_videos=bool(args.include_case),
        probe_metadata=bool(args.probe_metadata),
    )

    case_assets: List[Dict[str, Any]] = []
    if bool(args.scan_case_assets):
        case_assets = scan_case_assets(
            dataset_root=dataset_root,
            project_root=project_root,
            recursive=bool(args.recursive),
            case_limit=int(args.case_limit),
        )

    assets: List[Dict[str, Any]] = list(videos) + list(case_assets)
    summary = summarize(videos=videos, assets=assets, dataset_root=dataset_root)
    print_preview(videos=videos, summary=summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_json_path = output_dir / "dataset_index.json"
    index_csv_path = output_dir / "dataset_index.csv"
    summary_json_path = output_dir / "dataset_index_summary.json"
    summary_csv_path = output_dir / "dataset_index_summary.csv"

    now_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    index_payload = {
        "meta": {
            "scan_time": now_str,
            "dataset_root": str(dataset_root),
            "total_videos": len(videos),
            "view_counts": summary.get("video_view_counts", {}),
            "index_version": "2.0",
        },
        # compatibility for 02_batch_run.py
        "videos": videos,
        # optional compatibility for scripts expecting `items`
        "items": videos,
        # full auditable index
        "assets": assets,
        "summary": summary,
    }

    index_json_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if bool(args.write_csv):
        write_csv_rows(index_csv_path, assets, fieldnames=INDEX_COLUMNS)
        write_summary_csv(summary_csv_path, summary)

    print("\n[SUCCESS] dataset index generated")
    print(f"[OUT] {index_json_path}")
    if bool(args.write_csv):
        print(f"[OUT] {index_csv_path}")
    print(f"[OUT] {summary_json_path}")
    if bool(args.write_csv):
        print(f"[OUT] {summary_csv_path}")


if __name__ == "__main__":
    main()
