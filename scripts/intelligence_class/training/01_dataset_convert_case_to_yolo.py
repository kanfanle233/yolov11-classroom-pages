# -*- coding: utf-8 -*-
"""
Convert classroom jpg+json annotations into a YOLO detection dataset.

Key guarantees:
1) Stable JSON -> YOLO conversion.
2) Video-level split (never random by image).
3) Data cleaning + audit artifacts.

Input JSON format:
  {"labels":[{"name":"tt","x1":232,"y1":121,"x2":260,"y2":158}, ...]}

Output structure (default):
  data/processed/classroom_yolo/
    images/train/
    images/val/
    labels/train/
    labels/val/
    meta/
    dataset.yaml
    split_manifest.csv
    class_stats.json
    bad_samples.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image, UnidentifiedImageError  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    UnidentifiedImageError = Exception  # type: ignore


_this = Path(__file__).resolve()
for p in [_this] + list(_this.parents):
    if (p / "data").exists() and (p / "scripts").exists():
        sys.path.insert(0, str(p))
        break

from scripts.intelligence_class._utils.pathing import find_project_root

try:
    from scripts.intelligence_class._utils.pathing import resolve_under_project  # type: ignore
except Exception:  # pragma: no cover
    resolve_under_project = None  # type: ignore


# Fixed class order (must stay consistent for dataset.yaml and labels).
CLASS_NAMES: List[str] = ["tt", "dx", "dk", "zt", "xt", "js", "zl", "jz"]
CLASS_MAP: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRAME_TOKEN_RE = re.compile(r"^(?:f|frame)?\d{3,}$", re.IGNORECASE)
TRAILING_FRAME_SUFFIX_RE = re.compile(r"(?:[_\-](?:f|frame)?\d{1,8})+$", re.IGNORECASE)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    return find_project_root(Path(__file__).resolve())


def _resolve_path(project_root: Path, value: str) -> Path:
    if resolve_under_project is not None:
        return resolve_under_project(project_root, value)
    p = Path(value)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def setup_logger(log_file: Path, level: str = "INFO") -> logging.Logger:
    ensure_dir(log_file.parent)
    logger = logging.getLogger("convert_case_to_yolo")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _tokenize_stem(stem: str) -> List[str]:
    return [x for x in re.split(r"[_\-]+", stem) if x]


def _parse_video_id_by_prefix(stem: str) -> Optional[str]:
    """
    Priority-1: parse from filename prefix.
    """
    if "__" in stem:
        left, right = stem.split("__", 1)
        right_tokens = _tokenize_stem(right)
        kept: List[str] = []
        for tok in right_tokens:
            if FRAME_TOKEN_RE.match(tok):
                break
            kept.append(tok)
        if kept:
            return f"{left}__{'_'.join(kept)}"
        if left:
            return left

    tokens = _tokenize_stem(stem)
    kept2: List[str] = []
    for tok in tokens:
        if FRAME_TOKEN_RE.match(tok):
            break
        kept2.append(tok)

    if len(kept2) >= 2:
        return "_".join(kept2)

    # keep single non-numeric prefix only when a frame-like suffix exists
    if len(kept2) == 1 and not kept2[0].isdigit() and TRAILING_FRAME_SUFFIX_RE.search(stem):
        return kept2[0]
    return None


def infer_source_video_id(stem: str) -> str:
    """
    Robust source_video_id inference:
      1) parse by prefix
      2) fallback: remove trailing frame index pattern
      3) fallback: strip trailing long numeric suffix, else full stem
    """
    s = stem.strip()
    if not s:
        return "unknown_video"

    v1 = _parse_video_id_by_prefix(s)
    if v1:
        return v1

    v2 = TRAILING_FRAME_SUFFIX_RE.sub("", s)
    if v2 and v2 != s:
        return v2

    v3 = re.sub(r"(?:[_\-]?\d{3,})$", "", s)
    if v3:
        return v3
    return s


def verify_image_and_get_size(img_path: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    if Image is None:
        return None, None, "PIL not available"
    try:
        with Image.open(img_path) as im:
            im.verify()
        with Image.open(img_path) as im2:
            w, h = im2.size
        if int(w) <= 0 or int(h) <= 0:
            return None, None, f"invalid image size ({w}, {h})"
        return int(w), int(h), None
    except UnidentifiedImageError as e:
        return None, None, f"unidentified image: {e}"
    except Exception as e:
        return None, None, f"image read failed: {e}"


def load_json_labels(json_path: Path) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        return None, f"json parse failed: {e}"

    labels = data.get("labels", [])
    if not isinstance(labels, list):
        return None, "json 'labels' is not a list"
    out: List[Dict[str, Any]] = []
    for item in labels:
        if isinstance(item, dict):
            out.append(item)
    return out, None


def clip_and_normalize_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> Optional[Tuple[float, float, float, float]]:
    if width <= 0 or height <= 0:
        return None

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1c = max(0.0, min(float(width - 1), x1))
    y1c = max(0.0, min(float(height - 1), y1))
    x2c = max(0.0, min(float(width - 1), x2))
    y2c = max(0.0, min(float(height - 1), y2))

    bw = x2c - x1c
    bh = y2c - y1c
    if bw <= 0.0 or bh <= 0.0:
        return None

    cx = (x1c + x2c) * 0.5 / float(width)
    cy = (y1c + y2c) * 0.5 / float(height)
    nw = bw / float(width)
    nh = bh / float(height)
    if nw <= 0.0 or nh <= 0.0:
        return None
    return cx, cy, nw, nh


def _is_frame_like_token(token: str) -> bool:
    return bool(FRAME_TOKEN_RE.match(token))


def assign_video_splits(
    video_ids: Sequence[str],
    train_ratio: float,
    random_seed: int,
) -> Dict[str, str]:
    ids = sorted(set(video_ids))
    rng = random.Random(random_seed)
    rng.shuffle(ids)

    n = len(ids)
    if n == 0:
        return {}
    if n == 1:
        return {ids[0]: "train"}

    n_train = int(round(n * train_ratio))
    n_train = max(1, min(n - 1, n_train))

    train_ids = set(ids[:n_train])
    split_map: Dict[str, str] = {}
    for vid in ids:
        split_map[vid] = "train" if vid in train_ids else "val"
    return split_map


def write_dataset_yaml(dataset_root: Path, yaml_path: Path) -> None:
    lines = [
        f"path: {dataset_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for i, name in enumerate(CLASS_NAMES):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_component(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", text.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "sample"


def make_unique_basename(
    source_video_id: str,
    stem: str,
    used_names: set[str],
) -> str:
    base = f"{_safe_component(source_video_id)}__{_safe_component(stem)}"
    if base not in used_names:
        used_names.add(base)
        return base
    idx = 1
    while True:
        cand = f"{base}__{idx}"
        if cand not in used_names:
            used_names.add(cand)
            return cand
        idx += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert classroom jpg+json to YOLO dataset (video-level split).")
    parser.add_argument(
        "--in_dir",
        type=str,
        default="data/智慧课堂学生行为数据集/案例/智慧课堂学生行为数据集案例（正方视角）",
        help="input dir with image + same-name json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/classroom_yolo",
        help="output YOLO dataset root",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8, help="video-level train ratio (val=1-train)")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed for video split")
    parser.add_argument("--overwrite", type=int, default=0, help="1 to remove out_dir before running")
    parser.add_argument("--log_level", type=str, default="INFO", help="logging level")
    args = parser.parse_args()

    if not (0.0 < float(args.train_ratio) < 1.0):
        raise ValueError(f"--train_ratio must be in (0,1), got {args.train_ratio}")

    project_root = get_project_root()
    in_dir = _resolve_path(project_root, args.in_dir)
    out_dir = _resolve_path(project_root, args.out_dir)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {in_dir}")

    if out_dir.exists() and int(args.overwrite) == 1:
        shutil.rmtree(out_dir)

    ensure_dir(out_dir / "images" / "train")
    ensure_dir(out_dir / "images" / "val")
    ensure_dir(out_dir / "labels" / "train")
    ensure_dir(out_dir / "labels" / "val")
    ensure_dir(out_dir / "meta")

    logger = setup_logger(out_dir / "meta" / "convert.log", level=args.log_level)
    logger.info("Start conversion")
    logger.info("Input: %s", in_dir)
    logger.info("Output: %s", out_dir)
    logger.info("Class map: %s", CLASS_MAP)

    image_files = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS], key=lambda x: x.name)
    if not image_files:
        raise RuntimeError(f"No images found under: {in_dir}")

    bad_samples: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []

    class_total = {k: 0 for k in CLASS_NAMES}
    class_split = {
        "train": {k: 0 for k in CLASS_NAMES},
        "val": {k: 0 for k in CLASS_NAMES},
    }

    stats: Dict[str, int] = {
        "images_seen": 0,
        "images_kept": 0,
        "missing_json_images": 0,
        "corrupted_images": 0,
        "invalid_json_files": 0,
        "invalid_boxes": 0,
        "unknown_class_boxes": 0,
        "empty_label_images": 0,
        "total_boxes_kept": 0,
    }

    for img_path in image_files:
        stats["images_seen"] += 1
        stem = img_path.stem
        source_video_id = infer_source_video_id(stem)
        json_path = in_dir / f"{stem}.json"

        if not json_path.exists():
            stats["missing_json_images"] += 1
            rec = {
                "image": str(img_path),
                "json": str(json_path),
                "source_video_id": source_video_id,
                "reason": "missing_json",
            }
            bad_samples.append(rec)
            logger.warning("Missing json: %s", img_path.name)
            continue

        width, height, img_err = verify_image_and_get_size(img_path)
        if img_err is not None or width is None or height is None:
            stats["corrupted_images"] += 1
            rec = {
                "image": str(img_path),
                "json": str(json_path),
                "source_video_id": source_video_id,
                "reason": "corrupted_image",
                "detail": img_err,
            }
            bad_samples.append(rec)
            logger.warning("Corrupted image skipped: %s (%s)", img_path.name, img_err)
            continue

        labels, json_err = load_json_labels(json_path)
        if json_err is not None or labels is None:
            stats["invalid_json_files"] += 1
            rec = {
                "image": str(img_path),
                "json": str(json_path),
                "source_video_id": source_video_id,
                "reason": "invalid_json",
                "detail": json_err,
            }
            bad_samples.append(rec)
            logger.warning("Invalid json skipped: %s (%s)", json_path.name, json_err)
            continue

        yolo_lines: List[str] = []
        for idx, obj in enumerate(labels):
            name = str(obj.get("name", "")).strip()
            if name not in CLASS_MAP:
                stats["invalid_boxes"] += 1
                stats["unknown_class_boxes"] += 1
                bad_samples.append(
                    {
                        "image": str(img_path),
                        "json": str(json_path),
                        "source_video_id": source_video_id,
                        "reason": "unknown_class",
                        "label_index": idx,
                        "name": name,
                    }
                )
                continue

            try:
                x1 = float(obj.get("x1"))
                y1 = float(obj.get("y1"))
                x2 = float(obj.get("x2"))
                y2 = float(obj.get("y2"))
            except Exception:
                stats["invalid_boxes"] += 1
                bad_samples.append(
                    {
                        "image": str(img_path),
                        "json": str(json_path),
                        "source_video_id": source_video_id,
                        "reason": "invalid_bbox_fields",
                        "label_index": idx,
                        "label": obj,
                    }
                )
                continue

            norm = clip_and_normalize_bbox(x1, y1, x2, y2, width, height)
            if norm is None:
                stats["invalid_boxes"] += 1
                bad_samples.append(
                    {
                        "image": str(img_path),
                        "json": str(json_path),
                        "source_video_id": source_video_id,
                        "reason": "invalid_bbox_after_clip",
                        "label_index": idx,
                        "label": {"name": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }
                )
                continue

            cls_id = CLASS_MAP[name]
            cx, cy, bw, bh = norm
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            class_total[name] += 1
            stats["total_boxes_kept"] += 1

        empty_label = len(yolo_lines) == 0
        if empty_label:
            stats["empty_label_images"] += 1

        samples.append(
            {
                "img_path": img_path,
                "json_path": json_path,
                "stem": stem,
                "source_video_id": source_video_id,
                "width": width,
                "height": height,
                "yolo_lines": yolo_lines,
                "num_boxes": len(yolo_lines),
                "empty_label": empty_label,
            }
        )
        stats["images_kept"] += 1

    if not samples:
        raise RuntimeError("No valid samples after cleaning.")

    video_ids = [s["source_video_id"] for s in samples]
    split_map = assign_video_splits(video_ids, train_ratio=float(args.train_ratio), random_seed=int(args.random_seed))
    train_videos = sum(1 for _, sp in split_map.items() if sp == "train")
    val_videos = sum(1 for _, sp in split_map.items() if sp == "val")
    logger.info("Video split: train=%d, val=%d (total=%d)", train_videos, val_videos, len(split_map))

    used_names: set[str] = set()
    manifest_rows: List[Dict[str, Any]] = []

    for sample in samples:
        split = split_map[sample["source_video_id"]]
        suffix = sample["img_path"].suffix.lower()
        if suffix not in IMG_EXTS:
            suffix = ".jpg"

        base = make_unique_basename(sample["source_video_id"], sample["stem"], used_names)
        img_rel = Path("images") / split / f"{base}{suffix}"
        lbl_rel = Path("labels") / split / f"{base}.txt"
        img_out = out_dir / img_rel
        lbl_out = out_dir / lbl_rel

        ensure_dir(img_out.parent)
        ensure_dir(lbl_out.parent)
        shutil.copy2(sample["img_path"], img_out)
        lbl_out.write_text("\n".join(sample["yolo_lines"]) + ("\n" if sample["yolo_lines"] else ""), encoding="utf-8")

        for line in sample["yolo_lines"]:
            try:
                cid = int(line.split()[0])
                if 0 <= cid < len(CLASS_NAMES):
                    cname = CLASS_NAMES[cid]
                    class_split[split][cname] += 1
            except Exception:
                pass

        manifest_rows.append(
            {
                "image_path": img_rel.as_posix(),
                "label_path": lbl_rel.as_posix(),
                "source_video_id": sample["source_video_id"],
                "split": split,
                "width": sample["width"],
                "height": sample["height"],
                "num_boxes": sample["num_boxes"],
                "empty_label": str(bool(sample["empty_label"])).lower(),
            }
        )

    # Write manifest
    manifest_path = out_dir / "split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "label_path",
                "source_video_id",
                "split",
                "width",
                "height",
                "num_boxes",
                "empty_label",
            ],
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    # Write bad samples
    bad_samples_path = out_dir / "bad_samples.json"
    bad_payload = {
        "total_bad_records": len(bad_samples),
        "bad_samples": bad_samples,
    }
    bad_samples_path.write_text(json.dumps(bad_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write class stats
    class_stats = {
        "class_order": CLASS_NAMES,
        "class_box_count_total": class_total,
        "class_box_count_by_split": class_split,
        "empty_label_images": stats["empty_label_images"],
        "corrupted_images": stats["corrupted_images"],
        "missing_json_images": stats["missing_json_images"],
        "invalid_json_files": stats["invalid_json_files"],
        "invalid_boxes": stats["invalid_boxes"],
        "unknown_class_boxes": stats["unknown_class_boxes"],
        "images_seen": stats["images_seen"],
        "images_kept": stats["images_kept"],
        "total_boxes_kept": stats["total_boxes_kept"],
        "source_videos_total": len(split_map),
        "source_videos_train": train_videos,
        "source_videos_val": val_videos,
    }
    class_stats_path = out_dir / "class_stats.json"
    class_stats_path.write_text(json.dumps(class_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # Write dataset.yaml
    dataset_yaml_path = out_dir / "dataset.yaml"
    write_dataset_yaml(out_dir, dataset_yaml_path)

    # Write run meta
    run_meta = {
        "project_root": str(project_root),
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "train_ratio": float(args.train_ratio),
        "val_ratio": 1.0 - float(args.train_ratio),
        "random_seed": int(args.random_seed),
        "class_map": CLASS_MAP,
        "manifest_file": manifest_path.name,
        "class_stats_file": class_stats_path.name,
        "bad_samples_file": bad_samples_path.name,
    }
    (out_dir / "meta" / "run_config.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Done.")
    logger.info("dataset.yaml: %s", dataset_yaml_path)
    logger.info("split_manifest.csv: %s", manifest_path)
    logger.info("class_stats.json: %s", class_stats_path)
    logger.info("bad_samples.json: %s", bad_samples_path)
    logger.info(
        "Summary: seen=%d kept=%d boxes=%d empty_labels=%d bad_records=%d",
        stats["images_seen"],
        stats["images_kept"],
        stats["total_boxes_kept"],
        stats["empty_label_images"],
        len(bad_samples),
    )


if __name__ == "__main__":
    main()

