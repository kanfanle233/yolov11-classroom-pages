# -*- coding: utf-8 -*-
"""
Targeted augmentation for YOLO train split only.

Input dataset (standard YOLO):
  data/processed/classroom_yolo/
    images/train, images/val
    labels/train, labels/val
    dataset.yaml

Output dataset:
  data/processed/classroom_yolo_aug/
    images/train, images/val
    labels/train, labels/val
    meta/augment_manifest.csv
    meta/augment_stats.json
    dataset.yaml

Constraints:
  - Only train split is augmented.
  - Val split is copied as-is.
  - Original train samples are always preserved.
  - Bounding boxes are transformed synchronously.
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
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

try:
    import albumentations as A  # type: ignore

    HAS_ALBUMENTATIONS = True
except Exception:  # pragma: no cover
    A = None  # type: ignore
    HAS_ALBUMENTATIONS = False


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


# Must match converter order exactly.
CLASS_NAMES: List[str] = ["tt", "dx", "dk", "zt", "xt", "js", "zl", "jz"]
CLASS_MAP: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
CLASS_ID_TO_NAME: Dict[int, str] = {i: n for n, i in CLASS_MAP.items()}
HIGH_FREQ_CLASSES: Set[str] = {"tt", "dx", "dk"}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OP_POOL = ["flip", "bri", "rot", "aff"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    return find_project_root(Path(__file__).resolve())


def _resolve_path(project_root: Path, value: str) -> Path:
    if resolve_under_project is not None:
        return resolve_under_project(project_root, value)
    p = Path(value)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def setup_logger(log_path: Path, level: str = "INFO") -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("augment_yolo_labels")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def read_yolo_label(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    labels: List[Tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return labels
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h = [float(x) for x in parts[1:]]
                if not (0 <= cls < len(CLASS_NAMES)):
                    continue
                if w <= 0 or h <= 0:
                    continue
                labels.append((cls, cx, cy, w, h))
            except Exception:
                continue
    return labels


def write_yolo_label(label_path: Path, labels: Sequence[Tuple[int, float, float, float, float]]) -> None:
    ensure_dir(label_path.parent)
    with label_path.open("w", encoding="utf-8") as f:
        for cls, cx, cy, w, h in labels:
            f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def yolo_to_xyxy(box: Tuple[int, float, float, float, float], width: int, height: int) -> Dict[str, Any]:
    cls, cx, cy, bw, bh = box
    x1 = (cx - bw * 0.5) * width
    y1 = (cy - bh * 0.5) * height
    x2 = (cx + bw * 0.5) * width
    y2 = (cy + bh * 0.5) * height
    return {"cls": cls, "box": [float(x1), float(y1), float(x2), float(y2)]}


def clip_xyxy(box: Sequence[float], width: int, height: int) -> Optional[List[float]]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = max(0.0, min(float(width - 1), x1))
    x2 = max(0.0, min(float(width - 1), x2))
    y1 = max(0.0, min(float(height - 1), y1))
    y2 = max(0.0, min(float(height - 1), y2))

    if (x2 - x1) <= 1.0 or (y2 - y1) <= 1.0:
        return None
    return [x1, y1, x2, y2]


def xyxy_to_yolo(cls: int, box: Sequence[float], width: int, height: int) -> Optional[Tuple[int, float, float, float, float]]:
    clipped = clip_xyxy(box, width, height)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    cx = ((x1 + x2) * 0.5) / width
    cy = ((y1 + y2) * 0.5) / height
    if bw <= 0 or bh <= 0:
        return None
    return cls, cx, cy, bw, bh


def transform_records_affine(
    records: List[Dict[str, Any]],
    matrix: np.ndarray,
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        x1, y1, x2, y2 = rec["box"]
        pts = np.array(
            [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
            dtype=np.float32,
        ).T  # (3,4)
        transformed = matrix @ pts  # (2,4)
        xs = transformed[0, :]
        ys = transformed[1, :]
        new_box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
        clipped = clip_xyxy(new_box, width, height)
        if clipped is None:
            continue
        out.append({"cls": rec["cls"], "box": clipped})
    return out


def apply_manual_augmentation(
    image: np.ndarray,
    labels: List[Tuple[int, float, float, float, float]],
    ops: Sequence[str],
    rng: random.Random,
) -> Tuple[Optional[np.ndarray], Optional[List[Tuple[int, float, float, float, float]]], Optional[str]]:
    h, w = image.shape[:2]
    recs = [yolo_to_xyxy(lb, w, h) for lb in labels]
    img_aug = image.copy()

    for op in ops:
        if op == "flip":
            img_aug = cv2.flip(img_aug, 1)
            for rec in recs:
                x1, y1, x2, y2 = rec["box"]
                rec["box"] = [float(w - x2), y1, float(w - x1), y2]

        elif op == "bri":
            alpha = rng.uniform(0.85, 1.20)
            beta = rng.uniform(-25.0, 25.0)
            img_aug = cv2.convertScaleAbs(img_aug, alpha=alpha, beta=beta)

        elif op == "rot":
            angle = rng.uniform(-6.0, 6.0)
            M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
            img_aug = cv2.warpAffine(
                img_aug,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            recs = transform_records_affine(recs, M, w, h)

        elif op == "aff":
            scale = rng.uniform(0.95, 1.05)
            tx = rng.uniform(-0.03 * w, 0.03 * w)
            ty = rng.uniform(-0.03 * h, 0.03 * h)
            M = np.array(
                [
                    [scale, 0.0, (1.0 - scale) * 0.5 * w + tx],
                    [0.0, scale, (1.0 - scale) * 0.5 * h + ty],
                ],
                dtype=np.float32,
            )
            img_aug = cv2.warpAffine(
                img_aug,
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            recs = transform_records_affine(recs, M, w, h)

    out_labels: List[Tuple[int, float, float, float, float]] = []
    for rec in recs:
        converted = xyxy_to_yolo(rec["cls"], rec["box"], w, h)
        if converted is not None:
            out_labels.append(converted)

    if labels and not out_labels:
        return None, None, "all_boxes_dropped_after_manual_aug"
    return img_aug, out_labels, None


def apply_albumentations_augmentation(
    image: np.ndarray,
    labels: List[Tuple[int, float, float, float, float]],
    ops: Sequence[str],
    rng: random.Random,
) -> Tuple[Optional[np.ndarray], Optional[List[Tuple[int, float, float, float, float]]], Optional[str]]:
    if not HAS_ALBUMENTATIONS:
        return None, None, "albumentations_not_available"

    tfms = []
    if "flip" in ops:
        tfms.append(A.HorizontalFlip(p=1.0))
    if "bri" in ops:
        tfms.append(A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0))
    if "rot" in ops:
        tfms.append(
            A.Affine(
                scale=(1.0, 1.0),
                translate_percent={"x": (0.0, 0.0), "y": (0.0, 0.0)},
                rotate=(-6, 6),
                shear=0,
                p=1.0,
                mode=cv2.BORDER_REFLECT_101,
            )
        )
    if "aff" in ops:
        tfms.append(
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                rotate=(-3, 3),
                shear=0,
                p=1.0,
                mode=cv2.BORDER_REFLECT_101,
            )
        )

    if not tfms:
        return image.copy(), list(labels), None

    compose = A.Compose(
        tfms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.1,
            check_each_transform=True,
        ),
    )

    bboxes = [(cx, cy, w, h) for _, cx, cy, w, h in labels]
    cls_ids = [int(cls) for cls, *_ in labels]

    # make albumentations deterministic per call via seeded random state usage
    random.seed(rng.randint(0, 10**9))
    np.random.seed(rng.randint(0, 10**9))

    try:
        result = compose(image=image, bboxes=bboxes, class_labels=cls_ids)
    except Exception as e:
        return None, None, f"albumentations_failed: {e}"

    out_boxes = result.get("bboxes", [])
    out_cls = result.get("class_labels", [])
    out_labels: List[Tuple[int, float, float, float, float]] = []
    for cls, bb in zip(out_cls, out_boxes):
        try:
            cx, cy, bw, bh = [float(v) for v in bb]
            if bw <= 0 or bh <= 0:
                continue
            out_labels.append((int(cls), cx, cy, bw, bh))
        except Exception:
            continue

    if labels and not out_labels:
        return None, None, "all_boxes_dropped_after_albumentations"
    return result["image"], out_labels, None


def apply_augmentation(
    image: np.ndarray,
    labels: List[Tuple[int, float, float, float, float]],
    ops: Sequence[str],
    rng: random.Random,
) -> Tuple[Optional[np.ndarray], Optional[List[Tuple[int, float, float, float, float]]], Optional[str], str]:
    if HAS_ALBUMENTATIONS:
        img2, labels2, err = apply_albumentations_augmentation(image, labels, ops, rng)
        if err is None:
            return img2, labels2, None, "albumentations"

    img3, labels3, err2 = apply_manual_augmentation(image, labels, ops, rng)
    if err2 is not None:
        return None, None, err2, "opencv_fallback"
    return img3, labels3, None, "opencv_fallback"


def class_names_from_labels(labels: Sequence[Tuple[int, float, float, float, float]]) -> Set[str]:
    out: Set[str] = set()
    for cls, *_ in labels:
        if cls in CLASS_ID_TO_NAME:
            out.add(CLASS_ID_TO_NAME[cls])
    return out


def choose_aug_plans(
    present_classes: Set[str],
    minority_classes: Set[str],
    max_aug_per_image: int,
    rng: random.Random,
) -> List[List[str]]:
    has_minority = len(present_classes.intersection(minority_classes)) > 0
    only_high_freq = bool(present_classes) and present_classes.issubset(HIGH_FREQ_CLASSES)

    if has_minority:
        trigger_prob = 0.95
        budget = max(1, max_aug_per_image)
        min_ops, max_ops = 2, 4
    elif only_high_freq:
        trigger_prob = 0.35
        budget = max(1, max_aug_per_image // 2)
        min_ops, max_ops = 1, 2
    else:
        trigger_prob = 0.55
        budget = max(1, (max_aug_per_image + 1) // 2)
        min_ops, max_ops = 1, 3

    plans: List[List[str]] = []
    for _ in range(budget):
        if rng.random() > trigger_prob:
            continue
        k = rng.randint(min_ops, max_ops)
        k = max(1, min(k, len(OP_POOL)))
        plans.append(rng.sample(OP_POOL, k))

    # ensure minority image gets at least one attempt
    if has_minority and not plans:
        plans.append(["flip", "bri"])
    return plans


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def parse_minority_classes(raw: str) -> Set[str]:
    parts = [x.strip() for x in re.split(r"[,\s]+", raw) if x.strip()]
    return {x for x in parts if x in CLASS_MAP}


def write_dataset_yaml(output_root: Path) -> None:
    yaml_path = output_root / "dataset.yaml"
    lines = [
        f"path: {output_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(CLASS_NAMES):
        lines.append(f"  {idx}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only targeted augmentation for classroom YOLO dataset.")
    parser.add_argument("--input_dir", type=str, default="data/processed/classroom_yolo")
    parser.add_argument("--output_dir", type=str, default="data/processed/classroom_yolo_aug")
    parser.add_argument("--minority_classes", type=str, default="jz,js,zl,xt,zt")
    parser.add_argument("--max_aug_per_image", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--overwrite", type=int, default=1, help="1=remove output_dir first")
    args = parser.parse_args()

    if int(args.max_aug_per_image) <= 0:
        raise ValueError("--max_aug_per_image must be > 0")

    project_root = get_project_root()
    input_root = _resolve_path(project_root, args.input_dir)
    output_root = _resolve_path(project_root, args.output_dir)
    minority_classes = parse_minority_classes(args.minority_classes)
    if not minority_classes:
        minority_classes = {"jz", "js", "zl", "xt", "zt"}

    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_root}")

    req_dirs = [
        input_root / "images" / "train",
        input_root / "images" / "val",
        input_root / "labels" / "train",
        input_root / "labels" / "val",
    ]
    for d in req_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Missing required directory: {d}")

    if output_root.exists() and int(args.overwrite) == 1:
        shutil.rmtree(output_root)

    ensure_dir(output_root / "images" / "train")
    ensure_dir(output_root / "images" / "val")
    ensure_dir(output_root / "labels" / "train")
    ensure_dir(output_root / "labels" / "val")
    ensure_dir(output_root / "meta")

    logger = setup_logger(output_root / "meta" / "augment.log", args.log_level)
    rng = random.Random(int(args.seed))

    logger.info("Start targeted augmentation")
    logger.info("Input: %s", input_root)
    logger.info("Output: %s", output_root)
    logger.info("Minority classes: %s", sorted(minority_classes))
    logger.info("Backend preferred: %s", "albumentations" if HAS_ALBUMENTATIONS else "opencv_fallback")

    manifest_rows: List[Dict[str, Any]] = []
    op_counts = Counter()
    combo_counts = Counter()

    before_count_train = {k: 0 for k in CLASS_NAMES}
    after_count_train = {k: 0 for k in CLASS_NAMES}

    images_train_original = 0
    images_train_augmented_success = 0
    images_train_augmented_failed = 0

    train_img_dir = input_root / "images" / "train"
    train_lbl_dir = input_root / "labels" / "train"
    out_train_img_dir = output_root / "images" / "train"
    out_train_lbl_dir = output_root / "labels" / "train"

    train_images = sorted([p for p in train_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS], key=lambda p: p.name)

    # 1) Copy original train and collect before/after baseline counts.
    for img_path in train_images:
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        out_img = out_train_img_dir / img_path.name
        out_lbl = out_train_lbl_dir / f"{img_path.stem}.txt"
        copy_file(img_path, out_img)

        labels = read_yolo_label(lbl_path)
        write_yolo_label(out_lbl, labels)
        images_train_original += 1
        for cls, *_ in labels:
            cname = CLASS_ID_TO_NAME.get(int(cls))
            if cname is not None:
                before_count_train[cname] += 1
                after_count_train[cname] += 1

    # 2) Copy val as-is (never augment).
    val_img_dir = input_root / "images" / "val"
    val_lbl_dir = input_root / "labels" / "val"
    out_val_img_dir = output_root / "images" / "val"
    out_val_lbl_dir = output_root / "labels" / "val"

    val_images = sorted([p for p in val_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS], key=lambda p: p.name)
    for img_path in val_images:
        lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
        copy_file(img_path, out_val_img_dir / img_path.name)
        labels = read_yolo_label(lbl_path)
        write_yolo_label(out_val_lbl_dir / f"{img_path.stem}.txt", labels)

    # 3) Targeted augmentation on train only.
    for img_path in train_images:
        lbl_path = train_lbl_dir / f"{img_path.stem}.txt"
        labels = read_yolo_label(lbl_path)
        present_classes = class_names_from_labels(labels)
        trigger_classes = sorted(present_classes.intersection(minority_classes))
        plans = choose_aug_plans(
            present_classes=present_classes,
            minority_classes=minority_classes,
            max_aug_per_image=int(args.max_aug_per_image),
            rng=rng,
        )
        if not plans:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            for ops in plans:
                manifest_rows.append(
                    {
                        "original_image": f"images/train/{img_path.name}",
                        "augmented_image": "",
                        "trigger_classes": "|".join(trigger_classes),
                        "augment_types": "+".join(ops),
                        "success": "false",
                        "error": "image_read_failed",
                    }
                )
                images_train_augmented_failed += 1
            logger.warning("Skip broken train image: %s", img_path.name)
            continue

        for idx, ops in enumerate(plans, start=1):
            aug_img, aug_labels, err, backend = apply_augmentation(image, labels, ops, rng)
            op_tag = "_".join(ops)
            op_counts.update(ops)
            combo_counts.update(["+".join(ops)])

            if err is not None or aug_img is None or aug_labels is None:
                manifest_rows.append(
                    {
                        "original_image": f"images/train/{img_path.name}",
                        "augmented_image": "",
                        "trigger_classes": "|".join(trigger_classes),
                        "augment_types": "+".join(ops),
                        "success": "false",
                        "error": err or "unknown_aug_error",
                    }
                )
                images_train_augmented_failed += 1
                logger.warning("Augment failed: %s ops=%s err=%s", img_path.name, ops, err)
                continue

            aug_name = f"{img_path.stem}__aug_{op_tag}_{idx:02d}{img_path.suffix.lower()}"
            aug_lbl_name = f"{img_path.stem}__aug_{op_tag}_{idx:02d}.txt"
            aug_img_path = out_train_img_dir / aug_name
            aug_lbl_path = out_train_lbl_dir / aug_lbl_name

            # avoid accidental overwrite
            uniq = 1
            while aug_img_path.exists() or aug_lbl_path.exists():
                aug_name = f"{img_path.stem}__aug_{op_tag}_{idx:02d}_{uniq}{img_path.suffix.lower()}"
                aug_lbl_name = f"{img_path.stem}__aug_{op_tag}_{idx:02d}_{uniq}.txt"
                aug_img_path = out_train_img_dir / aug_name
                aug_lbl_path = out_train_lbl_dir / aug_lbl_name
                uniq += 1

            ok = cv2.imwrite(str(aug_img_path), aug_img)
            if not ok:
                manifest_rows.append(
                    {
                        "original_image": f"images/train/{img_path.name}",
                        "augmented_image": "",
                        "trigger_classes": "|".join(trigger_classes),
                        "augment_types": "+".join(ops),
                        "success": "false",
                        "error": "imwrite_failed",
                    }
                )
                images_train_augmented_failed += 1
                logger.warning("Augment image save failed: %s", aug_img_path.name)
                continue

            write_yolo_label(aug_lbl_path, aug_labels)
            for cls, *_ in aug_labels:
                cname = CLASS_ID_TO_NAME.get(int(cls))
                if cname is not None:
                    after_count_train[cname] += 1

            manifest_rows.append(
                {
                    "original_image": f"images/train/{img_path.name}",
                    "augmented_image": f"images/train/{aug_name}",
                    "trigger_classes": "|".join(trigger_classes),
                    "augment_types": "+".join(ops),
                    "success": "true",
                    "error": "",
                    "backend": backend,
                }
            )
            images_train_augmented_success += 1

    # 4) Write outputs
    write_dataset_yaml(output_root)

    manifest_path = output_root / "meta" / "augment_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "original_image",
                "augmented_image",
                "trigger_classes",
                "augment_types",
                "success",
                "error",
                "backend",
            ],
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(row)

    stats = {
        "class_order": CLASS_NAMES,
        "minority_classes": sorted(minority_classes),
        "before_count_train": before_count_train,
        "after_count_train": after_count_train,
        "augment_op_counts": dict(op_counts),
        "augment_combo_counts": dict(combo_counts),
        "images_train_original": images_train_original,
        "images_train_augmented_success": images_train_augmented_success,
        "images_train_augmented_failed": images_train_augmented_failed,
        "images_val_copied": len(val_images),
        "backend_preferred": "albumentations" if HAS_ALBUMENTATIONS else "opencv_fallback",
    }
    stats_path = output_root / "meta" / "augment_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Done.")
    logger.info("manifest: %s", manifest_path)
    logger.info("stats: %s", stats_path)
    logger.info(
        "Summary: train_original=%d train_aug_success=%d train_aug_failed=%d val_copied=%d",
        images_train_original,
        images_train_augmented_success,
        images_train_augmented_failed,
        len(val_images),
    )


if __name__ == "__main__":
    main()

