"""Split processed classroom_yolo dataset into train/val/test (70/15/15).

The original dataset only has train/val. This script:
1. Merges train+val images
2. Performs a stratified random split into 70/15/15
3. Copies images/labels to a test directory
4. Updates dataset.yaml

NOTE: The original case screenshots were extracted 1fps from 正方视角 videos.
Filenames do not encode source video ID, so a pure "by video" split is infeasible.
This random split is the standard approach used by comparable classroom behavior papers.
"""

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def main() -> None:
    parser = argparse.ArgumentParser(description="Split classroom_yolo into train/val/test (70/15/15)")
    parser.add_argument("--data_dir", default="data/processed/classroom_yolo")
    parser.add_argument("--out_dir", default="data/processed/classroom_yolo_test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", default="70,15,15")
    parser.add_argument("--dry_run", type=int, default=0)
    args = parser.parse_args()

    base = Path(args.data_dir)
    out_base = Path(args.out_dir)
    ratios = [float(x) / 100.0 for x in str(args.ratios).split(",")]
    assert len(ratios) == 3 and abs(sum(ratios) - 1.0) < 0.01
    random.seed(args.seed)

    # Gather all images from train+val
    all_items: List[Tuple[Path, Path, str]] = []  # (img, lbl, source_dir)
    for split in ["train", "val"]:
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            all_items.append((img_path, lbl_path if lbl_path.exists() else None, split))

    print(f"Total images: {len(all_items)} (from train+val)")

    # Count class distribution per image for stratification
    class_counts: Dict[str, Counter] = {}
    for img, lbl, _ in all_items:
        classes = []
        if lbl and lbl.exists():
            for line in lbl.read_text().strip().splitlines():
                if line.strip():
                    try:
                        cls_id = int(line.split()[0])
                        classes.append(cls_id)
                    except Exception:
                        pass
        class_counts[img.stem] = Counter(classes)

    # Simple stratification: group by the dominant class, then shuffle within groups
    by_dominant: Dict[int, List[Tuple[Path, Path, str]]] = defaultdict(list)
    for img, lbl, src in all_items:
        cnt = class_counts.get(img.stem, Counter())
        dominant = cnt.most_common(1)[0][0] if cnt else -1
        by_dominant[dominant].append((img, lbl, src))

    # Shuffle each group
    for k in by_dominant:
        random.shuffle(by_dominant[k])

    # Split
    train_imgs, val_imgs, test_imgs = [], [], []
    for dominant, items in sorted(by_dominant.items()):
        n = len(items)
        n_train = max(1, int(n * ratios[0]))
        n_val = max(1, int(n * ratios[1]))
        train_imgs.extend(items[:n_train])
        val_imgs.extend(items[n_train:n_train + n_val])
        test_imgs.extend(items[n_train + n_val:])

    # Ensure no overlap
    train_stems = {img.stem for img, _, _ in train_imgs}
    val_stems = {img.stem for img, _, _ in val_imgs}
    test_stems = {img.stem for img, _, _ in test_imgs}
    assert not (train_stems & val_stems), "train/val overlap"
    assert not (train_stems & test_stems), "train/test overlap"
    assert not (val_stems & test_stems), "val/test overlap"

    print(f"Split: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
    print(f"       train%={len(train_imgs)/len(all_items)*100:.1f}, val%={len(val_imgs)/len(all_items)*100:.1f}, test%={len(test_imgs)/len(all_items)*100:.1f}")

    if args.dry_run:
        print("[DRY RUN] No files copied.")
        return

    # Copy files
    for split_name, items in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        img_out = out_base / "images" / split_name
        lbl_out = out_base / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img, lbl, _ in items:
            shutil.copy2(img, img_out / img.name)
            if lbl and lbl.exists():
                shutil.copy2(lbl, lbl_out / lbl.name)
        print(f"  {split_name}: {len(items)} images copied")

    # Count class distribution in test set
    test_class_count: Counter = Counter()
    for _, lbl, _ in test_imgs:
        if lbl and lbl.exists():
            for line in lbl.read_text().strip().splitlines():
                if line.strip():
                    try:
                        test_class_count[int(line.split()[0])] += 1
                    except Exception:
                        pass

    # Class names
    names = {0: "tt", 1: "dx", 2: "dk", 3: "zt", 4: "xt", 5: "js", 6: "zl", 7: "jz"}
    print("\nTest set class distribution:")
    for cls_id in sorted(test_class_count):
        print(f"  {names.get(cls_id, cls_id)}: {test_class_count[cls_id]}")

    # Write dataset.yaml
    yaml_content = f"""# Classroom behavior detection - train/val/test split
# Split seed: {args.seed}, ratios: {args.ratios}
path: {out_base.resolve().as_posix()}
train: images/train
val: images/val
test: images/test
names:
  0: tt
  1: dx
  2: dk
  3: zt
  4: xt
  5: js
  6: zl
  7: jz
"""
    (out_base / "dataset.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"\n[DONE] dataset.yaml written to {out_base / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
