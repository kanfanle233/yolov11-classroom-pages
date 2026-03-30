# -*- coding: utf-8 -*-
"""
Train classroom behavior detector with Ultralytics YOLO.

Default dataset fallback order:
1) data/processed/classroom_yolo_aug/dataset.yaml
2) data/processed/classroom_yolo/dataset.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ultralytics import YOLO


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


def _resolve_path(project_root: Path, value: str) -> Path:
    if resolve_under_project is not None:
        return resolve_under_project(project_root, value)
    p = Path(value)
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def resolve_model_spec(project_root: Path, value: str) -> str:
    raw = str(value).strip()
    if not raw:
        raise ValueError("--model cannot be empty")

    p = Path(raw)
    candidate: Optional[Path] = None

    # Explicit filesystem path.
    if p.is_absolute():
        candidate = p
    elif raw.startswith(".") or ("/" in raw) or ("\\" in raw):
        candidate = (project_root / p).resolve()
    else:
        # If a same-name file exists under project root, use it.
        in_project = (project_root / p).resolve()
        if in_project.exists():
            candidate = in_project
        else:
            # Otherwise keep model alias for Ultralytics, e.g. yolo11s.pt.
            return raw

    resolved = candidate.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"model weight/config not found: {resolved}")
    return str(resolved)


def choose_default_dataset_yaml(project_root: Path) -> Path:
    aug = project_root / "data" / "processed" / "classroom_yolo_aug" / "dataset.yaml"
    base = project_root / "data" / "processed" / "classroom_yolo" / "dataset.yaml"
    if aug.exists():
        return aug.resolve()
    if base.exists():
        return base.resolve()
    raise FileNotFoundError(
        "No dataset.yaml found in fallback locations:\n"
        f"  - {aug}\n"
        f"  - {base}"
    )


def load_yaml(path: Path) -> Dict[str, Any]:
    # Prefer PyYAML when available.
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("dataset yaml must be a mapping/dict")
        return data
    except ModuleNotFoundError:
        # Minimal fallback parser for simple key/value + names block.
        raw = path.read_text(encoding="utf-8").splitlines()
        data: Dict[str, Any] = {}
        names: Dict[str, str] = {}
        in_names = False
        for line in raw:
            l = line.rstrip()
            if "#" in l:
                l = l.split("#", 1)[0].rstrip()
            if not l.strip():
                continue

            if l.strip() == "names:":
                in_names = True
                continue

            if in_names:
                # names block ends when indentation disappears.
                if not (l.startswith("  ") or l.startswith("\t")):
                    in_names = False
                else:
                    s = l.strip()
                    if ":" in s:
                        k, v = s.split(":", 1)
                        names[k.strip()] = v.strip().strip("'\"")
                    elif s.startswith("- "):
                        names[str(len(names))] = s[2:].strip().strip("'\"")
                    continue

            if ":" in l:
                k, v = l.split(":", 1)
                data[k.strip()] = v.strip().strip("'\"")

        if names:
            data["names"] = names
        return data


def normalize_names(dataset_meta: Dict[str, Any]) -> List[str]:
    names = dataset_meta.get("names")
    if isinstance(names, list):
        norm = [str(x) for x in names]
    elif isinstance(names, dict):
        def _key_order(k: Any) -> Tuple[int, str]:
            s = str(k)
            if s.isdigit():
                return (0, f"{int(s):08d}")
            return (1, s)

        norm = [str(v) for k, v in sorted(names.items(), key=lambda kv: _key_order(kv[0]))]
    else:
        raise ValueError("dataset.yaml missing valid 'names' field (list or dict expected)")

    nc = dataset_meta.get("nc")
    if nc is not None:
        try:
            nc_i = int(nc)
        except Exception as e:
            raise ValueError(f"Invalid nc in dataset.yaml: {nc}") from e
        if nc_i != len(norm):
            raise ValueError(
                f"dataset.yaml mismatch: nc={nc_i} but names count={len(norm)}"
            )
    return norm


def resolve_dataset_base(yaml_path: Path, dataset_meta: Dict[str, Any]) -> Path:
    base = dataset_meta.get("path")
    if base is None or str(base).strip() == "":
        return yaml_path.parent.resolve()
    p = Path(str(base))
    if p.is_absolute():
        return p.resolve()
    return (yaml_path.parent / p).resolve()


def labels_dir_from_images_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    for i, p in enumerate(parts):
        if p == "images":
            parts[i] = "labels"
            return Path(*parts)
    # fallback: sibling labels folder
    return images_dir.parent / "labels" / images_dir.name


def count_labels_per_class(label_dir: Path, num_classes: int) -> List[int]:
    counts = [0 for _ in range(num_classes)]
    if not label_dir.exists():
        return counts
    for txt in label_dir.glob("*.txt"):
        try:
            lines = txt.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in lines:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 1:
                continue
            try:
                cid = int(float(parts[0]))
            except Exception:
                continue
            if 0 <= cid < num_classes:
                counts[cid] += 1
    return counts


def to_name_count(names: Sequence[str], counts: Sequence[int]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, n in enumerate(names):
        out[str(n)] = int(counts[i] if i < len(counts) else 0)
    return out


def print_class_distribution(train_counts: Dict[str, int], val_counts: Dict[str, int]) -> None:
    print("\n" + "=" * 72)
    print("[Class Distribution] (box counts)")
    print("-" * 72)
    print(f"{'Class':<16}{'Train':>12}{'Val':>12}{'Total':>12}")
    print("-" * 72)
    names = sorted(set(train_counts.keys()) | set(val_counts.keys()))
    for name in names:
        tr = int(train_counts.get(name, 0))
        va = int(val_counts.get(name, 0))
        print(f"{name:<16}{tr:>12}{va:>12}{(tr + va):>12}")
    print("=" * 72 + "\n")


def maybe_load_meta_distribution(dataset_yaml: Path) -> Optional[Dict[str, Any]]:
    root = dataset_yaml.parent
    # Preferred for augmented dataset.
    p1 = root / "meta" / "augment_stats.json"
    if p1.exists():
        try:
            return json.loads(p1.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Preferred for converted dataset.
    p2 = root / "class_stats.json"
    if p2.exists():
        try:
            return json.loads(p2.read_text(encoding="utf-8"))
        except Exception:
            pass
    return None


def build_summary(
    save_dir: Path,
    dataset_yaml: Path,
    names: Sequence[str],
    train_count_by_name: Dict[str, int],
    val_count_by_name: Dict[str, int],
    args_dict: Dict[str, Any],
    extra_meta: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    best_pt = (save_dir / "weights" / "best.pt").resolve()
    last_pt = (save_dir / "weights" / "last.pt").resolve()
    results_csv = (save_dir / "results.csv").resolve()
    args_yaml = (save_dir / "args.yaml").resolve()

    return {
        "dataset_yaml": str(dataset_yaml.resolve()),
        "dataset_root": str(dataset_yaml.parent.resolve()),
        "class_names": list(names),
        "class_count_train": train_count_by_name,
        "class_count_val": val_count_by_name,
        "class_count_total": {k: int(train_count_by_name.get(k, 0) + val_count_by_name.get(k, 0)) for k in names},
        "train_params": args_dict,
        "best_weight_path": str(best_pt),
        "best_weight_exists": best_pt.exists(),
        "last_weight_path": str(last_pt),
        "last_weight_exists": last_pt.exists(),
        "results_csv_path": str(results_csv),
        "results_csv_exists": results_csv.exists(),
        "args_yaml_path": str(args_yaml),
        "args_yaml_exists": args_yaml.exists(),
        "meta_distribution_source": extra_meta if extra_meta is not None else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classroom case YOLO detector")
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="dataset yaml path; if empty, use fallback: classroom_yolo_aug -> classroom_yolo",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="start weight/model cfg, e.g. yolo11n.pt / yolo11s.pt / custom.pt",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0", help="0 / 0,1 / cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/classroom_case_yolo")
    parser.add_argument("--name", type=str, default="exp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cache", type=int, default=0, help="1=cache images in RAM")
    parser.add_argument("--resume", type=int, default=0, help="1=resume from last run")
    parser.add_argument(
        "--class_weights",
        type=str,
        default="",
        help="optional class weights file (reserved; not injected into YOLO internals for safety)",
    )
    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).resolve())

    if str(args.data).strip():
        data_yaml = _resolve_path(project_root, args.data)
    else:
        data_yaml = choose_default_dataset_yaml(project_root)

    model_spec = resolve_model_spec(project_root, args.model)
    runs_dir = _resolve_path(project_root, args.project)

    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_yaml}")

    dataset_meta = load_yaml(data_yaml)
    names = normalize_names(dataset_meta)
    if len(names) == 0:
        raise ValueError("dataset.yaml contains empty names")

    base = resolve_dataset_base(data_yaml, dataset_meta)
    train_ref = dataset_meta.get("train")
    val_ref = dataset_meta.get("val")
    if train_ref is None or val_ref is None:
        raise ValueError("dataset.yaml must contain both 'train' and 'val'")

    train_images_dir = (base / str(train_ref)).resolve()
    val_images_dir = (base / str(val_ref)).resolve()
    train_labels_dir = labels_dir_from_images_dir(train_images_dir)
    val_labels_dir = labels_dir_from_images_dir(val_images_dir)

    if not train_images_dir.exists():
        raise FileNotFoundError(f"train images dir not found: {train_images_dir}")
    if not val_images_dir.exists():
        raise FileNotFoundError(f"val images dir not found: {val_images_dir}")
    if not train_labels_dir.exists():
        raise FileNotFoundError(f"train labels dir not found: {train_labels_dir}")
    if not val_labels_dir.exists():
        raise FileNotFoundError(f"val labels dir not found: {val_labels_dir}")

    train_counts_raw = count_labels_per_class(train_labels_dir, len(names))
    val_counts_raw = count_labels_per_class(val_labels_dir, len(names))
    train_count_by_name = to_name_count(names, train_counts_raw)
    val_count_by_name = to_name_count(names, val_counts_raw)
    print_class_distribution(train_count_by_name, val_count_by_name)

    if str(args.class_weights).strip():
        print(
            "[INFO] --class_weights is provided but not injected into YOLO internals "
            "(to keep script stable and avoid unsafe source modifications)."
        )

    print("[INFO] Training start")
    print(f"[INFO] Dataset yaml : {data_yaml}")
    print(f"[INFO] Model        : {model_spec}")
    print(f"[INFO] Project/name : {runs_dir} / {args.name}")

    yolo = YOLO(str(model_spec))
    results = yolo.train(
        data=str(data_yaml),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        workers=int(args.workers),
        project=str(runs_dir),
        name=str(args.name),
        seed=int(args.seed),
        patience=int(args.patience),
        cache=bool(int(args.cache)),
        resume=bool(int(args.resume)),
    )

    save_dir: Optional[Path] = None
    if hasattr(results, "save_dir"):
        try:
            save_dir = Path(str(results.save_dir)).resolve()
        except Exception:
            save_dir = None
    if save_dir is None:
        save_dir = (runs_dir / args.name).resolve()

    save_dir.mkdir(parents=True, exist_ok=True)

    extra_meta = maybe_load_meta_distribution(data_yaml)
    args_dict = {k: v for k, v in vars(args).items()}
    summary = build_summary(
        save_dir=save_dir,
        dataset_yaml=data_yaml,
        names=names,
        train_count_by_name=train_count_by_name,
        val_count_by_name=val_count_by_name,
        args_dict=args_dict,
        extra_meta=extra_meta,
    )

    summary_path = save_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[INFO] Training done")
    print(f"[OUT] save_dir          : {save_dir}")
    print(f"[OUT] best.pt           : {summary['best_weight_path']}")
    print(f"[OUT] last.pt           : {summary['last_weight_path']}")
    print(f"[OUT] results.csv       : {summary['results_csv_path']}")
    print(f"[OUT] args.yaml         : {summary['args_yaml_path']}")
    print(f"[OUT] train_summary.json: {summary_path}")


if __name__ == "__main__":
    main()
