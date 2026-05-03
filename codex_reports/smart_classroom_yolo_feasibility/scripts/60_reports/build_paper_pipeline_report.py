from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


def _resolve_repo_root(anchor: Path) -> Path:
    for candidate in [anchor.resolve()] + list(anchor.resolve().parents):
        if (candidate / "data").exists() and (candidate / "scripts").exists():
            return candidate
    raise RuntimeError(f"Cannot resolve repo root from: {anchor}")


def _resolve(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_json(path: Path) -> Any:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _load_dataset_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {"raw_preview": text[:2000]}


def _copy_asset(src: Path, dst_dir: Path, figure_id: str, section: str, caption: str) -> Dict[str, Any]:
    dst = dst_dir / f"{figure_id}{src.suffix}"
    status = "missing"
    if src.exists() and src.is_file() and src.stat().st_size > 0:
        dst_dir.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        status = "ok"
    return {
        "figure_id": figure_id,
        "source_path": str(src),
        "target_path": str(dst),
        "section": section,
        "caption_draft": caption,
        "status": status,
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["figure_id", "source_path", "target_path", "section", "caption_draft", "status"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _metric_block(title: str, data: Dict[str, Any]) -> List[str]:
    lines = [f"### {title}"]
    if not data:
        lines.append("- No report found.")
        return lines
    for key, value in data.items():
        if isinstance(value, (dict, list)):
            continue
        lines.append(f"- `{key}`: `{value}`")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready assets and a pipeline report.")
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--run_id", default="", type=str)
    parser.add_argument("--dataset_yaml", default="data/processed/classroom_yolo/dataset.yaml", type=str)
    parser.add_argument("--train_run", action="append", default=[], help="YOLO training run directory")
    parser.add_argument("--paper_dir", default="", type=str)
    args = parser.parse_args()

    repo_root = _resolve_repo_root(Path(__file__))
    output_dir = _resolve(repo_root, args.output_dir)
    run_id = args.run_id.strip() or output_dir.parents[0].name
    paper_dir = _resolve(repo_root, args.paper_dir) if args.paper_dir else (
        repo_root / "codex_reports" / "smart_classroom_yolo_feasibility" / "paper_assets" / run_id
    )
    dataset_yaml = _resolve(repo_root, args.dataset_yaml)

    manifest = _load_json(output_dir / "pipeline_manifest.json") or {}
    fusion_report = _load_json(output_dir / "fusion_contract_report.json") or {}
    pipeline_report = _load_json(output_dir / "pipeline_contract_v2_report.json") or {}
    asr_report = _load_json(output_dir / "asr_quality_report.json") or {}
    verifier_eval = _load_json(output_dir / "verifier_eval_report.json") or {}
    dataset = _load_dataset_yaml(dataset_yaml)

    asset_rows: List[Dict[str, Any]] = []
    for idx, raw_run in enumerate(args.train_run):
        run_dir = _resolve(repo_root, raw_run)
        prefix = f"train{idx + 1}"
        for name, section, caption in [
            ("results.png", "training", "YOLO fine-tuning loss and metric curves"),
            ("confusion_matrix.png", "training", "YOLO validation confusion matrix"),
            ("val_batch0_pred.jpg", "qualitative", "Validation predictions sample batch 0"),
            ("val_batch1_pred.jpg", "qualitative", "Validation predictions sample batch 1"),
        ]:
            asset_rows.append(_copy_asset(run_dir / name, paper_dir, f"{prefix}_{Path(name).stem}", section, caption))

    for src, figure_id, section, caption in [
        (output_dir / "timeline_chart.png", "timeline_chart", "timeline", "Student-level translated visual behavior timeline"),
        (
            output_dir / "verifier_reliability_diagram.svg",
            "verifier_reliability_diagram",
            "verifier",
            "Verifier calibration reliability diagram",
        ),
    ]:
        asset_rows.append(_copy_asset(src, paper_dir, figure_id, section, caption))

    _write_csv(paper_dir / "paper_image_manifest.csv", asset_rows)

    dataset_names = dataset.get("names", {})
    if isinstance(dataset_names, list):
        dataset_classes = ", ".join(str(x) for x in dataset_names)
    elif isinstance(dataset_names, dict):
        dataset_classes = ", ".join(f"{k}:{v}" for k, v in dataset_names.items())
    else:
        dataset_classes = ""

    lines: List[str] = [
        f"# Paper Pipeline Report: {run_id}",
        "",
        "## Dataset",
        f"- Dataset YAML: `{dataset_yaml}`",
        f"- Classes: `{dataset_classes}`",
        "",
        "## Pipeline Artifacts",
    ]
    artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
    for key in [
        "pose_tracks_smooth",
        "behavior_det_semantic",
        "actions_fusion_v2",
        "event_queries_fusion_v2",
        "align_multimodal",
        "verified_events",
        "student_id_map",
        "timeline_students_csv",
        "timeline_chart_png",
    ]:
        lines.append(f"- `{key}`: `{artifacts.get(key, '')}`")

    lines.extend(["", "## Contract Summary"])
    lines.extend(_metric_block("Fusion Contract", fusion_report.get("counts", {}) if isinstance(fusion_report, dict) else {}))
    lines.extend(_metric_block("Pipeline Contract", pipeline_report.get("counts", {}) if isinstance(pipeline_report, dict) else {}))
    lines.extend(["", "## ASR Quality"])
    lines.extend(_metric_block("ASR", asr_report if isinstance(asr_report, dict) else {}))
    lines.extend(["", "## Verifier"])
    lines.extend(_metric_block("Verifier Counts", verifier_eval.get("counts", {}) if isinstance(verifier_eval, dict) else {}))
    lines.extend(_metric_block("Verifier Metrics", verifier_eval.get("metrics", {}) if isinstance(verifier_eval, dict) else {}))
    lines.extend(["", "## Paper Assets", f"- Manifest: `{paper_dir / 'paper_image_manifest.csv'}`"])

    paper_dir.mkdir(parents=True, exist_ok=True)
    report_path = paper_dir / "paper_pipeline_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"paper_dir": str(paper_dir), "report": str(report_path), "assets": len(asset_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
