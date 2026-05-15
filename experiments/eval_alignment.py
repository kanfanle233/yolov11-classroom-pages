import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts.schemas import (
    ARTIFACT_VERSION,
    SCHEMA_VERSION,
    validate_align_file,
    validate_json_file,
    write_json,
)
from verifier.model import action_match_score


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _resolve(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _run(label: str, cmd: List[str]) -> None:
    print(f"[RUN] {label}")
    print("      " + " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"{label} failed: exit={res.returncode}")


def _build_actions_from_align_sample(align_sample: Path, out_path: Path) -> Path:
    data = _load_json(align_sample)
    if not isinstance(data, list):
        raise ValueError("align sample must be a JSON list")
    seen = set()
    rows: List[Dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        for cand in row.get("candidates", []):
            if not isinstance(cand, dict):
                continue
            tid = cand.get("track_id")
            action = str(cand.get("action", "")).strip().lower()
            st = _safe_float(cand.get("start_time", 0.0), 0.0)
            ed = _safe_float(cand.get("end_time", st + 0.2), st + 0.2)
            conf = _safe_float(cand.get("action_confidence", 0.5), 0.5)
            if not isinstance(tid, int) or not action:
                continue
            key = (tid, action, round(st, 3), round(ed, 3))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "track_id": int(tid),
                    "action": action,
                    "start_time": st,
                    "end_time": max(st + 0.2, ed),
                    "action_confidence": conf,
                    "confidence": conf,
                    "conf": conf,
                }
            )
    rows.sort(key=lambda x: (x["start_time"], x["track_id"]))
    _write_jsonl(out_path, rows)
    return out_path


def _evaluate_alignment(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {
            "alignment_recall_at_1": 0.0,
            "alignment_recall_at_k": 0.0,
            "alignment_precision": 0.0,
            "mean_temporal_overlap": 0.0,
            "correct_action_in_topk": 0,
            "total_events": 0,
            "events_with_candidates": 0,
            "alignment_source_distribution": {},
        }

    top1_correct = 0
    topk_correct = 0
    overlap_sum = 0.0
    events_with_candidates = 0
    source_counts: Dict[str, int] = {}

    for row in rows:
        source = str(row.get("alignment_source", "")).strip() or "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1

        event_type = str(row.get("event_type", "unknown"))
        query_text = str(row.get("query_text", ""))
        candidates = row.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        if len(candidates) == 0:
            continue
        events_with_candidates += 1

        top = candidates[0] if isinstance(candidates[0], dict) else {}
        top_action = str(top.get("action", "")).strip().lower()
        top_overlap = _safe_float(top.get("overlap", 0.0), 0.0)
        overlap_sum += _clamp01(top_overlap)

        top1_match = action_match_score(event_type, query_text, top_action) >= 0.8
        if top1_match:
            top1_correct += 1

        any_match = False
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            action = str(cand.get("action", "")).strip().lower()
            if action_match_score(event_type, query_text, action) >= 0.8:
                any_match = True
                break
        if any_match:
            topk_correct += 1

    precision = top1_correct / max(1, events_with_candidates)
    recall_1 = top1_correct / total
    recall_k = topk_correct / total
    mean_overlap = overlap_sum / total
    return {
        "alignment_recall_at_1": round(recall_1, 6),
        "alignment_recall_at_k": round(recall_k, 6),
        "alignment_precision": round(precision, 6),
        "mean_temporal_overlap": round(mean_overlap, 6),
        "correct_action_in_topk": int(topk_correct),
        "total_events": int(total),
        "events_with_candidates": int(events_with_candidates),
        "alignment_source_distribution": source_counts,
    }


def _run_align_once(
    *,
    py_exe: str,
    align_script: Path,
    event_queries: Path,
    actions: Path,
    pose_uq: Path,
    out_path: Path,
    mode: str,
    query_offset: float,
    params: Dict[str, float],
    topk: int,
) -> List[Dict[str, Any]]:
    cmd = [
        py_exe,
        str(align_script),
        "--event_queries",
        str(event_queries),
        "--actions",
        str(actions),
        "--pose_uq",
        str(pose_uq),
        "--out",
        str(out_path),
        "--alignment_mode",
        mode,
        "--query_time_offset",
        str(float(query_offset)),
        "--base_window",
        str(float(params["base_window"])),
        "--alpha_motion",
        str(float(params["alpha_motion"])),
        "--beta_uq",
        str(float(params["beta_uq"])),
        "--min_window",
        str(float(params["min_window"])),
        "--max_window",
        str(float(params["max_window"])),
        "--fixed_window",
        str(float(params["fixed_window"])),
        "--fallback_window",
        str(float(params["fixed_window"])),
        "--enable_fallback",
        "1",
        "--topk",
        str(int(topk)),
    ]
    _run(f"alignment mode={mode} offset={query_offset:+.1f}s", cmd)
    ok, errors = validate_json_file(out_path, validate_align_file)
    if not ok:
        raise ValueError(f"align output failed schema validation: {errors[:3]}")
    data = _load_json(out_path)
    if not isinstance(data, list):
        raise ValueError("align output must be a JSON list")
    return data


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary(
    *,
    summary_path: Path,
    metrics: Dict[str, Any],
    sensitivity_rows: List[Dict[str, Any]],
    noise_rows: List[Dict[str, Any]],
) -> None:
    baseline = metrics["baseline_comparison"]
    lines = [
        "# Experiment A: UQ Adaptive Alignment",
        "",
        f"- Generated at (UTC): {metrics.get('generated_at')}",
        "- This run uses sample/weak labels and is **not final paper result**.",
        "",
        "## Baseline Comparison (offset=0s)",
        f"- adaptive recall@1: {baseline['adaptive_uq']['alignment_recall_at_1']}",
        f"- fixed recall@1: {baseline['fixed']['alignment_recall_at_1']}",
        f"- delta recall@1: {baseline['delta_adaptive_minus_fixed']['alignment_recall_at_1']}",
        f"- adaptive recall@k: {baseline['adaptive_uq']['alignment_recall_at_k']}",
        f"- fixed recall@k: {baseline['fixed']['alignment_recall_at_k']}",
        f"- delta recall@k: {baseline['delta_adaptive_minus_fixed']['alignment_recall_at_k']}",
        "",
        "## Noise Degradation",
    ]
    for row in noise_rows:
        if str(row.get("offset_sec")) == "0.0":
            continue
        lines.append(
            f"- offset={row['offset_sec']} mode={row['mode']} "
            f"recall@1={row['alignment_recall_at_1']} overlap={row['mean_temporal_overlap']}"
        )
    lines.append("")
    lines.append("## Sensitivity (adaptive_uq)")
    for row in sensitivity_rows:
        lines.append(
            f"- {row['parameter']}={row['value']} recall@1={row['alignment_recall_at_1']} "
            f"delta_vs_default={row['delta_recall_at_1_vs_default']}"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- alignment_source is emitted as fixed/adaptive_uq/fallback in align outputs.")
    lines.append("- verifier core decision logic is unchanged in this experiment.")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed vs adaptive UQ alignment with noise and sensitivity.")
    parser.add_argument("--py", default=sys.executable, type=str)
    parser.add_argument("--output_dir", default="output/paper_experiments/exp_a_uq_align", type=str)
    parser.add_argument("--event_queries", default="contracts/examples/event_queries.sample.jsonl", type=str)
    parser.add_argument("--pose_uq", default="contracts/examples/pose_tracks_smooth_uq.sample.jsonl", type=str)
    parser.add_argument("--align_sample", default="contracts/examples/align_multimodal.sample.json", type=str)
    parser.add_argument("--topk", default=8, type=int)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = _resolve(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    align_script = (repo_root / "scripts/xx_align_multimodal.py").resolve()
    event_queries = _resolve(repo_root, args.event_queries)
    pose_uq = _resolve(repo_root, args.pose_uq)
    align_sample = _resolve(repo_root, args.align_sample)
    actions_derived = output_dir / "actions.from_align.sample.jsonl"
    _build_actions_from_align_sample(align_sample, actions_derived)

    defaults = {
        "base_window": 1.0,
        "alpha_motion": 1.2,
        "beta_uq": 0.8,
        "min_window": 0.6,
        "max_window": 4.0,
        "fixed_window": 1.0,
    }

    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Baseline comparison at zero offset.
    fixed_rows = _run_align_once(
        py_exe=args.py,
        align_script=align_script,
        event_queries=event_queries,
        actions=actions_derived,
        pose_uq=pose_uq,
        out_path=tmp_dir / "align_fixed_offset0.json",
        mode="fixed",
        query_offset=0.0,
        params=defaults,
        topk=int(args.topk),
    )
    adaptive_rows = _run_align_once(
        py_exe=args.py,
        align_script=align_script,
        event_queries=event_queries,
        actions=actions_derived,
        pose_uq=pose_uq,
        out_path=tmp_dir / "align_adaptive_offset0.json",
        mode="adaptive_uq",
        query_offset=0.0,
        params=defaults,
        topk=int(args.topk),
    )
    fixed_metrics = _evaluate_alignment(fixed_rows)
    adaptive_metrics = _evaluate_alignment(adaptive_rows)

    # Noise curve: timestamp offsets.
    noise_rows: List[Dict[str, Any]] = []
    for offset in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        for mode in ["fixed", "adaptive_uq"]:
            rows = _run_align_once(
                py_exe=args.py,
                align_script=align_script,
                event_queries=event_queries,
                actions=actions_derived,
                pose_uq=pose_uq,
                out_path=tmp_dir / f"align_{mode}_offset_{offset:+.1f}.json",
                mode=mode,
                query_offset=offset,
                params=defaults,
                topk=int(args.topk),
            )
            m = _evaluate_alignment(rows)
            noise_rows.append(
                {
                    "mode": mode,
                    "offset_sec": float(offset),
                    "alignment_recall_at_1": m["alignment_recall_at_1"],
                    "alignment_recall_at_k": m["alignment_recall_at_k"],
                    "alignment_precision": m["alignment_precision"],
                    "mean_temporal_overlap": m["mean_temporal_overlap"],
                    "correct_action_in_topk": m["correct_action_in_topk"],
                    "total_events": m["total_events"],
                    "events_with_candidates": m["events_with_candidates"],
                }
            )

    # Sensitivity: one-factor-at-a-time on adaptive mode.
    sensitivity_grid = {
        "base_window": [0.6, 1.0, 1.4],
        "alpha_motion": [0.6, 1.2, 1.8],
        "beta_uq": [0.2, 0.8, 1.4],
        "min_window": [0.4, 0.6, 0.9],
        "max_window": [2.5, 4.0, 5.0],
    }
    sensitivity_rows: List[Dict[str, Any]] = []
    default_recall_1 = adaptive_metrics["alignment_recall_at_1"]
    default_recall_k = adaptive_metrics["alignment_recall_at_k"]
    for param, values in sensitivity_grid.items():
        for value in values:
            p = dict(defaults)
            p[param] = float(value)
            rows = _run_align_once(
                py_exe=args.py,
                align_script=align_script,
                event_queries=event_queries,
                actions=actions_derived,
                pose_uq=pose_uq,
                out_path=tmp_dir / f"align_adaptive_sens_{param}_{value}.json",
                mode="adaptive_uq",
                query_offset=0.0,
                params=p,
                topk=int(args.topk),
            )
            m = _evaluate_alignment(rows)
            sensitivity_rows.append(
                {
                    "parameter": param,
                    "value": float(value),
                    "alignment_recall_at_1": m["alignment_recall_at_1"],
                    "alignment_recall_at_k": m["alignment_recall_at_k"],
                    "alignment_precision": m["alignment_precision"],
                    "mean_temporal_overlap": m["mean_temporal_overlap"],
                    "correct_action_in_topk": m["correct_action_in_topk"],
                    "delta_recall_at_1_vs_default": round(m["alignment_recall_at_1"] - default_recall_1, 6),
                    "delta_recall_at_k_vs_default": round(m["alignment_recall_at_k"] - default_recall_k, 6),
                }
            )

    noise_csv = output_dir / "alignment_noise_curve.csv"
    sensitivity_csv = output_dir / "alignment_sensitivity.csv"
    metrics_json = output_dir / "metrics.json"
    summary_md = output_dir / "summary.md"

    _write_csv(
        noise_csv,
        [
            "mode",
            "offset_sec",
            "alignment_recall_at_1",
            "alignment_recall_at_k",
            "alignment_precision",
            "mean_temporal_overlap",
            "correct_action_in_topk",
            "total_events",
            "events_with_candidates",
        ],
        noise_rows,
    )
    _write_csv(
        sensitivity_csv,
        [
            "parameter",
            "value",
            "alignment_recall_at_1",
            "alignment_recall_at_k",
            "alignment_precision",
            "mean_temporal_overlap",
            "correct_action_in_topk",
            "delta_recall_at_1_vs_default",
            "delta_recall_at_k_vs_default",
        ],
        sensitivity_rows,
    )

    delta = {
        "alignment_recall_at_1": round(
            adaptive_metrics["alignment_recall_at_1"] - fixed_metrics["alignment_recall_at_1"], 6
        ),
        "alignment_recall_at_k": round(
            adaptive_metrics["alignment_recall_at_k"] - fixed_metrics["alignment_recall_at_k"], 6
        ),
        "alignment_precision": round(
            adaptive_metrics["alignment_precision"] - fixed_metrics["alignment_precision"], 6
        ),
        "mean_temporal_overlap": round(
            adaptive_metrics["mean_temporal_overlap"] - fixed_metrics["mean_temporal_overlap"], 6
        ),
        "correct_action_in_topk": int(
            adaptive_metrics["correct_action_in_topk"] - fixed_metrics["correct_action_in_topk"]
        ),
    }

    metrics_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "branch_name": "exp_a_uq_align",
        "objective": "Compare uncertainty-driven adaptive alignment against fixed-window baseline.",
        "data_mode": "sample_or_weak_labels",
        "generated_at": _now_iso(),
        "inputs": {
            "event_queries": str(event_queries),
            "pose_uq": str(pose_uq),
            "align_sample": str(align_sample),
            "actions_derived": str(actions_derived),
        },
        "default_params": defaults,
        "baseline_comparison": {
            "fixed": fixed_metrics,
            "adaptive_uq": adaptive_metrics,
            "delta_adaptive_minus_fixed": delta,
        },
        "noise_curve_file": str(noise_csv),
        "sensitivity_file": str(sensitivity_csv),
        "notes": [
            "Non-final paper result: uses sample/weak supervision proxies.",
            "correct_action_in_topk is computed via event_type-action alias matching.",
            "alignment_source is emitted by scripts/xx_align_multimodal.py as fixed/adaptive_uq/fallback.",
        ],
        "artifacts": {
            "metrics_json_sha256": _sha256(metrics_json) if metrics_json.exists() else "",
            "noise_csv_sha256": _sha256(noise_csv),
            "sensitivity_csv_sha256": _sha256(sensitivity_csv),
        },
    }
    write_json(metrics_json, metrics_payload)
    # Refresh hash after metrics file exists.
    metrics_payload["artifacts"]["metrics_json_sha256"] = _sha256(metrics_json)
    write_json(metrics_json, metrics_payload)

    _write_summary(summary_path=summary_md, metrics=metrics_payload, sensitivity_rows=sensitivity_rows, noise_rows=noise_rows)

    print(f"[DONE] metrics: {metrics_json}")
    print(f"[DONE] noise curve: {noise_csv}")
    print(f"[DONE] sensitivity: {sensitivity_csv}")
    print(f"[DONE] summary: {summary_md}")


if __name__ == "__main__":
    main()

