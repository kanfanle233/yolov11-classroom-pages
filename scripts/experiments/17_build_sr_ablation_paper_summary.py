import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_VARIANTS = [
    "A0_full_no_sr",
    "A1_full_sliced_no_sr",
    "A2_full_sliced_opencv",
    "A3_full_sliced_artifact_deblur_opencv",
    "A7_sliced_strong_seat_prior",
    "A8_adaptive_sliced_artifact_deblur_opencv",
    "A9_rear_dense_artifact_deblur_opencv",
    "A10_full_sliced_nvidia_vsr",
    "A11_full_sliced_maxine_vfx",
]


def _resolve(base_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(value: Any, base: Any) -> Any:
    if base in [None, "", 0] or value in [None, ""]:
        return None
    try:
        return round((float(value) - float(base)) / float(base) * 100.0, 2)
    except Exception:
        return None


def _ratio(delta_value: Any, delta_cost: Any) -> Any:
    if delta_value in [None, ""] or delta_cost in [None, "", 0]:
        return None
    try:
        return round(float(delta_value) / float(delta_cost), 6)
    except Exception:
        return None


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = [
        "case",
        "variant",
        "sr_backend",
        "sr_preprocess",
        "pipeline_status",
        "tracked_students",
        "tracked_students_delta_pct_vs_A0",
        "rear_pose_person_rows_proxy",
        "rear_proxy_delta_pct_vs_A0",
        "rear_avg_visible_keypoints",
        "rear_low_visible_keypoint_rate",
        "track_gap_count_proxy",
        "seat_anchor_jitter_mean",
        "pose_track_gap_count_proxy",
        "pose_track_max_gap_proxy",
        "pose_track_seat_prior_mode",
        "pose_track_max_dx_ratio",
        "matched_behavior_items",
        "matched_delta_pct_vs_A0",
        "unmatched_behavior_items",
        "unmatched_delta_pct_vs_A0",
        "actions_fusion_v2",
        "unlinked_rows_ge_200000",
        "person_recall",
        "person_precision",
        "person_f1",
        "gt_status",
        "AP50",
        "AP75",
        "mAP50_95",
        "PCK_0_10",
        "OKS_AP",
        "IDF1",
        "IDSW",
        "HOTA",
        "MOTA",
        "track_fragments",
        "behavior_macro_f1",
        "behavior_micro_f1",
        "segment_iou",
        "temporal_mAP_0_5",
        "frame_mAP_IoU0_5",
        "stage_runtime_sec",
        "effective_fps",
        "peak_vram_mb",
        "sr_cache_bytes",
        "cost_gain_ratio",
        "proxy_cost_gain_ratio",
        "sr_elapsed_sec",
    ]
    lines = [
        "# SR Ablation Paper Summary",
        "",
        "> No GT supplied yet: formal AP/PCK/IDF1/HOTA/behavior F1 remain placeholders. Proxy metrics are engineering evidence, not accuracy claims.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    lines += [
        "",
        "## Current Interpretation",
        "- `full_sliced` is the dominant improvement over full-frame inference on both front videos.",
        "- `artifact_deblur + opencv SR` is an input-enhancement ablation; accept it only when it improves proxy metrics without increasing unmatched items or track jitter.",
        "- Real-ESRGAN / BasicVSR++ / RealBasicVSR need external commands before their rows become valid.",
        "- NVIDIA VSR / Maxine VFX are supported as external-command adapters, but they require local SDK setup before their rows become valid.",
        "- `cost_gain_ratio` uses formal rear-person recall gain per extra runtime second; `proxy_cost_gain_ratio` uses proxy rear pose rows when GT is missing.",
        "- A paper-grade claim needs rear-row GT for detection, pose, behavior segments, and ID continuity.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build cross-case paper table for SR ablations.")
    parser.add_argument("--case", action="append", default=[], help="case_id=metrics_json_path")
    parser.add_argument("--out_dir", default="output/codex_reports")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--include_missing", type=int, default=0)
    args = parser.parse_args()

    cases = args.case or [
        "front_001=output/codex_reports/front_001_sr_ablation/sr_ablation_metrics.json",
        "front_002=output/codex_reports/front_002_sr_ablation/sr_ablation_metrics.json",
        "front_046=output/codex_reports/front_046_sr_ablation/sr_ablation_metrics.json",
    ]
    wanted = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    rows: List[Dict[str, Any]] = []
    for item in cases:
        case_id, raw_path = item.split("=", 1)
        payload = _read_json(_resolve(base_dir, raw_path))
        by_variant = {r.get("variant") or r.get("variant_id"): r for r in payload.get("variants", []) if isinstance(r, dict)}
        base = by_variant.get("A0_full_no_sr", {})
        for variant in wanted:
            r = by_variant.get(variant, {})
            if not r and int(args.include_missing) == 0:
                continue
            if int(args.include_missing) == 0 and not r.get("pipeline_status"):
                continue
            recall_delta = None
            runtime_delta = None
            proxy_delta = None
            try:
                if r.get("rear_person_recall") not in [None, ""] and base.get("rear_person_recall") not in [None, ""]:
                    recall_delta = float(r.get("rear_person_recall")) - float(base.get("rear_person_recall"))
            except Exception:
                recall_delta = None
            try:
                if r.get("stage_runtime_sec") not in [None, ""] and base.get("stage_runtime_sec") not in [None, ""]:
                    runtime_delta = float(r.get("stage_runtime_sec")) - float(base.get("stage_runtime_sec"))
                elif r.get("sr_elapsed_sec") not in [None, ""] and base.get("sr_elapsed_sec") not in [None, ""]:
                    runtime_delta = float(r.get("sr_elapsed_sec")) - float(base.get("sr_elapsed_sec"))
            except Exception:
                runtime_delta = None
            try:
                if r.get("rear_pose_person_rows_proxy") not in [None, ""] and base.get("rear_pose_person_rows_proxy") not in [None, ""]:
                    proxy_delta = float(r.get("rear_pose_person_rows_proxy")) - float(base.get("rear_pose_person_rows_proxy"))
            except Exception:
                proxy_delta = None
            rows.append(
                {
                    "case": case_id,
                    "variant": variant,
                    "sr_backend": r.get("sr_backend"),
                    "sr_preprocess": r.get("sr_preprocess", "off"),
                    "pipeline_status": r.get("pipeline_status"),
                    "tracked_students": r.get("tracked_students"),
                    "tracked_students_delta_pct_vs_A0": _pct(r.get("tracked_students"), base.get("tracked_students")),
                    "rear_pose_person_rows_proxy": r.get("rear_pose_person_rows_proxy"),
                    "rear_proxy_delta_pct_vs_A0": _pct(r.get("rear_pose_person_rows_proxy"), base.get("rear_pose_person_rows_proxy")),
                    "rear_avg_visible_keypoints": r.get("rear_avg_visible_keypoints"),
                    "rear_low_visible_keypoint_rate": r.get("rear_low_visible_keypoint_rate"),
                    "track_gap_count_proxy": r.get("track_gap_count_proxy"),
                    "seat_anchor_jitter_mean": r.get("seat_anchor_jitter_mean"),
                    "pose_track_gap_count_proxy": r.get("pose_track_gap_count_proxy"),
                    "pose_track_max_gap_proxy": r.get("pose_track_max_gap_proxy"),
                    "pose_track_seat_prior_mode": r.get("pose_track_seat_prior_mode"),
                    "pose_track_max_dx_ratio": r.get("pose_track_max_dx_ratio"),
                    "matched_behavior_items": r.get("matched_behavior_items"),
                    "matched_delta_pct_vs_A0": _pct(r.get("matched_behavior_items"), base.get("matched_behavior_items")),
                    "unmatched_behavior_items": r.get("unmatched_behavior_items"),
                    "unmatched_delta_pct_vs_A0": _pct(r.get("unmatched_behavior_items"), base.get("unmatched_behavior_items")),
                    "actions_fusion_v2": r.get("actions_fusion_v2"),
                    "timeline_student_rows": r.get("timeline_student_rows"),
                    "unlinked_rows_ge_200000": r.get("unlinked_rows_ge_200000"),
                    "person_recall": r.get("person_recall"),
                    "person_precision": r.get("person_precision"),
                    "person_f1": r.get("person_f1"),
                    "gt_status": r.get("gt_status"),
                    "AP50": r.get("AP50"),
                    "AP75": r.get("AP75"),
                    "mAP50_95": r.get("mAP50_95", r.get("mAP50-95")),
                    "PCK_0_10": r.get("PCK_0_1", r.get("PCK_0_10")),
                    "OKS_AP": r.get("OKS_AP", r.get("OKS-AP")),
                    "IDF1": r.get("IDF1"),
                    "IDSW": r.get("IDSW"),
                    "HOTA": r.get("HOTA"),
                    "MOTA": r.get("MOTA"),
                    "track_fragments": r.get("track_fragments"),
                    "behavior_macro_f1": r.get("behavior_macro_f1"),
                    "behavior_micro_f1": r.get("behavior_micro_f1"),
                    "segment_iou": r.get("segment_iou"),
                    "temporal_mAP_0_5": r.get("temporal_mAP_0_5"),
                    "frame_mAP_IoU0_5": r.get("frame_mAP_IoU0_5"),
                    "stage_runtime_sec": r.get("stage_runtime_sec"),
                    "effective_fps": r.get("effective_fps"),
                    "peak_vram_mb": r.get("peak_vram_mb"),
                    "sr_cache_bytes": r.get("sr_cache_bytes"),
                    "cost_gain_ratio": _ratio(recall_delta, runtime_delta),
                    "proxy_cost_gain_ratio": _ratio(proxy_delta, runtime_delta),
                    "sr_elapsed_sec": r.get("sr_elapsed_sec"),
                }
            )

    out_dir = _resolve(base_dir, args.out_dir)
    out_json = out_dir / "sr_ablation_paper_summary.json"
    out_csv = out_dir / "sr_ablation_paper_summary.csv"
    out_md = out_dir / "sr_ablation_paper_summary.md"
    out_json.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out_csv, rows)
    _write_md(out_md, rows)
    print(json.dumps({"status": "ok", "json": str(out_json), "csv": str(out_csv), "md": str(out_md), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
