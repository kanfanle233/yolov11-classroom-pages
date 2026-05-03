import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts.schemas import (
    ARTIFACT_VERSION,
    SCHEMA_VERSION,
    validate_verifier_calibration_report,
    write_json,
)
from verifier.metrics import apply_temperature, brier_score, ece_and_bins, fit_temperature_brier
from verifier.model import action_match_score


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
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


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return float(default)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _normalize_binary_label(value: Any) -> str:
    if isinstance(value, bool):
        return "match" if value else "mismatch"
    if isinstance(value, (int, float)):
        return "match" if float(value) >= 0.5 else "mismatch"
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"match", "matched", "positive", "pos", "1", "true"}:
            return "match"
        if v in {"mismatch", "negative", "neg", "0", "false", "not_match", "non_match"}:
            return "mismatch"
    return "mismatch"


def _build_action_index(aligned_rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in aligned_rows:
        event_id = str(row.get("event_id", row.get("query_id", "")))
        if not event_id:
            continue
        candidates = row.get("candidates", [])
        if not isinstance(candidates, list):
            continue
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            track_id = cand.get("track_id")
            if not isinstance(track_id, int):
                continue
            overlap = _clamp01(cand.get("overlap", 0.0), 0.0)
            key = (event_id, track_id)
            prev = index.get(key)
            prev_overlap = _clamp01(prev.get("overlap", 0.0), 0.0) if isinstance(prev, dict) else -1.0
            if overlap >= prev_overlap:
                index[key] = {
                    "action": str(cand.get("action", "")).strip().lower(),
                    "overlap": overlap,
                    "uq_track": _clamp01(cand.get("uq_track", cand.get("uq_score", 0.5)), 0.5),
                }
    return index


def _load_reference_labels(path: Path) -> Dict[Tuple[str, int], str]:
    labels: Dict[Tuple[str, int], str] = {}
    for row in _load_jsonl(path):
        event_id = str(row.get("event_id", row.get("query_id", ""))).strip()
        track_id = row.get("track_id")
        if not event_id or not isinstance(track_id, int):
            continue
        labels[(event_id, int(track_id))] = _normalize_binary_label(
            row.get("target_label", row.get("label", row.get("target", "mismatch")))
        )
    return labels


def _binary_metrics(scores: Sequence[float], targets: Sequence[int], threshold: float) -> Dict[str, float]:
    n = min(len(scores), len(targets))
    tp = fp = tn = fn = 0
    for i in range(n):
        pred = 1 if _clamp01(scores[i], 0.0) >= threshold else 0
        true = 1 if int(targets[i]) == 1 else 0
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 0:
            tn += 1
        else:
            fn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def _method_report(
    *,
    scores: Sequence[float],
    targets: Sequence[int],
    uq_scores: Sequence[float],
    threshold: float,
    num_bins: int,
    high_uq_threshold: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    base = _binary_metrics(scores, targets, threshold)
    ece, bins = ece_and_bins(scores, targets, num_bins=num_bins)
    brier = brier_score(scores, targets)

    high_scores: List[float] = []
    high_targets: List[int] = []
    for score, target, uq in zip(scores, targets, uq_scores):
        if _clamp01(uq, 0.0) >= high_uq_threshold:
            high_scores.append(_clamp01(score, 0.0))
            high_targets.append(1 if int(target) == 1 else 0)

    high_metrics = _binary_metrics(high_scores, high_targets, threshold) if high_scores else {
        "F1": 0.0,
        "false_positive_rate": 0.0,
    }

    report = {
        "Precision": round(base["Precision"], 6),
        "Recall": round(base["Recall"], 6),
        "F1": round(base["F1"], 6),
        "false_positive_rate": round(base["false_positive_rate"], 6),
        "false_negative_rate": round(base["false_negative_rate"], 6),
        "ECE": round(float(ece), 6),
        "Brier": round(float(brier), 6),
        "high_uq_subset_f1": round(float(high_metrics["F1"]), 6),
        "high_uq_subset_false_positive_rate": round(float(high_metrics["false_positive_rate"]), 6),
        "high_uq_subset_count": int(len(high_scores)),
        "sample_count": int(min(len(scores), len(targets))),
    }
    return report, bins


def _uq_bucket_name(uq: float, low_uq_threshold: float, high_uq_threshold: float) -> str:
    if uq < low_uq_threshold:
        return "low_uq"
    if uq < high_uq_threshold:
        return "mid_uq"
    return "high_uq"


def _write_svg(path: Path, bins: Sequence[Dict[str, Any]], title: str, color: str) -> None:
    width = 640
    height = 420
    margin_left = 70
    margin_right = 24
    margin_top = 36
    margin_bottom = 56
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def px_x(value: float) -> float:
        return margin_left + _clamp01(value, 0.0) * plot_w

    def px_y(value: float) -> float:
        return margin_top + (1.0 - _clamp01(value, 0.0)) * plot_h

    grid: List[str] = []
    ticks: List[str] = []
    for tick in range(6):
        v = tick / 5.0
        x = px_x(v)
        y = px_y(v)
        grid.append(
            f"<line x1='{x:.2f}' y1='{margin_top}' x2='{x:.2f}' y2='{height - margin_bottom}' "
            "stroke='#d9dde3' stroke-width='1' />"
        )
        grid.append(
            f"<line x1='{margin_left}' y1='{y:.2f}' x2='{width - margin_right}' y2='{y:.2f}' "
            "stroke='#d9dde3' stroke-width='1' />"
        )
        ticks.append(
            f"<text x='{x:.2f}' y='{height - margin_bottom + 22}' font-size='12' text-anchor='middle' "
            f"fill='#344054'>{v:.1f}</text>"
        )
        ticks.append(
            f"<text x='{margin_left - 14}' y='{y + 4:.2f}' font-size='12' text-anchor='end' "
            f"fill='#344054'>{v:.1f}</text>"
        )

    points: List[str] = []
    for row in bins:
        count = max(0, int(row.get("count", 0)))
        if count <= 0:
            continue
        conf = _clamp01(row.get("conf", 0.0), 0.0)
        acc = _clamp01(row.get("acc", 0.0), 0.0)
        radius = 3.0 + min(8.0, count / 25.0)
        points.append(
            f"<circle cx='{px_x(conf):.2f}' cy='{px_y(acc):.2f}' r='{radius:.2f}' "
            f"fill='{color}' fill-opacity='0.8' stroke='white' stroke-width='1' />"
        )

    diagonal = (
        f"<line x1='{px_x(0.0):.2f}' y1='{px_y(0.0):.2f}' "
        f"x2='{px_x(1.0):.2f}' y2='{px_y(1.0):.2f}' stroke='#98a2b3' "
        "stroke-width='2' stroke-dasharray='8 6' />"
    )
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>
  <rect width='{width}' height='{height}' fill='#f8fafc' />
  <text x='{margin_left}' y='22' font-size='18' font-weight='700' fill='#0f172a'>{title}</text>
  <text x='{margin_left}' y='{height - 16}' font-size='14' fill='#475467'>Predicted confidence</text>
  <text x='18' y='{margin_top + plot_h / 2:.2f}' font-size='14' fill='#475467'
        transform='rotate(-90 18 {margin_top + plot_h / 2:.2f})'>Empirical accuracy</text>
  {''.join(grid)}
  <rect x='{margin_left}' y='{margin_top}' width='{plot_w}' height='{plot_h}' fill='none' stroke='#101828' stroke-width='2' />
  {diagonal}
  {''.join(points)}
  {''.join(ticks)}
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def _write_summary(
    path: Path,
    *,
    method_metrics: Dict[str, Dict[str, Any]],
    temperature: float,
    label_sources: Dict[str, int],
    non_final_note: str,
) -> None:
    no_gate = method_metrics["no_uq_gate"]
    uq_gate = method_metrics["uq_gate"]
    calibrated = method_metrics["calibrated_uq_gate"]
    lines = [
        "# Exp-B Reliability Calibration (Smoke)",
        "",
        f"- label_sources: {label_sources}",
        f"- calibration_temperature: {temperature:.4f}",
        f"- no_uq_gate F1: {no_gate['F1']}",
        f"- uq_gate F1: {uq_gate['F1']}",
        f"- calibrated_uq_gate F1: {calibrated['F1']}",
        f"- no_uq_gate high_uq_subset_false_positive_rate: {no_gate['high_uq_subset_false_positive_rate']}",
        f"- uq_gate high_uq_subset_false_positive_rate: {uq_gate['high_uq_subset_false_positive_rate']}",
        f"- calibrated_uq_gate ECE: {calibrated['ECE']}",
        f"- calibrated_uq_gate Brier: {calibrated['Brier']}",
        "",
        "## Notes",
        "- This run is a smoke experiment for pipeline and metric packaging.",
        f"- {non_final_note}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate UQ-gated reliability and calibration for exp-b.")
    parser.add_argument("--verified", required=True, type=str)
    parser.add_argument("--aligned", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--reference_labels", default="", type=str)
    parser.add_argument("--score_threshold", default=0.60, type=float)
    parser.add_argument("--low_uq_threshold", default=0.40, type=float)
    parser.add_argument("--high_uq_threshold", default=0.55, type=float)
    parser.add_argument("--num_bins", default=10, type=int)
    parser.add_argument("--pseudo_match_threshold", default=0.80, type=float)
    parser.add_argument("--split", default="smoke", type=str)
    args = parser.parse_args()

    verified_path = Path(args.verified).resolve()
    aligned_path = Path(args.aligned).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reliability_bins_csv = out_dir / "reliability_bins.csv"
    calibration_report_path = out_dir / "calibration_report.json"
    reliability_before_svg = out_dir / "reliability_before.svg"
    reliability_after_svg = out_dir / "reliability_after.svg"
    metrics_raw_path = out_dir / "reliability_metrics_raw.json"
    summary_path = out_dir / "summary.md"

    verified_rows = _load_jsonl(verified_path)
    aligned_obj = _load_json(aligned_path)
    aligned_rows = aligned_obj if isinstance(aligned_obj, list) else []
    action_index = _build_action_index(aligned_rows)

    reference_labels: Dict[Tuple[str, int], str] = {}
    if args.reference_labels:
        reference_labels = _load_reference_labels(Path(args.reference_labels).resolve())

    scores_no_gate: List[float] = []
    scores_uq_gate: List[float] = []
    uq_scores: List[float] = []
    targets: List[int] = []
    label_sources: Counter[str] = Counter()

    for row in verified_rows:
        event_id = str(row.get("event_id", row.get("query_id", ""))).strip()
        track_id = _safe_int(row.get("track_id", -1), -1)
        if not event_id or track_id < 0:
            continue

        target_label = reference_labels.get((event_id, track_id))
        source = "reference_labels_file"
        if target_label is None:
            aligned_action = action_index.get((event_id, track_id), {}).get("action", "")
            event_type = str(row.get("event_type", "")).strip()
            query_text = str(row.get("query_text", "")).strip()
            semantic = action_match_score(event_type, query_text, str(aligned_action))
            target_label = "match" if semantic >= float(args.pseudo_match_threshold) else "mismatch"
            source = "pseudo_action_alias"

        evidence = row.get("evidence", {})
        uq_score = _clamp01(
            evidence.get("uq_score", action_index.get((event_id, track_id), {}).get("uq_track", row.get("uncertainty", 0.5))),
            0.5,
        )
        p_match = _clamp01(row.get("p_match", 0.0), 0.0)
        reliability = _clamp01(row.get("reliability_score", 0.0), 0.0)

        scores_no_gate.append(p_match)
        scores_uq_gate.append(reliability)
        uq_scores.append(uq_score)
        targets.append(1 if target_label == "match" else 0)
        label_sources[source] += 1

    threshold = _clamp01(args.score_threshold, 0.60)
    low_uq_threshold = _clamp01(args.low_uq_threshold, 0.40)
    high_uq_threshold = _clamp01(args.high_uq_threshold, 0.55)
    num_bins = max(2, int(args.num_bins))

    temperature, calibrated_scores = fit_temperature_brier(scores_uq_gate, targets)

    method_metrics: Dict[str, Dict[str, Any]] = {}
    method_bins: Dict[str, List[Dict[str, Any]]] = {}

    for method_name, method_scores in (
        ("no_uq_gate", scores_no_gate),
        ("uq_gate", scores_uq_gate),
        ("calibrated_uq_gate", calibrated_scores),
    ):
        report, bins = _method_report(
            scores=method_scores,
            targets=targets,
            uq_scores=uq_scores,
            threshold=threshold,
            num_bins=num_bins,
            high_uq_threshold=high_uq_threshold,
        )
        method_metrics[method_name] = report
        method_bins[method_name] = bins

    # UQ bucket analysis for low/mid/high.
    bucket_rows: List[Dict[str, Any]] = []
    for method_name, method_scores in (
        ("no_uq_gate", scores_no_gate),
        ("uq_gate", scores_uq_gate),
        ("calibrated_uq_gate", calibrated_scores),
    ):
        bucket_scores: Dict[str, List[float]] = {"low_uq": [], "mid_uq": [], "high_uq": []}
        bucket_targets: Dict[str, List[int]] = {"low_uq": [], "mid_uq": [], "high_uq": []}
        for score, target, uq in zip(method_scores, targets, uq_scores):
            b = _uq_bucket_name(_clamp01(uq, 0.0), low_uq_threshold, high_uq_threshold)
            bucket_scores[b].append(_clamp01(score, 0.0))
            bucket_targets[b].append(1 if int(target) == 1 else 0)

        for bucket in ("low_uq", "mid_uq", "high_uq"):
            scores_subset = bucket_scores[bucket]
            targets_subset = bucket_targets[bucket]
            base = _binary_metrics(scores_subset, targets_subset, threshold)
            ece_subset, _ = ece_and_bins(scores_subset, targets_subset, num_bins=max(2, min(5, num_bins)))
            brier_subset = brier_score(scores_subset, targets_subset)
            mean_score = _safe_div(sum(scores_subset), len(scores_subset))
            bucket_rows.append(
                {
                    "method": method_name,
                    "uq_bin": bucket,
                    "count": len(scores_subset),
                    "Precision": round(base["Precision"], 6),
                    "Recall": round(base["Recall"], 6),
                    "F1": round(base["F1"], 6),
                    "false_positive_rate": round(base["false_positive_rate"], 6),
                    "false_negative_rate": round(base["false_negative_rate"], 6),
                    "ECE": round(float(ece_subset), 6),
                    "Brier": round(float(brier_subset), 6),
                    "mean_score": round(float(mean_score), 6),
                }
            )

    with reliability_bins_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "uq_bin",
                "count",
                "Precision",
                "Recall",
                "F1",
                "false_positive_rate",
                "false_negative_rate",
                "ECE",
                "Brier",
                "mean_score",
            ],
        )
        writer.writeheader()
        writer.writerows(bucket_rows)

    calibration_report: Dict[str, Any] = {
        "split": str(args.split),
        "ece": float(method_metrics["calibrated_uq_gate"]["ECE"]),
        "brier": float(method_metrics["calibrated_uq_gate"]["Brier"]),
        "temperature": float(temperature),
        "temperature_scaling_enabled": True,
        "bin_stats": method_bins["calibrated_uq_gate"],
        "before_after": {
            "before": {
                "ece": float(method_metrics["uq_gate"]["ECE"]),
                "brier": float(method_metrics["uq_gate"]["Brier"]),
                "temperature": 1.0,
                "bin_stats": method_bins["uq_gate"],
            },
            "after": {
                "ece": float(method_metrics["calibrated_uq_gate"]["ECE"]),
                "brier": float(method_metrics["calibrated_uq_gate"]["Brier"]),
                "temperature": float(temperature),
                "bin_stats": method_bins["calibrated_uq_gate"],
            },
        },
        "config": {
            "verified": str(verified_path),
            "aligned": str(aligned_path),
            "reference_labels": str(Path(args.reference_labels).resolve()) if args.reference_labels else "",
            "score_threshold": threshold,
            "low_uq_threshold": low_uq_threshold,
            "high_uq_threshold": high_uq_threshold,
            "num_bins": num_bins,
            "label_source_counts": dict(label_sources),
        },
        "summary": {
            "num_samples": int(len(targets)),
            "ece_delta": round(
                float(method_metrics["calibrated_uq_gate"]["ECE"]) - float(method_metrics["uq_gate"]["ECE"]), 6
            ),
            "brier_delta": round(
                float(method_metrics["calibrated_uq_gate"]["Brier"]) - float(method_metrics["uq_gate"]["Brier"]), 6
            ),
            "warning": (
                "uses pseudo_action_alias labels; non-final paper result"
                if "pseudo_action_alias" in label_sources
                else "uses sample reference labels; non-final paper result"
            ),
        },
        "artifact_version": ARTIFACT_VERSION,
    }
    ok, msg = validate_verifier_calibration_report(calibration_report)
    if not ok:
        raise ValueError(f"invalid calibration_report payload: {msg}")
    write_json(calibration_report_path, calibration_report)

    _write_svg(reliability_before_svg, method_bins["uq_gate"], "Reliability Before Calibration", "#f97316")
    _write_svg(reliability_after_svg, method_bins["calibrated_uq_gate"], "Reliability After Calibration", "#2563eb")

    non_final_note = (
        "Non-final paper result: pseudo labels inferred from event-action alias rules."
        if "pseudo_action_alias" in label_sources
        else "Non-final paper result: sample reference labels only, not real held-out annotations."
    )
    _write_summary(
        summary_path,
        method_metrics=method_metrics,
        temperature=float(temperature),
        label_sources=dict(label_sources),
        non_final_note=non_final_note,
    )

    raw_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "method_metrics": method_metrics,
        "temperature": round(float(temperature), 6),
        "sample_count": int(len(targets)),
        "label_source_counts": dict(label_sources),
        "thresholds": {
            "score_threshold": threshold,
            "low_uq_threshold": low_uq_threshold,
            "high_uq_threshold": high_uq_threshold,
        },
        "warning": non_final_note,
        "outputs": {
            "reliability_bins_csv": str(reliability_bins_csv),
            "calibration_report_json": str(calibration_report_path),
            "reliability_before_svg": str(reliability_before_svg),
            "reliability_after_svg": str(reliability_after_svg),
            "summary_md": str(summary_path),
        },
    }
    write_json(metrics_raw_path, raw_payload)
    print(f"[DONE] reliability metrics: {metrics_raw_path}")
    print(f"[DONE] reliability bins: {reliability_bins_csv}")
    print(f"[DONE] calibration report: {calibration_report_path}")
    print(f"[DONE] reliability plots: {reliability_before_svg}, {reliability_after_svg}")


if __name__ == "__main__":
    main()
