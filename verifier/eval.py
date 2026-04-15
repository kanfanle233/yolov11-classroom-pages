import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts.schemas import ARTIFACT_VERSION, validate_verifier_eval_report, write_json
from verifier.metrics import (
    LABELS,
    brier_score,
    build_matrix_list,
    clamp01,
    confusion_matrix,
    ece_and_bins,
    label_distribution,
    metrics_from_confusion_matrix,
    parse_predicted_label,
    parse_reference_label,
    parse_score,
    pick_best_sweep,
    probability_consistency,
    threshold_sweep,
)


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


def _build_threshold_values(start: float, end: float, step: float) -> List[float]:
    if step <= 0:
        step = 0.05
    if end < start:
        start, end = end, start
    values: List[float] = []
    cur = float(start)
    while cur <= end + 1e-9:
        values.append(round(cur, 4))
        cur += step
    if not values:
        values = [0.60]
    return values


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def compute_auroc_binary(scores: Sequence[float], labels: Sequence[int]) -> float:
    n = min(len(scores), len(labels))
    if n <= 1:
        return 0.0
    paired = [(float(scores[i]), int(labels[i])) for i in range(n)]
    pos = sum(1 for _, y in paired if y == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return 0.0

    paired.sort(key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    rank = 1.0
    while i < n:
        j = i + 1
        while j < n and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            if paired[k][1] == 1:
                rank_sum_pos += avg_rank
        rank += (j - i)
        i = j
    return _safe_div(rank_sum_pos - (pos * (pos + 1) / 2.0), pos * neg)


def compute_binary_metrics(
    *,
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float = 0.5,
    num_bins: int = 10,
) -> Dict[str, float]:
    n = min(len(scores), len(labels))
    tp = fp = tn = fn = 0
    clipped_scores: List[float] = []
    clipped_labels: List[int] = []
    for i in range(n):
        s = clamp01(scores[i], default=0.0)
        y = 1 if int(labels[i]) == 1 else 0
        pred = 1 if s >= threshold else 0
        clipped_scores.append(s)
        clipped_labels.append(y)
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        else:
            fn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    ece, _ = ece_and_bins(clipped_scores, clipped_labels, num_bins=max(2, int(num_bins)))
    brier = brier_score(clipped_scores, clipped_labels)
    auroc = compute_auroc_binary(clipped_scores, clipped_labels)
    return {
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "AUROC": float(auroc),
        "ECE": float(ece),
        "Brier": float(brier),
        "sample_count": float(n),
    }


def compute_multiclass_macro_metrics(
    *,
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> Dict[str, float]:
    n = min(len(y_true), len(y_pred))
    if n <= 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "sample_count": 0.0}
    labels = sorted({str(y_true[i]) for i in range(n)} | {str(y_pred[i]) for i in range(n)})
    if not labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "sample_count": float(n)}
    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0
    for label in labels:
        tp = fp = fn = 0
        for i in range(n):
            yt = str(y_true[i])
            yp = str(y_pred[i])
            if yp == label and yt == label:
                tp += 1
            elif yp == label and yt != label:
                fp += 1
            elif yp != label and yt == label:
                fn += 1
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * p * r, p + r)
        macro_p += p
        macro_r += r
        macro_f1 += f1
    denom = float(len(labels))
    return {
        "precision": float(macro_p / denom),
        "recall": float(macro_r / denom),
        "f1": float(macro_f1 / denom),
        "sample_count": float(n),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate verified_events.jsonl and export verifier_eval_report.json")
    parser.add_argument("--verified", required=True, type=str, help="verified_events.jsonl")
    parser.add_argument("--out", required=True, type=str, help="verifier_eval_report.json")
    parser.add_argument("--split", default="val", type=str)
    parser.add_argument("--target_field", default="auto", type=str, help="ground-truth label field or auto")
    parser.add_argument("--score_field", default="reliability_score", type=str)
    parser.add_argument("--uncertain_margin", default=0.20, type=float)
    parser.add_argument("--sweep_start", default=0.30, type=float)
    parser.add_argument("--sweep_end", default=0.90, type=float)
    parser.add_argument("--sweep_step", default=0.05, type=float)
    args = parser.parse_args()

    verified_path = Path(args.verified).resolve()
    out_path = Path(args.out).resolve()
    rows = _load_jsonl(verified_path)

    y_true: List[str] = []
    y_pred: List[str] = []
    scores: List[float] = []
    ref_source_counts: Dict[str, int] = {}
    for row in rows:
        target_label, source = parse_reference_label(row, target_field=str(args.target_field))
        pred_label = parse_predicted_label(row)
        score = parse_score(row, score_field=str(args.score_field))
        y_true.append(target_label)
        y_pred.append(pred_label)
        scores.append(score)
        ref_source_counts[source] = ref_source_counts.get(source, 0) + 1

    cm = confusion_matrix(y_true, y_pred)
    metrics = metrics_from_confusion_matrix(cm)
    threshold_values = _build_threshold_values(float(args.sweep_start), float(args.sweep_end), float(args.sweep_step))
    sweep = threshold_sweep(
        y_true=y_true,
        scores=scores,
        thresholds=threshold_values,
        uncertain_margin=float(args.uncertain_margin),
    )
    best = pick_best_sweep(sweep, key="f1")
    prob_check = probability_consistency(rows)
    reference_quality = "weak_self_labels" if set(ref_source_counts.keys()) == {"self_label_fallback"} else "explicit_or_external_labels"

    report: Dict[str, Any] = {
        "split": str(args.split),
        "counts": {
            "total": len(rows),
            "reference_source": ref_source_counts,
            "reference_quality": reference_quality,
            "probability_consistency": prob_check,
        },
        "metrics": {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_label": metrics["per_label"],
        },
        "confusion_matrix": {
            "labels": list(LABELS),
            "matrix": build_matrix_list(cm),
            "as_dict": cm,
        },
        "threshold_sweep": sweep,
        "label_distribution": {
            "reference": label_distribution(y_true),
            "predicted": label_distribution(y_pred),
        },
        "config": {
            "score_field": str(args.score_field),
            "target_field": str(args.target_field),
            "uncertain_margin": float(args.uncertain_margin),
            "threshold_source": "val_f1_best",
            "best_threshold": {
                "match_threshold": float(best.get("match_threshold", 0.60)),
                "uncertain_threshold": float(best.get("uncertain_threshold", 0.40)),
                "f1": float(best.get("f1", 0.0)),
            },
            "source_file": str(verified_path),
        },
        "summary": {
            "best_threshold": {
                "match_threshold": float(best.get("match_threshold", 0.60)),
                "uncertain_threshold": float(best.get("uncertain_threshold", 0.40)),
                "f1": float(best.get("f1", 0.0)),
                "accuracy": float(best.get("accuracy", 0.0)),
            },
            "predicted_distribution": label_distribution(y_pred),
            "reference_distribution": label_distribution(y_true),
            "warning": (
                "metrics are based on self_label_fallback and are not a substitute for held-out ground truth"
                if reference_quality == "weak_self_labels"
                else ""
            ),
        },
        "artifact_version": ARTIFACT_VERSION,
    }

    ok, msg = validate_verifier_eval_report(report)
    if not ok:
        raise ValueError(f"invalid eval report payload: {msg}")
    write_json(out_path, report)
    print(f"[DONE] eval report: {out_path}")
    print(f"[INFO] total={len(rows)} f1={report['metrics']['f1']:.4f} e_prob={prob_check['mean_abs_error']:.4f}")
    if reference_quality == "weak_self_labels":
        print("[WARN] evaluation used self_label_fallback; metrics are weak proxies, not ground-truth evaluation")


if __name__ == "__main__":
    main()
