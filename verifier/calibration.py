import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts.schemas import ARTIFACT_VERSION, validate_verifier_calibration_report, write_json
from verifier.metrics import (
    binary_target_from_label,
    brier_score,
    ece_and_bins,
    fit_temperature_brier,
    parse_prob,
    parse_reference_label,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate verified_events probabilities and export verifier_calibration_report.json"
    )
    parser.add_argument("--verified", required=True, type=str, help="verified_events.jsonl")
    parser.add_argument("--out", required=True, type=str, help="verifier_calibration_report.json")
    parser.add_argument("--split", default="val", type=str)
    parser.add_argument("--target_field", default="auto", type=str, help="ground-truth label field or auto")
    parser.add_argument("--prob_field", default="p_match", type=str)
    parser.add_argument("--num_bins", default=10, type=int)
    parser.add_argument("--disable_temperature_scaling", type=int, default=0)
    args = parser.parse_args()

    verified_path = Path(args.verified).resolve()
    out_path = Path(args.out).resolve()
    rows = _load_jsonl(verified_path)

    probs: List[float] = []
    targets: List[int] = []
    ref_source_counts: Dict[str, int] = {}
    for row in rows:
        label, source = parse_reference_label(row, target_field=str(args.target_field))
        target = binary_target_from_label(label)
        prob = parse_prob(row, field=str(args.prob_field))
        probs.append(prob)
        targets.append(target)
        ref_source_counts[source] = ref_source_counts.get(source, 0) + 1

    before_ece, before_bins = ece_and_bins(probs, targets, num_bins=int(args.num_bins))
    before_brier = brier_score(probs, targets)

    use_scaling = int(args.disable_temperature_scaling) != 1
    if use_scaling:
        temperature, scaled_probs = fit_temperature_brier(probs, targets)
    else:
        temperature, scaled_probs = 1.0, list(probs)

    after_ece, after_bins = ece_and_bins(scaled_probs, targets, num_bins=int(args.num_bins))
    after_brier = brier_score(scaled_probs, targets)

    report: Dict[str, Any] = {
        "split": str(args.split),
        "ece": float(after_ece),
        "brier": float(after_brier),
        "temperature": float(temperature),
        "temperature_scaling_enabled": bool(use_scaling),
        "bin_stats": after_bins,
        "before_after": {
            "before": {
                "ece": float(before_ece),
                "brier": float(before_brier),
                "temperature": 1.0,
                "bin_stats": before_bins,
            },
            "after": {
                "ece": float(after_ece),
                "brier": float(after_brier),
                "temperature": float(temperature),
                "bin_stats": after_bins,
            },
        },
        "config": {
            "prob_field": str(args.prob_field),
            "target_field": str(args.target_field),
            "num_bins": int(args.num_bins),
            "source_file": str(verified_path),
            "reference_source": ref_source_counts,
        },
        "artifact_version": ARTIFACT_VERSION,
    }

    ok, msg = validate_verifier_calibration_report(report)
    if not ok:
        raise ValueError(f"invalid calibration report payload: {msg}")
    write_json(out_path, report)
    print(f"[DONE] calibration report: {out_path}")
    print(
        f"[INFO] total={len(rows)} ece_before={before_ece:.4f} ece_after={after_ece:.4f} "
        f"brier_before={before_brier:.4f} brier_after={after_brier:.4f} T={temperature:.3f}"
    )


if __name__ == "__main__":
    main()
