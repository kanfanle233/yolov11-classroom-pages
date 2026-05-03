import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contracts.schemas import validate_verifier_training_sample_record, write_jsonl
from verifier.dataset import build_negative_sampling_samples


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _validate_rows(rows: List[Dict[str, Any]]) -> None:
    for i, row in enumerate(rows):
        ok, msg = validate_verifier_training_sample_record(row)
        if not ok:
            raise ValueError(f"invalid training sample at idx={i}: {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build verifier training samples for exp-c negative sampling.")
    parser.add_argument("--event_queries", required=True, type=str)
    parser.add_argument("--aligned", required=True, type=str)
    parser.add_argument("--actions", required=True, type=str)
    parser.add_argument("--out", required=True, type=str, help="train_samples.jsonl")
    parser.add_argument("--train_ratio", default=0.8, type=float)
    parser.add_argument("--shift_1", default=1.0, type=float, help="temporal shift seconds")
    parser.add_argument("--shift_2", default=2.0, type=float, help="temporal shift seconds")
    args = parser.parse_args()

    event_queries = Path(args.event_queries).resolve()
    aligned = Path(args.aligned).resolve()
    actions = Path(args.actions).resolve()
    out = Path(args.out).resolve()

    rows = build_negative_sampling_samples(
        event_queries_path=event_queries,
        aligned_path=aligned,
        actions_path=actions,
        train_ratio=_safe_float(args.train_ratio, 0.8),
        shift_seconds=(_safe_float(args.shift_1, 1.0), _safe_float(args.shift_2, 2.0)),
    )
    if not rows:
        raise RuntimeError("no samples generated")

    _validate_rows(rows)
    write_jsonl(out, rows)

    neg_counter = Counter(str(r.get("negative_type", "unknown")) for r in rows)
    split_counter = Counter(str(r.get("split", "unknown")) for r in rows)
    print(f"[DONE] samples: {out}")
    print(f"[INFO] total={len(rows)} by_negative_type={dict(neg_counter)} by_split={dict(split_counter)}")


if __name__ == "__main__":
    main()
