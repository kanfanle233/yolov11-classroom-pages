import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts.schemas import (
    validate_event_query_record,
    validate_jsonl_file,
    validate_pose_uq_record,
    validate_verified_event_record,
)


def _report(title: str, ok: bool, count: int, errors) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {title}: {count} rows")
    if errors:
        for e in errors[:5]:
            print(f"  - {e}")
        if len(errors) > 5:
            print(f"  ... {len(errors) - 5} more")
    return ok


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate fixed-contract JSONL outputs.")
    parser.add_argument("--event_queries", type=str, default="")
    parser.add_argument("--pose_uq", type=str, default="")
    parser.add_argument("--verified_events", type=str, default="")
    args = parser.parse_args()

    event_path = Path(args.event_queries) if args.event_queries else (base_dir / "output" / "event_queries.jsonl")
    pose_uq_path = Path(args.pose_uq) if args.pose_uq else (base_dir / "output" / "pose_tracks_smooth_uq.jsonl")
    verified_path = Path(args.verified_events) if args.verified_events else (base_dir / "output" / "verified_events.jsonl")

    if not event_path.is_absolute():
        event_path = (base_dir / event_path).resolve()
    if not pose_uq_path.is_absolute():
        pose_uq_path = (base_dir / pose_uq_path).resolve()
    if not verified_path.is_absolute():
        verified_path = (base_dir / verified_path).resolve()

    ok_event, n_event, err_event = validate_jsonl_file(event_path, validate_event_query_record)
    ok_pose, n_pose, err_pose = validate_jsonl_file(pose_uq_path, validate_pose_uq_record)
    ok_ver, n_ver, err_ver = validate_jsonl_file(verified_path, validate_verified_event_record)

    all_ok = True
    all_ok &= _report("event_queries", ok_event, n_event, err_event)
    all_ok &= _report("pose_tracks_smooth_uq", ok_pose, n_pose, err_pose)
    all_ok &= _report("verified_events", ok_ver, n_ver, err_ver)

    if not all_ok:
        raise SystemExit(1)
    print("[DONE] all schema checks passed.")


if __name__ == "__main__":
    main()
