from pathlib import Path
from typing import List, Tuple

from contracts.schemas import (
    validate_align_record,
    validate_event_query_record,
    validate_jsonl_file,
    validate_pose_uq_record,
    validate_verified_event_record,
)


def verify_event_queries(path: Path) -> Tuple[bool, int, List[str]]:
    return validate_jsonl_file(path, validate_event_query_record)


def verify_pose_uq(path: Path) -> Tuple[bool, int, List[str]]:
    return validate_jsonl_file(path, validate_pose_uq_record)


def verify_aligned(path: Path) -> Tuple[bool, int, List[str]]:
    return validate_jsonl_file(path, validate_align_record)


def verify_verified_events(path: Path) -> Tuple[bool, int, List[str]]:
    return validate_jsonl_file(path, validate_verified_event_record)
