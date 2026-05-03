from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path


def _repo_root() -> Path:
    for p in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents):
        if (p / "data").exists() and (p / "scripts").exists():
            return p
    raise RuntimeError("repo root not found")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    repo = _repo_root()
    py = sys.executable
    fusion_dir = repo / "codex_reports" / "smart_classroom_yolo_feasibility" / "scripts" / "50_fusion_contract"
    taxonomy = repo / "codex_reports" / "smart_classroom_yolo_feasibility" / "profiles" / "action_semantics_8class.yaml"

    tmp_parent = repo / "output" / "_tmp_fusion_contract_tests"
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_parent / f"fusion_contract_v2_{uuid.uuid4().hex}"
    tmp.mkdir(parents=True, exist_ok=True)
    behavior = tmp / "behavior_det.jsonl"
    behavior_sem = tmp / "behavior_det.semantic.jsonl"
    actions_sem = tmp / "actions.behavior.semantic.jsonl"
    actions_fusion = tmp / "actions.fusion_v2.jsonl"
    asr_events = tmp / "event_queries.jsonl"
    events = tmp / "event_queries.fusion_v2.jsonl"
    visual_events = tmp / "event_queries.visual_fallback.jsonl"
    align = tmp / "align_multimodal.json"
    verified = tmp / "verified_events.jsonl"
    report = tmp / "fusion_contract_report.json"

    _write_jsonl(
        behavior,
        [
            {
                "frame": 0,
                "t": 0.0,
                "behaviors": [
                    {"label": "jz", "action": "listen", "conf": 0.9, "bbox": [0, 0, 10, 10]},
                    {"label": "zt", "action": "distract", "conf": 0.8, "bbox": [20, 0, 30, 10]},
                    {"label": "dx", "action": "note", "conf": 0.7, "bbox": [40, 0, 50, 10]},
                ],
            }
        ],
    )

    subprocess.run(
        [
            py,
            str(fusion_dir / "semanticize_behavior_det.py"),
            "--in",
            str(behavior),
            "--out",
            str(behavior_sem),
            "--taxonomy",
            str(taxonomy),
        ],
        check=True,
    )
    sem = _read_jsonl(behavior_sem)[0]["behaviors"]
    assert sem[0]["semantic_id"] == "teacher_interaction", sem[0]
    assert sem[1]["semantic_id"] == "turn_head", sem[1]
    assert sem[2]["semantic_id"] == "write", sem[2]

    subprocess.run(
        [
            py,
            str(fusion_dir / "behavior_det_to_actions_v2.py"),
            "--in",
            str(behavior_sem),
            "--out",
            str(actions_sem),
            "--fps",
            "25",
        ],
        check=True,
    )
    action_rows = _read_jsonl(actions_sem)
    assert {row["semantic_id"] for row in action_rows} == {"teacher_interaction", "turn_head", "write"}

    _write_jsonl(tmp / "actions.jsonl", [{"track_id": 1, "action": "listen", "conf": 0.5, "start_time": 0.0, "end_time": 0.2}])
    subprocess.run(
        [
            py,
            str(fusion_dir / "merge_fusion_actions_v2.py"),
            "--actions",
            str(tmp / "actions.jsonl"),
            "--behavior_actions",
            str(actions_sem),
            "--out",
            str(actions_fusion),
            "--taxonomy",
            str(taxonomy),
        ],
        check=True,
    )
    fusion_rows = _read_jsonl(actions_fusion)
    assert all(row.get("semantic_id") and row.get("behavior_code") for row in fusion_rows)

    _write_jsonl(
        asr_events,
        [
            {
                "event_id": "e_asr_empty",
                "query_id": "e_asr_empty",
                "schema_version": "2026-04-01",
                "event_type": "unknown",
                "query_text": "[ASR_EMPTY:empty_whisper_result]",
                "trigger_text": "[ASR_EMPTY]",
                "trigger_words": [],
                "timestamp": 0.1,
                "t_center": 0.1,
                "start": 0.0,
                "end": 0.2,
                "confidence": 0.1,
                "source": "asr",
            }
        ],
    )
    subprocess.run(
        [
            py,
            str(fusion_dir / "build_event_queries_fusion_v2.py"),
            "--event_queries",
            str(asr_events),
            "--actions",
            str(actions_fusion),
            "--out",
            str(events),
            "--visual_out",
            str(visual_events),
            "--min_asr_queries",
            "2",
            "--visual_topk",
            "3",
        ],
        check=True,
    )
    assert len(_read_jsonl(visual_events)) > 0, "ASR_EMPTY must produce visual fallback events"
    align.write_text(json.dumps([{"event_id": "e0", "query_text": "teacher interaction", "event_type": "teacher_interaction", "candidates": [{"track_id": 100001}]}]), encoding="utf-8")
    _write_jsonl(verified, [{"event_id": "e0", "track_id": 100001}])
    subprocess.run(
        [
            py,
            str(fusion_dir / "check_fusion_contract.py"),
            "--output_dir",
            str(tmp),
            "--report",
            str(report),
            "--strict",
            "1",
        ],
        check=True,
    )

    bad = tmp / "bad_actions.jsonl"
    row = dict(fusion_rows[0])
    row.pop("semantic_id", None)
    _write_jsonl(bad, [row])
    actions_fusion.write_text(bad.read_text(encoding="utf-8"), encoding="utf-8")
    failed = subprocess.run(
        [
            py,
            str(fusion_dir / "check_fusion_contract.py"),
            "--output_dir",
            str(tmp),
            "--strict",
            "1",
        ],
        check=False,
    )
    assert failed.returncode != 0, "contract check must fail when semantic_id is missing"

    print("[DONE] fusion_contract_v2 tests passed")


if __name__ == "__main__":
    main()
