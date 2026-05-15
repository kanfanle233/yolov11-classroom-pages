# A/B Compare Summary (run_full_001 vs run_full_wisdom8_001)

## Scope
- Baseline run: `output/codex_reports/run_full_001/full_integration_001`
- New run: `output/codex_reports/run_full_wisdom8_001/full_integration_001`
- New behavior detector: `runs/detect/wisdom8_yolo11s_detect_v1/weights/best.pt`

## Key counts
- `actions.jsonl`: 75 -> 75 (delta 0)
- `actions.behavior.jsonl`: 129 -> 131 (delta +2)
- `actions.behavior_aug.jsonl`: 204 -> 206 (delta +2)
- `behavior_det.jsonl`: 408 -> 408 (delta 0)
- `event_queries.jsonl`: 1 -> 1 (delta 0)
- `verified_events.jsonl`: 1 -> 1 (delta 0)

## Verifier metrics
- `f1`: 0.3333 -> 0.3333 (no change)
- `accuracy`: 1.0000 -> 1.0000 (no change)
- `total samples`: 1 -> 1

## Limitation (important)
- Both runs have ASR placeholder only:
  - `transcript.jsonl`: `[ASR_EMPTY:empty_whisper_result]`
- Because textual signal is empty, this case cannot meaningfully validate semantic-side improvements.

## Practical conclusion
- New detector improves visual action evidence volume slightly (+2 merged actions).
- Current sample does not support strong "vision-semantic dual verification" claims.
- Next recommended action: run the same pipeline on at least one classroom video with non-empty transcript.
