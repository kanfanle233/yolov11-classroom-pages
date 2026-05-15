# YOLO Pose + Object/Behavior Fusion Interface Audit (2026-04-25)

## Current Chain
The current full pipeline in `scripts/09_run_pipeline.py` is:

1. `02_export_keypoints_jsonl.py` -> `pose_keypoints_v2.jsonl`
2. `03_track_and_smooth.py` -> `pose_tracks_smooth.jsonl`
3. `03c_estimate_track_uncertainty.py` -> `pose_tracks_smooth_uq.jsonl`
4. optional object evidence: `02b_export_objects_jsonl.py` -> `objects.jsonl`
5. pose/rule action recognition: `05_slowfast_actions.py` -> `actions.jsonl` or `actions.raw.jsonl`
6. optional object fusion: `05b_fuse_actions_with_objects.py` -> `actions.jsonl`
7. behavior detector: `02d_export_behavior_det_jsonl.py` -> `behavior_det.jsonl`
8. behavior-to-action conversion: `05c_behavior_det_to_actions.py` -> `actions.behavior.jsonl`
9. action merge: `05d_merge_action_sources.py` -> `actions.behavior_aug.jsonl`
10. ASR/event extraction -> `transcript.jsonl`, `event_queries.jsonl`
11. alignment: `xx_align_multimodal.py` consumes `actions_for_downstream`
12. verifier: `07_dual_verification.py` -> `verified_events.jsonl`, `per_person_sequences.json`
13. semantic backfill currently runs after the full pipeline -> `*.semantic.*`

## Observed Output Health
For `output/codex_reports/run_full_e150_001/full_integration_001`:

- `behavior_det.jsonl`: 408 rows, non-empty.
- `actions.behavior.jsonl`: 137 rows, non-empty.
- `actions.behavior_aug.jsonl`: 212 rows, non-empty.
- `align_multimodal.json`: non-empty.
- `verified_events.jsonl`: 1 row.
- `*.semantic.*`: present and checked.

The full chain is not failing structurally, but the text side is weak in this sample because Whisper produced an empty transcript placeholder, so downstream verification only has one generic event query.

## Main Interface Risks
1. `02d_export_behavior_det_jsonl.py` still maps behavior labels to old action aliases:
   - `zt -> distract`
   - `xt -> chat`
   - `jz -> listen`
   - `dx -> note`

2. `05c_behavior_det_to_actions.py` only keeps `action`, `conf`, time fields and track fields. It loses the original YOLO class code (`tt/dx/dk/zt/xt/js/zl/jz`).

3. Because `jz` is converted to `listen`, once it enters `actions.behavior.jsonl`, it cannot be reliably recovered as `teacher_interaction`.

4. `semantic_bridge` currently runs after the full pipeline when called by the codex orchestrator. That is useful for paper assets and backfill, but it does not help `xx_align_multimodal.py` or `07_dual_verification.py` during the same run unless semantic fields are already present upstream.

5. The current `full_integration_001.yaml` and `action_semantics_8class.yaml` contain garbled Chinese text in some fields. The English semantic IDs still work, but Chinese labels for paper/timeline display should be repaired to UTF-8 Chinese.

6. `check_outputs.py` only checks file existence and non-zero size. It does not prove row counts, candidate counts, semantic coverage, or that `verified_events.jsonl` is meaningful rather than a single placeholder row.

## Recommendation
Use a two-layer integration, not post-hoc semantic repair only.

Layer 1 should be an upstream semantic carrier:

- Modify `02d_export_behavior_det_jsonl.py` so every detection writes:
  - `behavior_code`
  - `behavior_label_zh`
  - `behavior_label_en`
  - `semantic_id`
  - `semantic_label_zh`
  - `semantic_label_en`
  - `taxonomy_version`
- Keep the old `label` and `action` fields for compatibility.
- Set canonical mappings directly:
  - `tt -> listen`
  - `dx -> write`
  - `dk -> read`
  - `zt -> turn_head`
  - `xt -> group_discussion`
  - `js -> raise_hand`
  - `zl -> stand`
  - `jz -> teacher_interaction`

Layer 2 should preserve those fields through the chain:

- Modify `05c_behavior_det_to_actions.py` to propagate semantic fields from each detection into `actions.behavior.jsonl`.
- Modify `05d_merge_action_sources.py` to preserve semantic fields during merge.
- Ensure `actions_for_downstream` points to a semantic-aware action file before step 66 alignment.
- Keep `semantic_bridge.py` as a backfill/check/report stage, not as the only source of semantics.

## Suggested Stable Runtime Contract
The downstream action rows should always contain at least:

- `track_id`
- `action`
- `semantic_id`
- `behavior_code`
- `conf`
- `start_time`
- `end_time`
- `start_frame`
- `end_frame`
- `source`

`xx_align_multimodal.py`, `verifier/infer.py`, `07_dual_verification.py`, and `10_visualize_timeline.py` already prefer `semantic_id` when present, so the key remaining work is to guarantee it exists before alignment.

## Zero-Result Prevention Gates
Add a strict contract check between every major boundary:

1. After `behavior_det.jsonl`: require row count > 0 and behavior item count > 0.
2. After `actions.behavior.jsonl`: require row count > 0 and semantic coverage = 100%.
3. Before `xx_align_multimodal.py`: require `event_queries.jsonl`, `actions_for_downstream`, and `pose_tracks_smooth_uq.jsonl` all non-empty.
4. After `align_multimodal.json`: require at least one event and at least one candidate unless the video truly has no visual events.
5. After `verified_events.jsonl`: warn when rows <= 1 and ASR transcript is `[ASR_EMPTY]`, because this is technically valid but not useful for paper metrics.

## Final Engineering Advice
Do not change the YOLO detection head or switch to RNN/LSTM/Transformer for this integration problem yet. The current failure mode is not model capacity; it is contract propagation and semantic identity loss between detector output, action conversion, alignment, and verifier.

The next implementation step should be `semantic_contract_v2`: make semantics available before alignment, repair UTF-8 labels, and add row-count/coverage gates. After that, rerun `run_full_e150_001` with both object evidence and behavior detection enabled, then compare against the current post-hoc semantic run.
