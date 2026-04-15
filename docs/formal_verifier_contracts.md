# Formal Verifier Contracts (Iteration 1 Freeze)

This document freezes the paper-main contracts for the first iteration.
The frozen schema date is `2026-04-01`.

## Version Rules
- `schema_version`: must be `2026-04-01` or `2026-04-01+<patch>`
- `artifact_version`: semantic tag for report artifacts, recommended:
  `formal_verifier_contracts@2026-04-01`
- New fields can be appended in later iterations, but existing field meaning must not change.

## 1) `event_queries.jsonl`
One JSON object per line.

Required fields:
- `event_id` `str`: stable id used across align/train/verify
- `schema_version` `str`: version string
- `query_text` `str`: normalized query text
- `event_type` `str`: event class name
- `trigger_words` `list[str]`: extracted trigger tokens
- `timestamp` `float`: query center time
- `start` `float`: query start time
- `end` `float`: query end time
- `confidence` `float[0,1]`: extraction confidence
- `source` `str`: source tag (`asr`, etc.)

## 2) `pose_tracks_smooth_uq.jsonl`
One frame-level JSON object per line.

Required fields:
- `schema_version` `str`
- `uq_type` `str`: `heuristic_statistical | learned_variance_head | probabilistic_presence_aware`
- `uq_scope` `str`
- `presence_aware` `bool`
- `variance_head` `bool`
- `frame` `int`: frame index
- `t` `float`: seconds
- `persons` `list[object]`: track UQ list
- `uq_frame` `float[0,1]`: frame-level UQ summary

Required per `persons[]`:
- `track_id` `int`
- `uq_track` `float[0,1]`
- `uq_conf` `float[0,1]`
- `uq_motion` `float[0,1]`
- `uq_kpt` `float[0,1]`
- `log_sigma2` `float`

Optional per `persons[]`:
- `uq_track_heuristic` `float[0,1]`: heuristic baseline before learned blending
- `uq_variance` `float[0,1]`: variance-head-derived uncertainty
- `sigma2` `float >= 0`: scalar variance estimate

Current iteration note:
- Default mode is `heuristic_statistical`.
- When `--variance_model` is provided, `03c_estimate_track_uncertainty.py` switches to `learned_variance_head` and blends heuristic UQ with a minimal learned scalar variance head trained on pseudo temporal residuals.
- This is still not RTMO/ProbPose-style probabilistic pose output. It is a lightweight sequence-level variance head, not a YOLO pose-head rewrite.
- Presence probability / out-of-window probability / visibility branches are not part of the current contract.

## 3) `align_multimodal.json`
Top-level JSON list, each item is an aligned query block.

Required fields per item:
- `event_id` `str`
- `query_text` `str`
- `event_type` `str`
- `window_center` `float`
- `window_start` `float`
- `window_end` `float`
- `window_size` `float`
- `basis_motion` `float[0,1]`
- `basis_uq` `float[0,1]`
- `candidates` `list[object]`

Recommended per `candidates[]`:
- `track_id` `int`
- `action` `str`
- `start_time` `float`
- `end_time` `float`
- `overlap` `float[0,1]`
- `action_confidence` `float[0,1]`
- `uq_track` `float[0,1]`
- `uq_variance` `float[0,1]`
- `log_sigma2` `float`

## 4) `verifier_samples_train.jsonl`
One training sample per line.

Required fields:
- `sample_id` `str`
- `event_id` `str`
- `sample_type` `str`: `positive | temporal_shift | semantic_mismatch`
- `query_text` `str`
- `event_type` `str`
- `track_id` `int`
- `clip_start` `float`
- `clip_end` `float`
- `target_label` `str`: `match | mismatch`
- `negative_kind` `str`
- `provenance` `object`: sample trace metadata

## 5) `verified_events.jsonl`
Paper-main result file. One validated result per line.

Required fields:
- `event_id` `str`
- `track_id` `int`
- `event_type` `str`
- `query_text` `str`
- `query_time` `float`
- `window_start` `float`
- `window_end` `float`
- `p_match` `float[0,1]`
- `p_mismatch` `float[0,1]`
- `reliability_score` `float[0,1]`
- `uncertainty` `float[0,1]`
- `label` `str`: `match | uncertain | mismatch`
- `threshold_source` `str`
- `model_version` `str`

## 6) `verifier_eval_report.json`
Required fields:
- `split` `str`
- `counts` `object`
- `metrics` `object`
- `confusion_matrix` `object`
- `threshold_sweep` `list[object]`
- `label_distribution` `object`
- `config` `object`
- `artifact_version` `str`

`metrics` must include:
- `precision` `float[0,1]`
- `recall` `float[0,1]`
- `f1` `float[0,1]`
- `accuracy` `float[0,1]`
- `macro_f1` `float[0,1]`

`confusion_matrix` must include:
- `labels` `list[str]`
- `matrix` `list[list[int]]`

## 7) `verifier_calibration_report.json`
Required fields:
- `split` `str`
- `ece` `float >= 0`
- `brier` `float >= 0`
- `temperature` `float > 0`
- `temperature_scaling_enabled` `bool`
- `bin_stats` `list[object]`
- `before_after` `object` with `before` and `after`
- `artifact_version` `str`

Supporting artifact:
- `verifier_reliability_diagram.svg`: optional visualization exported from the same calibration run.

Supporting files for learned variance head:
- `variance_head.pt`: optional minimal scalar variance head checkpoint
- `variance_head_report.json`: training summary for the variance head


## 8) `pipeline_manifest.json`
Required fields:
- `case_id` `str`
- `video_id` `str`
- `schema_version` `str`
- `artifacts` `object`: artifact path map
- `config_snapshot` `object`: run args snapshot

## Reference Samples
Frozen sample fixtures are in:
- `contracts/examples/*.sample.jsonl`
- `contracts/examples/*.sample.json`
