# 目录说明：`scripts/pipeline`

- 代码文件数：`24`
- 代码行数（LOC）：`8192`

## 目录职责
- 入口脚本职责：Build rear-row ROI super-resolution frame cache.
- 入口脚本职责：Export 8-class behavior detections to jsonl.
- 入口脚本职责：Estimate track uncertainty and export fixed UQ schema.
- 入口脚本职责：Track students from semantic behavior detections.

## 入口脚本
- `scripts/pipeline/01_pose_video_demo.py` (line 103)
- `scripts/pipeline/02_export_keypoints_jsonl.py` (line 286)
- `scripts/pipeline/02b_export_objects_jsonl.py` (line 218)
- `scripts/pipeline/02c_build_rear_roi_sr_cache.py` (line 243)
- `scripts/pipeline/02d_export_behavior_det_jsonl.py` (line 280)
- `scripts/pipeline/03_track_and_smooth.py` (line 492)
- `scripts/pipeline/03c_estimate_track_uncertainty.py` (line 206)
- `scripts/pipeline/03e_track_behavior_students.py` (line 1083)
- `scripts/pipeline/04_action_rules.py` (line 191)
- `scripts/pipeline/04_complex_logic.py` (line 205)
- `scripts/pipeline/05_slowfast_actions.py` (line 419)
- `scripts/pipeline/05b_fuse_actions_with_objects.py` (line 159)
- `scripts/pipeline/06_api_asr_realtime.py` (line 217)
- `scripts/pipeline/06_asr_whisper_to_jsonl.py` (line 432)
- `scripts/pipeline/06_overlay_pose_behavior_video.py` (line 592)
- `scripts/pipeline/06b_event_query_extraction.py` (line 206)
- `scripts/pipeline/06c_asr_openai_to_jsonl.py` (line 205)
- `scripts/pipeline/06d_build_rear_row_contact_sheet.py` (line 177)
- `scripts/pipeline/06e_extract_instruction_context.py` (line 189)
- `scripts/pipeline/06f_llm_semantic_fusion.py` (line 299)
- `scripts/pipeline/07_dual_verification.py` (line 407)
- `scripts/pipeline/08_overlay_sequences.py` (line 411)
- `scripts/pipeline/10_visualize_timeline.py` (line 881)
- `scripts/pipeline/xx_align_multimodal.py` (line 266)

## 上游依赖
- 自动扫描未命中固定规则

## 下游产物
- 自动扫描未命中固定规则

## 文件清单
| file | LOC | entry | description |
| --- | --- | --- | --- |
| 01_pose_video_demo.py | 104 | yes |  |
| 02_export_keypoints_jsonl.py | 287 | yes |  |
| 02b_export_objects_jsonl.py | 220 | yes |  |
| 02c_build_rear_roi_sr_cache.py | 244 | yes | Build rear-row ROI super-resolution frame cache. |
| 02d_export_behavior_det_jsonl.py | 281 | yes | Export 8-class behavior detections to jsonl. |
| 03_track_and_smooth.py | 493 | yes |  |
| 03c_estimate_track_uncertainty.py | 207 | yes | Estimate track uncertainty and export fixed UQ schema. |
| 03e_track_behavior_students.py | 1084 | yes | Track students from semantic behavior detections. |
| 04_action_rules.py | 192 | yes |  |
| 04_complex_logic.py | 206 | yes |  |
| 05_slowfast_actions.py | 420 | yes |  |
| 05b_fuse_actions_with_objects.py | 160 | yes | Fuse SlowFast actions with object detection evidence |
| 06_api_asr_realtime.py | 218 | yes | Realtime ASR (DashScope) with non-empty transcript guarantee. |
| 06_asr_whisper_to_jsonl.py | 433 | yes | Whisper ASR to transcript.jsonl with non-empty guarantee. |
| 06_overlay_pose_behavior_video.py | 593 | yes | Overlay pose skeletons and 8-class behavior labels into one demo video. |
| 06b_event_query_extraction.py | 207 | yes | Extract structured event queries from transcript. |
| 06c_asr_openai_to_jsonl.py | 206 | yes | OpenAI ASR to transcript.jsonl with non-empty guarantee. |
| 06d_build_rear_row_contact_sheet.py | 178 | yes | Build rear-row enhancement comparison contact sheet. |
| 06e_extract_instruction_context.py | 190 | yes | Extract teacher instruction context from ASR transcript. |
| 06f_llm_semantic_fusion.py | 300 | yes | LLM Semantic Fusion for audio-visual classroom analysis. |
| 07_dual_verification.py | 408 | yes | Step07 orchestration: query + align + verifier -> verified_events.jsonl |
| 08_overlay_sequences.py | 412 | yes |  |
| 10_visualize_timeline.py | 882 | yes |  |
| xx_align_multimodal.py | 267 | yes | Adaptive multimodal aligner (query + visual candidates + UQ basis). |
