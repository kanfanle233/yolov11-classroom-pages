# Paper Pipeline Report: run_full_paper_mainline_001

## Dataset
- Dataset YAML: `F:\PythonProject\pythonProject\YOLOv11\data\processed\classroom_yolo\dataset.yaml`
- Classes: `0:tt, 1:dx, 2:dk, 3:zt, 4:xt, 5:js, 6:zl, 7:jz`

## Pipeline Artifacts
- `pose_tracks_smooth`: `F:\PythonProject\pythonProject\YOLOv11\output\codex_reports\run_full_paper_mainline_001\full_integration_001\pose_tracks_smooth.jsonl`
- `behavior_det_semantic`: `F:\PythonProject\pythonProject\YOLOv11\output\codex_reports\run_full_paper_mainline_001\full_integration_001\behavior_det.semantic.jsonl`
- `actions_fusion_v2`: `F:\PythonProject\pythonProject\YOLOv11\output\codex_reports\run_full_paper_mainline_001\full_integration_001\actions.fusion_v2.jsonl`
- `event_queries_fusion_v2`: `F:\PythonProject\pythonProject\YOLOv11\output\codex_reports\run_full_paper_mainline_001\full_integration_001\event_queries.fusion_v2.jsonl`
- `align_multimodal`: `F:\PythonProject\pythonProject\YOLOv11\output\codex_reports\run_full_paper_mainline_001\full_integration_001\align_multimodal.json`
- `verified_events`: `F:\PythonProject\pythonProject\YOLOv11\output\codex_reports\run_full_paper_mainline_001\full_integration_001\verified_events.jsonl`
- `student_id_map`: `F:\PythonProject\pythonProject\YOLOv11\output\codex_reports\run_full_paper_mainline_001\full_integration_001\student_id_map.json`
- `timeline_students_csv`: `F:\PythonProject\pythonProject\YOLOv11\output\codex_reports\run_full_paper_mainline_001\full_integration_001\timeline_students.csv`
- `timeline_chart_png`: `F:\PythonProject\pythonProject\YOLOv11\output\codex_reports\run_full_paper_mainline_001\full_integration_001\timeline_chart.png`

## Contract Summary
### Fusion Contract
- `actions_fusion_v2`: `186`
- `actions_fusion_v2_semantic_valid`: `186`
- `event_queries_fusion_v2`: `12`
- `verified_events`: `12`
- `align_events`: `12`
- `align_total_candidates`: `96`
### Pipeline Contract
- `pose_rows`: `408`
- `pose_track_rows`: `408`
- `tracked_students`: `11`
- `behavior_items`: `14785`
- `actions_fusion_v2`: `186`
- `actions_fusion_v2_semantic_valid`: `186`
- `event_queries_fusion_v2`: `12`
- `align_events`: `12`
- `align_total_candidates`: `96`
- `verified_events`: `12`
- `student_count`: `11`
- `timeline_student_rows`: `30`

## ASR Quality
### ASR
- `video`: `F:\PythonProject\pythonProject\YOLOv11\data\智慧课堂学生行为数据集\正方视角\001.mp4`
- `wav`: `F:\PythonProject\pythonProject\YOLOv11\output\codex_reports\run_full_paper_mainline_001\full_integration_001\asr_audio_16k.wav`
- `transcript`: `F:\PythonProject\pythonProject\YOLOv11\output\codex_reports\run_full_paper_mainline_001\full_integration_001\transcript.jsonl`
- `model`: `medium`
- `device`: `cuda`
- `compute_type`: `float16`
- `beam_size`: `10`
- `vad_filter`: `0`
- `condition_on_previous_text`: `0`
- `audio_filter`: ``
- `segments_raw`: `3`
- `segments_accepted`: `0`
- `segments_rejected`: `3`
- `status`: `placeholder`

## Verifier
### Verifier Counts
- `total`: `12`
### Verifier Metrics
- `precision`: `0.3333333333333333`
- `recall`: `0.3333333333333333`
- `f1`: `0.3333333333333333`
- `accuracy`: `1.0`
- `macro_f1`: `0.3333333333333333`

## Paper Assets
- Manifest: `F:\PythonProject\pythonProject\YOLOv11\codex_reports\smart_classroom_yolo_feasibility\paper_assets\run_full_paper_mainline_001\paper_image_manifest.csv`
