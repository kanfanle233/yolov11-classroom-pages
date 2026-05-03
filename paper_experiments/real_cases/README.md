# Real Case Bridge

This directory bridges `data/智慧课堂学生行为数据集` into paper-experiment case handling.

## What is here

- `index.json`
  - Full bridge index with scan summary, class-map notes, and selected case records.
- `case_inventory.csv`
  - Flat inventory for downstream batch scripts and manual review.
- `annotation_candidates.jsonl`
  - Candidate list for later human annotation.

## Selection policy

- 5 views are scanned: `正方视角`, `后方视角`, `教师视角`, `斜上方视角1`, `斜上方视角2`.
- 4 raw videos are selected per view.
- Every selected video exists and has non-zero size.
- The `case_id` is traceable from the view code plus the raw stem.
- These rows are candidates only. They are not gold labels.

## Canonical model and label mapping

- Use `output/case_yolo/data.yaml` with `runs/detect/case_yolo_train/weights/best.pt`.
- Keep the class order from `output/case_yolo/data.yaml` when running the current pipeline.
- Treat `data/processed/classroom_yolo/dataset.yaml` as a separate processed dataset with a different class order.

## Case output path policy

- Unified output root: `output/智慧课堂学生行为数据集/real_pilot/<view_code>/<case_id>/`
- Example:
  - `output/智慧课堂学生行为数据集/real_pilot/front/front__001`
  - `output/智慧课堂学生行为数据集/real_pilot/rear/rear__0001`
- Do not scatter case outputs directly under `output/智慧课堂学生行为数据集`.

## How to use

1. Run `scripts/intelligence_class/pipeline/01_run_single_video.py` on each selected `video_path`.
2. Refresh the resulting case directory with `scripts/09c_refresh_case_outputs.py` under `real_pilot/<view_code>/<case_id>`.
3. Keep gold annotation separate from model output and manual review.

## Boundary

- No system prediction is promoted to gold here.
- No manual annotation is fabricated here.
- This directory only defines the bridge inventory and selection metadata.
