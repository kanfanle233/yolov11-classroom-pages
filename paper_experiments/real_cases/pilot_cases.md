# Pilot Case Selection

Generated from `paper_experiments/real_cases/annotation_candidates.jsonl`.

## Selection policy

- Exactly 5 cases in total.
- Exactly 1 case per view.
- Prefer `candidate_for_demo=true`.
- If a view has no demo candidate, fallback to `candidate_for_gold_annotation=true`.
- Do not run all 20 candidates in this pilot.

## Selected pilot cases

| case_id | view | view_code | video_path | duration_if_available | file_size_bytes | reason |
| --- | --- | --- | --- | ---: | ---: | --- |
| front__001 | 正方视角 | front | data/智慧课堂学生行为数据集/正方视角/001.mp4 | 16.32 | 1247600 | demo candidate |
| rear__0001 | 后方视角 | rear | data/智慧课堂学生行为数据集/后方视角/0001.mp4 | 9.233 | 978698 | demo candidate |
| teacher__0001 | 教师视角 | teacher | data/智慧课堂学生行为数据集/教师视角/0001.mp4 | 1.7 | 107094 | demo candidate |
| top1__0001 | 斜上方视角1 | top1 | data/智慧课堂学生行为数据集/斜上方视角1/0001.mp4 | 3.1 | 173753 | demo candidate |
| top2__250184 | 斜上方视角2 | top2 | data/智慧课堂学生行为数据集/斜上方视角2/250184.mp4 | 25.367 | 1940995 | demo candidate |

## Pipeline constraints for this pilot

- Keep detector weights on `runs/detect/case_yolo_train/weights/best.pt`.
- Keep class-map source aligned with `output/case_yolo/data.yaml`.
- Do not switch to `data/processed/classroom_yolo/dataset.yaml` class order.
- Do not treat any pipeline output as gold.

## Output layout

`output/智慧课堂学生行为数据集/real_pilot/<view_code>/<case_id>/`
