# 目录说明：`scripts/experiments`

- 代码文件数：`6`
- 代码行数（LOC）：`2392`

## 目录职责
- 入口脚本职责：Run and collect rear-row ROI SR ablation metrics.
- 入口脚本职责：Build cross-case paper table for SR ablations.
- 入口脚本职责：Create a rear-row GT annotation template JSONL and frame images.
- 入口脚本职责：Evaluate rear-row detection, pose, tracking, behavior, SR, and runtime metrics.

## 入口脚本
- `scripts/experiments/16_run_rear_row_sr_ablation.py` (line 808)
- `scripts/experiments/17_build_sr_ablation_paper_summary.py` (line 252)
- `scripts/experiments/18_build_rear_row_gt_template.py` (line 163)
- `scripts/experiments/19_eval_rear_row_metrics.py` (line 876)
- `scripts/experiments/25_split_test_set.py` (line 150)
- `scripts/experiments/26_diag_tracking_quality.py` (line 137)

## 上游依赖
- 自动扫描未命中固定规则

## 下游产物
- 自动扫描未命中固定规则

## 文件清单
| file | LOC | entry | description |
| --- | --- | --- | --- |
| 16_run_rear_row_sr_ablation.py | 809 | yes | Run and collect rear-row ROI SR ablation metrics. |
| 17_build_sr_ablation_paper_summary.py | 253 | yes | Build cross-case paper table for SR ablations. |
| 18_build_rear_row_gt_template.py | 164 | yes | Create a rear-row GT annotation template JSONL and frame images. |
| 19_eval_rear_row_metrics.py | 877 | yes | Evaluate rear-row detection, pose, tracking, behavior, SR, and runtime metrics. |
| 25_split_test_set.py | 151 | yes | Split classroom_yolo into train/val/test (70/15/15) |
| 26_diag_tracking_quality.py | 138 | yes | Diagnose tracking fragmentation and quality. |
