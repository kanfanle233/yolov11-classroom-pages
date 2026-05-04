# 14 scripts 清理清单

更新时间：2026-05-04

## 当前总原则

- 不按照旧清单做批量删除。
- 先以“是否仍被主流程、实验、bundle、前端、论文材料引用”为准。
- 当前仓库仍在快速演进，很多脚本虽然不是主入口，但仍承担实验、汇总、打包或文档引用角色。

## 当前明确必须保留

### 主流程

- `scripts/main/09_run_pipeline.py`
- `scripts/main/09b_run_pipeline.py`
- `scripts/pipeline/01_pose_video_demo.py`
- `scripts/pipeline/02_export_keypoints_jsonl.py`
- `scripts/pipeline/02c_build_rear_roi_sr_cache.py`
- `scripts/pipeline/02d_export_behavior_det_jsonl.py`
- `scripts/pipeline/03_track_and_smooth.py`
- `scripts/pipeline/03c_estimate_track_uncertainty.py`
- `scripts/pipeline/03e_track_behavior_students.py`
- `scripts/pipeline/05_slowfast_actions.py`
- `scripts/pipeline/06_asr_whisper_to_jsonl.py`
- `scripts/pipeline/06b_event_query_extraction.py`
- `scripts/pipeline/06e_extract_instruction_context.py`
- `scripts/pipeline/06f_llm_semantic_fusion.py`
- `scripts/pipeline/07_dual_verification.py`
- `scripts/pipeline/10_visualize_timeline.py`

### 实验与评估

- `scripts/experiments/16_run_rear_row_sr_ablation.py`
- `scripts/experiments/17_build_sr_ablation_paper_summary.py`
- `scripts/experiments/18_build_rear_row_gt_template.py`
- `scripts/experiments/19_eval_rear_row_metrics.py`
- `scripts/experiments/25_split_test_set.py`
- `scripts/experiments/26_diag_tracking_quality.py`

### 前端与论文

- `scripts/frontend/20_build_frontend_data_bundle.py`
- `scripts/paper/30_build_paper_figures_tables.py`
- `scripts/paper/31_build_paper_figures_curated.py`
- `scripts/paper/32_build_d3og_selected_figures.py`
- `scripts/paper/33_build_repo_report_alignment.py`

## 当前明确不应按旧清单批量删除的原因

根据仓库内现有引用，以下类别仍在被使用或被文档、前端、汇总逻辑引用：

- `03e_track_behavior_students.py`
- `16/17/18/19` 实验脚本
- `20_build_frontend_data_bundle.py`
- `25_split_test_set.py`
- `index_v2.html` / BFF 相关前端数据链

因此，旧版“建议删除 25 个脚本”的清单现在已经不安全。

## 当前只建议做的低风险清理

### 非源码缓存

- `scripts/__pycache__/`
- 其它自动生成的 `__pycache__`

### 输出目录中的临时结果

- `output/_tmp_*`
- 明确标注为 smoke / 临时 / verify 的中间目录

### 重新生成型索引或报告

- 仅在确认可重新生成时，清理旧的自动生成报告

## 当前需要二次核验后才能决定的对象

| 类别 | 当前建议 |
| --- | --- |
| 旧实验脚本 | 先做引用扫描，再决定 |
| 旧训练脚本 | 若仍被论文或复现实验依赖，则保留 |
| 旧前端或调试脚本 | 若仍被 `server/app.py`、模板或 bundle 生成链引用，则保留 |
| 文档中提到但不常运行的脚本 | 在更新文档后再次核验 |

## 当前推荐的清理流程

1. 先跑全文引用扫描。
2. 把“主流程依赖”“实验依赖”“论文依赖”“前端依赖”分开。
3. 仅对无引用、无产物、无文档依赖的对象提删除建议。
4. 删除前单独在分支上操作，不和索引更新混在一起。

## 当前结论

- 旧版 `14_scripts清理清单.md` 的大规模删除建议不再适用。
- 这轮索引更新完成后，应该先基于新索引再做一次引用审计。
- 在那之前，最安全的清理范围只有缓存、临时输出和可重生成中间文件。
