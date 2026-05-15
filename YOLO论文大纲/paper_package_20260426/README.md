# 论文资料包说明

生成时间：2026-04-26  
根目录：`F:/PythonProject/pythonProject/YOLOv11/codex_reports/smart_classroom_yolo_feasibility/paper_package_20260426`

## 目录结构

| 目录 | 内容 | 用途 |
|---|---|---|
| `00_source_docs` | 用户上传的 `论文准备.md`、全量遍历审计报告、数据集统计 | 写作依据与背景材料 |
| `01_figures_detection` | YOLO11s e150 训练曲线、混淆矩阵、PR/F1/P/R 曲线、预测样例 | 主文实验图与附录图 |
| `02_figures_pipeline` | 主线 timeline、reliability diagram、旧版 e150 timeline | 系统效果图与可视化分析图 |
| `03_metrics_tables` | results.csv、run comparison、contract report、ASR report、verifier report、timeline CSV | 表格、指标、实验对比 |
| `04_references` | 参考文献清单 | 引言和相关工作 |
| `05_outline` | 论文主线与大纲 | 正式写作骨架 |
| `06_demo_materials` | `actions.fusion_v2.jsonl`、`verified_events.jsonl`、timeline JSON | 录屏、系统展示、复现实验 |

## 推荐主文图表

| 图表 | 文件 | 放置章节 |
|---|---|---|
| 系统总体框架图 | 建议根据 `05_outline/paper_mainline.md` 中流程重新绘制 | 方法 |
| YOLO 检测训练曲线 | `01_figures_detection/e150_results.png` | 实验 |
| 混淆矩阵 | `01_figures_detection/e150_confusion_matrix.png` 或 normalized 版本 | 实验 |
| PR/F1 曲线 | `01_figures_detection/e150_BoxPR_curve.png`, `01_figures_detection/e150_BoxF1_curve.png` | 实验/附录 |
| 定性检测样例 | `01_figures_detection/e150_val_batch0_pred.jpg` 等 | 实验 |
| 学生行为时间线 | `02_figures_pipeline/mainline_timeline_chart.png` | 系统与可视化分析 |
| Verifier 可靠性图 | `02_figures_pipeline/mainline_verifier_reliability_diagram.svg` | 校准/附录 |

## 关键指标来源

| 指标 | 文件 |
|---|---|
| e150 检测指标 | `03_metrics_tables/e150_effect_summary.md` |
| epoch 级训练日志 | `03_metrics_tables/e150_results.csv` |
| 多 run 对比 | `03_metrics_tables/run_comparison.md` |
| 主线链路结果 | `03_metrics_tables/pipeline_contract_v2_report.json` |
| fusion v2 语义覆盖 | `03_metrics_tables/fusion_contract_report.json` |
| ASR 质量门控 | `03_metrics_tables/asr_quality_report.json` |
| 学生 ID 与动作时间线 | `03_metrics_tables/student_id_map.json`, `03_metrics_tables/timeline_students.csv` |

