# 实验指标摘要

## YOLO11s e150 行为检测

来源文件：`e150_effect_summary.md`, `e150_results.csv`

| 指标 | 数值 | 论文使用方式 |
|---|---:|---|
| mAP50 | 0.933 | 主文报告 |
| mAP50-95 | 0.804 | 主文报告 |
| Precision | 0.887 | 主文报告 |
| Recall | 0.894 | 主文报告 |

说明：该结果对应 `runs/detect/official_yolo11s_detect_e150_v1/weights/best.pt`，可作为当前论文主线行为检测模型。

## 训练运行对比

来源文件：`run_comparison.md`, `run_comparison.json`

| Run | Epoch | mAP50 | mAP50-95 | 结论 |
|---|---:|---:|---:|---|
| old_case_baseline | 80 | 0.93345 | 0.81140 | 历史参考；数据/类别映射不一定完全同源 |
| wisdom8_current | 80 | 0.93183 | 0.79836 | 同 split 官方微调基线 |
| wisdom8_ft70 | 70 | 0.92612 | 0.79521 | 未超过 80 epoch 基线 |
| official_smoke10 | 10 | 0.91846 | 0.76591 | 冒烟训练，不作为正式模型 |
| official_yolo11s_detect_e150_v1 | 150 | 0.933 | 0.804 | 当前主线模型 |

## 主线 pipeline 结果

来源文件：`pipeline_contract_v2_report.json`, `fusion_contract_report.json`

| 指标 | 数值 | 解释 |
|---|---:|---|
| tracked_students | 11 | pose tracking 后识别出的学生轨迹数 |
| student_count | 11 | 映射成 S01/S02/... 的学生数 |
| timeline_student_rows | 30 | 学生级 timeline 动作片段数 |
| actions_fusion_v2 | 186 | 进入下游 align/verifier 的融合动作条数 |
| actions_fusion_v2_semantic_valid | 186 | 语义字段完整的动作条数 |
| event_queries_fusion_v2 | 12 | 融合后的事件查询条数 |
| align_total_candidates | 96 | 对齐阶段候选总数 |
| verified_events | 12 | 双重验证输出事件数 |

## ASR 质量门控

来源文件：`asr_quality_report.json`

| 指标 | 数值 | 解释 |
|---|---:|---|
| model | medium | Whisper 模型 |
| device | cuda | 推理设备 |
| compute_type | float16 | 推理精度 |
| segments_raw | 3 | Whisper 原始片段数 |
| segments_accepted | 0 | 通过质量门控片段数 |
| segments_rejected | 3 | 被拒绝片段数 |
| status | placeholder | 文本低可信，触发视觉 fallback |

论文写法：这组结果不能写成“文本模态显著提升准确率”；应写为“系统具备 ASR 质量审计与低质文本兜底能力”。

## 推荐放入论文主文的表格

1. 表 1：8 类课堂行为 taxonomy。
2. 表 2：YOLO11s e150 行为检测指标。
3. 表 3：pipeline contract 结果，包括学生数、动作数、事件数、候选数。
4. 表 4：消融实验规划或初步对比。

