# 论文主线

## 推荐题目

中文题目：  
**面向智慧课堂的视觉行为序列与文本语义流双重验证框架**

英文题目：  
**A Dual-Verification Framework for Aligning Visual Classroom Behavior Sequences with Textual Semantic Streams**

## 一句话主线

本文不是单纯“改 YOLO 检测头”的论文，而是提出一条面向智慧课堂的系统级多模态感知链路：用微调 YOLOv11 检测学生 8 类课堂行为，用 YOLO pose 生成学生轨迹 ID，用 ASR/文本事件提供语义流，再通过语义桥接、UQ 驱动对齐和双重验证，把“谁在什么时间做什么动作”转化为可审计、可解释、可视化的 timeline。

## 核心研究问题

1. 复杂课堂环境中，单一视觉检测容易受到遮挡、低光、多人重叠和类别相似动作影响。
2. 课堂语音/文本流可能为空、低质或存在幻觉，不能无条件进入下游推理。
3. YOLO 输出的 `tt/dx/dk/zt/xt/js/zl/jz` 对人和 NLP 模块都不可读，需要稳定语义协议。
4. 检测结果必须落到学生级 timeline，否则难以支撑智慧教育分析。

## 本文解决方案

| 层级 | 方案 | 当前产物 |
|---|---|---|
| 视觉检测 | YOLO11s 微调识别 8 类课堂行为 | `01_figures_detection/e150_results.png`, `03_metrics_tables/e150_results.csv` |
| 姿态轨迹 | YOLO pose + tracking 生成 `track_id` | `03_metrics_tables/student_id_map.json` |
| 语义桥接 | 将短码转为中英双语 canonical id | `06_demo_materials/actions.fusion_v2.jsonl` |
| 文本语义 | Whisper ASR + 质量门控；低质则视觉 fallback | `03_metrics_tables/asr_quality_report.json` |
| 跨模态对齐 | UQ 驱动时间窗口生成候选 | `03_metrics_tables/pipeline_contract_v2_report.json` |
| 双重验证 | 输出 match/mismatch/uncertain 及可靠性 | `03_metrics_tables/verifier_eval_report.json` |
| 可视分析 | 学生级 timeline 和论文图表 | `02_figures_pipeline/mainline_timeline_chart.png`, `03_metrics_tables/timeline_students.csv` |

## 创新点定位

1. **语义桥接协议**：把检测类别从 `tt/dx/...` 变成可被后续模块直接消费的 `semantic_id + bilingual label`。
2. **UQ 驱动跨模态对齐**：把 pose tracking 不确定性引入文本事件与视觉动作的时间窗口。
3. **ASR 质量门控与视觉兜底**：不让低质量 ASR 幻觉污染 verifier；文本低可信时自动使用视觉事件。
4. **学生级 timeline 可视分析**：学生 ID 不进入 YOLO 检测头，而在 tracking 后处理层生成，形成 `S01/S02/...` 的动作序列。
5. **可失败的工程协议**：缺文件、空结果、语义缺失、align 无候选时严格失败，避免“空数据还继续处理”。

## 当前实验证据

| 证据 | 数值/结论 | 来源 |
|---|---|---|
| e150 检测模型 | mAP50=0.933, mAP50-95=0.804, Precision=0.887, Recall=0.894 | `03_metrics_tables/e150_effect_summary.md` |
| 主线学生数 | 11 个学生 ID | `03_metrics_tables/pipeline_contract_v2_report.json` |
| fusion 动作 | 186 条，语义有效 186 条 | `03_metrics_tables/fusion_contract_report.json` |
| 事件与候选 | event queries=12, align candidates=96, verified events=12 | `03_metrics_tables/pipeline_contract_v2_report.json` |
| ASR 状态 | medium/cuda/float16 产生 3 段 raw，0 段 accepted，触发视觉 fallback | `03_metrics_tables/asr_quality_report.json` |
| timeline | 30 条学生动作片段 | `03_metrics_tables/timeline_students.csv` |

## 不能这样写

1. 不能写“本文提出新的 YOLOv11 检测头”，因为当前主线使用的是官方 YOLO11s 微调，不是改内核后的公平训练结果。
2. 不能写“文本模态显著提升性能”，因为当前样例 ASR 被质量门控判为低质，主要走视觉 fallback。
3. 不能写“已完成统计显著验证”，因为当前 verified 事件样本量仍小，需要扩大 case 和人工 gold label。
4. 不能把 LLM、Video-LLaMA、InternVideo2 写成本文已实现方法，只能放相关工作和未来展望。

