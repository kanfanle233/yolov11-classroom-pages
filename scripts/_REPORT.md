# scripts/ 目录 — 完整维护报告

> **位置**：`F:\PythonProject\pythonProject\YOLOv11\scripts\`
> **报告生成日期**：2026-05-03

## 一、大致背景

`scripts/` 是项目的**核心代码目录**，包含162个Python脚本（数据来自`报告.md`的审阅统计）。这是整个课堂行为分析流水线的代码实现主体，涵盖了从视频预处理到最终可视化的完整链路。同时也是代码量最大、结构最复杂、最需要维护关注的目录。

## 二、位置与目录结构

```
scripts/
├── yolo11n.pt                        # YOLO权重（scripts目录下也有一份）
│
├── main/                             # 主线入口（2个文件）
│   ├── 09_run_pipeline.py            # 主线A：算法入口（主要使用）
│   └── 09b_run_pipeline.py           # 主线A变体/备份
│
├── pipeline/                         # 流水线核心步骤（24个文件）⭐核心
│   ├── 01_pose_video_demo.py         # 01-姿态视频演示
│   ├── 02_export_keypoints_jsonl.py  # 02-关键点导出
│   ├── 02b_export_objects_jsonl.py   # 02b-目标检测导出
│   ├── 02c_build_rear_roi_sr_cache.py# 02c-后方视角ROI超分缓存
│   ├── 02d_export_behavior_det_jsonl.py# 02d-行为检测导出
│   ├── 03_track_and_smooth.py        # 03-轨迹跟踪与平滑
│   ├── 03c_estimate_track_uncertainty.py# 03c-跟踪不确定性估计
│   ├── 03e_track_behavior_students.py# 03e-学生行为跟踪
│   ├── 04_action_rules.py            # 04-规则化动作判定
│   ├── 04_complex_logic.py           # 04-复杂逻辑判定
│   ├── 05_slowfast_actions.py        # 05-SlowFast动作识别（深度学习）
│   ├── 05b_fuse_actions_with_objects.py# 05b-动作与目标融合
│   ├── 06_api_asr_realtime.py        # 06-实时API ASR
│   ├── 06_asr_whisper_to_jsonl.py    # 06-Whisper ASR转JSONL
│   ├── 06_overlay_pose_behavior_video.py# 06-姿态行为覆盖视频
│   ├── 06b_event_query_extraction.py # 06b-事件查询抽取
│   ├── 06c_asr_openai_to_jsonl.py    # 06c-OpenAI ASR转JSONL
│   ├── 06d_build_rear_row_contact_sheet.py# 06d-后方视角联系表
│   ├── 06e_extract_instruction_context.py# 06e-指令上下文抽取
│   ├── 06f_llm_semantic_fusion.py    # 06f-LLM语义融合
│   ├── 07_dual_verification.py       # 07-双重验证（规则+可学习验证器）
│   ├── 08_overlay_sequences.py       # 08-序列覆盖视频
│   ├── 10_visualize_timeline.py      # 10-时间线可视化
│   └── xx_align_multimodal.py        # xx-多模态对齐
│
├── experiments/                      # 实验脚本（6个文件）
│   ├── 16_run_rear_row_sr_ablation.py    # 16-后方列超分消融
│   ├── 17_build_sr_ablation_paper_summary.py# 17-超分消融论文汇总
│   ├── 18_build_rear_row_gt_template.py # 18-后方列GT模板
│   ├── 19_eval_rear_row_metrics.py      # 19-后方列指标评估
│   ├── 25_split_test_set.py             # 25-测试集分割
│   └── 26_diag_tracking_quality.py      # 26-跟踪质量诊断
│
├── paper/                            # 论文图表生成（4个文件）
│   ├── 30_build_paper_figures_tables.py  # 30-论文图表表格生成
│   ├── 31_build_paper_figures_curated.py # 31-论文精选图表
│   ├── 32_build_d3og_selected_figures.py # 32-D3图表生成
│   └── 33_build_repo_report_alignment.py # 33-仓库报告对齐
│
├── frontend/                         # 前端数据打包（1个文件）
│   └── 20_build_frontend_data_bundle.py  # 前端数据打包
│
├── utils/                            # 工具脚本（6个文件）
│   ├── 02b_check_jsonl_schema.py         # JSONL Schema校验
│   ├── 12_export_features.py             # 特征导出
│   ├── 13_semantic_projection.py         # 语义投影
│   ├── build_verifier_training_samples.py# 构建验证器训练样本
│   ├── object_evidence_mapping.py        # 目标证据映射
│   └── sliced_inference_utils.py         # 切片推理工具
│
├── intelligence_class/               # 智慧课堂工程化模块（大型子目录）
│   ├── pipeline/  (~14个文件)        # 工程化流水线（与主pipeline有重复）
│   ├── tools/      (~11个文件)       # 辅助工具
│   ├── training/   (~3个文件)        # Case YOLO训练
│   ├── _utils/     (~2个文件)        # 内部工具
│   └── web_ui/     (1个文件)         # Web界面
│
├── ultralytics/                      # Vendored Ultralytics代码 (~95个文件)
│   ├── engine/    (~8个文件)         # 模型引擎
│   ├── nn/        (~11个文件)        # 神经网络模块
│   ├── trackers/  (~9个文件)         # 目标跟踪
│   ├── utils/     (~41个文件)        # 工具函数
│   ├── solutions/ (~20个文件)        # 解决方案
│   ├── hub/       (~5个文件)         # Hub集成
│   └── cfg/       (1个文件)          # 配置
│
├── models/                           # 模型定义（已迁移到上级models/目录）
│   ├── __init__.py
│   ├── interaction_graph.py
│   ├── igformer.py
│   ├── mllm_inference.py
│   └── cca_module.py
│
├── modules/                          # 模块化组件
│   ├── __init__.py
│   └── peer_context.py
│
├── training/                         # 训练脚本
│   └── train_classroom_yolo.py
│
├── 000.py                            # 实验性脚本
├── 09b_run_pipeline.py               # 主线入口变体
├── 09c_refresh_case_outputs.py
├── 11_debug_pipeline_check.py
├── 11_group_stgcn.py
├── 12_export_features.py
├── 13_semantic_projection.py
├── 14_mllm_semantic_verify.py
├── 99_debug_objects_stats.py
└── export_code.py
```

## 三、是干什么的（按流水线阶段）

### 阶段0：基础检测
- `01_pose_video_demo.py` — 快速姿态检测演示
- `02_export_keypoints_jsonl.py` — YOLO-Pose关键点提取（流水线第一步）
- `02b_export_objects_jsonl.py` — YOLO-Detect目标检测（流水线第一步）

### 阶段1：轨迹跟踪
- `03_track_and_smooth.py` — 核心：ByteTrack跟踪 + 卡尔曼平滑
- `03c_estimate_track_uncertainty.py` — 跟踪不确定性估计(UQ)

### 阶段2：行为识别
- `04_action_rules.py` — 基于规则的动作判定（快速但粗糙）
- `04_complex_logic.py` — 复杂逻辑规则
- `05_slowfast_actions.py` — SlowFast深度学习动作识别（核心）
- `05b_fuse_actions_with_objects.py` — 目标检测结果与动作融合

### 阶段3：语音处理
- `06_api_asr_realtime.py` — 实时API语音转写
- `06_asr_whisper_to_jsonl.py` — Whisper本地语音转写
- `06c_asr_openai_to_jsonl.py` — OpenAI语音转写
- `06b_event_query_extraction.py` — 从转录文本中抽取事件查询

### 阶段4：多模态融合与验证
- `xx_align_multimodal.py` — 多模态时序对齐
- `07_dual_verification.py` — 双重验证（规则+可学习验证器）
- `06f_llm_semantic_fusion.py` — LLM语义融合

### 阶段5：可视化
- `08_overlay_sequences.py` — 行为序列视频覆盖
- `10_visualize_timeline.py` — D3.js时间线可视化
- `06_overlay_pose_behavior_video.py` — 姿态行为覆盖视频

## 四、有什么用

1. **完整流水线**：从原始视频到可视化报告的端到端处理
2. **模块化设计**：每个步骤独立可替换（如可切换Whisper/OpenAI ASR）
3. **多种行为识别方案**：规则基础(04) → 深度学习(05 SlowFast) → 融合验证(07) → 大模型语义(06f)
4. **实验支撑**：experiments/ 和 paper/ 子目录直接支撑论文实验和图表

## 五、维护注意事项

### 关键警告
- **脚本总数162个**，但核心主线仅约22个脚本（详见`报告.md`第5节分类）
- **双入口并存**：`scripts/main/09_run_pipeline.py` 和 `intelligence_class/pipeline/01_run_single_video.py` 功能重叠
- **重复实现**：`02_export_keypoints_jsonl.py` 等脚本在 `pipeline/` 和 `intelligence_class/pipeline/` 各有一份
- **Vendored Ultralytics**：`scripts/ultralytics/` 是完整的Ultralytics代码副本(~95个文件)，如果后续只需推理，可改为pip依赖

### 建议阅读顺序（新人上手）
1. `报告.md` — 了解162个脚本的分类（核心/辅助/可删）
2. `09_run_pipeline.py` — 理解流水线编排逻辑
3. `pipeline/02_export_keypoints_jsonl.py` — 理解数据起点
4. `pipeline/07_dual_verification.py` — 理解核心验证逻辑
5. `pipeline/10_visualize_timeline.py` — 理解最终输出
