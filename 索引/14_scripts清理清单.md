# 14 Scripts 清理清单

> 生成时间：2026-05-03
> 原则：删除后不影响主线 pipeline、论文实验、D3 可视化的任何功能

---

## 🔴 建议删除（25 个）—— 旧实验/废弃/不可运行

### 不可运行的占位符（import 不存在的模块）
| 文件 | 原因 |
|---|---|
| `14_mllm_semantic_verify.py` | import `models.cca_module` / `models.mllm_inference`，模块不存在，必 crash |
| `11_group_stgcn.py` | ST-GCN 已被 IGFormer plan 替代，未跑通 |

### 旧实验脚本（论文不需要）
| 文件 | 原因 |
|---|---|
| `17_run_exp_c_negative_sampling.py` | exp-c 负采样实验，不再需要 |
| `18_run_exp_d_semantic_embedding.py` | exp-d 语义嵌入实验，不再需要 |
| `19_run_exp_e_object_evidence.py` | exp-e 物体证据实验，不再需要 |
| `15_run_paper_experiment.py` | 旧 paper 实验入口，已被 16/17/19 替代 |
| `16_run_paper_baseline.py` | 旧 baseline 实验，不再需要 |
| `21_run_asr_backend_ablation.py` | ASR 后端消融，不再需要 |
| `22_run_real_manifest_batch.py` | 旧 real data 批次处理 |
| `25_validate_real_gold.py` | 旧 gold 验证脚本 |
| `26_prepare_gold_peer_review.py` | 旧 peer review 准备 |
| `27_plan_second_batch_gold.py` | 旧第二批 gold 计划 |
| `28_generate_overlay_batch.py` | 旧叠加视频批处理 |

### 旧/重复功能（已被新脚本替代）
| 文件 | 替代者 |
|---|---|
| `29_build_analysis_bundle.py` | `20_build_frontend_data_bundle.py` |
| `02c_export_objects_jsonl_custom.py` | `02d_export_behavior_det_jsonl.py` |
| `02c_objects_video_demo.py` | `01_pose_video_demo.py` |
| `03b_objects_video_demo.py` | `06_overlay_pose_behavior_video.py` |
| `05c_behavior_det_to_actions.py` | fusion contract v2 scripts |
| `05d_merge_action_sources.py` | fusion contract v2 scripts |
| `05_overlay_action_video.py` | `06_overlay_pose_behavior_video.py` |
| `09c_refresh_case_outputs.py` | `20_build_frontend_data_bundle.py` |

### 调试/工具脚本（非核心功能）
| 文件 | 原因 |
|---|---|
| `11_debug_pipeline_check.py` | 调试脚本 |
| `99_debug_objects_stats.py` | 调试脚本 |
| `export_code.py` | 代码导出工具（非核心） |
| `23_compare_case_metrics.py` | 旧指标对比 |

---

## 🟡 可选删除（5 个）—— 很少使用但保留无妨

| 文件 | 原因 |
|---|---|
| `000.py` | 旧 dataset index 扫描器，不再使用 |
| `14_mllm_semantic_verify.py` | 已在🔴中列出 |
| `20_validate_real_bridge_config.py` | 旧 bridge 配置验证 |
| `24_generate_real_gold_from_tasks.py` | 旧 gold 生成 |
| `03d_train_track_variance_head.py` | 方差头训练（未使用） |

---

## 🟢 必须保留（42 个）—— 核心管线 + 论文实验

### 主线流水线（14 个）
- `09_run_pipeline.py` — 主编排器
- `01_pose_video_demo.py` — Step 1
- `02_export_keypoints_jsonl.py` — Step 2 关键点
- `02d_export_behavior_det_jsonl.py` — 行为检测
- `02b_export_objects_jsonl.py` — 目标检测
- `03_track_and_smooth.py` — 跟踪平滑
- `03c_estimate_track_uncertainty.py` — UQ 估计
- `03e_track_behavior_students.py` — 学生跟踪
- `05_slowfast_actions.py` — 动作识别
- `06_asr_whisper_to_jsonl.py` — ASR
- `06b_event_query_extraction.py` — 事件抽取
- `07_dual_verification.py` — 双重验证
- `08_overlay_sequences.py` — 视频叠加
- `10_visualize_timeline.py` — 时间线

### 论文实验（9 个）
- `16_run_rear_row_sr_ablation.py` — SR 消融
- `17_build_sr_ablation_paper_summary.py` — 论文总表
- `18_build_rear_row_gt_template.py` — GT 模板
- `19_eval_rear_row_metrics.py` — 正式评估
- `20_build_frontend_data_bundle.py` — D3 bundle
- `25_split_test_set.py` — Test set 拆分
- `26_diag_tracking_quality.py` — 跟踪诊断
- `06f_llm_semantic_fusion.py` — LLM 融合
- `06e_extract_instruction_context.py` — 指令上下文

### 功能增强（6 个）
- `sliced_inference_utils.py` — 切片推理工具库
- `02c_build_rear_roi_sr_cache.py` — SR 缓存
- `06d_build_rear_row_contact_sheet.py` — 联系表
- `06_overlay_pose_behavior_video.py` — 融合视频
- `04_complex_logic.py` — 复杂规则
- `04_action_rules.py` — 动作规则

### 辅助保留（7 个）
- `xx_align_multimodal.py` — 跨模态对齐
- `05b_fuse_actions_with_objects.py` — 物体融合
- `06_api_asr_realtime.py` — API ASR
- `06c_asr_openai_to_jsonl.py` — OpenAI ASR
- `12_export_features.py` — 特征导出
- `13_semantic_projection.py` — 语义投影
- `09b_run_pipeline.py` — 别名入口

### 论文图表（4 个）
- `30_build_paper_figures_tables.py`
- `31_build_paper_figures_curated.py`
- `32_build_d3og_selected_figures.py`
- `33_build_repo_report_alignment.py`

### 模型/训练（2 个）
- `build_verifier_training_samples.py`
- `object_evidence_mapping.py`

---

---

## 子文件夹分析

### `scripts/ultralytics/` — 6.6MB，185 文件
**可删除 ✅**
- pip 已安装 ultralytics（`F:\miniconda\envs\pytorch_env\lib\site-packages\ultralytics\`）
- 没有脚本显式引用这个 vendored 版本
- 当初 vendored 是为了实现 plan 中的 ASPN/DySnakeConv 修改（升级四），但从未开始
- **删除影响**：零

### `scripts/models/` — 54KB，5 文件
**可删除 ✅**
- `cca_module.py`, `igformer.py`, `interaction_graph.py`, `mllm_inference.py` — 全部来自 implementation_plan 四大升级，**从未实现**
- 导入会直接 crash（依赖不存在的模块）
- `__init__.py` — 空包初始化

### `scripts/modules/` — 18KB，2 文件
**可删除 ✅**
- `peer_context.py` — Plan 升级二的同伴感知，从未实现
- `__init__.py` — 空包初始化

### `scripts/intelligence_class/` — 1.2MB，31 文件
**全部保留，但可精简**
- 旧版 IC 系统，与主 pipeline 并行
- `web_ui/app.py` — Flask 前端（仍有用）
- `pipeline/` — 批量处理、调试工具（部分有用）
- `tools/` — 后处理工具（部分被新脚本替代）
- `training/` — 数据转换工具（可能有用）
- 🟡 可删：`debug_pose_chain.py`, `debug_pose_tracks.py`, `00_collect_code_for_ai.py`, `01_dump_py_only.py`, `check.py`, `move.py`

### `scripts/runs/` — 空目录
**可删除 ✅**

### `scripts/output/` — 空目录
**可删除 ✅**

### `scripts/__pycache__/` — 936KB
**可删除 ✅** — Python 缓存，自动重新生成

### `scripts/training/` — 8KB
**保留** — `train_classroom_yolo.py` 仍可能使用

---

## 清理统计

| 类别 | 数量 | 大小 |
|---|---|---|
| 🔴 建议删除（顶层 .py） | **25** | ~300KB |
| 🔴 建议删除（ultralytics/） | **185** | **6.6MB** |
| 🔴 建议删除（models/modules/） | **7** | 72KB |
| 🔴 建议删除（空目录/缓存） | **3** | 936KB |
| 🟡 可选删除（IC 调试） | **4** | ~50KB |
| 🟢 必须保留 | **42** + IC 22 | ~1.2MB |

**总回收：~8MB + 185 个冗余文件**

---

## 删除命令

```powershell
cd F:\PythonProject\pythonProject\YOLOv11\scripts

# === 顶层废弃脚本 ===
del 14_mllm_semantic_verify.py
del 11_group_stgcn.py
del 17_run_exp_c_negative_sampling.py
del 18_run_exp_d_semantic_embedding.py
del 19_run_exp_e_object_evidence.py
del 15_run_paper_experiment.py
del 16_run_paper_baseline.py
del 21_run_asr_backend_ablation.py
del 22_run_real_manifest_batch.py
del 25_validate_real_gold.py
del 26_prepare_gold_peer_review.py
del 27_plan_second_batch_gold.py
del 28_generate_overlay_batch.py
del 29_build_analysis_bundle.py
del 02c_export_objects_jsonl_custom.py
del 02c_objects_video_demo.py
del 03b_objects_video_demo.py
del 05c_behavior_det_to_actions.py
del 05d_merge_action_sources.py
del 05_overlay_action_video.py
del 09c_refresh_case_outputs.py
del 11_debug_pipeline_check.py
del 99_debug_objects_stats.py
del export_code.py
del 23_compare_case_metrics.py

# === 未实现的 plan 模块 ===
rmdir /s /q models
rmdir /s /q modules

# === 冗余的 vendored ultralytics ===
rmdir /s /q ultralytics

# === 空目录 + 缓存 ===
rmdir /s /q runs
rmdir /s /q output
rmdir /s /q __pycache__

# === IC 调试脚本（可选） ===
del intelligence_class\pipeline\debug_pose_chain.py
del intelligence_class\pipeline\debug_pose_tracks.py
del intelligence_class\tools\00_collect_code_for_ai.py
del intelligence_class\tools\01_dump_py_only.py
del intelligence_class\tools\check.py
del intelligence_class\tools\move.py
```

> ⚠️ 请确认后执行。建议 `git rm -r` 保留 git 历史。
