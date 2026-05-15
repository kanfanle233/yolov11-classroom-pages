# paper_experiments/ 目录 — 完整维护报告

> **位置**：`F:\PythonProject\pythonProject\YOLOv11\paper_experiments\`
> **报告生成日期**：2026-05-03

## 一、大致背景

`paper_experiments/` 是**论文实验的核心数据目录**，存放所有实验的配置文件、运行日志、Gold标注数据、真实案例分析数据。这是论文"实验"章节的数据来源，也是项目中最需要妥善保管的目录之一。

## 二、位置与目录结构

```
paper_experiments/
│
├── configs/                           # 实验配置文件(6个JSON)
│   ├── exp_uq_adaptive_alignment.json             # Exp A: UQ自适应对齐
│   ├── exp_dual_verifier_reliability.json         # Dual Verifier可靠性
│   ├── exp_object_action_fusion.json              # 目标-动作融合
│   ├── exp_b_reliability_calibration.json         # Exp B: 可靠性校准
│   ├── real_exp_a_uq_alignment.json               # 真实数据Exp A
│   └── real_exp_b_reliability_calibration.json    # 真实数据Exp B
│
├── samples/                           # 数据样例(6个JSONL)
│   ├── event_queries.sample.jsonl
│   ├── pose_tracks_smooth_uq.sample.jsonl
│   ├── actions_dual.sample.jsonl
│   ├── actions_object_fusion.sample.jsonl
│   ├── objects_object_fusion.sample.jsonl
│   └── reference_labels.reliability.sample.jsonl
│
├── run_logs/                          # 运行日志(~130+个.log文件)
│   ├── random6_20260420/              # Random6主实验(30 log)
│   ├── random6_20260420_trackfix/     # 跟踪修正(7 log)
│   ├── random6_20260420_yolo11x_object/ # YOLO11x目标(30 log)
│   ├── asr_ablation_smoke_20260420_asr/ # ASR冒烟测试
│   ├── asr_ablation_smoke_20260420_asr_front/
│   ├── asr_ablation_random6_small_cpu/   # CPU小模型消融
│   ├── asr_ablation_random6_small_cuda/  # CUDA消融(30+ log)
│   └── *.csv (10个CSV汇总文件)
│
├── real_cases/                        # 真实案例分析数据
│   ├── README.md                      # 真实案例说明
│   ├── output_path_policy.md          # 输出路径策略
│   ├── pilot_cases.md                 # 试点案例
│   ├── index.json / pilot_cases.json  # 案例索引
│   ├── pilot_run_status.json          # 运行状态
│   ├── annotation_candidates.jsonl    # 标注候选
│   └── ~15个CSV文件(案例清单、指标汇总、覆盖度等)
│
└── gold/                              # Gold标注数据
    ├── README.md                      # Gold标注说明
    ├── annotation_checklist.md        # 标注检查清单
    ├── annotation_guideline.real.md   # 标注指南
    ├── gold_events.schema.json        # Gold事件Schema
    ├── gold_events.real.schema.json   # 真实数据Gold Schema
    ├── gold_events.sample.jsonl       # Gold样例
    ├── gold_events.real.jsonl         # 真实Gold数据
    ├── gold_events.real.template.jsonl# Gold模板
    ├── splits.sample.json             # 训练/测试分割样例
    ├── splits.real.template.json      # 真实数据分割模板
    ├── annotation_packet_manifest.json # 标注包清单
    ├── second_batch_candidates.csv    # 第二批标注候选
    │
    ├── annotation_workbench/          # Wave 0: 19个证据包
    │   ├── README.md
    │   ├── tasks.jsonl, tasks.csv, case_summary.csv
    │   └── evidence/ (19个子目录, 编号1-19)
    │       └── 每个子目录含: frame_*.jpg, contact_sheet.jpg, evidence.json, README.md
    │
    ├── annotation_workbench_wave1/    # Wave 1
    ├── annotation_workbench_wave2/    # Wave 2
    └── annotation_workbench_wave3/    # Wave 3
```

## 三、是干什么的

### configs/ — 实验配置
每个JSON文件定义一个实验的参数：模型选择、数据集划分、超参数、评估指标等。被 `scripts/experiments/` 下的脚本读取执行。

### run_logs/ — 运行日志
记录每次实验执行的完整输出，包括每帧处理结果、错误信息、性能数据。用于调试和结果复现。

### real_cases/ — 真实案例
从真实课堂视频中选取的案例，包含案例索引、运行状态、标注候选和各类指标CSV。

### gold/ — Gold标注
**论文实验的核心**。Gold标注是人工标注的"标准答案"，用于训练和评估可学习验证器(verifier/)。

- `annotation_workbench/` — 标注工作台(Wave 0)，19个标注任务包，每个任务含关键帧图片、联系表、证据JSON和说明
- `annotation_workbench_wave1/2/3/` — 后续批次的标注任务

## 四、有什么用

1. **实验复现**：configs + run_logs 确保实验可追溯复现
2. **Gold标准**：人工标注数据是verifier训练和评估的基础
3. **论文数据**：所有图表和表格的数据来源
4. **持续标注**：annotation_workbench支持多批次标注工作流

## 五、维护注意事项

- **不要修改Gold标注数据**（gold/下的.jsonl文件），除非经过正式的标注修订流程
- run_logs/ 下的.log文件是自动生成的，但建议保留以便追溯
- annotation_workbench/ 下的证据图片(contact_sheet.jpg, frame_*.jpg) 占用大量空间
- 每个实验配置与 `docs/branch_reports/` 下的报告一一对应
- CSV汇总文件(如case_metrics.csv)通常由 `scripts/paper/` 下的脚本自动生成
