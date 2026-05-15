# codex_reports/ 目录 — 完整维护报告

> **位置**：`F:\PythonProject\pythonProject\YOLOv11\codex_reports\`
> **报告生成日期**：2026-05-03

## 一、大致背景

`codex_reports/` 是使用 **Codex CLI** 工具编排的实验管理和自动化执行模块。它提供了一种声明式的方式来定义实验流水线的执行计划，支持dry-run（预演）和full-run（完整执行）模式。

## 二、位置与目录结构

```
codex_reports/smart_classroom_yolo_feasibility/
│
├── feasibility_report.md              # 可行性分析报告
├── project_execution_plan.md          # 项目执行计划
├── concrete_execution_steps.md        # 具体执行步骤
├── manual_run_guide.md                # 手动运行指南
├── integration_interface_audit_20260425.md  # 集成接口审计
├── semantic_bridge_debug_report_20260425.md # 语义桥接调试报告
├── orchestrator.py                    # 编排器核心脚本
│
├── profiles/                          # 流水线配置文件(3个YAML)
│   ├── full_integration_001.yaml      # 全量集成配置
│   ├── full_fusion_contract_v2.yaml   # 融合契约V2配置
│   └── action_semantics_8class.yaml   # 8类动作语义配置
│
├── scripts/                           # Codex编排的执行脚本
│   ├── 00_common/                     # 公共基础(5个文件)
│   │   ├── pathing.py                 # 路径管理
│   │   ├── manifest_store.py          # Manifest存储
│   │   ├── command_executor.py        # 命令执行器
│   │   ├── log_utils.py               # 日志工具
│   │   └── config_loader.py           # 配置加载器
│   │
│   ├── 10_train/train_wisdom.py       # 智慧课堂数据训练
│   ├── 20_scb/scb_tasks.py            # SCB数据集任务
│   ├── 30_infer/run_full_integration.py # 全量推理
│   ├── 40_semantics/                  # 语义处理(2个文件)
│   │   ├── semantic_bridge.py         # 语义桥接
│   │   └── backfill_semantic_runs.py  # 语义回填
│   ├── 50_fusion_contract/            # 融合契约(8个文件)
│   │   ├── fusion_utils.py            # 融合工具
│   │   ├── semanticize_behavior_det.py # 行为检测语义化
│   │   ├── semanticize_objects.py     # 目标语义化
│   │   ├── check_fusion_contract.py   # 融合契约检查
│   │   ├── check_pipeline_contract_v2.py # 流水线契约检查V2
│   │   ├── build_event_queries_fusion_v2.py # 事件查询构建
│   │   ├── run_full_fusion_v2.py      # 全量融合运行
│   │   ├── merge_fusion_actions_v2.py # 融合动作合并
│   │   └── behavior_det_to_actions_v2.py # 行为检测→动作
│   ├── 60_reports/build_paper_pipeline_report.py # 论文报告生成
│   └── 90_tests/                      # 测试(4个文件)
│       ├── check_outputs.py           # 输出检查
│       ├── test_stage_linkage.py      # 阶段链接测试
│       ├── check_semantic_outputs.py  # 语义输出检查
│       └── test_fusion_contract_v2.py # 融合契约测试V2
│
├── runs/                              # 运行记录(8个子目录)
│   ├── dryrun_full_integration_001/   # 预演运行
│   ├── dryrun_full_integration_002/
│   ├── dryrun_full_integration_003/
│   ├── linkage_failure_001/           # 链接失败记录
│   ├── manual_ready_001/              # 手动准备记录
│   ├── run_full_001/                  # 完整运行
│   ├── run_full_wisdom8_001/          # Wisdom8运行
│   └── dryrun_semantic_bridge_001/    # 语义桥接预演
│
├── paper_assets/                      # 论文资产(3个运行目录)
│   ├── run_full_wisdom8_001/          # 16个文件(PNG, SVG, JSON, CSV, MD)
│   ├── run_full_e150_001/             # 14个文件
│   └── run_full_paper_mainline_001/   # 7个文件
│
├── paper_package_20260426/            # 20260426论文打包
│   ├── asset_manifest.csv
│   ├── 01_figures_detection/          # 12个检测图表(PNG, JPG)
│   ├── 02_figures_pipeline/           # 4个流水线图表(SVG, PNG)
│   └── 03_metrics_tables/            # 4个指标表格(MD, CSV, JSON)
│
└── project_audit_20260426/            # 20260426项目审计
    ├── file_inventory.csv
    └── unreadable_files.csv
```

## 三、是干什么的

- **orchestrator.py** — 编排器，读取profiles配置，按顺序执行流水线步骤
- **profiles/***.yaml — 声明式定义流水线的步骤、输入输出、参数
- **scripts/** — 各阶段的执行脚本（与主 `scripts/pipeline/` 功能有重叠但更工程化）
- **runs/** — 每次运行的记录（manifest.json记录步骤和状态，commands.ps1是生成的命令）
- **paper_assets/** — 从运行结果中提取的论文图表和数据

## 四、有什么用

1. **实验自动化**：通过profiles声明式定义实验，减少手动操作
2. **dry-run支持**：预演模式在真正执行前验证流水线配置
3. **可追溯性**：每次运行生成manifest.json记录完整执行链路
4. **论文资产提取**：自动从运行结果中提取图表和表格

## 五、维护注意事项

- 与 `scripts/pipeline/` 有功能重叠（如推理、融合），需注意同步
- profiles/ 的YAML配置是实验的"入口"，修改时需确保对应的scripts存在
- runs/ 下的每个子目录是某次运行的记录，占用空间较大，可清理旧的dry-run
- paper_package/ 是论文投稿的素材包，不要随意修改结构
- 这个目录下还有独立的Python脚本集（scripts/），与主项目脚本有依赖关系
