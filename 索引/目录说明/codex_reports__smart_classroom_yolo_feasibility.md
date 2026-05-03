# 目录说明：`codex_reports/smart_classroom_yolo_feasibility`

- 代码文件数：`1`
- 代码行数（LOC）：`608`

## 目录职责
- 统一编排入口和 profile 驱动的实验任务目录。
- 把阶段命令、输入输出和状态写入 manifest，支持 dry-run、emit-only 与验证。

## 入口脚本
- `codex_reports/smart_classroom_yolo_feasibility/orchestrator.py` (line 607)

## 上游依赖（静态线索）
- profiles/*.yaml
- scripts/pipeline/00_common

## 下游产物（静态线索）
- runs/* manifest/commands/logs

## 文件清单
| 文件 | LOC | 入口 | 说明 |
| --- | --- | --- | --- |
| orchestrator.py | 608 | yes | Unified codex_reports orchestrator. |
