# 索引文档体系

本目录由 `refresh_index.py` 自动生成或刷新，用于让后续协作者快速理解项目结构、主流程、脚本定位、前后端现状和风险边界。

## 一键刷新

```powershell
& F:/miniconda/envs/pytorch_env/python.exe ./索引/refresh_index.py --dry_run 1
& F:/miniconda/envs/pytorch_env/python.exe ./索引/refresh_index.py --dry_run 0
```

## 关键输出
- `00_全局总览.md`
- `01_运行流程总图.md`
- `02_快速定位.md`
- `03_风险与异常.md`
- `目录说明/*.md`
- `data/file_inventory_all.csv`
- `data/source_scope_inventory.csv`
- `data/output_inventory.csv`
- `data/output_summary.json`
- `data/entrypoints.json`
- `data/pipeline_steps.json`
- `data/index_stats.json`
- `data/risks.json`
- `data/server_frontend_state.json`
- `data/yolo_paper_docs.json`

## 口径说明
- `roots` 默认扫描 `codex_reports scripts output server web_viz yolo论文`。
- `scope=self_first`：目录说明跳过缓存目录与 `scripts/pipeline/ultralytics`。
- `yolo论文/` 中的背景/研究报告可用于解释语境，但方案文档默认按“规划/待落地”处理。
