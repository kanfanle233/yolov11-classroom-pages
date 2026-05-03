# 索引文档体系

本目录由 `refresh_index.py` 自动生成或刷新，用于让后续 Codex 快速理解项目结构、主流程、脚本定位、后排增强实验和当前风险。

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
- `data/entrypoints.json`
- `data/pipeline_steps.json`
- `data/index_stats.json`
- `data/risks.json`

## 口径说明
- `roots` 默认只扫描 `codex_reports scripts`。
- `scope=self_first`：目录说明跳过 `scripts/pipeline/ultralytics` 和缓存目录。
- `loc_mode=full`：全量 LOC 只输出一个主指标；自研 LOC 仅作为目录说明辅助。
- 当前推荐主线：`track_backend=hybrid`，pose track 负责学生身份，8 类行为检测负责行为语义挂载。
- 当前后排增强主线：`full_sliced/rear_adaptive + 可选 ROI SR/去噪去模糊 + pose-backbone fusion`。
