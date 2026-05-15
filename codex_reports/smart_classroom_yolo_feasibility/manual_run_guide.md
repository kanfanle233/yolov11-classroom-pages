# Smart Classroom Orchestrator 执行记录与开跑指令

更新日期：2026-04-23

## 1. 当前状态

- 已完成分层目录与脚本编排：
  - `scripts/00_common`
  - `scripts/10_train`
  - `scripts/20_scb`
  - `scripts/30_infer`
  - `scripts/90_tests`
  - `profiles/full_integration_001.yaml`
  - `orchestrator.py`
- 已实现统一 CLI：
  - `--plan`
  - `--stage`
  - `--dry_run`
  - `--emit_only`
  - `--run_id`
- 已实现阶段间关联协议（manifest）：
  - `stage_results[stage_name].inputs`
  - `stage_results[stage_name].outputs`
  - `stage_results[stage_name].command`
  - `stage_results[stage_name].status`
  - `stage_results[stage_name].started_at`
  - `stage_results[stage_name].finished_at`

## 2. 已完成测试

- 编排冒烟：
  - `dryrun_full_integration_003` 运行成功，生成：
    - `runs/dryrun_full_integration_003/manifest.json`
    - `runs/dryrun_full_integration_003/commands.ps1`
- 命令完整性检查：
  - `commands.ps1` 含 `asr_backend=whisper`
  - `commands.ps1` 含视频路径 `data/智慧课堂学生行为数据集/正方视角/001.mp4`
- 链路缺失输入测试：
  - 人工删除 `infer_full_pipeline.meta.required_relpaths` 后，
  - `--stage validate_outputs` 正确失败（非静默继续）。
- 编码兼容修复：
  - `commands.ps1` 已改为 `UTF-8 BOM` 写出，兼容 Windows PowerShell 非 ASCII 路径。

## 3. 已生成可直接使用的命令清单

- 推荐直接使用：
  - `codex_reports/smart_classroom_yolo_feasibility/runs/manual_ready_001/commands.ps1`

该 run 是通过 `--emit_only 1` 生成，不会触发重任务。

## 4. 手动开跑（你执行）

先进入项目根目录：

```powershell
Set-Location "F:\PythonProject\pythonProject\YOLOv11"
```

### 4.1 先仅生成命令（不执行）

```powershell
& "F:\miniconda\envs\pytorch_env\python.exe" `
  ".\codex_reports\smart_classroom_yolo_feasibility\orchestrator.py" `
  --plan full_integration_001 `
  --emit_only 1 `
  --run_id manual_ready_001
```

### 4.2 手动执行完整集成推理（重任务）

```powershell
& "F:\miniconda\envs\pytorch_env\python.exe" `
  ".\codex_reports\smart_classroom_yolo_feasibility\orchestrator.py" `
  --plan full_integration_001 `
  --stage infer_full_pipeline `
  --run_id run_full_001
```

### 4.3 执行输出校验

```powershell
& "F:\miniconda\envs\pytorch_env\python.exe" `
  ".\codex_reports\smart_classroom_yolo_feasibility\orchestrator.py" `
  --plan full_integration_001 `
  --stage validate_outputs `
  --run_id run_full_001
```

### 4.4 可选：单独手动训练（默认未自动开启）

```powershell
& "F:\miniconda\envs\pytorch_env\python.exe" `
  ".\codex_reports\smart_classroom_yolo_feasibility\orchestrator.py" `
  --plan full_integration_001 `
  --stage train_wisdom_s `
  --run_id train_s_001
```

```powershell
& "F:\miniconda\envs\pytorch_env\python.exe" `
  ".\codex_reports\smart_classroom_yolo_feasibility\orchestrator.py" `
  --plan full_integration_001 `
  --stage train_wisdom_m `
  --run_id train_m_001
```

## 5. 目标产物（validate_outputs 将严格检查）

- `pose_keypoints_v2.jsonl`
- `pose_tracks_smooth.jsonl`
- `transcript.jsonl`
- `event_queries.jsonl`
- `align_multimodal.json`
- `verified_events.jsonl`
- `pipeline_manifest.json`
