# 具体执行步骤

生成时间：2026-04-23  
项目根目录：`F:\PythonProject\pythonProject\YOLOv11`

## 总目标

先把项目做成一个能跑通、能出实验结果、能支撑论文叙事的深度学习系统：

视觉检测 baseline -> SCB 清洗与外部验证 -> 姿态序列推理 -> 文本语义流处理 -> 视觉-语义双重验证 -> 消融实验与论文图表。

## Step 0：固定实验目录和类别映射

目的：先防止类别顺序混乱。

必须统一记录两套 YAML：

| 数据集 | YAML | 类别顺序 |
|---|---|---|
| 智慧课堂 processed | `data\processed\classroom_yolo\dataset.yaml` | `tt, dx, dk, zt, xt, js, zl, jz` |
| 旧 case_yolo | `output\case_yolo\data.yaml` | `dx, dk, tt, zt, js, zl, xt, jz` |

执行要求：

| 项 | 要求 |
|---|---|
| 新训练 | 优先使用 `data\processed\classroom_yolo\dataset.yaml` |
| 旧权重解释 | `runs\detect\case_yolo_train\weights\best.pt` 必须配 `output\case_yolo\data.yaml` |
| 禁止事项 | 不要拿旧权重按 processed 的类别顺序解释 |

验收标准：每个训练 run 的 `args.yaml` 里能看到唯一数据 YAML。

## Step 1：训练智慧课堂 8 类 YOLOv11 baseline

目的：得到项目主视觉模型。

运行命令：

```powershell
$env:YOLO_CONFIG_DIR="F:\PythonProject\pythonProject\YOLOv11\.ultralytics"
& "F:\miniconda\envs\pytorch_env\python.exe" `
  ".\scripts\intelligence_class\training\03_train_case_yolo.py" `
  --data "data\processed\classroom_yolo\dataset.yaml" `
  --model "yolo11s.pt" `
  --epochs 80 `
  --imgsz 832 `
  --batch 8 `
  --device 0 `
  --workers 4 `
  --name "wisdom8_yolo11s_detect_v1"
```

输出：

```text
runs\detect\wisdom8_yolo11s_detect_v1\weights\best.pt
runs\detect\wisdom8_yolo11s_detect_v1\results.csv
runs\detect\wisdom8_yolo11s_detect_v1\confusion_matrix.png
```

验收标准：

| 指标 | 最低要求 |
|---|---|
| 训练完成 | 有 `best.pt` |
| mAP50 | 不低于旧模型附近水平，目标约 0.90+ |
| 少数类 | 单独检查 `zt/js/zl/xt/jz` AP |
| 类别解释 | 预测图类别必须符合 `tt,dx,dk,zt,xt,js,zl,jz` 顺序 |

## Step 2：训练更大模型做对比

目的：判断 YOLOv11s 是否够用。

建议只做一个更大模型，先选 `yolo11m.pt`，不要一开始就堆很多模型。

```powershell
$env:YOLO_CONFIG_DIR="F:\PythonProject\pythonProject\YOLOv11\.ultralytics"
& "F:\miniconda\envs\pytorch_env\python.exe" `
  ".\scripts\intelligence_class\training\03_train_case_yolo.py" `
  --data "data\processed\classroom_yolo\dataset.yaml" `
  --model "yolo11m.pt" `
  --epochs 80 `
  --imgsz 832 `
  --batch 6 `
  --device 0 `
  --workers 4 `
  --name "wisdom8_yolo11m_detect_v1"
```

验收标准：如果 `mAP50-95` 和少数类 AP 没明显提升，就保留 `yolo11s` 作为主模型，论文里说明轻量模型更适合实时课堂场景。

## Step 3：清洗 SCB-Dataset3

目的：把 SCB 变成可复现实验数据，而不是直接混进智慧课堂训练集。

需要修复的问题：

| 子集 | 问题 |
|---|---|
| `0.355k_university_yolo_Dataset` | 1 行负宽度框 |
| `0.671k_university_yolo_Dataset` | 2 行负宽度框，目录多一层 |
| `5k_HRW_yolo_Dataset` | 图片目录不标准，1 行坐标越界 |

建议产出新目录，不修改原始数据：

```text
data\processed\scb_yolo_clean\
  0.355k_university\
  0.671k_university\
  5k_HRW\
```

清洗规则：

| 标签问题 | 处理方式 |
|---|---|
| `w <= 0` 或 `h <= 0` | 删除该框 |
| `cx/cy/w/h` 超出 `[0,1]` | 优先裁剪到边界，裁剪后无效则删除 |
| 图片无标签 | 保留空 txt，但单独统计 |
| 图片和标签不配对 | 不进入 clean 数据 |

验收标准：

| 项 | 要求 |
|---|---|
| 图片/标签 | 100% 配对 |
| 标签列数 | 每行 5 列 |
| 坐标范围 | `cx,cy,w,h` 全部在 `[0,1]` |
| 宽高 | `w > 0` 且 `h > 0` |
| YAML | 每个子集有自己的 `data.yaml` |

## Step 4：SCB 单独训练，不与智慧课堂直接合并

目的：把 SCB 作为外部数据能力验证。

先分别训练：

```powershell
$env:YOLO_CONFIG_DIR="F:\PythonProject\pythonProject\YOLOv11\.ultralytics"
& "F:\miniconda\envs\pytorch_env\python.exe" `
  ".\scripts\intelligence_class\training\03_train_case_yolo.py" `
  --data "data\processed\scb_yolo_clean\5k_HRW\data.yaml" `
  --model "yolo11s.pt" `
  --epochs 80 `
  --imgsz 832 `
  --batch 8 `
  --device 0 `
  --workers 4 `
  --name "scb_hrw_yolo11s_detect_v1"
```

验收标准：得到 SCB 自己的检测模型和指标，但不把它当作智慧课堂 8 类模型。

## Step 5：做 SCB -> 智慧课堂预训练微调实验

目的：验证外部课堂数据是否提升智慧课堂模型。

流程：

| 子步骤 | 操作 |
|---|---|
| 5.1 | 用清洗后的 SCB 训练 `scb_pretrain.pt` |
| 5.2 | 用 `scb_pretrain.pt` 作为初始权重训练智慧课堂 8 类 |
| 5.3 | 与 Step 1 的 `yolo11s.pt -> 智慧课堂` baseline 对比 |

示例命令：

```powershell
$env:YOLO_CONFIG_DIR="F:\PythonProject\pythonProject\YOLOv11\.ultralytics"
& "F:\miniconda\envs\pytorch_env\python.exe" `
  ".\scripts\intelligence_class\training\03_train_case_yolo.py" `
  --data "data\processed\classroom_yolo\dataset.yaml" `
  --model "runs\detect\scb_hrw_yolo11s_detect_v1\weights\best.pt" `
  --epochs 80 `
  --imgsz 832 `
  --batch 8 `
  --device 0 `
  --workers 4 `
  --name "wisdom8_from_scb_yolo11s_detect_v1"
```

验收标准：如果少数类 AP 或跨视频泛化提升，就把它作为论文的跨域预训练实验；如果没有提升，就作为负结果说明类别不一致导致迁移有限。

## Step 6：跑姿态序列，不直接微调 pose

目的：获得论文中“姿态序列”的视觉证据。

先使用现有 pose 模型推理：

```powershell
$env:YOLO_CONFIG_DIR="F:\PythonProject\pythonProject\YOLOv11\.ultralytics"
& "F:\miniconda\envs\pytorch_env\python.exe" `
  ".\scripts\02_export_keypoints_jsonl.py" `
  --video "data\智慧课堂学生行为数据集\正方视角\某个视频.mp4" `
  --model "yolo11x-pose.pt" `
  --out "output\pose_keypoints_v2.jsonl"
```

再做跟踪和平滑：

```powershell
& "F:\miniconda\envs\pytorch_env\python.exe" `
  ".\scripts\03_track_and_smooth.py" `
  --in "output\pose_keypoints_v2.jsonl" `
  --out "output\pose_tracks_smooth.jsonl"
```

验收标准：

| 项 | 要求 |
|---|---|
| 关键点 | 每帧能输出人体 17 点 |
| track | 同一学生 track 不频繁跳变 |
| 可视化 | 姿态 overlay 视频能看出动作变化 |

## Step 7：加入行为检测输出到视频流程

目的：把 Step 1 的行为检测模型接入现有管线。

示例：

```powershell
$env:YOLO_CONFIG_DIR="F:\PythonProject\pythonProject\YOLOv11\.ultralytics"
& "F:\miniconda\envs\pytorch_env\python.exe" `
  ".\scripts\02d_export_behavior_det_jsonl.py" `
  --video "data\智慧课堂学生行为数据集\正方视角\某个视频.mp4" `
  --model "runs\detect\wisdom8_yolo11s_detect_v1\weights\best.pt" `
  --out "output\behavior_det.jsonl"
```

验收标准：每个检测事件包含 frame、time、bbox、class、confidence。

## Step 8：做视觉事件融合

目的：把姿态、行为检测、物品检测合成视觉候选事件。

建议先用规则，不急着上复杂模型：

| 事件 | 视觉规则 |
|---|---|
| 举手 | `js` 检测框 + 手腕/肘部高于肩部 |
| 低头写字 | `dx` 检测框 + 头部下倾 + 手部靠近桌面 |
| 低头看书 | `dk` 检测框 + 书本区域约束 |
| 小组讨论 | `xt` 检测框 + 多人头部朝向变化 + 相邻轨迹 |
| 教师指导 | `jz` 检测框 + 教师接近学生区域 |

输出：

```text
output\actions_fused.jsonl
```

验收标准：每条事件能回溯到至少一种视觉证据，最好包含姿态和行为检测两个来源。

## Step 9：处理文本语义流

目的：把课堂音频或 OCR 文本变成可对齐事件。

输出格式建议：

```json
{"t0": 12.4, "t1": 16.8, "event": "teacher_question", "text": "谁来回答这个问题"}
```

事件类型先定义为：

| 事件类型 | 关键词示例 |
|---|---|
| `teacher_question` | 谁来回答、请你说、举手 |
| `writing_instruction` | 写下来、记录、抄写 |
| `reading_instruction` | 看书、读一读、翻到 |
| `discussion_instruction` | 同桌讨论、小组交流 |
| `teacher_guidance` | 这里、这个地方、注意看 |

验收标准：每条文本事件有时间窗、事件类型、原始文本。

## Step 10：做视觉-语义双重验证

目的：完成论文核心创新。

融合规则：

| 情况 | 判断 |
|---|---|
| 视觉强、文本强 | 高可信事件 |
| 视觉强、文本弱 | 保留为视觉事件，但置信度降低 |
| 视觉弱、文本强 | 标记为待复核候选 |
| 视觉与文本冲突 | 标记为冲突样本，用于误判分析 |

输出：

```text
output\verified_events.jsonl
output\align_multimodal.json
```

验收标准：每个最终事件都能解释“视觉证据是什么、文本证据是什么、为什么通过或不通过验证”。

## Step 11：做消融实验

目的：证明系统级框架确实比单一 YOLO 检测更强。

至少做 4 组：

| 实验 | 使用模块 |
|---|---|
| A | YOLO 行为检测 |
| B | YOLO 行为检测 + pose 序列 |
| C | YOLO 行为检测 + pose 序列 + 物品约束 |
| D | YOLO 行为检测 + pose 序列 + 物品约束 + 文本语义 |

指标：

| 指标 | 说明 |
|---|---|
| Precision | 是否减少误判 |
| Recall | 是否减少漏检 |
| F1 | 综合表现 |
| Conflict Rate | 视觉和文本冲突比例 |
| Case Study | 典型误判纠正案例 |

验收标准：D 组至少在误判减少或解释性上明显优于 A 组。

## Step 12：论文成果整理

最终项目表：

| 成果 | 文件/目录 |
|---|---|
| 智慧课堂主检测权重 | `runs\detect\wisdom8_yolo11s_detect_v1\weights\best.pt` |
| SCB 清洗数据 | `data\processed\scb_yolo_clean` |
| SCB 预训练实验 | `runs\detect\scb_*` |
| 姿态序列 | `pose_tracks_smooth.jsonl` |
| 行为检测事件 | `behavior_det.jsonl` |
| 融合行为事件 | `actions_fused.jsonl` |
| 文本事件 | `event_queries.jsonl` |
| 双重验证结果 | `verified_events.jsonl` |
| 论文图表 | 混淆矩阵、PR 曲线、消融表、案例图 |

论文叙事顺序：

| 章节 | 内容 |
|---|---|
| 方法 | YOLOv11 视觉候选、姿态序列、文本语义流、双重验证 |
| 数据 | 智慧课堂为主，SCB 为外部/迁移数据 |
| 实验 | baseline、SCB 迁移、融合消融 |
| 讨论 | 遮挡、少数类、跨域、文本冲突样本 |

## 最小可行版本

如果只做最小版本，按这个顺序执行：

| 顺序 | 必做项 |
|---:|---|
| 1 | 训练 `wisdom8_yolo11s_detect_v1` |
| 2 | 在 3-5 个真实课堂视频上导出行为检测 JSONL |
| 3 | 用 `yolo11x-pose.pt` 导出姿态轨迹 |
| 4 | 人工写 20-50 条文本事件 query |
| 5 | 做视觉-语义对齐和验证 |
| 6 | 输出单模态 vs 双模态的对比表 |

这个版本最适合作为课程深度学习项目或论文初稿原型。
