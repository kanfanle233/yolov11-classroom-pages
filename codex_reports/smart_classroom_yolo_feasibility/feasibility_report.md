# 智慧课堂 YOLO 微调可行性检验

生成时间：2026-04-23  
项目根目录：`F:\PythonProject\pythonProject\YOLOv11`

## 1. 结论

可以微调 YOLO，但当前数据最适合微调 **YOLO 检测模型**，不适合直接微调 `yolo11s-pose.pt`。

原因很明确：`data\智慧课堂学生行为数据集\案例\智慧课堂学生行为数据集案例（正方视角）` 里的标注 JSON 只有行为类别和人体框：

```json
{"labels":[{"name":"js","x1":664,"y1":240,"x2":732,"y2":313}]}
```

这能转换成 YOLO detect 的 `class x_center y_center width height` 标签，但没有 17 个关键点坐标和可见性字段，因此不能作为 pose 任务的监督数据。当前姿态模型可以继续作为预训练推理模块使用；若要微调 pose，需要重新标注 COCO-pose 风格的关键点数据。

## 2. 本地数据遍历结果

### 2.1 `data` 总体

`data` 下共 60,423 个文件、43 个目录：

| 顶层目录 | 文件数 | 说明 |
|---|---:|---|
| `智慧课堂学生行为数据集` | 30,606 | 原始视频 + 案例截图标注 |
| `processed` | 17,772 | 已转换 YOLO 数据集 |
| `SCB-Dataset3 yolo dataset` | 12,045 | 另一个 YOLO 格式数据源 |

按后缀统计：`.jpg` 18,459、`.txt` 14,906、`.mp4` 12,836、`.json` 8,887、`.png` 5,329、`.zip` 3、`.yaml` 1、`.csv` 1、`.log` 1。

### 2.2 智慧课堂原始数据

`data\智慧课堂学生行为数据集\readme.txt` 说明：数据来自 2019 年度小学语文 1-6 年级部级优课课堂视频，共 131 节，按镜头切割，包含四类视角：

| 视角 | 片段数 | 当前状态 |
|---|---:|---|
| 正方视角 | 593 | 原始 `.mp4` |
| 斜上方视角 | 4,781 | 原始 `.mp4`，分在 `斜上方视角1/2` |
| 后方视角 | 4,544 | 原始 `.mp4` |
| 教师视角 | 2,918 | 原始 `.mp4` |

真正带行为框标注的是 `案例` 子目录，不是所有视频目录。

### 2.3 案例截图标注

`案例\README.txt` 说明：案例数据从正方视角视频每秒抽取一张截图并标注 8 类学生行为：

| 类别 | 编码 | README 标注数 |
|---|---|---:|
| 低头写字 | `dx` | 72,462 |
| 低头看书 | `dk` | 58,932 |
| 抬头听课 | `tt` | 117,528 |
| 转头 | `zt` | 5,339 |
| 举手 | `js` | 4,183 |
| 站立 | `zl` | 4,101 |
| 小组讨论 | `xt` | 4,663 |
| 教师指导 | `jz` | 680 |

文件配对检查：

| 项 | 数量 |
|---|---:|
| `.jpg` | 8,884 |
| `.json` | 8,884 |
| 成功配对 | 8,883 |
| 缺 JSON | `7355.jpg` |
| 缺图片 | `bb_7355.json` |

### 2.4 已转换 YOLO 数据

`data\processed\classroom_yolo` 已经是可训练数据：

| 项 | 数量 |
|---|---:|
| 图片 | 8,883 |
| 标签 | 8,883 |
| train 图片/标签 | 7,416 / 7,416 |
| val 图片/标签 | 1,467 / 1,467 |
| 保留框 | 267,861 |
| 空标签图片 | 0 |
| 损坏图片 | 0 |
| 缺失 JSON 图片 | 1 |
| 无效框/未知类 | 0 / 0 |

类别顺序是：

```yaml
0: tt
1: dx
2: dk
3: zt
4: xt
5: js
6: zl
7: jz
```

注意：这套类别顺序和 `output\case_yolo\data.yaml` 不同。

### 2.5 既有训练集与权重

`output\case_yolo` 也是 YOLO detect 数据集：

| split | 图片数 | 标签数 |
|---|---:|---:|
| train | 7,106 | 7,106 |
| val | 888 | 888 |
| test | 889 | 889 |

类别顺序是：

```yaml
0: dx
1: dk
2: tt
3: zt
4: js
5: zl
6: xt
7: jz
```

已有权重：

```text
runs\detect\case_yolo_train\weights\best.pt
runs\detect\case_yolo_train\weights\last.pt
```

本地读取 `best.pt` 的 `names` 为：

```python
{0: 'dx', 1: 'dk', 2: 'tt', 3: 'zt', 4: 'js', 5: 'zl', 6: 'xt', 7: 'jz'}
```

所以：现有 `best.pt` 只能按 `output\case_yolo\data.yaml` 的类别顺序解释；不能直接拿 `data\processed\classroom_yolo\dataset.yaml` 的顺序解释，否则 `tt/dx/dk/js/zl/xt` 会错位。

既有训练参数：`yolo11s.pt`、detect 任务、80 epochs、`imgsz=832`、`batch=8`、`device=0`。最后一轮指标：precision 0.88629、recall 0.88692、mAP50 0.93345、mAP50-95 0.81140。

### 2.6 本地运行环境

```text
ultralytics 8.3.252
torch 2.7.1+cu118
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

训练/验证前需要设置：

```powershell
$env:YOLO_CONFIG_DIR="F:\PythonProject\pythonProject\YOLOv11\.ultralytics"
```

否则 `ultralytics` 会尝试访问用户目录 settings，当前沙箱会报权限错误。

## 3. 可行性判断

### 3.1 可立即做的方向

可以立即做 8 类课堂行为检测微调。推荐基座是 `yolo11s.pt` 或 `yolo11m.pt`，任务是 detect，不是 pose。

最稳妥的第一轮是基于 `data\processed\classroom_yolo\dataset.yaml` 重新训练一个新 run，避免覆盖已有 `case_yolo_train`：

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
  --name "classroom_yolo_video_split_s"
```

训练后必须用新 run 的 `weights\best.pt` 和同一份 `dataset.yaml` 绑定解释类别。

### 3.2 不建议直接做的方向

不建议直接用这些 JSON 微调 `yolo11s-pose.pt`。当前 JSON 没有关键点字段，只有检测框。Ultralytics pose 数据集需要在每个对象后包含关键点坐标和可见性，并在 YAML 中定义 `kpt_shape`。如果硬把检测框数据喂给 pose 训练，任务类型和标签结构不匹配。

### 3.3 图中方案的落地边界

图片里的方案是“姿态 + 物品 + 时序/融合”的路线，工程上可行，但不能只靠当前案例 JSON 一步到位。

可落地的拆分是：

| 模块 | 当前是否具备监督数据 | 建议 |
|---|---|---|
| 行为框检测 | 具备 | 直接微调 YOLO detect |
| 人体姿态关键点 | 不具备 | 先用 `yolo11s-pose.pt`/`yolo11x-pose.pt` 推理；若要微调需重标 17 点 |
| 物品检测 | 部分具备，另有 SCB 数据 | 需要统一类别和数据质量后再训练 |
| 时序动作 | 当前案例为抽帧标注 | 用跟踪/规则/时序模型融合，不能只看单帧 |
| 场景标签/区域标签 | 当前没有明确监督 | 需要额外标注教师区、学生区、课桌区域等 |

## 4. 主要风险

1. **类别不均衡严重**：`jz` 只有 680 框，`js/zl/xt/zt` 都在 4k-5k 量级，而 `tt` 超过 117k。直接训练会偏向 `tt/dx/dk`。

2. **类别映射已经存在两套**：`data\processed\classroom_yolo` 和 `output\case_yolo` 的 class id 顺序不同。所有评估、可视化、论文表格必须绑定权重对应的 YAML。

3. **现有 `output\case_yolo` 是随机图片级切分**：同一课堂视频附近帧可能同时出现在 train/val/test，指标可能偏乐观。论文或严谨实验应优先用视频级/片段级切分。

4. **当前数据只有正方视角行为框标注**：如果目标是多视角泛化，后方、教师、斜上方视频目前没有同等行为框标注，直接泛化会有域偏移。

5. **单帧检测不能充分表达时序行为**：如“转头”“小组讨论”“教师指导”在短时序上更可靠，建议与跟踪、姿态和规则融合。

## 5. 推荐执行路线

第一阶段：不改模型结构，先用 `data\processed\classroom_yolo` 训练 YOLO11s detect baseline，并固定类别映射。

第二阶段：做严格验证。按视频/片段划分 train/val/test，报告每类 AP、混淆矩阵，重点看 `jz/js/zl/xt/zt` 少数类。

第三阶段：补强少数类。对 `jz/js/zl/xt/zt` 做定向采样和轻量增强，避免用过强旋转/裁剪破坏课堂几何。

第四阶段：融合现有姿态管线。保持 pose 模型为推理模块，输出关键点和 track，再把行为检测结果映射到人轨迹上。

第五阶段：如果确实要“微调 pose”，重新标注一小批课堂场景 17 点关键点，并按 Ultralytics pose 格式单独建数据集。

## 6. 参考资料

- Ultralytics 检测数据集格式：<https://docs.ultralytics.com/datasets/detect/>
- Ultralytics 姿态数据集格式：<https://docs.ultralytics.com/datasets/pose/>
- Ultralytics 训练文档：<https://docs.ultralytics.com/modes/train/>
- Ultralytics YOLO11 模型文档：<https://docs.ultralytics.com/models/yolo11/>
- SCB-dataset GitHub：<https://github.com/Whiffe/SCB-dataset>
- SCB-dataset arXiv：<https://arxiv.org/abs/2304.02488>
