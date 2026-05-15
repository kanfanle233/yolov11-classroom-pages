# 视觉-语义双重验证项目执行方案

生成时间：2026-04-23  
项目根目录：`F:\PythonProject\pythonProject\YOLOv11`

## 1. 项目定位

本项目建议定位为：基于 YOLOv11 的课堂多模态感知与视觉-语义双重验证框架。

核心目标不是只做一个学生行为检测器，而是构建一个系统级感知流程：视觉侧输出学生姿态序列、行为框、物品证据和时序轨迹；语义侧输出课堂文本流、事件描述和敏感语义片段；最后通过时间戳、空间位置和语义规则完成跨模态验证，降低遮挡、误检和单模态偏差。

## 2. 数据集关系判断

### 2.1 智慧课堂数据集

可用部分：

| 数据 | 当前内容 | 可训练任务 |
|---|---|---|
| `data\智慧课堂学生行为数据集\案例` | 8,884 张正方视角截图 + 8,884 个 JSON | YOLO detect 行为检测 |
| `data\processed\classroom_yolo` | 已转换 YOLO 数据，8,883 图，267,861 框 | YOLO detect 行为检测 |
| 正方/后方/教师/斜上方原始视频 | 12,836 段 `.mp4` | 推理、验证、抽帧补标，不是直接训练标签 |

当前 8 类行为：

| 编码 | 中文语义 | 训练建议 |
|---|---|---|
| `dx` | 低头写字 | 主类 |
| `dk` | 低头看书 | 主类 |
| `tt` | 抬头听课 | 主类 |
| `zt` | 转头 | 少数类，需重点评估 |
| `js` | 举手 | 少数类，需重点评估 |
| `zl` | 站立 | 少数类，需重点评估 |
| `xt` | 小组讨论 | 少数类，需时序辅助 |
| `jz` | 教师指导 | 极少数类，需补标或采样增强 |

### 2.2 SCB-Dataset3 yolo dataset

本地检查结果：

| 子集 | 图片/标签 | 类别数 | 是否可直接训练 | 问题 |
|---|---:|---:|---|---|
| `0.355k_university_yolo_Dataset` | 335 / 335 | 6 | 基本可以 | 1 行负宽度框 |
| `0.671k_university_yolo_Dataset` | 671 / 671 | 6 | 基本可以 | 2 行负宽度框，目录多套一层 |
| `5k_HRW_yolo_Dataset` | 5,015 / 5,015 | 3 | 需整理后可以 | 图片目录不标准，1 行坐标越界 |

### 2.3 两种数据集能否一起训练

结论：不能直接混在一起训练；可以在清洗、重映射后间接互通。

原因：

| 维度 | 智慧课堂 | SCB-Dataset3 | 结论 |
|---|---|---|---|
| 任务类型 | detect | detect | 任务类型兼容 |
| 标签格式 | YOLO 5 列框 | YOLO 5 列框 | 格式大体兼容 |
| 类别空间 | 8 类 | 本地为 6 类/3 类 | 不能直接合并 |
| 类别名称文件 | 明确 | 本地缺少统一 names YAML | 不能盲目映射 |
| 场景 | 小学语文课堂，正方视角为主 | 大学/HRW 等课堂场景 | 适合做域扩展或预训练 |
| 标注质量 | 有 1 对缺失 | 有少量非法框 | 合并前必须清洗 |

推荐用法：

| 方案 | 做法 | 推荐度 |
|---|---|---|
| 独立训练 | 智慧课堂训练 8 类主模型，SCB 单独训练对比模型 | 高 |
| 预训练再微调 | 先用清洗后的 SCB 训练课堂通用检测特征，再用智慧课堂 8 类微调 | 高 |
| 外部验证 | 用 SCB 或原始多视角视频测试主模型泛化能力 | 高 |
| 直接合并训练 | 把两个数据集简单放一起训练 | 不推荐 |
| 重映射合并 | 找到 SCB 类别语义后，把重叠类映射到 `dx/dk/tt/zt/js/zl/xt/jz` | 中，前提是 class names 明确 |

## 3. 模型路线

### 3.1 视觉侧

| 模块 | 模型 | 输入 | 输出 | 训练状态 |
|---|---|---|---|---|
| 行为检测 | YOLOv11s/m detect | 课堂帧 | 8 类行为框 | 可立即微调 |
| 姿态估计 | YOLOv11s-pose/x-pose | 视频帧 | 17 点人体关键点 | 先推理，不直接微调 |
| 物品检测 | YOLOv11 detect | 课堂帧 | 书本、手机、课桌、椅子等 | 需补物品标签或引入外部数据 |
| 跟踪平滑 | ByteTrack/BoT-SORT + 自定义平滑 | 检测框/关键点 | person track | 可工程实现 |
| 时序行为 | 规则/轻量 MLP/ST-GCN | 姿态序列 + 行为框 | 时序动作置信度 | 第二阶段实现 |

### 3.2 语义侧

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| ASR | 课堂音频 | 带时间戳文本 | 提供实时文本语义流 |
| OCR/板书识别 | 视频帧 | 屏幕/黑板文字 | 补充课堂内容线索 |
| 敏感文本识别 | ASR/OCR 文本 | 敏感词、事件 query | 触发验证需求 |
| 语义事件抽取 | 文本流 | 事件类型、时间窗、关键词 | 与视觉候选对齐 |

### 3.3 双重验证融合

| 层级 | 视觉证据 | 语义证据 | 融合判断 |
|---|---|---|---|
| 同桌讨论 | 多人头部朝向、相邻学生姿态变化 | 文本出现“讨论/交流/同桌” | 提升讨论事件可信度 |
| 举手回答 | 手腕/肘部高于头肩、YOLO 行为框为 `js` | 文本出现“请回答/谁来说” | 区分举手和伸手整理物品 |
| 教师指导 | 教师接近学生区域、学生附近出现 `jz` | 文本出现“看这里/这个地方” | 提升教师指导事件可信度 |
| 低头写字 | 头部下倾、手部靠近桌面、书本/纸张存在 | 文本出现“写下来/记录” | 区分写字和低头看书 |

## 4. 执行步骤

| 阶段 | 目标 | 输入 | 产出 | 验收标准 |
|---|---|---|---|---|
| 0 | 固化数据清单 | `data` 目录 | 数据审计表、类别映射表 | 每个权重绑定唯一 YAML |
| 1 | 训练 8 类行为检测 baseline | `data\processed\classroom_yolo` | YOLOv11s 行为检测权重 | mAP50、每类 AP、混淆矩阵完整 |
| 2 | 整理 SCB 数据 | `SCB-Dataset3 yolo dataset` | 标准 YOLO 目录 + data.yaml | 无负宽度/越界框，图片标签全配对 |
| 3 | SCB 预训练/外部验证 | 清洗后的 SCB | SCB 权重、跨域验证报告 | 明确是否提升少数类/泛化 |
| 4 | 姿态序列推理 | 原始视频 + pose 模型 | `pose_tracks_smooth.jsonl` | track 连续、关键点可视化正常 |
| 5 | 物品检测补强 | COCO 预训练或少量补标 | 物品检测 JSONL | 书本/手机/课桌等证据可输出 |
| 6 | 文本流处理 | ASR/OCR 文本 | `event_queries.jsonl` | 每条事件有时间窗和语义类型 |
| 7 | 视觉-语义对齐 | 行为/姿态/物品/文本 | `align_multimodal.json` | 事件能追溯到视觉与文本证据 |
| 8 | 双重验证评估 | gold 标注或人工复核 | 准确率、召回率、误判案例 | 单模态 vs 双模态有对比 |
| 9 | 论文实验整理 | 所有结果 | 消融表、流程图、案例图 | 能支撑“系统级感知架构”叙事 |

## 5. 推荐实验表

| 实验编号 | 实验名 | 训练数据 | 模型 | 目的 |
|---|---|---|---|---|
| E1 | Wisdom-Detect Baseline | 智慧课堂 8 类 | YOLOv11s | 主模型基线 |
| E2 | Wisdom-Detect Larger | 智慧课堂 8 类 | YOLOv11m/l | 验证模型容量收益 |
| E3 | SCB-Clean Baseline | 清洗 SCB | YOLOv11s | 外部课堂数据检测能力 |
| E4 | SCB-to-Wisdom Finetune | SCB 预训练，再智慧课堂微调 | YOLOv11s | 验证跨数据预训练是否有效 |
| E5 | Pose-Rule Fusion | 智慧课堂视频 | YOLOv11-pose + 规则 | 验证姿态序列对行为分类的补益 |
| E6 | Object-Constraint Fusion | 行为检测 + 物品检测 | YOLOv11 detect | 验证书本/手机/桌面约束 |
| E7 | Vision-Semantic Dual | 视觉候选 + ASR/OCR 事件 | 对齐/验证模块 | 验证双重验证框架 |
| E8 | Ablation | 去掉姿态/物品/文本 | 同上 | 证明每个模态的贡献 |

## 6. 关键命令建议

训练智慧课堂 8 类主模型：

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
  --name "wisdom8_yolo11s_detect"
```

如果使用已有 `output\case_yolo\data.yaml`，必须使用它对应的类别顺序：

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

如果使用 `data\processed\classroom_yolo\dataset.yaml`，类别顺序是：

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

这两套顺序不能混用。

## 7. 不建议做的事

| 不建议项 | 原因 |
|---|---|
| 用当前 JSON 直接微调 `yolo11s-pose.pt` | 没有关键点标签 |
| 把 SCB 和智慧课堂直接合并 | 类别空间不同，SCB 本地缺 names YAML |
| 只报告 mAP50 | 少数类和跨域泛化问题会被掩盖 |
| 用随机帧切分当论文强结论 | 相邻帧泄漏会导致指标偏乐观 |
| 只做单帧行为检测 | 无法体现“视觉-语义双重验证”的论文创新 |

## 8. 论文叙事建议

主线可以这样写：

1. YOLOv11 行为检测提供课堂视觉候选事件。
2. YOLOv11-pose 提供人体关键点序列和轨迹级姿态证据。
3. 物品检测提供书本、手机、课桌等上下文约束。
4. ASR/OCR 文本流提供课堂语义事件和敏感语义触发。
5. 跨模态对齐模块将视觉候选与文本事件映射到同一时间轴。
6. 双重验证模块根据视觉置信度、姿态逻辑、物品约束和语义一致性给出最终事件判断。

这样论文贡献点不是“我训练了一个 YOLO 模型”，而是“我构建了一个面向复杂课堂场景的多模态协同验证架构”。

## 9. 参考资料

- Ultralytics YOLO11 文档：<https://docs.ultralytics.com/models/yolo11/>
- Ultralytics 检测数据集格式：<https://docs.ultralytics.com/datasets/detect/>
- Ultralytics 姿态数据集格式：<https://docs.ultralytics.com/datasets/pose/>
- Ultralytics 训练文档：<https://docs.ultralytics.com/modes/train/>
- SCB-Dataset arXiv：<https://arxiv.org/abs/2304.02488>
- SCB-Dataset GitHub：<https://github.com/Whiffe/SCB-dataset>
