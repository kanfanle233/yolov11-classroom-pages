# 论文大纲

目标篇幅：中文 8 页以内或英文 8 页以内；若面向 ChinaVis/可视分析方向，必须把“可视分析系统”和“教育分析价值”写清楚，而不是只写检测精度。

## 0. 摘要与关键词

### 摘要结构

1. **背景**：智慧课堂需要自动理解学生行为、教师互动和课堂事件，但单一视觉检测容易受遮挡、多人重叠和动作相似性影响。
2. **问题**：现有课堂行为检测多停留在视觉-only，缺少视觉行为序列与课堂文本/语音语义流之间的可靠对齐和验证。
3. **方法**：提出一个视觉-语义双重验证框架，结合 YOLOv11 微调行为检测、YOLO pose tracking、Whisper ASR 质量门控、语义桥接协议、UQ 驱动跨模态对齐和 timeline 可视分析。
4. **结果**：在当前智慧课堂数据上，e150 YOLO11s 行为检测达到 mAP50=0.933、mAP50-95=0.804；主线 demo 生成 11 个学生 ID、186 条融合动作、12 个验证事件和学生级 timeline。
5. **意义**：系统将底层检测结果提升为可审计、可解释、可视化的课堂行为分析链路。

### 关键词

智慧课堂；YOLOv11；学生行为检测；多模态感知；跨模态对齐；语义验证；可视分析；timeline

## 1. 引言

### 1.1 研究背景

智慧课堂分析从“是否检测到人/动作”转向“谁在什么时候发生了什么教学行为，以及该事件是否被其他模态支持”。课堂环境具有遮挡、多学生密集、动作细粒度相似、音频噪声大等特点，因此单一视觉或单一文本都不稳定。

### 1.2 现有问题

1. 视觉-only 方法可检测行为，但难以解释课堂语义。
2. ASR 或文本流可能为空或低质，不能直接作为事实。
3. 检测输出短码不适合进入后续语义推理。
4. 缺少学生级 timeline，难以服务教学可视分析。

### 1.3 本文贡献

1. 构建 YOLOv11 课堂行为微调模型，识别 8 类课堂动作。
2. 设计 `fusion_contract_v2` 语义桥接协议，将视觉短码转化为中英双语语义字段。
3. 提出 UQ 驱动的视觉-文本事件对齐和双重验证流程。
4. 构建学生级 timeline，把 track ID 映射为 S01/S02 等可读学生编号。
5. 给出可审计的工程链路和论文图表资产，保证中间文件不为空、不缺字段、不静默失败。

## 2. 相关工作

### 2.1 课堂行为检测

介绍 YOLOv5/YOLOv8/SlowFast 等在课堂行为检测中的使用，指出它们主要解决视觉检测精度，但较少关注语义流对齐和可视分析闭环。重点引用 Applied Sciences 2022、Sensors 2023 YOLOv8、MSTA-SlowFast。

### 2.2 多模态音视频理解

介绍 MUSIC-AVQA、MAViL、AVMIT、Video-LLaMA、InternVideo2 等工作。强调这些工作证明音视频多模态有价值，但多数不是面向课堂细粒度学生行为 timeline。

### 2.3 ASR 与文本语义流

介绍 Whisper 的鲁棒 ASR 能力，同时指出课堂音频可能噪声大、说话人远、回声强，因此需要质量门控和 fallback。

### 2.4 可视分析与智慧教育系统

强调 ChinaVis/可视分析导向下，系统不仅要输出模型指标，还要支持 timeline、学生行为分布、事件回放和证据链追踪。

## 3. 方法

### 3.1 系统总览

输入为课堂视频；输出包括行为检测结果、学生轨迹、ASR 文本、融合动作、验证事件和 timeline 图表。建议主文放一张总流程图。

### 3.2 YOLOv11 行为检测

使用 `official_yolo11s_detect_e150_v1` 作为行为检测模型，类别为：

| 代码 | 中文 | 英文 | semantic id |
|---|---|---|---|
| tt | 听课 | listening | listen |
| dx | 写字 | writing | write |
| dk | 看书 | reading | read |
| zt | 转头 | turning head | turn_head |
| xt | 小组讨论 | group discussion | group_discussion |
| js | 举手 | raise hand | raise_hand |
| zl | 站立 | standing | stand |
| jz | 教师互动 | teacher interaction | teacher_interaction |

### 3.3 Pose Tracking 与学生 ID

YOLO pose 负责人体关键点，tracking 负责生成 `track_id`。论文中要明确：学生 ID 不作为 YOLO 类别训练，不进入检测头；它是后处理层的稳定编号。

### 3.4 语义桥接协议

定义每条动作必须包含：

`track_id, action, behavior_code, semantic_id, semantic_label_zh, semantic_label_en, conf, start_time, end_time, source, taxonomy_version`

缺失这些字段则 contract 检查失败。

### 3.5 ASR 质量门控与视觉 fallback

Whisper 输出不仅保存文本，还保存 `avg_logprob/no_speech_prob/compression_ratio` 等质量信息。低质量文本写入 `asr_quality_report.json`，不进入下游语义验证；此时从高置信视觉动作生成 `event_queries.visual_fallback.jsonl`。

### 3.6 UQ 驱动跨模态对齐

给出公式：

\[
\Delta_q = clip(\Delta_0 + \alpha M_q + \beta U_q, \Delta_{min}, \Delta_{max})
\]

\[
W_q=[t_q-\Delta_q,t_q+\Delta_q]
\]

其中 \(M_q\) 表示动作/轨迹运动不稳定性，\(U_q\) 表示 pose 不确定性。

### 3.7 双重验证

定义视觉置信度：

\[
C_v=\lambda_d\bar{p}_{det}+\lambda_p(1-\bar{u}_{pose})+\lambda_o p_{obj}
\]

定义文本置信度：

\[
C_t=\sigma(w_1\cdot avg\_logprob-w_2\cdot no\_speech\_prob-w_3\cdot compression\_ratio)
\]

定义跨模态一致性：

\[
S=\alpha C_v+\beta C_t+\gamma Sim(a,q)+\delta O(a,q)-\eta UQ
\]

最终输出：

\[
D(S)=
\begin{cases}
match, & S \ge \tau_m \\
mismatch, & S < \tau_u \\
uncertain, & \tau_u \le S < \tau_m
\end{cases}
\]

## 4. 实验

### 4.1 数据集

当前数据集为 `data/processed/classroom_yolo`，已有 train/val，但 test 为空。论文正式版建议按视频或 case 重切 `train/val/test`，避免相邻帧泄漏。

### 4.2 训练设置

主模型为 YOLO11s，输入尺寸 832，epoch 150，行为模型权重为 `official_yolo11s_detect_e150_v1/weights/best.pt`。

### 4.3 对比基线

1. Vision-only：只用 YOLO 行为检测。
2. Pose-rule：只用 pose rules 生成动作。
3. Text-only：只用 ASR 事件抽取。
4. Late fusion：行为检测与文本事件后融合。
5. Ours：fusion_contract_v2 + UQ align + dual verification。

### 4.4 指标

视觉检测：mAP50、mAP50-95、Precision、Recall、F1、confusion matrix、FPS。  
跨模态验证：Accuracy、Precision、Recall、F1、Top-k candidate recall、FPR、FNR。  
校准：ECE、Brier、reliability diagram。  
系统：Latency、FPS、产物完整率、语义覆盖率、timeline 学生覆盖数。

### 4.5 当前结果

当前可写入阶段性实验：

| 项目 | 结果 |
|---|---|
| YOLO11s e150 mAP50 | 0.933 |
| YOLO11s e150 mAP50-95 | 0.804 |
| Precision / Recall | 0.887 / 0.894 |
| 主线学生 ID 数 | 11 |
| fusion 动作条数 | 186 |
| 语义有效动作条数 | 186 |
| event queries | 12 |
| align candidates | 96 |
| verified events | 12 |
| timeline 学生动作片段 | 30 |

注意：当前 ASR 在 demo 视频上被质量门控判为低可信，因此不能把这组结果写成“文本模态显著提升准确率”，只能写成“系统具备文本质量门控与视觉 fallback 能力”。

## 5. 可视分析系统

### 5.1 系统界面目标

展示学生位置、学生 ID、行为时间线、事件验证结果、文本/视觉证据链。

### 5.2 Timeline 设计

每行对应一个学生，例如 S01、S02；每个色块表示动作片段，如听课、写字、站立。点击片段可回看视频帧和 verifier 证据。

### 5.3 教育分析价值

1. 教师可以观察课堂参与度。
2. 研究者可以查看模型误检和不确定事件。
3. 系统可以解释“为什么某个事件被判定为 match/mismatch/uncertain”。

## 6. 讨论

### 6.1 当前优势

1. 检测模型已有较完整训练结果和图表。
2. 主线链路已跑通，并有 strict contract 防止空数据伪成功。
3. timeline 解决了学生级解释问题。

### 6.2 当前短板

1. 缺独立 test split。
2. `jz` 教师互动类别样本偏少。
3. ASR 目前在示例视频上质量低。
4. verifier 的 gold label 和统计样本不足。

## 7. 结论

本文构建了一个面向智慧课堂的视觉-语义双重验证框架。与传统视觉-only 行为检测不同，本文将 YOLOv11 行为检测、pose tracking、ASR 质量门控、语义桥接、UQ 对齐和学生级 timeline 统一为一条可审计链路。实验表明，该系统能够输出稳定的行为检测结果、学生级动作时间线和跨模态验证事件，为智慧课堂行为分析和可视化教学评估提供基础。

## 8. 未来展望

1. 构建独立 test split 和更多 case 的人工 gold label。
2. 引入更强 ASR 或音频增强，降低视觉 fallback 触发率。
3. 对 `jz/zt/js/xt` 长尾类别做针对性采样和增强。
4. 加入 temporal transformer 或 lightweight action localization 作为动作时序模块，而不是替代 YOLO 检测器。
5. 将系统整理为 GitHub Pages 或本地 Web demo，支持交互式 timeline、事件回放和证据链查看。

