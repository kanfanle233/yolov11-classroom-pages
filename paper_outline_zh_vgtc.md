# 面向 IEEE VGTC 英文稿件的中文论文大纲

## 0. Executive Summary

本文最稳妥的投稿定位不是“改进 YOLOv11 检测器”的算法论文，而是面向 IEEE VGTC / IEEE VIS / TVCG 风格的多模态可视分析系统论文。核心问题是：在真实智慧课堂中，视觉行为检测、学生轨迹、教师语音/文本语义流和模型不确定性如何被组织成一条可审计、可解释、可验证的事件分析链路。

推荐主线为：**visual analytics + multimodal verification + interpretable classroom behavior timeline**。YOLOv11s 和 YOLO pose 是基础视觉感知器；论文创新应放在语义桥接、pipeline contract、不确定性感知时间对齐、动态双重验证、ASR quality gate 与 visual fallback、学生级 timeline 和证据链可视分析。

当前仓库能支持的主文贡献：

| 状态 | 可写内容 | 证据来源 |
| --- | --- | --- |
| 已实现，可作为主线 | 8 类课堂行为检测、pose tracking、UQ 估计、event query、adaptive alignment、dual verification、timeline 输出 | `scripts/main/09_run_pipeline.py`, `scripts/pipeline/*`, `verifier/infer.py`, `contracts/schemas.py` |
| 已实现，可作为系统贡献 | FastAPI + D3/VSumVis 风格前端、case list、timeline、projection/glyph、evidence panel、video linked interaction | `server/app.py`, `web_viz/templates/front_vsumvis.html`, `scripts/frontend/20_build_frontend_data_bundle.py` |
| 已实现但实验不足 | MLP verifier、student verifier、calibration/ECE/Brier、rear-row ROI sliced/SR ablation | `verifier/model.py`, `verifier/train.py`, `verifier/calibration.py`, `output/codex_reports/front_046_sr_ablation/` |
| 原型/可选扩展 | LLM semantic fusion，当前默认 simulate 且真实 API 调用未实现 | `scripts/pipeline/06f_llm_semantic_fusion.py` |
| 仅规划或未形成主线证据 | OCR、敏感文本识别、IGFormer、ST-GCN、MLLM 视觉推理、ASPN/DySnake/GLIDE 主线训练 | `models/yolov11_classroom/`, `implementation_plan.md`, 相关报告 |

写作边界：所有“提升”“泛化”“显著优于”的 claim 必须有正式 split、人工 gold label 或实验表支撑。当前 detection 指标存在报告冲突，event-level gold label 规模较小，ASR case 质量不一致，因此正式英文稿前必须二次核对。

## 1. Paper Positioning

### 1.1 当前论文最稳妥定位

推荐定位：**面向智慧课堂的多模态验证与可视分析系统论文**。

英文定位句：

> This paper presents an auditable multimodal visual analytics framework that verifies classroom visual behavior sequences against noisy textual semantic streams and exposes the full evidence chain through a student-level timeline interface.

不推荐定位：

- 不推荐写成 YOLOv11 检测器改进论文。当前主线使用 Ultralytics YOLO 系列权重与微调结果，仓库中虽有 ASPN、DySnakeConv、GLIDE Loss 等模块，但没有形成主线公平训练和消融证据。
- 不推荐写成纯多模态大模型论文。`06f_llm_semantic_fusion.py` 的真实 LLM API 调用仍是 placeholder，默认通过 `simulate_llm_response()` 产生规则式结果。
- 不推荐写成 ASR 带来显著提升的论文。当前 ASR 既有 `front_1885_full`、`front_26729_full` 的 accepted segments，也有 `ASR_EMPTY` 和 placeholder case；更稳妥的贡献是 ASR quality gate 与 visual fallback 保护 verifier。

### 1.2 面向 IEEE VGTC 的强调点

IEEE VGTC/VIS/TVCG 读者更关心可解释、人机协同、可视分析任务和系统证据链。因此主文应强调：

1. 教师/研究者需要从“检测框”转向“学生级事件 timeline”。
2. 系统保留 query、alignment candidates、modality scores、UQ、label 和 source files，可被审计。
3. 可视界面支持 case comparison、timeline browsing、video seeking、evidence inspection 和 reliability/uncertainty interpretation。
4. 方法章节服务系统目标，不把普通前端功能包装成算法贡献。

### 1.3 主文与附录分工

适合主文：

- 论文问题定义与系统架构。
- 语义桥接与 fusion contract。
- UQ-guided adaptive alignment 与 dual verification scoring。
- ASR quality gate 与 visual fallback 的设计动机。
- VSumVis 风格 timeline/evidence interface。
- 检测指标、pipeline contract、case study、有限 gold label 结果、rear-row ablation 的保守呈现。

适合 supplementary material：

- 完整 pipeline command 与所有 CLI 参数。
- 每个 case 的完整 artifact list。
- 更长的 failure case gallery。
- 完整 per-class AP、confusion matrix、reliability bins。
- 全量 references BibTeX。
- 原型 LLM prompt 和 simulate 输出样例。
- 未纳入主文的 ASPN/DySnake/GLIDE 代码说明。

## 2. Candidate Titles

### 2.1 中文题目候选

1. **《面向智慧课堂视觉行为序列与文本语义流的可审计双重验证框架》**  
   适配理由：准确覆盖 visual behavior sequence、textual semantic stream、dual verification 和 auditability，适合作为系统论文题目。

2. **《面向智慧课堂多模态行为分析的学生级时间线验证与可视分析系统》**  
   适配理由：更偏 IEEE VGTC，突出 student-level timeline 和 visual analytics，但弱化了文本语义流。

3. **《融合不确定性感知对齐的智慧课堂视觉-语义事件验证框架》**  
   适配理由：更方法导向，强调 UQ alignment 和 event verification，适合方法章节较强时使用。

4. **《面向噪声课堂场景的视觉行为与教师语义流双重验证方法》**  
   适配理由：突出遮挡、远场语音、ASR 噪声等场景，但可视分析系统感略弱。

5. **《可审计的智慧课堂多模态感知与行为时间线分析系统》**  
   适配理由：最偏系统展示，适合若投稿强调 demo、interface 和 case study。

### 2.2 英文题目候选

1. **An Auditable Dual-Verification Framework for Smart Classroom Visual Behavior Sequences and Textual Semantic Streams**  
   理由：与当前代码和报告证据最吻合，覆盖 dual verification、visual/text streams 和 auditability。

2. **A Visual Analytics Framework for Verifying Student-Level Classroom Behavior Timelines with Noisy Semantic Streams**  
   理由：更贴近 IEEE VIS/VGTC，强调 visual analytics 与 noisy semantic streams。

3. **Uncertainty-Aware Visual-Semantic Verification for Smart Classroom Behavior Analysis**  
   理由：更短、更方法化，适合 Method 比 System 更强的稿件。

4. **From Classroom Detections to Auditable Behavior Timelines: Multimodal Verification of Visual Actions and Textual Events**  
   理由：叙事性强，突出从 detection 到 timeline 的转化。

5. **Dual Verification of Classroom Behavior Events via Visual Sequences, Textual Streams, and Reliability-Aware Timelines**  
   理由：突出 event-level verification 和 reliability-aware timeline。

### 2.3 推荐题目

推荐中文题目：**《面向智慧课堂视觉行为序列与文本语义流的可审计双重验证框架》**。

推荐英文题目：**An Auditable Dual-Verification Framework for Smart Classroom Visual Behavior Sequences and Textual Semantic Streams**。

推荐理由：该题目不会误导审稿人以为本文提出新检测器结构；同时覆盖系统的三个核心对象：视觉行为序列、文本语义流、可审计验证框架。

### 2.4 不建议出现在题目中的表达

- “改进 YOLOv11 检测器”
- “新型 YOLOv11 网络”
- “YOLOv11-based multimodal LLM reasoning”
- “real-time production classroom deployment”
- “OCR-enhanced classroom understanding”
- “large-scale LLM semantic verification”
- “end-to-end multimodal foundation model”

这些表达容易把本文引向未被当前主线实验支撑的方向。

## 3. Abstract Draft

### 3.1 中文摘要草稿

智慧课堂行为分析需要从复杂课堂视频中理解学生动作、教师指令和课堂事件，但单一视觉检测在遮挡、远距离、多人重叠、小动作相似和低光条件下容易产生不稳定判断。音频和文本语义流能够为课堂事件提供上下文，但远场语音、环境噪声和 ASR 错识别也可能将不可靠文本证据引入分析流程。

本文提出一种面向智慧课堂视觉行为序列与文本语义流的可审计双重验证框架。系统首先使用 YOLOv11s 行为检测器和 YOLO pose 构建 8 类课堂行为片段与学生轨迹，并基于关键点可见率、运动稳定性和边界框稳定性估计 track-level uncertainty。随后，系统通过语义桥接协议将检测短码映射为带有 `semantic_id`、中英文语义标签和 taxonomy version 的可验证事件表示；文本侧通过 Whisper ASR、质量门控和事件查询抽取构建 textual semantic stream，并在 ASR 不可靠时触发 visual fallback。跨模态对齐阶段根据运动不稳定性和 UQ 自适应调整候选时间窗口，双重验证阶段使用视觉分数、文本分数、动态模态权重和不确定性惩罚输出 `match`、`mismatch` 或 `uncertain`，同时保留完整证据链。

当前仓库中的阶段性结果表明，该 pipeline 已能生成学生级 timeline、event queries、top-k alignment candidates、verified events、pipeline contract reports 和 VSumVis 风格可视分析界面。行为检测指标在不同报告中存在验证集与测试集口径冲突：官方微调对比报告给出 `mAP50≈0.93`、`mAP50-95≈0.80` 的验证结果，而部分写作报告记录 `test mAP50=0.9806`、`mAP50-95=0.8782`；正式投稿前需按视频级独立 split 再次核对。本文的贡献在于将课堂视觉检测结果转化为学生级、事件级、可靠性可解释的可审计分析结果，而不是提出新的检测器结构。局限在于 event-level gold label 规模仍小、ASR 质量受场景影响明显、LLM 模块当前仍处于原型状态。

### 3.2 英文摘要写作结构

- Background: Smart classroom analytics requires interpretable event-level understanding rather than raw detections.
- Problem: Visual-only detectors and noisy ASR streams are both unreliable under occlusion, distant students, overlapping speakers, and low-quality audio.
- Method: Introduce semantic contract, UQ-guided adaptive alignment, dual verification, ASR gate/fallback, student-level timeline interface.
- Results: Report available validation evidence and pipeline outputs; flag metric conflicts before final submission.
- Contributions: Auditable multimodal verification and visual analytics framework.
- Limitation sentence: Current evidence is limited by test split risk, small event gold labels, and prototype LLM extension.

## 4. Index Terms

中文关键词：

智慧课堂分析；多模态验证；课堂行为检测；视觉行为序列；文本语义流；不确定性感知对齐；双重验证；可视分析；学生级时间线；可靠性评估

English Index Terms:

Smart classroom analytics; multimodal verification; visual behavior analysis; visual analytics; uncertainty-aware alignment; classroom behavior detection; temporal event verification; student-level timeline; reliability scoring; human-centered AI.

## 5. Introduction Outline

### 5.1 第一段：智慧课堂行为分析的实际需求

写作目标：从教育场景需求进入，而不是从检测器进入。

要点：

- 智慧课堂分析需要理解学生参与、教师互动、课堂秩序和学习活动节奏。
- 真实应用中，教师或研究者关心的不是单帧检测框，而是“哪个学生在什么时间发生了什么行为，该判断是否可靠”。
- 因此，课堂行为分析需要从 frame-level detections 转化为 student-level temporal events 和可解释 timeline。

### 5.2 第二段：视觉-only 方法的问题

要点：

- 课堂视觉场景具有多人密集、遮挡、后排远距离、小动作差异、低光、运动模糊等挑战。
- 听课、写字、看书、转头、讨论等类别视觉上相近，单帧检测置信度不等于事件真实性。
- 现有 YOLO-based classroom behavior recognition 多关注 mAP 和实时性，较少保留 verification evidence、uncertainty 和 teacher semantic context。

### 5.3 第三段：音频/文本语义流的价值和风险

要点：

- 教师指令、课堂提问和语音转写可以提供事件语义线索，例如“开始讨论”“请举手”“看黑板”。
- 但远场语音、回声、多人说话和录音质量会导致 ASR 空文本、错识别或低置信片段。
- 盲目融合低质量 ASR 会污染 verifier，因此本文引入 ASR quality gate 和 visual fallback。

### 5.4 第四段：本文不是简单融合，而是 dual verification

要点：

- 本文不将视觉和文本做固定权重后融合，而是提出“视觉候选 + 文本查询”的双重验证。
- 对齐窗口由 motion uncertainty 和 track UQ 自适应调整。
- 最终输出不是二分类硬标签，而是 `match / mismatch / uncertain`、`p_match`、`reliability_score` 和 evidence。

### 5.5 第五段：系统输出

要点：

- 输出包括 `verified_events.jsonl`、`align_multimodal.json`、`timeline_students.csv/json`、`student_id_map.json`、`pipeline_contract_v2_report.json`。
- 前端通过 case list、timeline view、semantic stream、evidence panel、video seeking 和 projection/glyph view 支持审计。
- 论文案例应展示 query text、candidate actions、UQ、visual/text scores、final label 和视频片段。

### 5.6 最后一段：贡献列表

建议贡献不超过 5 条：

1. 提出一个面向智慧课堂视觉行为序列与文本语义流的双重验证框架，将检测结果转化为事件级、学生级、可审计输出。
2. 设计语义桥接与 fusion contract，将课堂行为短码统一为可被文本查询、verifier 和 timeline 使用的语义协议。
3. 提出不确定性感知的自适应时间对齐，根据 motion instability 和 track UQ 动态检索 top-k 视觉候选。
4. 实现 confidence-weighted dual verification 与 reliability scoring，在 ASR 可用、ASR 不可靠和 visual fallback 情况下输出三值决策。
5. 构建 VSumVis 风格学生级 timeline 与证据链可视分析界面，用于解释、审计和对比课堂行为事件。

后排 ROI sliced/SR ablation 可作为实验贡献或 supplementary，不建议放入主贡献列表，除非补足 GT 与正式统计。

## 6. Related Work Outline

### 6.1 Classroom Behavior Detection and Smart Classroom Analytics

已有工作：自建课堂数据集、学生行为识别、课堂参与度分析、教师行为分析和教育数据挖掘。

最接近工作：SCB-Dataset、MM-TBA、课堂行为检测数据集与教师行为数据集。

本文区别：本文不是只输出行为类别，而是将行为片段与教师语义流对齐，并输出 event-level verification、reliability 和 student timeline。

避免同质化风险：不声称“首次做智慧课堂行为检测”，而强调 noisy classroom 场景下的可审计多模态验证。

### 6.2 YOLO-based Real-time Detection and Classroom Behavior Recognition

已有工作：改进 YOLOv5/YOLOv8、轻量网络、注意力机制、多尺度融合、实时课堂行为检测。

最接近工作：

- Classroom Behavior Detection Based on Improved YOLOv5 Algorithm Combining Multi-Scale Feature Fusion and Attention Mechanism, Applied Sciences, 2022, DOI: `10.3390/app12136790`。
- Student Behavior Detection in the Classroom Based on Improved YOLOv8, Sensors, 2023, DOI: `10.3390/s23208385`。
- BiTNet, JKSUCI, 2023, DOI: `10.1016/j.jksuci.2023.101670`。

本文区别：YOLOv11s 是基础视觉感知器，不是创新点。本文关注 detection outputs 如何进入 tracking、semantic contract、verification 和 visual analytics。

避免同质化风险：不要比较谁的检测器结构更复杂；要比较是否支持 text stream、temporal alignment、UQ、verification output 和 visual analytics。

### 6.3 Pose Tracking and Uncertainty Estimation

已有工作：Kalman filter、Hungarian matching、ByteTrack、多目标跟踪、pose-based action rules、track smoothing。

当前实现证据：

- `scripts/pipeline/03_track_and_smooth.py` 使用 OpenCV Kalman filter 与 `linear_sum_assignment`。
- `scripts/pipeline/03e_track_behavior_students.py` 实现 behavior detection tracking 和 student mapping。
- `scripts/pipeline/03c_estimate_track_uncertainty.py` 基于 keypoint visibility、motion stability、bbox stability 输出 UQ。

本文区别：UQ 不只用于过滤，而进入 temporal window、modality weight 和 reliability penalty。

避免同质化风险：不把启发式 UQ 写成贝叶斯严格不确定性估计；应称为 heuristic statistical track uncertainty。

### 6.4 Audio-Visual / Video-Language Multimodal Understanding

已有工作：CLIP、ImageBind、MAViL、CMR-AVE、LanguageBind、Video-LLaMA、InternVideo2 等证明跨模态语义对齐与音视频理解价值。

最接近工作：音视频事件定位和课堂音视频事件检测。

本文区别：本文不训练大规模 foundation model，而是在 classroom-specific pipeline 中做 event query、candidate retrieval、dual verification 和 timeline audit。

避免同质化风险：不要声称本文是 MLLM 或 Video-Language foundation model；LLM 模块仅为 semantic reasoning prototype。

### 6.5 ASR and Speech-based Classroom Understanding

已有工作：Whisper、Wav2Vec 2.0、AV-ASR、课堂语音/教师行为数据集。

当前实现证据：

- `scripts/pipeline/06_asr_whisper_to_jsonl.py` 使用 `faster_whisper`，保存 `avg_logprob`、`no_speech_prob`、`compression_ratio`。
- 输出 `asr_quality_report.json`，含 `segments_raw`、`segments_accepted`、`segments_rejected` 和 `status`。

本文区别：ASR 不是无条件增强信号，而是被 quality gate 管理；当 ASR 低质量时，系统回退到 visual query，避免错误文本污染。

避免同质化风险：不要写“文本模态显著提升性能”，除非后续有 gold transcript 和 ASR ablation。

### 6.6 Cross-modal Temporal Alignment and Event Verification

已有工作：固定窗口音视频事件定位、跨模态 attention、audio-visual event localization。

当前实现证据：`scripts/pipeline/xx_align_multimodal.py` 以 query time 为中心，使用 `window_size = clip(base + alpha_motion * motion + beta_uq * uq)` 检索 top-k candidates，并按 `overlap` 和 `action_confidence` 排序。

本文区别：对齐窗口由 track uncertainty 和 motion cues 共同驱动，输出候选证据而不是直接吞并到单个融合分数。

避免同质化风险：不把当前 heuristic adaptive window 写成学习式 alignment network。

### 6.7 Reliability, Calibration, and Uncertainty-aware Decision Making

已有工作：ECE、Brier Score、temperature scaling、reliability diagram、uncertainty-aware decision making。

当前实现证据：

- `verifier/model.py` 实现 `expected_calibration_error()` 和 `brier_score()`。
- `verifier/calibration.py` 支持 temperature scaling 与 reliability diagram。
- `contracts/schemas.py` 验证 calibration report schema。

本文区别：reliability 被用于课堂事件审计，不只作为分类器置信度校准指标。

避免同质化风险：不要将小样本 self-label fallback 的 eval report 写成正式 verifier accuracy。

### 6.8 Visual Analytics for Education and Human-centered AI Systems

已有工作：教育可视分析、timeline visualization、human-centered AI audit、interactive visual analytics。

当前实现证据：

- `server/app.py` 提供 `/api/v2/vsumvis/*`、`/api/case/{case_id}/evidence/{event_id}`、`/api/case/{case_id}/alignment/{event_id}`。
- `web_viz/templates/front_vsumvis.html` 实现 case list、summary chips、video panel、projection/glyph、timeline、evidence panel。

本文区别：系统把行为 detection、semantic stream、verification evidence 和 video interaction 组织为统一审计界面。

避免同质化风险：不把普通 UI 页面写成“新可视化算法”，而应写成 design goals 与 task-driven visual analytics design。

### 6.9 相关工作对比表设计方案

表 1：Related Work Comparison。

列：

`Method`, `Year`, `Venue`, `Scenario`, `Visual Input`, `Audio/Text Input`, `Temporal Alignment`, `Uncertainty Modeling`, `Verification Output`, `Visual Analytics Interface`, `Classroom-specific`, `Difference from Ours`

候选行：

- Improved YOLOv5 classroom behavior detection, 2022, Applied Sciences。
- Improved YOLOv8 classroom behavior detection, 2023, Sensors。
- MSTA-SlowFast, 2023, Sensors。
- SCB-Dataset, 2023, arXiv。
- BiTNet, 2023, JKSUCI。
- CLIP, 2021, ICML。
- CMR-AVE, 2020, ACM MM。
- MAViL, 2023, NeurIPS。
- Whisper, 2022, arXiv/ICML technical report。
- AVDor / Multimodal Audio-Visual Detection in Classroom, 2025, Scientific Reports。
- Ours。

## 7. Problem Definition

### 7.1 输入

课堂视频帧序列：

$$
\mathcal{V} = \{I_t\}_{t=1}^{T}, \quad t \in [1,T].
$$

音频/文本语义流：

$$
\mathcal{A} = \{a_k=(t_k^s,t_k^e,z_k,m_k)\}_{k=1}^{K},
$$

其中 $z_k$ 为 ASR 文本，$m_k$ 包含 `avg_logprob`、`no_speech_prob`、`compression_ratio` 等质量指标。

### 7.2 学生轨迹

学生轨迹集合：

$$
\mathcal{T} = \{\tau_s\}_{s=1}^{N}, \quad
\tau_s = \{(b_{s,t}, p_{s,t}, \rho_{s,t})\}_{t \in \Omega_s},
$$

其中 $b_{s,t}$ 是 bbox，$p_{s,t}$ 是 pose keypoints，$\rho_{s,t}$ 是 track confidence 或可见性统计。

### 7.3 视觉动作片段集合

视觉动作片段：

$$
\mathcal{X} = \{x_i=(s_i, a_i, t_i^s, t_i^e, c_i, u_i, \sigma_i)\}_{i=1}^{M},
$$

其中 $s_i$ 为 track/student id，$a_i$ 为行为语义标签，$c_i$ 为 action confidence，$u_i$ 为 track uncertainty，$\sigma_i$ 为语义字段，如 `semantic_id`、`semantic_label_zh`、`semantic_label_en`。

### 7.4 文本事件查询集合

文本事件查询：

$$
\mathcal{Q} = \{q_j=(e_j,\hat{t}_j,r_j,\kappa_j)\}_{j=1}^{J},
$$

其中 $e_j$ 是 event type 或 semantic id，$\hat{t}_j$ 是 query time，$r_j$ 是 text/audio reliability，$\kappa_j$ 记录 query source，如 ASR、teacher instruction、visual fallback。

### 7.5 候选对齐窗口

对每个查询 $q_j$，系统计算自适应窗口：

$$
\Delta_j = \mathrm{clip}(\Delta_0 + \alpha_m M_j + \beta_u U_j, \Delta_{\min}, \Delta_{\max}).
$$

当前实现参数：$\Delta_0=1.0$，$\alpha_m=1.2$，$\beta_u=0.8$，$\Delta_{\min}=0.6$，$\Delta_{\max}=4.0$，Top-K=8。来源：`scripts/pipeline/xx_align_multimodal.py`。

候选集合：

$$
\mathcal{C}_j = \{x_i \in \mathcal{X} \mid \mathrm{overlap}([\hat{t}_j-\Delta_j,\hat{t}_j+\Delta_j],[t_i^s,t_i^e]) > 0\}.
$$

候选按 `overlap` 和 `action_confidence` 排序。

### 7.6 验证输出

最终输出：

$$
y_j \in \{\mathrm{match}, \mathrm{mismatch}, \mathrm{uncertain}\},
$$

并保存：

$$
o_j=(q_j, x_j^\*, p_j^{match}, p_j^{mismatch}, R_j, U_j, y_j, E_j),
$$

其中 $E_j$ 是完整 evidence object，包括 query、selected candidate、top-k candidates、visual/text scores、modality weights、timeline segment 和 source files。

### 7.7 任务目标

给定 $\mathcal{V}$ 与 $\mathcal{A}$，目标不是最大化单帧检测 mAP，而是构建一个可审计映射：

$$
F: (\mathcal{V},\mathcal{A}) \rightarrow \{o_j\}_{j=1}^{J},
$$

使每个课堂语义事件都有候选检索、双重验证、可靠性估计和可视化解释。

## 8. Method Outline

### 8.1 System Overview

输入：

- classroom video。
- optional classroom audio。
- YOLO behavior detector weights。
- YOLO pose model。
- semantic taxonomy。

输出：

- visual behavior stream: `actions.fusion_v2.jsonl`。
- pose uncertainty stream: `pose_tracks_smooth_uq.jsonl`。
- textual semantic stream: `transcript.jsonl`、`event_queries.fusion_v2.jsonl`。
- alignment output: `align_multimodal.json`。
- verification output: `verified_events.jsonl`。
- visual analytics output: `timeline_chart.json/png`、`timeline_students.csv/json`、`student_id_map.json`、frontend bundle。

三条数据流：

1. Visual behavior stream：YOLOv11 behavior detection + pose tracking + behavior segment construction。
2. Pose uncertainty stream：keypoint visibility、motion stability、bbox stability。
3. Textual semantic stream：Whisper ASR、quality gate、event query extraction、instruction context。

### 8.2 Visual Behavior Sequence Construction

8 类行为 taxonomy：

| Code | 中文语义 | English Semantic Label | 说明 |
| --- | --- | --- | --- |
| `tt` | 抬头听课 | listen | 常见多数类，视觉上与看黑板/听讲相关 |
| `dx` | 低头写字 | write | 与看书相近，需要上下文区分 |
| `dk` | 低头看书 | read | 与写字相近，类别边界需谨慎 |
| `zt` | 转头 | turn_head | 小动作，易受姿态/角度影响 |
| `xt` | 小组讨论 | group_discussion | 可能需要多人关系和语音上下文 |
| `js` | 举手 | raise_hand | pose rule 与 detector 都可支撑 |
| `zl` | 站立 | stand | 与 bbox height/pose 相关 |
| `jz` | 教师互动 | teacher_interaction | 长尾类，数据最少，需限制 claim |

代码证据：

- `scripts/pipeline/02d_export_behavior_det_jsonl.py` 使用 Ultralytics YOLO 导出 behavior detections，支持 `full`、`sliced`、`full_sliced`。
- `scripts/pipeline/02_export_keypoints_jsonl.py` 使用 YOLO pose 导出 keypoints，支持 full/sliced/ROI SR sliced。
- `scripts/pipeline/03_track_and_smooth.py` 使用 Kalman filter、Hungarian matching、seat prior、EMA smoothing。
- `scripts/pipeline/03e_track_behavior_students.py` 将 behavior detections 映射为 track/student-level segments，并保留 semantic fields。

写作注意：

- 学生 ID 是 tracking 后处理结果，不是 YOLO 类别。
- object evidence 若使用，应写为辅助 evidence，不写成主线核心贡献，除非给出正式消融。

### 8.3 Semantic Bridging and Fusion Contract

动机：

- `tt/dx/dk` 等短码适合检测训练，但不适合 ASR query、语义验证和 timeline 展示。
- 需要稳定字段连接视觉、文本、verifier 和 UI。

关键字段：

- `semantic_id`
- `semantic_label_zh`
- `semantic_label_en`
- `taxonomy_version`
- `schema_version`

实现证据：

- `scripts/pipeline/03e_track_behavior_students.py` 在 action records 中保存 semantic fields。
- `contracts/schemas.py` 定义 schema validators，`SCHEMA_VERSION="2026-04-01"`，verified labels 为 `match/uncertain/mismatch`。
- `scripts/main/09_run_pipeline.py` 支持 `--semantic_taxonomy` 并在主线中传递 taxonomy。

论文写法：

- 该协议是系统级贡献，可写成 **semantic bridging protocol** 或 **fusion contract**。
- fail-fast 机制：缺字段或类型错误应被 contract 检查发现；当前报告中已有多个 contract 输出目录。
- 下游 verifier 与 timeline 使用 semantic fields 避免短码不可读。

### 8.4 Pose Uncertainty Estimation

当前 UQ 是 heuristic statistical track uncertainty，不是学习式 Bayesian uncertainty。

实现公式：

$$
v_{s,t} = \frac{\#\{k: conf(k)\ge 0.25\}}{\#\{k\}},
$$

$$
U_{s,t} = \mathrm{clip}_{[0,1]}\left(
0.45(1-v_{s,t}) + 0.35(1-m_{s,t}) + 0.20(1-b_{s,t})
\right),
$$

其中 $m_{s,t}$ 为 motion stability，$b_{s,t}$ 为 bbox stability。

代码证据：`scripts/pipeline/03c_estimate_track_uncertainty.py`。

UQ 进入：

- alignment window：`beta_u * uq_basis`。
- verifier weight：`w_visual = 1 - uq`。
- reliability penalty：`R = p_match * (1 - uq_gate * uq)`。
- UI：evidence panel 和 timeline reliability/uncertainty display。

### 8.5 ASR Quality Control and Event Query Construction

ASR 模块：

- Whisper / faster-whisper backend。
- 音频 RMS 预检查。
- segment metrics: `avg_logprob`、`no_speech_prob`、`compression_ratio`。
- quality tier: `good`、`low_conf`、`reject`。
- report: `asr_quality_report.json`。

代码证据：`scripts/pipeline/06_asr_whisper_to_jsonl.py`。

Event query extraction：

- `scripts/pipeline/06b_event_query_extraction.py` 通过关键词/规则抽取 `raise_hand`、`head_down`、`discussion`、`respond_call`、`teacher_instruction` 等事件。
- 低质量或空 ASR 时，系统应使用 visual fallback，论文写成“防止不可靠文本污染 verifier”。

教师指令上下文：

- `scripts/pipeline/06e_extract_instruction_context.py` 已实现规则式 teacher instruction context，可写成 lightweight context extractor。
- 若没有正式 ablation，应放在 method detail 或 optional evidence，不写为核心贡献。

### 8.6 UQ-guided Adaptive Multimodal Alignment

固定窗口问题：

- 课堂事件可能存在语音和动作延迟。
- 遮挡或运动剧烈时，视觉 track 不稳定，需要更宽候选窗口。
- 固定窗口可能漏掉真实候选或引入太多误匹配。

实现：

$$
\Delta_j=\mathrm{clip}(1.0+1.2M_j+0.8U_j,0.6,4.0).
$$

候选检索：

- 对每个 query，计算 window start/end。
- 检索与窗口重叠的 action segments。
- 保存 `overlap`、`action_confidence`、`uq_track`、semantic fields。
- 按 `(overlap, action_confidence)` 降序取 Top-K，默认 8。

代码证据：`scripts/pipeline/xx_align_multimodal.py`。

### 8.7 Dual Verification and Reliability Scoring

特征：

`verifier/model.py` 的 feature vector 为：

$$
\phi(q,x)=[\mathrm{overlap},\mathrm{action\_confidence},\mathrm{text\_score},1-uq].
$$

当前 runtime verifier：

视觉分数：

$$
C_v = \mathrm{clip}_{[0,1]}(0.65 \cdot \mathrm{overlap} + 0.35 \cdot \mathrm{action\_confidence}).
$$

真实音频可用时：

$$
w_v=1-uq,\quad w_a=\mathrm{audio\_confidence},
$$

$$
p_{match}=\mathrm{clip}_{[0,1]}\left(
\frac{w_v C_v+w_a C_t}{w_v+w_a+\epsilon}
\right).
$$

visual fallback 或无可用音频时：

$$
p_{match}=C_v.
$$

可靠性：

$$
R=p_{match}(1-\lambda_u uq),\quad \lambda_u=0.60.
$$

三值决策：

$$
y=
\begin{cases}
\mathrm{match}, & R \ge 0.60,\\
\mathrm{uncertain}, & 0.40 \le R < 0.60,\\
\mathrm{mismatch}, & R < 0.40.
\end{cases}
$$

代码证据：`verifier/infer.py`、`verifier/model.py`。

MLP verifier：

- `verifier/model.py` 定义 `VerifierMLP`。
- `verifier/train.py`、`verifier/eval.py`、`verifier/calibration.py` 已实现训练、评估、校准。
- 由于 gold label 规模有限，应写成 optional learnable verifier / future extension，或仅作为 exploratory baseline。

### 8.8 Student-level Timeline and Visual Analytics Interface

输出：

- `student_id_map.json`：track 到学生显示 ID 的映射。
- `timeline_students.csv/json`：学生级动作片段。
- `timeline_chart.json/png`：timeline 图表数据和静态图。
- `verified_events.jsonl`：验证事件 overlay。

系统实现：

- `scripts/pipeline/10_visualize_timeline.py` 生成 timeline assets。
- `scripts/frontend/20_build_frontend_data_bundle.py` 打包 frontend bundle，schema version 为 `2026-05-01+frontend_bundle_v2`。
- `server/app.py` 提供 case、contract、ASR quality、evidence、alignment、VSumVis routes。
- `web_viz/templates/front_vsumvis.html` 使用 D3 实现 case list、video panel、projection/glyph、timeline 和 evidence panel。

写作边界：

- 可以说 interface supports audit and interpretation。
- 不要说前端本身提出新降维算法或新 glyph 算法，除非后续补充可视编码创新与用户研究。

## 9. Visual Analytics System / Interface Outline

### 9.1 Design Goals

DG1. 从检测框转向学生级事件解释：支持教师/研究者查看每个学生的行为时间线。

DG2. 保留证据链：每个 verified event 可追溯到 query、alignment candidates、视觉候选、scores、UQ 和 source files。

DG3. 暴露不确定性：用 reliability、uncertainty、match/mismatch/uncertain 显示事件可信度。

DG4. 支持跨 case 对比：通过 case list、summary chips、SR ablation endpoints 和 projection view 比较不同视频或推理设置。

DG5. 支持视频联动审计：点击 timeline 或 event 后跳转到对应视频时间。

### 9.2 Data Processing Pipeline

界面数据由 pipeline artifacts 与 frontend bundle 组成：

- raw case dir: `output/codex_reports/front_45618_full` 等。
- bundle dir: `output/frontend_bundle/*`。
- server normalization: `server/app.py` 中 `_read_timeline_segments_front()`、`_normalize_timeline_segments()`、`_build_feature_rows()`、`_build_front_projection()`。

### 9.3 Visual Encoding

建议正式稿描述：

- 颜色：match / uncertain / mismatch 使用不同色相。
- 时间位置：x 轴表示视频时间。
- 行位置：y 轴表示 student/track。
- 矩形段：行为片段，长度表示持续时间。
- 透明度或边框：表示 reliability 或 uncertainty。
- glyph/projection：用于 case/event-level overview 和聚类趋势观察。

当前实现状态：

- timeline、projection、glyph、parallel-like feature view 已在 `front_vsumvis.html` 中实现。
- exact visual encoding 需在图注中以真实截图为准。

### 9.4 Timeline View

内容：

- 学生级行为片段。
- verified event overlay。
- click-to-seek。
- event selection and filtering。

证据来源：

- `timeline_chart.json`
- `timeline_students.csv/json`
- `/api/v2/vsumvis/timeline/{case_id}`

实现状态：已实现，但 bundle 与 raw case 路径可能存在字段不一致，论文需说明 server 做 normalization。

### 9.5 Semantic Stream View

内容：

- ASR transcript。
- event queries。
- query source: ASR / placeholder / visual fallback。
- ASR quality status。

证据来源：

- `transcript.jsonl`
- `event_queries.fusion_v2.jsonl`
- `asr_quality_report.json`
- `/api/case/{case_id}/asr-quality`

实现状态：部分实现。前端展示需确认是否完整显示 ASR quality gate；若截图不完整，可写成 system design and partial prototype。

### 9.6 Evidence Panel

内容：

- query text。
- selected visual candidate。
- top-k alignment candidates。
- visual_score、text_score、uq_score。
- w_visual、w_audio。
- p_match、p_mismatch、reliability、label。
- source files。

证据来源：

- `/api/case/{case_id}/evidence/{event_id}`
- `/api/case/{case_id}/alignment/{event_id}`
- `verified_events.jsonl`
- `align_multimodal.json`

实现状态：已实现，`front_vsumvis.html` 中 `fetchEvidenceAPI()` 异步拉取 evidence。

### 9.7 Case Comparison View

内容：

- case list。
- contract status chips。
- verified count、student count、timeline count。
- SR ablation comparison。

证据来源：

- `/api/v2/vsumvis/cases`
- `/api/v2/vsumvis/ablation/sr`
- `/api/v2/vsumvis/compare/sr`

实现状态：已实现，但正式论文中的 quantitative comparison 需使用正式表格而不是 UI 数字。

### 9.8 Video-linked Interaction

内容：

- video player。
- timeline click-to-seek。
- selected event display。
- media fallback source selection。

证据来源：`front_vsumvis.html` 中 video player、`/api/media/{video_id}`、timeline click handlers。

### 9.9 Reliability and Uncertainty Visualization

内容：

- event color or bar for label。
- uncertainty/reliability in evidence panel。
- calibration reliability diagram as static figure。

证据来源：

- `verified_events.jsonl`
- `verifier_calibration_report.json`
- `verifier_reliability_diagram.svg`

实现状态：reliability diagram exists in multiple output dirs, but sample size must be stated.

### 9.10 How the Interface Supports Audit and Interpretation

写作段落建议：

> The interface supports audit by linking every visual mark to the corresponding query, alignment candidates, verification scores, and source artifacts. A teacher or analyst can inspect why an event is matched, uncertain, or mismatched, replay the video around the event time, and compare the selected visual candidate against alternative candidates.

## 10. Experiments Outline

### 10.1 Dataset and Preprocessing

需要写：

- 数据来源：智慧课堂学生行为数据集，正方/后方/顶部等视角按实际 manifest 核对。
- detection dataset：`data/processed/classroom_yolo/dataset.yaml`。
- 当前统计：train 7416 images，val 1467 images，total 8883 images，total boxes 267861。
- test split 风险：`YOLO论文大纲/project_audit_20260426/dataset_classroom_yolo_summary.json` 显示 test images 为 0。
- 8 类行为定义与类别不平衡：`jz` 仅 680 boxes，最大/最小类比例约 172.835。
- 是否按视频划分：正式投稿前必须核对；若仍为 frame-level split，必须作为 limitation。
- 隐私与匿名化：说明视频仅用于研究，输出中使用 track/student id，不暴露学生身份；正式稿需补伦理与授权说明。

### 10.2 Implementation Details

视觉：

- behavior detector: YOLOv11s / Ultralytics YOLO, `runs/detect/official_yolo11s_detect_e150_v1/weights/best.pt` 或正式锁定权重。
- pose model: YOLO11 pose，如 `yolo11x-pose.pt` 或 `yolo11s-pose.pt`，按实验记录核对。
- inference modes: full / sliced / full_sliced / roi_sr_sliced。
- SR backend: off/opencv/realesrgan/basicvsrpp 等，但主文只报告已成功运行的 variant。

文本：

- ASR backend: faster-whisper / Whisper。
- metrics: avg_logprob、no_speech_prob、compression_ratio、audio RMS。
- event extraction: rule-based event query extraction。

Pipeline：

- main command: `scripts/main/09_run_pipeline.py`。
- outputs: actions, UQ, transcript, queries, align, verified events, timeline, contract reports。
- device、input size、thresholds、FPS/latency：正式稿前从 run logs 和 config 中核对。

### 10.3 Evaluation Metrics

视觉检测：

- mAP50。
- mAP50-95。
- Precision。
- Recall。
- F1。
- per-class AP。
- confusion matrix。
- FPS / latency。

跨模态验证：

- event-level accuracy。
- precision / recall / F1。
- candidate recall。
- Top-k recall。
- mismatch false positive rate。
- uncertain ratio。
- selected candidate rank。

可靠性：

- ECE。
- Brier Score。
- reliability diagram。
- calibration before/after temperature scaling。

ASR：

- raw segments。
- accepted segments。
- rejected segments。
- fallback rate。
- WER，若有 gold transcript。

可视分析：

- timeline coverage。
- student count。
- action segment count。
- ID switches，若有 GT。
- case completion。
- qualitative expert review，若补用户研究。

### 10.4 Baselines

必设 baseline：

1. Visual-only YOLO：仅用 behavior detector 和 confidence 输出事件。
2. Pose-rule baseline：`scripts/pipeline/04_action_rules.py`。
3. ASR/Text-only baseline：仅基于 event queries，不使用视觉候选，适合说明 ASR 噪声风险。
4. Fixed-window alignment：固定 1s/2s 窗口。
5. Static late fusion：固定 visual/audio 权重。
6. Fusion contract without UQ：保留语义字段但不使用 UQ。
7. Ours full model：semantic contract + UQ adaptive alignment + dynamic verification + ASR gate/fallback + timeline。

可选 baseline：

- SlowFast action recognition，若 `05_slowfast_actions.py` 有可复现实验。
- MLP verifier，若有足够 human gold labels。

### 10.5 Ablation Studies

建议消融：

- without semantic bridging。
- without ASR quality gate。
- without visual fallback。
- fixed window vs adaptive window。
- static fusion vs dynamic fusion。
- without uncertainty penalty。
- with/without rear-row slicing。
- with/without super-resolution。
- with/without teacher instruction context。
- LLM semantic prototype，写为 qualitative/prototype ablation，不写主表强结论。

### 10.6 Robustness Experiments

场景：

- occlusion。
- low light。
- motion blur。
- keypoint dropout。
- time offset。
- ASR noise。
- missing audio。
- rear-row small-object condition。

实现建议：

- 从现有 case 中标记场景标签。
- 对 ASR 施加时间偏移或文本噪声。
- 对 pose keypoints 做 dropout simulation。
- 对 rear-row case 比较 full、sliced、SR variants。

### 10.7 Case Study

推荐主案例：`output/codex_reports/front_45618_full`。

原因：

- `backend_output_review_for_paper.md` 记录其包含 `actions.fusion_v2.jsonl`、`event_queries.fusion_v2.jsonl`、`align_multimodal.json`、`verified_events.jsonl`、`timeline_chart.json`、`timeline_students.csv`、`student_id_map.json` 和 contract reports。
- API 示例显示该 case 可返回 timeline segments、verified events 和 event queries。
- 第一个 verified event 具有 `p_match`、`p_mismatch`、`reliability_score`、`uncertainty`、`label` 和 evidence scores。

辅助案例：

- `output/codex_reports/front_1885_full`：schema pass 与真实 ASR accepted segments。
- `output/codex_reports/front_046_sr_ablation`：rear-row enhancement ablation。
- `output/codex_reports/run_full_001/full_integration_001`：ASR placeholder/fallback failure-style case。

### 10.8 Efficiency and Scalability

已有线索：

- `front_046_sr_ablation` 表含 `stage_runtime_sec` 和 `effective_fps`。
- real experiment summaries 有 elapsed seconds。

待补：

- 标准硬件：GPU、CPU、RAM。
- per-stage runtime breakdown。
- detector FPS 与 pipeline end-to-end FPS。
- server/interface loading time。

## 11. Results and Analysis Plan

### 11.1 Behavior Detection Results

可写阶段性结果：

- `official_yolo_finetune_compare/reports/current_assessment.md`：`wisdom8_yolo11s_detect_v1` final `mAP50=0.93183`，`mAP50-95=0.79836`。
- `runs/detect/official_yolo11s_detect_e150_v1/results.csv` 最后若干行显示 val 指标约 `mAP50=0.92755`、`mAP50-95=0.79949`。
- `runs/detect/case_yolo_train/results.csv` 历史 run 显示 `mAP50=0.93345`、`mAP50-95=0.81140`，但 `current_assessment.md` 说明该 run 绑定不同 dataset yaml/class mapping，只能作 historical reference。

需二次核对：

- `YOLO论文大纲/full_research_report.md` 和 `yolo论文/深度研究报告_2026-05-03.md` 报告 `test mAP50=0.9806`、`mAP50-95=0.8782`。
- `YOLO论文大纲/project_audit_20260426/dataset_classroom_yolo_summary.json` 显示 test images 为 0。
- 正式论文不能在未核对 split 前写“独立测试集结果”。

### 11.2 Pipeline Contract Results

可写：

- `backend_output_review_for_paper.md` 记录 22 个目录具备 fusion v2、timeline、verified events 和 contract 组合。
- `front_1885_full` 通过真实 schema 检查。
- `contracts/schemas.py` 定义 event query、pose UQ、align、verified event、eval/calibration report 和 manifest validators。

写法：

- 主文写 pipeline contract 保障 artifact completeness 和 auditability。
- 附录列出完整 schema fields。

### 11.3 Alignment and Verification Results

可写：

- `front_45618_full/align_multimodal.json` 包含 event id、query text、window、basis_motion、basis_uq 和 candidates。
- `front_45618_full/verified_events.jsonl` 包含 `p_match`、`p_mismatch`、`reliability_score`、`uncertainty`、`label`、evidence。
- 示例事件：`event_id=e_000000_00`，`p_match=0.7132`，`reliability_score=0.6603`，`label=match`。来源：`backend_output_review_for_paper.md`。

限制：

- 若 eval report 使用 self-label fallback，不可写为正式 accuracy。
- 必须用 `paper_experiments/gold/gold_validation_report.json` 或新 gold label 做正式事件级指标。

### 11.4 ASR Quality Gate and Fallback Results

可写：

- `front_1885_full/asr_quality_report.json`：`segments_raw=12`、`segments_accepted=12`、`segments_rejected=0`、`status=ok`。
- `front_26729_full/asr_quality_report.json`：`segments_raw=22`、`segments_accepted=22`、`segments_rejected=0`、`status=ok`。
- `run_full_001/full_integration_001/transcript.jsonl`：`[ASR_EMPTY:empty_whisper_result]`。

论文结论：

- ASR quality gate 能区分可用文本与 placeholder/fallback 情况。
- 不写 ASR 显著提升；写“防止不可靠 textual evidence 污染验证链”。

### 11.5 Rear-row Enhancement Ablation

可写：

- `output/codex_reports/front_046_sr_ablation/sr_ablation_compare_table.md` 中 A0/A1/A8 对比显示 tracked_students、rear pose rows、actions_fusion_v2、timeline rows 等 proxy 随 sliced/SR 设置变化。
- A0 full no SR: tracked_students=20，actions_fusion_v2=332，timeline rows=77。
- A1 full_sliced: tracked_students=33，actions_fusion_v2=713，timeline rows=167。
- A8 adaptive_sliced_artifact_deblur_opencv: tracked_students=37，actions_fusion_v2=1609，timeline rows=276。

限制：

- 表中明确说明多数 metrics 是 proxy，正式 claim 需要 GT annotations、runtime cost 和 failure cases。
- 不应写“后排检出率提升 xx%”作为正式结论，除非补 formal GT。

### 11.6 Tracking Ablation

可写：

- `scripts/pipeline/03_track_and_smooth.py` 和 `03e_track_behavior_students.py` 支持 Kalman/Hungarian/seat prior/ByteTrack-like tracking。
- 现有报告中有 track gap、seat anchor jitter、track count 等 diagnostics。

待补：

- IDSW、IDF1、HOTA、MOTA 需要正式 tracking GT。

### 11.7 Interface Case Study

可写：

- `server/app.py` 已有 `/api/case/{case_id}/evidence/{event_id}`、`/api/case/{case_id}/alignment/{event_id}`。
- `front_vsumvis.html` 显示 case list、video、projection/glyph、timeline、evidence panel、Top-K Alignment Candidates。

论文写法：

- 用截图展示如何从 event glyph/timeline segment 打开 evidence panel，并查看 selected candidate 与 alternatives。

### 11.8 Failure Analysis

必须写：

- 检测 split 风险。
- 长尾类 `jz`。
- ASR placeholder 与 text_score 语义不一致。
- event-level gold label 太少。
- bundle/raw case 字段不完全一致。
- LLM prototype 不足以作为主贡献。

## 12. Case Study Plan

### 12.1 主案例选择

主案例：`output/codex_reports/front_45618_full`。

选择理由：

- 证据链最完整，包含 visual actions、event queries、alignment、verified events、timeline、student map 和 contract。
- 后端报告已记录 API 可返回 timeline segments、verified events、event queries。
- 适合展示“从 query 到 top-k candidates 到 verified event 到 timeline overlay”的完整链路。

### 12.2 Case Study 写法

1. 输入视频：说明视角、长度、课堂活动类型，避免暴露身份信息。
2. 视觉检测结果：展示若干行为片段和 student tracks。
3. text/ASR 状态：说明 ASR 是否可用，accepted/rejected/fallback 状态。
4. alignment candidates：展示一个 query 的 adaptive window 和 top-k candidates。
5. verified events：列出 selected candidate、`p_match`、`reliability_score`、`uncertainty`、label。
6. timeline：展示该事件如何叠加到 student-level timeline。
7. reliability explanation：解释 UQ 如何降低 reliability 或为何输出 uncertain。
8. failure/uncertain example：必须展示至少一个 `uncertain` 或 `mismatch`，避免只展示成功案例。

### 12.3 建议图文材料

- 截图 1：VSumVis interface overview。
- 截图 2：Evidence panel with top-k alignment candidates。
- 表格：单事件证据链字段。
- 小图：video frame + selected student bbox/track。
- 小图：timeline segment with verification label。

## 13. Discussion

### 13.1 Why Verification Matters More Than Detection Confidence

课堂分析面向事件解释，而 detection confidence 只说明单个视觉框的分类置信度。教师和研究者需要知道事件是否被文本语义、时间上下文和轨迹稳定性共同支持。因此 verification 比单帧 confidence 更符合教育分析任务。

### 13.2 Why ASR Should Be Gated Rather Than Blindly Fused

ASR 在课堂中受远场语音、噪声、回声、多说话人影响。低质量 ASR 若被固定权重融合，会产生伪语义证据。quality gate 和 visual fallback 的意义是保护 pipeline，而不是证明文本一定提升性能。

### 13.3 Why Uncertainty Should Affect Temporal Alignment

当 track 不稳定或学生动作剧烈时，视觉行为片段的时间边界更不可靠。自适应窗口可增加 candidate recall，同时通过 top-k candidates 和 reliability penalty 控制误匹配风险。

### 13.4 Why Visual Analytics Matters

可视分析界面将模型内部结果转化为可检查证据链。教师/分析者可以看到何时发生事件、哪个学生相关、系统为什么判断 match/uncertain/mismatch，以及原始视频是否支持该判断。

### 13.5 Generalization Risks

- 单一教室/视角数据可能导致 domain bias。
- 类别长尾影响少数行为 recall。
- ASR 质量受麦克风和教室声学环境影响。
- 后排增强参数可能对不同教室布局不稳定。

### 13.6 Ethical and Privacy Considerations

- 学生视频涉及隐私，正式论文需说明授权、匿名化和数据访问边界。
- timeline 不应直接用于高风险学生评价。
- 系统输出应作为教学分析辅助，保留 human review。
- 对低置信和 uncertain 事件应避免自动化惩罚性使用。

### 13.7 Deployment Limitations

- 当前系统是离线/准离线 pipeline 和本地 FastAPI prototype。
- 生产实时部署、权限管理、数据安全、跨校区运维尚未实现。

## 14. Limitations

1. **Dataset split risk**：当前 processed dataset 记录 train=7416、val=1467、test=0，正式泛化结论需要视频级独立 test split。
2. **Event-level gold label insufficiency**：`paper_experiments/gold/gold_validation_report.json` 仅 19 条 peer-reviewed gold rows，不足以支撑强统计结论。
3. **ASR quality limitations**：存在 accepted ASR case，也存在 placeholder/empty case，不能写文本模态稳定提升。
4. **LLM module prototype status**：`06f_llm_semantic_fusion.py` 默认 simulate，真实 LLM API 调用未实现。
5. **No OCR implementation**：当前没有形成 OCR 主线模块，敏感文本识别只能放 future work。
6. **No production real-time deployment**：当前是 pipeline + local server prototype，不是生产系统。
7. **Cross-classroom generalization not fully verified**：尚缺跨教室、跨学校、跨摄像机配置验证。
8. **Long-tail category risk**：`jz` 类样本极少，少数类指标需要单独报告。
9. **Tracking GT missing**：IDF1、HOTA、MOTA 等正式 tracking 指标需要人工 GT。
10. **Possible bias and privacy issues**：学生行为分析存在误判、偏差和隐私风险，必须限制用途并加入人工审核。

## 15. Conclusion Draft

本文提出了一个面向智慧课堂视觉行为序列与文本语义流的可审计双重验证框架。该框架以 YOLOv11 行为检测和 pose tracking 构建学生级视觉行为序列，以 ASR quality gate 和事件查询构建文本语义流，并通过语义桥接协议、不确定性感知自适应对齐和动态可靠性评分输出 `match`、`mismatch` 与 `uncertain` 三类事件判断。与仅报告检测框或分类置信度的方法不同，本文将底层视觉检测结果转化为学生级、事件级、可靠性可解释的 timeline，并通过可视分析界面暴露 query、候选、分数、标签和源文件组成的完整证据链。当前实验和系统输出表明该框架具备可行性，但正式投稿仍需补充视频级独立测试集、更大规模人工事件标注、跨场景验证和更强的学习式 verifier。

## 16. Figures and Tables Plan

| 编号 | 中文标题 | English Title | 放置章节 | 内容 | 数据来源文件 | 服务论点 | 已有素材 | 需新生成 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Figure 1 | 整体框架图 | Overall Framework | Introduction / Method | video/audio 输入，visual stream，text stream，UQ，alignment，verification，timeline UI | `scripts/main/09_run_pipeline.py`, `backend_output_review_for_paper.md` | 本文是系统级 dual-verification framework | 部分已有 pipeline 图 | 需要 IEEE 双栏版 |
| Figure 2 | 视觉行为序列构建 | Visual Behavior Sequence Construction | Method 4.2 | YOLO behavior detections、pose tracks、student mapping、action segments | `02d_export_behavior_det_jsonl.py`, `03_track_and_smooth.py`, `03e_track_behavior_students.py` | 从 detection 到 student-level sequence | 有检测/timeline 输出 | 需要整理 |
| Figure 3 | UQ 引导的自适应对齐 | UQ-guided Adaptive Alignment | Method 4.6 | query time、adaptive window、top-k candidates、UQ/motion basis | `xx_align_multimodal.py`, `align_multimodal.json` | 说明不是固定窗口 | 有 JSON | 需要绘制 |
| Figure 4 | 双重验证评分 | Dual Verification Scoring | Method 4.7 | visual_score、text_score、weights、p_match、reliability、三值标签 | `verifier/infer.py`, `verifier/model.py` | 解释 verifier 公式 | 无 | 需要绘制 |
| Figure 5 | 学生级 timeline 界面 | Student-level Timeline Interface | System / Case Study | VSumVis page, video, timeline, evidence panel | `front_vsumvis.html`, `server/app.py` | 可视分析与审计 | 已有前端 | 需要截图 |
| Figure 6 | 案例证据链 | Case Study Evidence Chain | Case Study | query、candidate、scores、label、timeline overlay | `front_45618_full/verified_events.jsonl`, `align_multimodal.json` | 展示可审计性 | 有数据 | 需要合成图 |
| Figure 7 | 可靠性图 | Reliability Diagram | Experiments / Results | calibration bins, ECE, Brier | `verifier_calibration_report.json`, `verifier_reliability_diagram.svg` | 可靠性评估 | 有 SVG | 需确认样本数 |
| Figure 8 | 后排增强对比 | Rear-row Enhancement Comparison | Ablation / Supplement | A0/A1/A8 frame/timeline/proxy metrics | `front_046_sr_ablation/` | 后排场景鲁棒性 | 有表格 | 需图形化 |
| Table 1 | 相关工作对比 | Related Work Comparison | Related Work | 方法、模态、alignment、UQ、verification、VA interface | 本地 references/report | 研究空白 | 无 | 需要 |
| Table 2 | 行为分类体系 | Behavior Taxonomy | Method | 8 类 code、中英标签、样本数、风险 | `dataset_readiness.md`, taxonomy scripts | 语义桥接基础 | 有数据 | 需要 |
| Table 3 | 数据集统计 | Dataset Statistics | Experiments | train/val/test、boxes、class imbalance | `dataset_readiness.md`, `dataset_classroom_yolo_summary.json` | 数据规模与 split 风险 | 有 | 需要三线表 |
| Table 4 | Pipeline Contract 输出 | Pipeline Contract Outputs | Method / Results | artifact、schema、status、source | `contracts/schemas.py`, contract reports | 可审计工程链 | 有 | 需要 |
| Table 5 | 检测结果 | Detection Results | Results | precision、recall、mAP50、mAP50-95、per-class AP | `current_assessment.md`, `runs/detect/*/results.csv` | 视觉基础能力 | 有冲突 | 需核对 |
| Table 6 | 消融实验 | Ablation Study | Experiments / Results | semantic bridge、ASR gate、adaptive window、dynamic fusion、UQ penalty | scripts/experiments, future runs | 方法有效性 | 部分 | 需补实验 |
| Table 7 | 鲁棒性结果 | Robustness Results | Experiments | occlusion、low light、ASR noise、missing audio、rear-row | existing/future case labels | 噪声场景讨论 | 部分 | 需补标注 |
| Table 8 | 局限与实现状态 | Limitations and Implementation Status | Discussion / Supplement | implemented、under-evaluated、prototype、planned | code/reports | 防止过度声明 | 可写 | 需要 |

## 17. Mathematical Formulas Plan

### 17.1 视觉动作片段

放置：Problem Definition。

$$
\mathcal{X} = \{x_i=(s_i,a_i,t_i^s,t_i^e,c_i,u_i,\sigma_i)\}_{i=1}^{M}.
$$

对应实现：`actions.fusion_v2.jsonl`、`03e_track_behavior_students.py`。

### 17.2 文本事件查询

放置：Problem Definition。

$$
\mathcal{Q} = \{q_j=(e_j,\hat{t}_j,r_j,\kappa_j)\}_{j=1}^{J}.
$$

对应实现：`06b_event_query_extraction.py`、`event_queries.fusion_v2.jsonl`。

### 17.3 UQ 估计

放置：Method 4.4。

$$
U_{s,t}=0.45(1-v_{s,t})+0.35(1-m_{s,t})+0.20(1-b_{s,t}).
$$

对应实现：`03c_estimate_track_uncertainty.py`。

### 17.4 ASR 文本置信度

放置：Method 4.5。

建议写成门控函数而非强公式：

$$
C_t = g(\mathrm{avg\_logprob},\mathrm{no\_speech\_prob},\mathrm{compression\_ratio},\mathrm{source}).
$$

说明：当前实现主要是 quality tier 和 audio confidence，不应硬写成学习式概率。

### 17.5 自适应时间窗口

放置：Method 4.6。

$$
\Delta_j=\mathrm{clip}(\Delta_0+\alpha_m M_j+\beta_u U_j,\Delta_{\min},\Delta_{\max}).
$$

当前参数：$\Delta_0=1.0$，$\alpha_m=1.2$，$\beta_u=0.8$，$\Delta_{\min}=0.6$，$\Delta_{\max}=4.0$。

### 17.6 候选重叠

放置：Method 4.6。

$$
O_{ij}=
\frac{|[t_i^s,t_i^e]\cap[\hat{t}_j-\Delta_j,\hat{t}_j+\Delta_j]|}
\min(t_i^e-t_i^s,2\Delta_j)+\epsilon}.
$$

对应实现：`scripts/pipeline/xx_align_multimodal.py` 的 `_interval_overlap()` 使用 intersection duration 除以两个区间中较短的时长，而不是 IoU。

### 17.7 视觉分数

放置：Method 4.7。

$$
C_v=\mathrm{clip}_{[0,1]}(0.65O_{ij}+0.35C_i).
$$

对应实现：`verifier/infer.py` 中 `visual_score`。

### 17.8 动态融合权重

放置：Method 4.7。

$$
w_v=1-uq,\quad w_a=C_a.
$$

对应实现：`verifier/infer.py` 中 `w_visual` 和 `w_audio`。

### 17.9 p_match

放置：Method 4.7。

$$
p_{match}=
\frac{w_v C_v+w_a C_t}{w_v+w_a+\epsilon}.
$$

visual fallback 情况：

$$
p_{match}=C_v.
$$

### 17.10 reliability

放置：Method 4.7。

$$
R=p_{match}(1-\lambda_u uq),\quad \lambda_u=0.60.
$$

对应实现：`verifier/infer.py`。

### 17.11 三值决策

放置：Method 4.7。

$$
y=
\begin{cases}
\mathrm{match}, & R \ge \tau_m,\\
\mathrm{uncertain}, & \tau_u \le R < \tau_m,\\
\mathrm{mismatch}, & R < \tau_u.
\end{cases}
$$

默认：$\tau_m=0.60$，$\tau_u=0.40$。来源：`verifier/model.py`。

## 18. References Planning

### 18.1 YOLO 系列与课堂行为检测

引用位置：Related Work 2.1/2.2，Experiments detector baseline。

- Wang, C.-Y., Bochkovskiy, A., Liao, H.-Y. M. **YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors**. CVPR 2023.
- Wang, A., Chen, H., Liu, L., et al. **YOLOv10: Real-Time End-to-End Object Detection**. arXiv 2024, `arXiv:2405.14458`.
- Ultralytics YOLO / YOLO11 documentation or model card. 用于说明基础 detector 来源，正式稿需补官方 citation 或 software citation。
- **Classroom Behavior Detection Based on Improved YOLOv5 Algorithm Combining Multi-Scale Feature Fusion and Attention Mechanism**. Applied Sciences, 2022. DOI: `10.3390/app12136790`.
- **Student Behavior Detection in the Classroom Based on Improved YOLOv8**. Sensors, 2023. DOI: `10.3390/s23208385`.
- **BiTNet: A Lightweight Object Detection Network for Real-time Classroom Behavior Recognition**. Journal of King Saud University - Computer and Information Sciences, 2023. DOI: `10.1016/j.jksuci.2023.101670`.
- **SCB-Dataset: Student Classroom Behavior Dataset**. arXiv 2023, `arXiv:2304.02488`.

本地来源：`yolo论文/深度研究报告_2026-05-03.md`，`YOLO论文大纲/论文准备.md`，`YOLO论文大纲/.../04_references/references.md`。

### 18.2 Pose Tracking / Action Recognition

引用位置：Related Work 2.3，Method 4.2。

- Kalman filter 经典引用：Kalman, 1960，正式 BibTeX 待补。
- Hungarian matching 经典引用：Kuhn, 1955 或 Munkres, 1957，正式 BibTeX 待补。
- ByteTrack：Zhang et al., **ByteTrack: Multi-Object Tracking by Associating Every Detection Box**, ECCV 2022，正式 BibTeX 待补。
- Feichtenhofer et al., **SlowFast Networks for Video Recognition**, ICCV 2019，正式 BibTeX 待补。
- **MSTA-SlowFast: A Student Behavior Detector for Classroom Environments**. Sensors, 2023. DOI: `10.3390/s23115205`.
- **A Spatio-Temporal Attention-Based Method for Detecting Student Classroom Behaviors**. arXiv 2023, `arXiv:2310.02523`.

本地来源：`scripts/pipeline/03_track_and_smooth.py`，`scripts/pipeline/05_slowfast_actions.py`，`scripts/_REPORT.md`，`yolo论文/深度研究报告_2026-05-03.md`。

### 18.3 Whisper / ASR / Classroom Speech

引用位置：Related Work 2.5，Method 4.5，ASR quality gate。

- Radford, A., Kim, J. W., Xu, T., et al. **Robust Speech Recognition via Large-Scale Weak Supervision**. arXiv 2022, `arXiv:2212.04356`.
- Baevski et al. **Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**. NeurIPS 2020, `arXiv:2006.11477`.
- AV-ASR with Whisper / Dual-Use AV-ASR, arXiv 2026, citation details must be verified before formal submission because it is recent.
- MM-TBA / teacher behavior dataset, Scientific Data 2025, details must be verified before formal citation.

本地来源：`scripts/pipeline/06_asr_whisper_to_jsonl.py`，`YOLO论文大纲/论文准备.md`，`yolo论文/深度研究报告_2026-05-03.md`。

### 18.4 Multimodal Video-Language / Audio-Visual Understanding

引用位置：Related Work 2.4/2.6，Discussion。

- Radford et al. **Learning Transferable Visual Models From Natural Language Supervision**. ICML 2021, `arXiv:2103.00020`.
- Nagrani et al. / attention bottlenecks: **Attention Bottlenecks for Multimodal Fusion**. NeurIPS 2021, `arXiv:2107.00135`.
- Girdhar et al. **ImageBind: One Embedding Space To Bind Them All**. CVPR 2023, `arXiv:2305.05665`.
- Tian et al. **CMR-AVE: Cross-Modal Relation-Aware Network for Audio-Visual Event Localization**. ACM MM 2020, `arXiv:2008.00836`.
- Zhai et al. **MAViL: Masked Audio-Video Learners**. NeurIPS 2023.
- LanguageBind, ICLR 2024, `arXiv:2310.01852`.
- Video-LLaMA, InternVideo2, CAT, Meerkat：放 Related Work 或 Future Work，不放 Method 主贡献。

本地来源：`yolo论文/深度研究报告_2026-05-03.md`，`YOLO论文大纲/.../04_references/references.md`。

### 18.5 Calibration / Uncertainty / Reliability

引用位置：Related Work 2.7，Method 4.4/4.7，Experiments reliability metrics。

- Guo et al. **On Calibration of Modern Neural Networks**. ICML 2017，正式 BibTeX 待补。
- Minderer et al. **Revisiting the Calibration of Modern Neural Networks**. NeurIPS 2021 / arXiv `2106.07998`，按最终版本核对。
- Brier, G. W. **Verification of forecasts expressed in terms of probability**. Monthly Weather Review, 1950，Brier score 源头引用。
- Temperature scaling、ECE、reliability diagram 相关标准引用。

本地来源：`verifier/model.py`，`verifier/calibration.py`，`yolo论文/深度研究报告_2026-05-03.md`。

### 18.6 SAHI / Sliced Inference / Super-Resolution

引用位置：Method 4.2，Ablation，Supplement。

- SAHI: slicing aided hyper inference，正式 BibTeX 待补。
- Real-ESRGAN：Wang et al., ICCV Workshops 2021 或 arXiv，正式 BibTeX 待补。
- BasicVSR++ / RealBasicVSR：正式 BibTeX 待补。
- OpenCV interpolation 不需要论文引用，但需说明是 engineering baseline。

本地来源：`utils/sliced_inference_utils.py`，`02_export_keypoints_jsonl.py`，`02d_export_behavior_det_jsonl.py`，`output/codex_reports/front_046_sr_ablation/`。

### 18.7 Visual Analytics / Education Analytics / Human-centered AI

引用位置：Related Work 2.8，System Design，Discussion。

- IEEE VIS/VGTC visual analytics system papers on timeline/event sequence analysis，需后续检索和 BibTeX 化。
- Education visual analytics / learning analytics dashboards，需后续补齐。
- Human-centered AI audit / uncertainty visualization，需后续补齐。
- VAST / TVCG papers on provenance and explainable visual analytics，可作为 interface audit 论据。

本地来源：`YOLO论文大纲/论文准备.md` 提到 ChinaVis/IEEE VGTC 模板与可视分析范围；正式英文稿需补更强 VIS/TVCG references。

### 18.8 IEEE VGTC Writing Style References

引用位置：不一定进入 References，可作为 formatting preparation。

- IEEE VGTC / TVCG LaTeX template。
- IEEE VIS author kit。
- ChinaVis 2026 英文稿若采用 IEEE VGTC 模板，需按最新 CFP 核对页数和匿名要求。

注意：模板和 author kit 属格式依据，不一定作为论文 reference。

### 18.9 待清洗 BibTeX 列表

后续任务：

1. 从 `YOLO论文大纲/.../04_references/references.md` 抽取已列条目。
2. 从 `论文准备.md` 抽取 links。
3. 从 `scripts/pipeline/06f_llm_semantic_fusion.py` 抽取 Demirel et al. 2025 late multimodal sensor fusion 引用，并核对真实性。
4. 从 `verifier/infer.py` 注释抽取 Kiziltepe et al. IEEE Access 2024 与 CMU-MMAC Survey arXiv 2025，并核对正式题名、作者和 DOI/arXiv。
5. 对所有 2025/2026 文献做二次核验，避免引用不存在或信息不稳的论文。

## 19. IEEE VGTC Formatting Checklist

- 最终英文稿使用 IEEE VGTC / TVCG LaTeX 模板。
- 正文结构建议：Abstract + Index Terms + Introduction + Related Work + Problem Definition + Method + Visual Analytics System + Experiments + Case Study + Discussion + Conclusion + References。
- 图表使用英文 Figure / Table 标题。
- 所有公式统一编号，并在正文逐一解释变量。
- 算法伪代码使用 Algorithm 环境，可命名为 `Dual-Verification Pipeline`。
- 引用使用 BibTeX，不手写 URL-only references。
- 表格尽量使用三线表风格。
- 主要框架图可跨双栏，其他图适配单栏或双栏宽度。
- 贡献列表控制在 4–5 条。
- 每个 claim 必须对应代码、实验或文献证据。
- detection 指标必须标注 split 和 source，不混用 val/test。
- LLM、OCR、MLLM、ASPN/DySnake/GLIDE 等未形成主线证据的内容只能放 limitations/future work。
- ASR 写成 quality-gated semantic stream，不写成稳定提升模块。
- Figure 1/5/6 应优先准备高质量矢量图或高清截图，保证双栏缩放后可读。
- Supplementary material 可包含完整 schema、artifact inventory、case videos、extra timelines 和 BibTeX。

## 20. Codex Follow-up Task List

1. 核对最新 detection metrics：锁定 `official_yolo11s_detect_e150_v1` 或新的正式 run，导出 precision、recall、mAP50、mAP50-95、per-class AP 和 confusion matrix。
2. 核对 train/val/test split 是否按视频划分：若 test 仍为 0 或 frame-level split，创建 video-level independent test split。
3. 从代码注释和 references 文件中提取 BibTeX：覆盖 YOLO、Whisper、SlowFast、ByteTrack、SAHI/SR、CLIP/audio-visual、calibration、visual analytics。
4. 生成 IEEE VGTC 图表资产：Figure 1 framework、Figure 3 adaptive alignment、Figure 4 verification scoring、Figure 5 interface screenshot。
5. 补充 event-level gold labels：至少覆盖 match/mismatch/uncertain、多视角、ASR ok/fallback、后排场景。
6. 运行 fixed-window vs adaptive-window ablation：报告 candidate recall、Top-k recall、false match rate、uncertain ratio。
7. 运行 static fusion vs dynamic fusion ablation：比较 fixed visual/text weights 与 `w_v=1-uq, w_a=audio_confidence`。
8. 生成 reliability diagram：使用人工 gold 或明确标注 exploratory labels，输出 ECE、Brier 和 calibration plot。
9. 导出 per-class AP 表：特别标注 `jz` 长尾类和类别不平衡风险。
10. 整理 case study evidence：选择 `front_45618_full`，导出 query、alignment candidates、verified event、timeline overlay、video frame。
11. 生成英文 LaTeX 初稿 skeleton：按 IEEE VGTC 模板创建 `main.tex`、`sections/`、`figures/`、`tables/` 和 `refs.bib`。
