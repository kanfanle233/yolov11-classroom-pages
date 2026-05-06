# 面向 IEEE VGTC 英文稿件的中文论文大纲（8 页正文压缩版）

> 目标稿件类型：IEEE VGTC / IEEE VIS conference-style paper  
> 当前语言：中文写作版，后续翻译为英文正式稿  
> 正文页数约束：不含参考文献，正文不超过 8 页  
> 写作定位：可视分析 + 多模态验证系统论文，而不是 YOLOv11 检测器改进论文

## 题目与论文主线

**推荐中文题目**  
《面向智慧课堂视觉行为序列与文本语义流的可审计双重验证框架》

**推荐英文题目**  
*An Auditable Dual-Verification Framework for Smart Classroom Visual Behavior Sequences and Textual Semantic Streams*

**论文主线**  
本文研究真实智慧课堂中“视觉行为片段是否与课堂语义事件一致”的验证问题。系统以 YOLOv11 系列检测器和 YOLO pose 构建学生级视觉行为序列，在后排模糊区域加入 ROI 分辨率增强与切片推理分支，再从 Whisper ASR 和事件抽取模块得到文本语义流。随后，系统用语义桥接协议统一视觉短码和文本事件，用轨迹不确定性调节跨模态时间窗口，并结合 LLM 辅助语义加权思路与蒸馏轻量 verifier 输出 `match`、`mismatch` 或 `uncertain`。最终结果以学生级时间线和证据面板呈现，支持教师或研究者审计模型判断。

**核心写作原则**

- YOLOv11s 是基础检测器，不写成本文创新点。
- ASR 是有质量门控的文本来源，不写成稳定提升模块。
- 可以把 LLM 相关内容写入主创新，但必须区分“在线 LLM 语义打分原型”和“蒸馏后的轻量 verifier 主流程分支”。
- 不得写成“LLM 替代人工 gold 标注”；更准确的说法是“LLM 银标辅助训练轻量 verifier，人工 gold 仍作为评测参考”。
- OCR、MLLM、ST-GCN、IGFormer、ASPN/DySnake/GLIDE 等未形成主线实验的内容不写入主贡献。
- 所有实验结论采用保守口径。当前检测结果主写验证集证据 `mAP50≈0.93`、`mAP50-95≈0.80`；旧报告中的 `test mAP50=0.9806` 仅作为待复核冲突，不作为正式主结果。

## 摘要草稿

智慧课堂行为分析不仅需要识别学生动作，还需要判断这些动作是否与课堂语义事件相互印证。单一视觉检测在遮挡、远距离、多人重叠和小动作相似的场景中容易产生不稳定判断；语音转写虽然提供了教师指令和课堂活动的语义线索，但远场噪声、回声和 ASR 错识别也可能引入不可靠文本证据。为解决这一问题，本文提出一个面向智慧课堂视觉行为序列与文本语义流的可审计双重验证框架。系统以 YOLOv11 系列检测器和 YOLO pose 为基础构建 8 类课堂行为片段与学生轨迹，在后排模糊区域引入 ROI 分辨率增强与切片推理分支，以改善小目标和模糊行为的可见性；同时基于关键点可见率、运动稳定性和边界框稳定性估计轨迹不确定性。随后，系统通过语义桥接协议将视觉短码转换为可验证的中英文语义标签，在文本侧使用 Whisper ASR、质量门控和事件查询抽取构建语义流，并引入 LLM 辅助语义打分原型与蒸馏轻量 verifier 分支，用于补充语义加权和降低运行时对大模型的依赖。跨模态对齐阶段根据运动状态和轨迹不确定性自适应调整候选时间窗口；验证阶段根据视觉分数、文本分数、动态模态权重与轻量 verifier 输出三值决策，并保留完整证据链。当前本地实验表明，该框架能够生成学生级时间线、事件查询、对齐候选、验证事件和 VSumVis 风格可视分析界面；后排增强分支显著提高了可观测片段和时间线覆盖，学生模型分支已进入主流程并支持替代在线 LLM 推理。行为检测主线已有验证集证据支持，但独立测试集划分、事件级人工金标规模以及在线 LLM 直接打分的定量效果仍需补强。因此，本文的主要贡献在于把课堂视觉检测结果转化为可审计、可解释、可交互分析的事件级结果，并探索后排增强与语义蒸馏在课堂验证任务中的系统价值。

**Index Terms**  
Smart classroom analytics; multimodal verification; visual behavior analysis; visual analytics; uncertainty-aware alignment; classroom behavior detection; temporal event verification; student-level timeline; reliability scoring.

## 8 页正文预算

| 模块 | 建议页数 | 写作重点 |
| --- | ---: | --- |
| Abstract + Index Terms | 0.4 | 问题、方法、阶段性证据、局限 |
| 1. Introduction | 0.9 | 需求、挑战、双重验证思想、贡献 |
| 2. Related Work | 0.8 | 三类相关工作，差异由表格承担 |
| 3. Problem Definition | 0.6 | 输入、视觉片段、文本查询、验证输出 |
| 4. Framework and Method | 1.7 | 后排增强、语义桥接、UQ 对齐、动态验证 |
| 5. Visual Analytics System | 1.1 | 设计目标、视图、交互、审计任务 |
| 6. Experiments and Results | 1.6 | 检测、pipeline、ASR gate、SR 与 student verifier 证据 |
| 7. Case Study and Discussion | 0.6 | 单案例证据链与方法讨论 |
| 8. Limitations and Conclusion | 0.3 | 诚实局限与收束 |

---

# 1. Introduction

## 写作目的

本节需要说明为什么智慧课堂行为分析不能停留在单帧检测框和分类置信度上。读者应在引言结束时理解：本文研究的是“课堂事件是否可靠”的系统问题，而不是“检测器结构是否更强”的模型问题。

## 章节草稿

智慧课堂分析的目标，是帮助教师和研究者理解课堂中真实发生的学习行为。相比“某一帧中检测到一个学生在写字”，教学分析更关心的是：哪个学生在什么时间持续参与了什么活动，这个判断是否与教师指令和课堂语义一致，以及系统为什么给出这样的判断。这样的需求要求模型输出从 frame-level detections 转向 student-level temporal events。

仅依赖视觉检测会遇到明显限制。真实课堂中常见后排远距离、多人遮挡、低光、运动模糊和小动作相似等情况。听课、低头写字、低头看书和转头在视觉上往往非常接近。即使检测器给出较高置信度，这个置信度也只能说明单帧视觉分类相对可信，并不能说明该行为是否构成一个真实课堂事件。

文本语义流可以补充这一缺口。教师语音中常包含“请举手”“开始讨论”“看黑板”等指令，这些信息能够帮助解释视觉行为。但是，课堂语音通常是远场录制，伴随回声、环境噪声和多人说话。ASR 结果可能为空，也可能包含错识别文本。如果把低质量 ASR 直接和视觉结果融合，反而会污染事件判断。

基于这一观察，本文提出一个可审计的视觉-语义双重验证框架。系统先构建学生级视觉行为序列，再从 ASR 和事件抽取模块得到文本语义查询。随后，系统通过语义桥接协议统一视觉和文本的事件表示，并根据轨迹不确定性动态调整时间对齐窗口。最终，系统输出 `match`、`mismatch` 或 `uncertain`，同时保留 query、候选片段、模态分数、可靠性和源文件。

本文的贡献可概括为四点。第一，提出一个面向智慧课堂的视觉行为序列与文本语义流双重验证框架，将检测结果转化为可审计事件。第二，提出面向后排模糊目标的 ROI 分辨率增强与切片推理分支，用于提高后排学生行为片段和时间线证据的可观测性。第三，引入 LLM 辅助语义加权思路，并将其蒸馏为轻量 verifier 分支，使系统在保留语义判断能力的同时减少运行时对大模型的依赖。第四，构建学生级时间线与证据链可视分析界面，使教师和研究者能够检查模型判断的来源。

## 可引用证据

- 主流程编排：`scripts/main/09_run_pipeline.py`。
- 语义桥接和验证输出：`contracts/schemas.py`、`verifier/infer.py`。
- 学生级时间线：`output/codex_reports/front_45618_full/timeline_chart.png`、`timeline_students.csv`。

## 图表占位

**[Fig. 1: Overall Framework，双栏图，放在 Introduction 末尾或 Method 开头]**  
建议新绘制。内容包括 video/audio 输入、visual behavior stream、textual semantic stream、UQ stream、adaptive alignment、dual verification 和 visual analytics interface。依据 `scripts/main/09_run_pipeline.py` 与 `paper_outline_zh_vgtc.md` 绘制。

---

# 2. Related Work

## 写作目的

本节压缩为三组相关工作。不要写成完整综述。重点是说明已有方法为什么不足以解决本文的问题，并用一张对比表承担细节差异。

## 2.1 Classroom Behavior Detection and YOLO-based Recognition

课堂行为检测研究大多围绕视觉检测器展开。已有工作通过改进 YOLOv5、YOLOv8、轻量网络或注意力模块提升课堂行为识别精度。这类方法对检测性能有价值，但通常将输出停留在行为类别和检测置信度上。它们较少处理教师语义流、跨模态时间对齐、事件级可靠性和可审计时间线。

本文与这些工作不同。YOLOv11s 在本文中是基础视觉感知器，用于产生 8 类课堂行为片段。本文不把检测器结构作为创新点，而是研究检测结果如何进入学生跟踪、语义桥接、事件验证和可视分析界面。

## 2.2 Multimodal Event Understanding and ASR

多模态学习和音视频事件理解表明，视觉、音频和语言信号可以相互补充。CLIP、ImageBind、CMR-AVE、MAViL 等工作证明了跨模态语义对齐的价值。Whisper 和 Wav2Vec 2.0 等 ASR 模型则为课堂语义流提供了技术基础。

但智慧课堂中的文本流并不稳定。教师语音可能被噪声、回声和多人说话干扰。本文因此不做盲目融合，而是先对 ASR 片段进行质量门控，再把可靠文本转换为事件查询。当文本不可用时，系统使用视觉回退，保证验证流程不会被错误文本牵引。

## 2.3 Reliability-aware Visual Analytics Systems

可视分析系统强调人在模型循环中的解释、审计和决策支持。对智慧课堂而言，教师和研究者需要看到的不只是模型结论，还包括结论背后的时间、学生、候选证据和不确定性。本文将 verified events、alignment candidates、reliability scores 和 student-level timeline 组织到同一界面中，使用户能够检查每个事件的判断依据。

## 可引用证据

- 本地参考文献计划：`paper_outline_zh_vgtc.md` 的 References Planning。
- 课堂行为检测文献来源：`yolo论文/深度研究报告_2026-05-03.md`。
- ASR 与 multimodal 相关引用：`YOLO论文大纲/论文准备.md`。

## 图表占位

**[Table 1: Related Work Comparison，放在本节末尾]**  
列建议：`Method`、`Scenario`、`Visual Input`、`Audio/Text Input`、`Temporal Alignment`、`Uncertainty Modeling`、`Verification Output`、`Visual Analytics Interface`、`Difference from Ours`。  
正文中只保留最关键的 6-8 个方法，不要铺开长表。

---

# 3. Problem Definition

## 写作目的

本节用少量公式定义任务边界。重点是让读者明白：本文输入是视频帧和文本语义流，输出是带有可靠性和证据链的事件级验证结果。

## 3.1 Input Streams

设课堂视频为帧序列：

$$
\mathcal{V}=\{I_t\}_{t=1}^{T}.
$$

音频或文本语义流表示为：

$$
\mathcal{A}=\{a_k=(t_k^s,t_k^e,z_k,m_k)\}_{k=1}^{K},
$$

其中 $z_k$ 是 ASR 文本，$m_k$ 包含 `avg_logprob`、`no_speech_prob`、`compression_ratio` 等质量指标。

## 3.2 Visual Behavior Segments

学生轨迹集合为：

$$
\mathcal{T}=\{\tau_s\}_{s=1}^{N},\quad
\tau_s=\{(b_{s,t},p_{s,t},\rho_{s,t})\}_{t\in\Omega_s}.
$$

其中 $b_{s,t}$ 是边界框，$p_{s,t}$ 是人体关键点，$\rho_{s,t}$ 表示可见性或轨迹质量统计。

视觉动作片段集合为：

$$
\mathcal{X}=\{x_i=(s_i,a_i,t_i^s,t_i^e,c_i,u_i,\sigma_i)\}_{i=1}^{M}.
$$

这里，$s_i$ 是学生或轨迹编号，$a_i$ 是行为语义标签，$c_i$ 是动作置信度，$u_i$ 是轨迹不确定性，$\sigma_i$ 是语义字段。

## 3.3 Textual Event Queries and Verification Output

文本事件查询集合为：

$$
\mathcal{Q}=\{q_j=(e_j,\hat{t}_j,r_j,\kappa_j)\}_{j=1}^{J}.
$$

其中 $e_j$ 是事件类型或语义编号，$\hat{t}_j$ 是查询时间，$r_j$ 是文本可靠性，$\kappa_j$ 表示查询来源，如 ASR、教师指令或视觉回退。

系统最终输出：

$$
y_j\in\{\mathrm{match},\mathrm{mismatch},\mathrm{uncertain}\},
$$

并保存证据对象：

$$
o_j=(q_j,x_j^\*,p_j^{match},p_j^{mismatch},R_j,U_j,y_j,E_j).
$$

$R_j$ 是可靠性分数，$E_j$ 包含 query、候选片段、模态分数、权重、时间线片段和源文件。

## 3.4 Adaptive Candidate Window

对每个文本查询，系统根据运动不稳定性和轨迹不确定性计算候选窗口：

$$
\Delta_j=\mathrm{clip}(\Delta_0+\alpha_mM_j+\beta_uU_j,\Delta_{\min},\Delta_{\max}).
$$

当前代码默认参数为 $\Delta_0=1.0$、$\alpha_m=1.2$、$\beta_u=0.8$、$\Delta_{\min}=0.6$、$\Delta_{\max}=4.0$。候选重叠不是 IoU，而是交集长度除以两个时间区间中较短的长度：

$$
O_{ij}=
\frac{|[t_i^s,t_i^e]\cap[\hat{t}_j-\Delta_j,\hat{t}_j+\Delta_j]|}
{\min(t_i^e-t_i^s,2\Delta_j)+\epsilon}.
$$

这一点与 `scripts/pipeline/xx_align_multimodal.py` 中 `_interval_overlap()` 的实现一致。

## 可引用证据

- 自适应窗口：`scripts/pipeline/xx_align_multimodal.py`。
- UQ 估计：`scripts/pipeline/03c_estimate_track_uncertainty.py`。
- 验证输出字段：`contracts/schemas.py`。

## 图表占位

本节不单独放图。公式较多，建议保持短小。相关流程在 Fig. 2 中统一展示。

---

# 4. Framework and Method

## 写作目的

本节是方法核心。写作时要围绕“从检测到验证”的证据链展开，而不是罗列 pipeline 脚本。建议按五个模块写：视觉行为序列、后排增强分支、语义桥接、UQ 对齐、双重验证与轻量 verifier。

## 4.1 Visual Behavior Sequence Construction

系统首先从课堂视频中构建视觉行为序列。行为检测器识别 8 类课堂行为：抬头听课、低头写字、低头看书、转头、小组讨论、举手、站立和教师互动。YOLO pose 模型提取人体关键点，tracking 模块基于 Kalman 滤波、匈牙利匹配和座位先验生成稳定轨迹。随后，系统将帧级 detections 聚合为学生级行为片段。

这一步的目标不是提出新的检测器，而是为后续验证提供结构化视觉证据。每个视觉片段都包含学生编号、行为标签、起止时间、动作置信度和语义字段。

## 4.2 Rear-row Resolution Enhancement Branch

后排学生常同时面临尺寸小、模糊强和遮挡重的问题。为此，系统在主检测分支外增加 ROI 分辨率增强与切片推理分支。该分支先根据后排区域生成密集切片，再选择 `opencv`、`realesrgan`、`basicvsrpp`、`realbasicvsr` 或外部 `nvidia_vsr` / `maxine_vfx` 适配器对后排 ROI 做增强，最后将增强结果送入 pose 与行为检测。论文中不应把它写成通用“超分提升精度”结论，而应更准确地表述为：它提高了后排可观测人数、行为片段召回代理指标和时间线覆盖率。

这一分支的价值主要体现在困难场景，而不是替代基础检测器。正文应明确：后排增强是系统级鲁棒性设计，既服务于行为检测，也服务于后续跟踪、对齐和可视审计。

## 4.3 Semantic Bridging Protocol

检测模型输出的 `tt`、`dx`、`dk` 等短码对训练有效，但对文本验证和时间线展示并不友好。系统因此加入语义桥接协议，将短码转换为 `semantic_id`、`semantic_label_zh`、`semantic_label_en` 和 `taxonomy_version`。这些字段使视觉行为、文本查询、验证器和前端界面使用同一套语义标识。

语义桥接也承担 contract 功能。若下游缺少必要字段，schema validator 可以暴露问题。这样，论文中的每个 verified event 都能追溯到统一语义协议，而不是散落的脚本输出。

## 4.4 Track Uncertainty Estimation

系统用启发式统计方法估计轨迹不确定性。该方法不是贝叶斯不确定性模型，而是根据关键点可见率、运动稳定性和边界框稳定性计算：

$$
U_{s,t}=0.45(1-v_{s,t})+0.35(1-m_{s,t})+0.20(1-b_{s,t}).
$$

其中 $v_{s,t}$ 是关键点可见率，$m_{s,t}$ 是运动稳定性，$b_{s,t}$ 是边界框稳定性。该不确定性会进入三个位置：调节对齐窗口、调整视觉模态权重、惩罚最终可靠性。

## 4.5 UQ-guided Alignment and Dual Verification

固定时间窗口难以适应真实课堂。学生动作可能滞后于教师指令，遮挡和运动也会让视觉片段边界不稳定。因此，系统根据运动状态和 UQ 自适应调整时间窗口，并保留 Top-K 候选，而不是直接选一个结果。

验证阶段使用视觉分数、文本分数和动态模态权重。启发式主分支的运行时实现为：

$$
C_v=\mathrm{clip}_{[0,1]}(0.65O_{ij}+0.35c_i),
$$

$$
w_v=1-uq,\quad w_a=C_a,
$$

$$
p_{match}=\frac{w_vC_v+w_aC_t}{w_v+w_a+\epsilon}.
$$

当没有可靠音频或触发视觉回退时，系统令 $p_{match}=C_v$。最终可靠性为：

$$
R=p_{match}(1-\lambda_u uq),\quad \lambda_u=0.60.
$$

决策规则为：

$$
y=
\begin{cases}
\mathrm{match}, & R\ge 0.60,\\
\mathrm{uncertain}, & 0.40\le R<0.60,\\
\mathrm{mismatch}, & R<0.40.
\end{cases}
$$

这一设计的重点不是追求单一高分，而是让系统在证据不足时能够输出 `uncertain`，并把判断依据交给用户审计。

## 4.6 LLM-assisted Semantic Weighting and Distilled Student Verifier

在语义侧，系统保留了一个在线 LLM 辅助语义打分原型。它读取结构化的教师指令、行为标签、时间重叠和启发式 verifier 分数，给出小幅度的语义 boost 或 dampen，用于说明“语义如何影响融合判断”。该原型当前主要用于语义分析、队列构建和前端证据展示，不宜把它写成完全成熟的在线主评分器。

为了让这一路径进入实际主流程，系统进一步使用 LLM teacher 产生的银标训练轻量 student verifier，并在 `verifier/infer.py` 中通过 `--llm_student_model` 覆盖启发式或 MLP 输出。这样，运行时主流程可以使用轻量模型直接输出 `p_match`、`p_mismatch` 和 `uncertain`，同时保留 `student_model_path`、`teacher_source` 和 `teacher_dataset` 等追溯字段。论文中应把这一点写成“LLM 辅助语义判断被蒸馏为轻量 verifier 分支”，而不是“LLM 替代人工 gold 标注”。

## 可引用证据

- 视觉行为导出：`scripts/pipeline/02d_export_behavior_det_jsonl.py`。
- 跟踪与平滑：`scripts/pipeline/03_track_and_smooth.py`。
- 行为学生映射：`scripts/pipeline/03e_track_behavior_students.py`。
- 后排增强：`scripts/experiments/16_run_rear_row_sr_ablation.py`、`scripts/utils/sliced_inference_utils.py`。
- UQ：`scripts/pipeline/03c_estimate_track_uncertainty.py`。
- 对齐：`scripts/pipeline/xx_align_multimodal.py`。
- LLM 语义原型：`scripts/pipeline/06f_llm_semantic_fusion.py`。
- 轻量 verifier：`verifier/infer.py`、`verifier/model.py`、`output/llm_judge_pipeline/teacher_labels/silver_label_summary.json`。

## 图表占位

**[Fig. 2: UQ-guided Alignment and Dual Verification，放在本节中部]**  
建议新绘制合并图。左侧显示 query time、自适应窗口和 Top-K 候选；右侧显示 `visual_score`、`text_score`、`w_v`、`w_a`、`p_match`、`reliability` 和三值标签。依据 `xx_align_multimodal.py` 与 `verifier/infer.py`。

**[Table 2: Dataset and Behavior Taxonomy，放在本节末或实验节开头]**  
数据来自 `official_yolo_finetune_compare/reports/dataset_readiness.md`。表中列 8 类行为、训练/验证样本或框数、类别不平衡风险。

---

# 5. Visual Analytics System

## 写作目的

本节要写得更像 IEEE VGTC。重点不是“我们做了一个前端”，而是说明界面如何支持分析任务、证据审计和人机协同。

## 5.1 Design Goals and User Tasks

系统面向教师、课堂研究者和模型开发者。它支持三类任务：

- **T1：定位课堂事件。** 用户需要知道事件在什么时间发生、涉及哪个学生或轨迹，以及行为持续了多久。
- **T2：解释验证结果。** 用户需要查看为什么一个事件被判为 `match`、`mismatch` 或 `uncertain`，包括视觉候选、文本查询、模态分数和不确定性。
- **T3：比较案例和设置。** 用户需要比较不同视频、不同推理设置或后排增强方案下的行为片段、验证事件和时间线覆盖情况。

## 5.2 Interface Components

界面以 case 为入口。左侧 case list 显示案例状态和 contract 状态。主视图包含视频面板、投影或 glyph 概览、学生级时间线和证据面板。用户点击时间线片段或事件点后，系统显示 query、selected candidate、Top-K alignment candidates、visual/text scores、UQ、modality weights、reliability 和 label。

这套界面把模型输出从分散文件转化为可检查的分析对象。用户既能看结果，也能看结果背后的候选证据和源文件。

## 5.3 Visual Encodings

时间线横轴表示视频时间，纵向通道表示学生或轨迹。行为片段用矩形表示，长度对应持续时间。`match`、`uncertain` 和 `mismatch` 用不同颜色区分。可靠性和不确定性可以通过透明度、边框或证据面板中的条形编码呈现。投影或 glyph 视图用于快速查看事件分布和 case 差异。

## 5.4 Audit Workflow

一次典型审计流程如下。用户先在 case list 中选择 `front_45618_full`，查看全局时间线。随后，用户点击一个 verified event，界面跳转到对应视频时间，并打开证据面板。用户可以比较 selected candidate 与其他 Top-K candidates，检查该事件为什么被判为 `match` 或 `uncertain`。如果 ASR 质量较低，界面应标出 query source 或 fallback 状态，提醒用户不要过度解读文本证据。

## 可引用证据

- 后端接口：`server/app.py` 中 `/api/v2/vsumvis/*`、`/api/case/{case_id}/evidence/{event_id}`、`/api/case/{case_id}/alignment/{event_id}`。
- 前端模板：`web_viz/templates/front_vsumvis.html`。
- 前端 bundle：`scripts/frontend/20_build_frontend_data_bundle.py`。
- 主案例资产：`output/codex_reports/front_45618_full/`。

## 图表占位

**[Fig. 3: Visual Analytics Interface，放在本节中部]**  
插入本地前端真实截图。建议运行 `server/app.py` 后打开 `/paper/front-vsumvis`，选择 `front_45618_full`，截取包含 case list、video panel、timeline 和 evidence panel 的页面。

---

# 6. Experiments and Results

## 写作目的

本节用本地已有结果支撑“系统可行性”，但不写成强泛化结论。所有指标必须注明来源和口径。

## 6.1 Dataset and Implementation

当前行为检测数据集包含 7416 张训练图、1467 张验证图，共 8883 张图像和 267861 个标注框。数据覆盖 8 类课堂行为，但类别分布明显不均衡，最大类与最小类框数比例约为 172.835。当前 processed dataset 中 test split 为 0，因此检测结果应称为验证集证据，不应写成独立测试集结论。

系统实现使用 YOLOv11 系列行为检测器、YOLO pose、tracking and smoothing、Whisper ASR、事件查询抽取、自适应对齐、dual verifier、可选后排 ROI 增强分支，以及 D3/FastAPI 可视分析界面。运行时 verifier 支持启发式分支、MLP 分支和 `--llm_student_model` 轻量学生模型分支。主流程由 `scripts/main/09_run_pipeline.py` 编排。

## 6.2 Detection and Pipeline Results

本地官方微调对比报告显示，`wisdom8_yolo11s_detect_v1` 的 final `mAP50=0.93183`，`mAP50-95=0.79836`。`runs/detect/official_yolo11s_detect_e150_v1/results.csv` 的最后阶段结果也保持在相近范围，验证集 `mAP50≈0.93`，`mAP50-95≈0.80`。这些结果说明基础视觉感知器能够为后续系统提供可用输入。

需要特别说明，部分旧报告记录了 `test mAP50=0.9806`、`mAP50-95=0.8782`。但本地 dataset summary 同时显示 test images 为 0，因此该指标不能作为正式主结果。正式投稿前必须重新核对 split 和评估脚本。

## 6.3 ASR Gate, Alignment, and Verification Outputs

ASR 质量在不同案例中差异明显。`front_1885_full/asr_quality_report.json` 记录 `segments_raw=12`、`segments_accepted=12`、`segments_rejected=0`。`front_26729_full` 中 accepted segments 为 22。另一方面，`run_full_001/full_integration_001/transcript.jsonl` 出现 `[ASR_EMPTY:empty_whisper_result]`。因此，论文应强调 ASR quality gate 和 visual fallback，而不是声称 ASR 稳定提升性能。

在验证输出方面，`front_45618_full` 包含 `align_multimodal.json` 和 `verified_events.jsonl`。报告中一个示例事件具有 `p_match=0.7132`、`reliability_score=0.6603`、`uncertainty=0.3397` 和 `label=match`。这说明系统已经能保留事件级证据链。`verifier/infer.py` 还支持通过 `--llm_student_model` 载入蒸馏后的 student judge，在运行时覆盖启发式或 MLP 分数，并把 `student_model_path`、`teacher_source` 和 `teacher_dataset` 写回事件证据。由于事件级人工金标仍少，正式统计指标应以补充标注后的结果为准。

## 6.4 Ablation and Robustness Evidence

后排增强实验可作为鲁棒性分析。`front_046_sr_ablation` 中，A0 full no SR 的 `tracked_students=20`、`actions_fusion_v2=332`、`timeline_student_rows=77`；A1 full_sliced 提升到 `tracked_students=33`、`actions_fusion_v2=713`、`timeline_student_rows=167`；A8 adaptive sliced + artifact deblur opencv 达到 `tracked_students=37`、`actions_fusion_v2=1609`、`timeline_student_rows=276`。这些是有价值的 proxy evidence，但表格也说明正式 AP、IDF1、HOTA 等指标需要 GT 标注。因此正文应写为“后排增强提高了可观测片段和时间线覆盖”，不要写成正式精度提升。

LLM 相关证据应拆成两层写。第一层是在线语义加权原型：`06f_llm_semantic_fusion.py` 当前仍包含 `simulate_llm_response()` 和未完成的真实 API 占位，因此它更适合支撑“语义加权思路与前端可解释展示”。第二层是蒸馏后的轻量 verifier：`output/llm_judge_pipeline/teacher_labels/silver_label_summary.json` 已明确这些标签是 silver labels，用于训练 student judge，而不是 gold labels。正文可以把这一分支写成主创新，但必须配套说明它降低的是在线大模型依赖，不是替代人工评测基准。

## 可引用证据

- 数据统计：`official_yolo_finetune_compare/reports/dataset_readiness.md`。
- 检测结果：`official_yolo_finetune_compare/reports/current_assessment.md`、`runs/detect/official_yolo11s_detect_e150_v1/results.csv`。
- 检测图：`codex_reports/smart_classroom_yolo_feasibility/paper_assets/run_full_e150_001/e150_results.png`、`e150_confusion_matrix_normalized.png`。
- ASR：`output/codex_reports/front_1885_full/asr_quality_report.json`、`output/codex_reports/front_26729_full/asr_quality_report.json`。
- 消融：`output/codex_reports/front_046_sr_ablation/sr_ablation_compare_table.md`。
- LLM/student：`scripts/pipeline/06f_llm_semantic_fusion.py`、`output/llm_judge_pipeline/teacher_labels/silver_label_summary.json`、`verifier/infer.py`。

## 图表占位

**[Table 3: Main Results and Ablation Summary，放在本节]**  
合并三类结果：检测验证结果、pipeline/evidence 输出、后排 proxy ablation。表中必须标注 `validation evidence`、`proxy metrics` 和 `needs GT`。

**[Fig. 5: Reliability or Rear-row Ablation，放在本节末]**  
优先插入 `output/codex_reports/front_45618_full/verifier_reliability_diagram.svg`。若论文更强调后排场景，则改用 `output/codex_reports/front_046_sr_ablation/sr_ablation_contact_sheet.jpg`。

---

# 7. Case Study and Discussion

## 写作目的

本节通过一个完整案例说明系统如何把视觉、文本、对齐、验证和界面串起来。讨论部分要自然承接实验局限，而不是重复方法。

## 7.1 Case Selection

主案例建议选择 `front_45618_full`。该案例包含 `actions.fusion_v2.jsonl`、`event_queries.fusion_v2.jsonl`、`align_multimodal.json`、`verified_events.jsonl`、`timeline_chart.png`、`timeline_students.csv`、`student_id_map.json` 和 contract reports。它适合展示从 query 到 candidate，再到 verified event 和 timeline overlay 的完整链路。

## 7.2 Evidence Chain

案例写作可以围绕一个具体事件展开。首先展示 ASR 或事件查询文本。然后展示系统如何根据 query time 和 UQ 计算候选窗口，并返回 Top-K 视觉候选。接着展示 selected candidate 的视觉行为标签、时间范围、overlap、action confidence 和 UQ。最后给出 `p_match`、`p_mismatch`、`reliability_score`、`uncertainty` 和三值标签。

这个案例应同时展示一个 uncertain 或 mismatch 事件。这样可以避免论文只展示成功结果，也能说明系统为什么需要三值验证而不是简单二分类。

## 7.3 Discussion

该案例说明，课堂行为分析中的关键问题不是检测框是否存在，而是事件是否可靠。双重验证把视觉行为片段和文本语义查询放到同一证据链中。UQ 对齐让系统在轨迹不稳定时扩大候选搜索范围，可靠性惩罚则避免高不确定性片段被过度确信。ASR gate 的作用也很清楚：当文本可靠时，它提供语义线索；当文本不可靠时，系统回退到视觉证据，避免错误文本污染判断。

可视分析界面的价值在于让这些中间证据可见。用户可以检查候选片段、分数和视频画面，而不是只接受最终标签。这一点是本文区别于普通课堂行为检测论文的重要部分。

## 可引用证据

- 主案例：`output/codex_reports/front_45618_full/`。
- 案例静态时间线：`output/codex_reports/front_45618_full/timeline_chart.png`。
- 验证事件：`output/codex_reports/front_45618_full/verified_events.jsonl`。
- 对齐候选：`output/codex_reports/front_45618_full/align_multimodal.json`。

## 图表占位

**[Fig. 4: Case Evidence Chain，放在本节]**  
建议合成一张多面板图：左侧 query text，中间 Top-K candidates，右侧 verified label 和 timeline overlay，下方放 video frame 或 evidence metrics。数据来自 `front_45618_full`。

---

# 8. Limitations and Conclusion

## 写作目的

本节要诚实收束。不要把局限写成附带说明，而要说明这些局限如何限定本文结论。

## 8.1 Limitations

第一，当前检测数据仍存在 split 风险。本地 processed dataset 显示 test split 为 0，因此检测结果应解释为验证集证据，不能写成独立测试集泛化结论。后续需要按视频级重新划分 train/val/test。

第二，事件级人工金标仍不足。`paper_experiments/gold/gold_validation_report.json` 中 peer-reviewed gold rows 规模较小，无法支撑强统计显著结论。当前 verifier 指标更适合作为系统可行性证据。

第三，ASR 质量受场景影响明显。部分案例有 accepted segments，部分案例为空或 placeholder。因此本文强调 ASR gate 和 visual fallback，而不是强调文本模态稳定提升。

第四，在线 LLM 直接语义打分仍以原型实现为主。当前真正进入运行主流程的是蒸馏后的轻量 verifier 分支，而不是完整的大模型在线推理。因此，正文可以强调“LLM-assisted semantic weighting + distilled student verifier”，但不能把在线 LLM 直接打分写成已充分验证的独立结论。

第五，LLM 银标不能替代人工 gold。`silver_label_summary.json` 已明确这些标签仅用于训练 student judge，不应用作最终评测真值。当前学生模型分支仍需要更大规模、人工复核后的 gold 数据来证明泛化能力。

第六，当前系统是离线或准离线 pipeline 加本地 FastAPI 原型。生产级实时部署、权限管理、跨教室泛化和隐私治理仍需进一步研究。

## 8.2 Conclusion

本文提出了一个面向智慧课堂视觉行为序列与文本语义流的可审计双重验证框架。该框架把 YOLOv11 系列行为检测、后排 ROI 分辨率增强、pose tracking、ASR 质量门控、语义桥接、不确定性感知对齐以及蒸馏轻量 verifier 组织成一条完整证据链，并通过学生级时间线和可视分析界面呈现事件判断。当前本地实验表明，后排增强分支能够提高困难场景中的可观测行为片段和时间线覆盖，学生模型分支能够进入主流程承担事件评分，而系统整体仍保持可追溯的 query、候选、分数和标签输出。未来工作将重点补充视频级独立测试集、更大规模人工事件标注、在线 LLM 语义打分的系统性消融，以及跨场景验证。

---

# 写作备忘

## 主文保留图表

| 编号 | 标题 | 插入位置 | 本地来源 |
| --- | --- | --- | --- |
| Fig. 1 | Overall Framework | Introduction / Method | 新绘制，依据 `scripts/main/09_run_pipeline.py` |
| Fig. 2 | UQ-guided Alignment and Dual Verification | Method | 新绘制，依据 `xx_align_multimodal.py`、`verifier/infer.py` |
| Fig. 3 | Visual Analytics Interface | System | 运行 `web_viz/templates/front_vsumvis.html` 对 `front_45618_full` 截图 |
| Fig. 4 | Case Evidence Chain | Case Study | `output/codex_reports/front_45618_full/align_multimodal.json`、`verified_events.jsonl`、`timeline_chart.png` |
| Fig. 5 | Reliability / Rear-row Ablation | Results | `front_45618_full/verifier_reliability_diagram.svg` 或 `front_046_sr_ablation/sr_ablation_contact_sheet.jpg` |
| Table 1 | Related Work Comparison | Related Work | 本地 references planning |
| Table 2 | Dataset and Behavior Taxonomy | Method / Experiments | `official_yolo_finetune_compare/reports/dataset_readiness.md` |
| Table 3 | Main Results and Ablation Summary | Results | `current_assessment.md`、`results.csv`、`sr_ablation_compare_table.md` |

## 不进入正文主贡献的内容

- OCR、敏感文本识别。
- MLLM 视觉推理。
- ST-GCN、IGFormer。
- ASPN、DySnakeConv、GLIDE Loss 的主线贡献。
- 在线 LLM semantic fusion 的大规模定量结论。
- `test mAP50=0.9806` 的正式主结果表述。

## 后续英文稿提醒

- 使用 IEEE VGTC conference template。
- 正文图表标题全部用英文。
- 公式统一编号。
- 引用统一 BibTeX。
- 正文贡献控制在 4 条左右。
- 每个实验 claim 都要标注 split、source 和是否为 proxy metric。
