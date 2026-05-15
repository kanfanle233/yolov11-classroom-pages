# 面向智慧课堂的视觉行为序列与文本语义流双重验证框架论文大纲

生成日期：2026-04-27  
建议文件位置：`YOLO论文/paper_outline_zh.md`  
写作定位：中文初稿大纲，后续可扩写为中文论文，再翻译为英文论文。  
阅读范围说明：本大纲依据 `YOLO论文/论文准备.md`、`YOLO论文/full_research_report.md`、`YOLO论文/project_audit_20260426/full_research_report.md`、`YOLO论文/paper_package_20260426/README.md`、`YOLO论文/paper_package_20260426/03_metrics_tables/metrics_summary.md`、`YOLO论文/paper_package_20260426/05_outline/paper_outline.md` 等文本材料整理。未读取体积较大的 PNG 图片和大型 CSV 原始内容，只根据资料包 README、指标摘要和审计报告中给出的文件名、统计值和用途进行引用规划。

---

## 0. 论文定位判断

### 0.1 是否可以开始写论文

判断：可以开始写初稿，但实验还需补强。

原因如下。

1. 项目已经具备清晰的论文主线。当前主线不是单纯改 YOLO 检测头，而是把 YOLO11 行为检测、pose tracking、ASR 质量门控、事件查询、UQ 自适应对齐、dual verification 和学生级 timeline 组织成一条可审计的视觉-语义双重验证流程。
2. 仓库中已经有可用的阶段性结果。资料显示 YOLO11s e150 行为检测已经达到 mAP50 0.933、mAP50-95 0.804、Precision 0.887、Recall 0.894。主线 pipeline 已经生成 11 个学生 ID、186 条融合动作、12 条事件查询、96 个对齐候选、12 条验证事件和 30 条学生级 timeline 动作片段。
3. 论文的创新点可以成立，但需要克制表达。语义桥接、UQ 自适应窗口、ASR 质量门控与视觉 fallback、学生级 timeline、pipeline contract 都有较明确的代码或报告证据。
4. 正式投稿还不能只依赖当前 demo。风险集中在独立 test split 缺失、ASR 样例质量不足、verifier 事件级 gold label 太少、长尾类别不足、跨模态指标尚未形成大规模统计。因此现在适合写中文初稿和方法章节，但正式投稿前必须补齐实验。

### 0.2 面向 ChinaVis 2026 的定位

如果面向 ChinaVis 2026，可将论文定位为智能教育场景下的可视分析与多模态感知系统。写作重点应包括：

- 视觉检测结果如何被组织为学生级行为序列。
- 文本语义事件如何与视觉动作候选进行验证。
- 不确定性、ASR 质量和 contract 如何减少空结果或低质结果造成的误导。
- timeline 和 case evidence 如何支撑教学行为解释和系统审计。

不要把 GitHub Pages、前端 demo 或普通可视化页面写成核心算法贡献。它们应放在系统实现、可视分析案例或附录中。

---

## 1. 题目候选

### 1.1 中文题目候选

1. 面向智慧课堂的视觉行为序列与文本语义流双重验证框架
2. 基于 YOLO11 的课堂学生行为序列与语义事件一致性验证方法
3. 面向噪声鲁棒课堂感知的视觉-语义双重验证与学生级时间线分析
4. 融合不确定性感知对齐与 ASR 质量门控的智慧课堂行为验证框架

推荐使用第 1 个。它最符合当前仓库实际完成情况，也避免把贡献写成单一检测器改进。

### 1.2 英文题目候选

1. A Dual-Verification Framework for Aligning Visual Classroom Behavior Sequences with Real-Time Textual Semantic Streams
2. Uncertainty-Aware Visual-Semantic Verification for Robust Classroom Behavior Analysis
3. A YOLO11-based Visual-Semantic Verification Pipeline for Student Behavior Timeline Analysis
4. Robust Classroom Behavior Understanding via Visual Action Sequences, Textual Event Streams, and Dual Verification

推荐使用第 1 或第 2 个。第 1 个更完整，第 2 个更适合英文论文的精炼标题风格。

---

# Title

## 本段写作目标

给出论文中文题目和英文题目，突出课堂场景、视觉行为序列、文本语义流和 dual verification 四个关键词。

## 应展开的核心内容

题目不应强调“改进 YOLO11 结构”，因为主线中自定义 YOLO 结构还没有完整接入训练和消融。题目应强调系统级方法，即视觉识别结果是否可靠，以及文本事件如何验证视觉判断。

## 未来正文可以扩写的段落要点

- 中文题目使用“视觉行为序列”和“文本语义流”，体现视频时间线和 ASR 事件流。
- 英文题目保留 dual-verification、aligning、visual classroom behavior sequences、textual semantic streams。
- 标题中避免出现“大模型验证”“扩散模型”“自定义 YOLO 模块”等当前证据不足的内容。

## 需要引用的项目证据或实验结果

- `scripts/09_run_pipeline.py` 被审计报告标记为全链路编排脚本，连接 pose、tracking、ASR、行为检测、fusion、align、verifier 和 timeline。
- `scripts/07_dual_verification.py` 已输出 `p_match/p_mismatch/reliability/uncertainty/label`。
- `scripts/xx_align_multimodal.py` 已实现 UQ 驱动窗口和候选对齐。

## 需要插入的图、表或公式标注

本节通常不插入图表。题目页可以在后续模板中只保留中英文标题和作者信息。

---

# Abstract

## 本段写作目标

用 150 到 250 字概括问题、方法、结果和意义。摘要要突出“可靠性判断”和“跨模态验证”，不能写成普通目标检测摘要。

## 应展开的核心内容

1. 复杂课堂中，单一视觉检测会受遮挡、多人重叠、动作相似和低光影响。
2. ASR 文本流也会因课堂噪声、远场语音、回声和漏识别而不可靠。
3. 本文提出视觉行为序列与文本语义流的 dual verification 框架。
4. 方法包括 YOLO11 行为检测、pose tracking、UQ 估计、ASR 质量门控、视觉事件 fallback、UQ 自适应时间窗、跨模态一致性打分和学生级 timeline。
5. 当前可以报告阶段性结果，但必须说明是阶段性实验。不能声称已经完成大规模统计验证。

## 未来正文可以扩写的段落要点

摘要建议结构如下。

第一句写背景和挑战。  
第二句写核心问题。  
第三句写方法总览。  
第四句写主要输出和阶段性结果。  
第五句写意义和局限，说明后续会通过独立 test split 和人工 gold label 进一步验证。

## 需要引用的项目证据或实验结果

- YOLO11s e150 行为检测 mAP50 为 0.933，mAP50-95 为 0.804。
- 主线 pipeline 生成 11 个学生 ID、186 条 fusion 动作、12 条 event queries、96 个 align candidates、12 条 verified events。
- ASR 当前样例中 raw segments 为 3，accepted 为 0，rejected 为 3，状态为 placeholder，应写成“触发质量门控和视觉 fallback 的案例”，不能写成文本模态提升。

## 需要插入的图、表或公式标注

摘要不插图。可在摘要末尾避免出现过多数字，详细数值放到 Experiments 和 Results。

### 摘要草稿

智慧课堂行为分析需要在复杂场景中判断学生动作、教师互动和课堂事件是否可靠。然而，单一视觉检测容易受到遮挡、多人重叠、动作相似和关键点缺失影响，课堂 ASR 文本流也可能因远场语音和环境噪声产生空文本或低可信片段。为此，本文提出一种面向智慧课堂的视觉行为序列与文本语义流双重验证框架。该框架首先使用 YOLO11 构建 8 类课堂行为检测结果，并结合 pose tracking 形成学生级行为片段；随后通过 fusion contract v2 将短码动作转化为可验证的中英双语语义标签；在文本侧，系统对 ASR 输出进行质量门控，并在文本不可用时生成视觉事件兜底查询；最后通过 UQ 驱动的自适应时间窗口完成事件对齐，并输出 p_match、p_mismatch、reliability、uncertainty 和 verified label。阶段性实验显示，当前 YOLO11s 行为检测模型在验证集上达到 mAP50 0.933 和 mAP50-95 0.804，主线 pipeline 能生成学生级 timeline 和可审计验证事件。本文为噪声课堂环境下的多模态行为验证和可视化教学分析提供了一个可扩展基础，但正式投稿仍需补充独立测试集、人工事件级标注和更充分的 ASR 消融实验。

---

# Keywords

## 本段写作目标

列出中文关键词和英文关键词，为中文初稿和英文稿翻译做准备。

## 应展开的核心内容

关键词应覆盖任务、模型、方法和系统输出。

## 未来正文可以扩写的段落要点

中文关键词建议 6 到 8 个，英文关键词与之对应。

## 需要引用的项目证据或实验结果

不需要单独引用。关键词来自论文主线。

## 需要插入的图、表或公式标注

不需要图表。

### 关键词建议

智慧课堂；课堂行为识别；YOLO11；多模态感知；视觉-语义验证；不确定性估计；ASR 质量门控；学生级时间线

### English Keywords

Smart classroom; classroom behavior recognition; YOLO11; multimodal perception; visual-semantic verification; uncertainty estimation; ASR quality control; student-level timeline

---

# 1. Introduction

## 本段写作目标

说明为什么课堂行为识别不能只做视觉检测，为什么需要视觉行为序列与文本语义流的双重验证。引言要收敛到一个主问题：系统如何判断视觉行为识别结果是否可靠，并利用文本语义事件对视觉判断进行验证。

## 应展开的核心内容

### 1.1 背景

智慧课堂分析希望从视频中理解学生参与度、课堂互动和教学节奏。传统方法通常将课堂行为识别建模为图像检测、姿态动作判断或视频动作分类。但真实课堂中存在多个学生、桌椅遮挡、视角受限、动作幅度小、动作类别相似等问题。例如听课、看书、写字、转头等动作在局部视觉上可能非常接近。

### 1.2 问题

单一视觉模型输出的是候选框、类别和置信度，但这个置信度并不等同于课堂语义上的可靠性。文本或语音信息可以提供事件语义，但 ASR 在课堂环境中也可能因为远场声音、噪声和多人同时说话而失败。因此，关键不在于简单融合两个模态，而在于判断各模态是否可信，并让两个模态在时间和语义层面互相验证。

### 1.3 本文思路

本文以 YOLO11 行为检测和 pose tracking 构建视觉行为序列，以 Whisper ASR 和事件查询构建文本语义流。然后通过 UQ 驱动的自适应时间窗寻找视觉动作候选，并用 dual verification 输出 match、mismatch 或 uncertain。系统最终生成 verified events 和学生级 timeline。

### 1.4 创新点组织

可写 4 个贡献，建议如下。

1. 提出面向智慧课堂的视觉行为序列与文本语义流双重验证框架。  
   已有证据较强。它体现了全链路 pipeline，不是单个检测模型。
2. 设计 fusion contract v2 语义桥接协议。  
   已有证据较强。该协议将 `tt/dx/dk/zt/xt/js/zl/jz` 转为中英双语 semantic id，便于文本事件验证和 timeline 解释。
3. 引入 UQ 驱动的自适应跨模态时间窗口。  
   已有代码证据。可作为方法创新，但实验上仍需固定窗口对比和参数消融。
4. 设计 ASR 质量门控与视觉事件 fallback。  
   已有工程证据。当前样例中 ASR 被拒绝，适合作为鲁棒性机制说明，但不能写成文本模态带来显著提升。
5. 构建学生级可解释 timeline 与可审计 pipeline contract。  
   已有输出证据。应作为系统解释性和可视分析贡献，不要写成核心检测算法贡献。

## 未来正文可以扩写的段落要点

第一段写智慧课堂和课堂行为理解需求。  
第二段写视觉-only 的局限。  
第三段写 ASR 文本流的潜力和噪声问题。  
第四段写本文的 dual verification 思路。  
第五段列贡献。贡献要用 “we propose / we design / we implement / we evaluate” 这类 CVPR 风格表达。

## 需要引用的项目证据或实验结果

- 数据集 8 类行为包括 `tt/dx/dk/zt/xt/js/zl/jz`。
- 主线模型结果 mAP50 0.933、mAP50-95 0.804。
- pipeline contract 结果显示 186 条 fusion 动作和 12 条 verified events。
- ASR 质量门控结果显示当前 demo 中 accepted 为 0，说明系统必须处理低可信 ASR。

## 需要插入的图、表或公式标注

【建议插入图 1：整体方法框架图】  
放置章节：Introduction 末尾或 Method 3.1。  
主要内容：视频输入、YOLO11 行为检测、YOLO pose、tracking、UQ、ASR、ASR quality gate、visual fallback、event query、UQ adaptive alignment、dual verifier、verified events、student timeline。  
数据来源：`paper_package_20260426/README.md` 中的目录说明、`full_research_report.md` 中的主线脚本索引。  
服务论点：本文不是视觉-only 检测，而是可审计的视觉-语义双重验证框架。

---

# 2. Related Work

## 本段写作目标

把相关工作分成几条线，并明确本文与现有工作的区别。不要写成文献堆砌。每个小节最后都要指出 gap。

## 应展开的核心内容

### 2.1 Classroom Behavior Detection

重点讨论课堂行为检测和学生行为识别。可覆盖改进 YOLOv5、YOLOv8、SlowFast、时空注意力、轻量检测网络等工作。已有课堂行为检测工作多关注视觉检测精度、实时性和遮挡处理，但通常缺少文本事件验证、ASR 质量门控和学生级可解释 timeline。

### 2.2 YOLO 系列和实时目标检测

说明使用 YOLO11 的原因是实时性、工程可用性和与检测数据集格式兼容。可以引用 YOLOv7、YOLOv10、YOLO 系列综述等。本文不声称提出新的 YOLO 结构。当前主线使用的是 YOLO11s 官方微调权重，而不是自定义 ASPN、DySnake、GLIDE 等结构。

### 2.3 Pose Estimation and Uncertainty Modeling

介绍 pose estimation 和 tracking 在行为理解中的作用。课堂中人的关键点可能因遮挡、低光、坐姿和多人重叠缺失，因此 pose UQ 可以作为后续对齐和验证的可靠性信号。本文将 UQ 用于自适应时间窗，而不是只作为可视化指标。

### 2.4 Multimodal Video-Language Understanding

讨论 CLIP、UniVL、Video-LLaMA、InternVideo2、MAViL、Meerkat、RCAD 等视频-语言或音视频理解工作。强调它们证明跨模态语义对齐有价值，但多数依赖大规模预训练或通用数据集，不直接解决课堂多学生 timeline 和 ASR 低质量下的可靠性验证。

### 2.5 ASR and Speech-based Classroom Analysis

讨论 Whisper 和课堂语音分析。Whisper 是鲁棒 ASR 基座，但课堂环境常见远场、噪声和多人说话，直接将 ASR 文本作为事实会产生风险。本文采用质量门控和 fallback，避免低质文本污染 verifier。

### 2.6 Temporal Grounding and Verification

介绍 temporal grounding、event localization 和 cross-modal retrieval。本文的区别是用轻量、可审计的 event query 和 action candidate 对齐，不依赖端到端大模型进行黑箱判断。

### 2.7 Reliability, Calibration and Uncertainty Evaluation

讨论 ECE、Brier score、reliability diagram 等。本文中 verifier 输出 reliability 和 uncertainty，应通过校准指标验证其可信度。当前已有 calibration 相关代码和可靠性图资产，但需要更多人工标注样本。

### 2.8 Visualization and Smart Classroom Analytics

讨论可视分析如何帮助教师理解课堂行为。本文的 timeline 和 case evidence 用于解释“哪个学生在什么时候做了什么，以及该判断是否可靠”。

## 未来正文可以扩写的段落要点

每个小节可用 1 到 2 段。相关工作表应放在 2.8 后，用三线表对比本文和已有方法。重点列出是否使用视觉、文本、跨模态验证、UQ、课堂场景、timeline。

## 需要引用的项目证据或实验结果

- `论文准备.md` 已整理多篇 2022 到 2026 年文献，包括 MAViL、CAT、Meerkat、MUSIC-AVQA、AVMIT、MM-TBA、RCAD 等。
- `full_research_report.md` 中已有课堂行为检测和多模态理解论文矩阵。
- 审计报告明确提醒不要把 Video-LLaMA、InternVideo2 这类 foundation model 写成本项目已实现。

## 需要插入的图、表或公式标注

【建议插入三线表 1：相关工作对比表】  
放置章节：Related Work 末尾。  
主要内容：年份、方法、任务、是否使用视觉、是否使用文本、是否进行跨模态验证、是否建模不确定性、是否面向课堂场景、是否输出学生级 timeline、与本文差异。  
数据来源：`论文准备.md` 文献表、`full_research_report.md` 论文矩阵。  
服务论点：本文区别于视觉-only 课堂检测，也区别于大模型式视频语言理解，核心是课堂场景下的可审计双重验证。

---

# 3. Problem Definition

## 本段写作目标

用形式化方式定义输入、输出和核心任务。让后续方法中的视觉序列、文本事件、对齐窗口、验证标签都有清晰符号。

## 应展开的核心内容

给定一段课堂视频 \(V\) 和可选音频轨道 \(A\)，系统需要输出学生级行为序列和验证事件。视觉侧生成一组动作片段：

\[
\mathcal{X}=\{x_i=(s_i,a_i,t_i^s,t_i^e,c_i,u_i)\}
\]

其中 \(s_i\) 是学生或 track，\(a_i\) 是行为语义标签，\(t_i^s,t_i^e\) 是时间范围，\(c_i\) 是视觉置信度，\(u_i\) 是不确定性。

文本侧生成一组事件查询：

\[
\mathcal{Q}=\{q_j=(e_j,\tau_j,r_j)\}
\]

其中 \(e_j\) 是事件语义，\(\tau_j\) 是时间，\(r_j\) 是 ASR 质量或事件置信度。

目标是为每个文本事件或视觉 fallback 事件找到候选视觉动作，并输出：

\[
y_j \in \{\text{match},\text{mismatch},\text{uncertain}\}
\]

同时输出 `p_match`、`p_mismatch`、`reliability`、`uncertainty` 和 `verified_label`。

## 未来正文可以扩写的段落要点

- 说明本文不是单纯分类任务，也不是纯目标检测任务。
- 核心任务是可靠性验证和时间语义对齐。
- 对低质量 ASR，文本事件可被视觉 fallback 替代，此时系统验证的是“视觉高置信事件是否能构成可审计事件流”。

## 需要引用的项目证据或实验结果

- `event_queries.fusion_v2`、`actions.fusion_v2`、`aligned`、`verified_events` 是当前主线文件。
- `07_dual_verification.py` 输出字段包括 `p_match/p_mismatch/reliability/uncertainty/label`。
- `10_visualize_timeline.py` 输出 `timeline_chart.png/json`、`timeline_students.csv`、`student_id_map.json`。

## 需要插入的图、表或公式标注

【建议插入公式 1：任务输入输出定义】  
放置章节：Problem Definition。  
主要内容：定义视觉动作集合、文本事件集合和验证标签。  
数据来源：主线 pipeline 产物字段。  
服务论点：本文任务不是普通检测，而是多模态事件验证。

---

# 4. Method

## 本段写作目标

细化方法模块。每个模块都要说明输入、处理、输出、实现状态和论文表达边界。

---

## 4.1 Overall Framework

### 本段写作目标

说明整体输入输出和流程，给读者建立端到端系统概念。

### 应展开的核心内容

输入为课堂视频。视觉侧包括 YOLO11 pose、tracking、smooth、pose UQ、YOLO11s 8 类行为检测、行为框动作片段生成、物品证据融合。文本侧包括 Whisper ASR、质量门控、文本事件抽取和视觉 fallback。融合侧包括 fusion contract v2、UQ adaptive alignment、dual verification 和 timeline visualization。

### 未来正文可以扩写的段落要点

- 首先介绍为什么要分成视觉序列构建、文本事件构建和跨模态验证三层。
- 再说明每一层的输入输出文件。
- 最后强调 strict contract 防止空文件、缺字段和无候选情况下静默成功。

### 需要引用的项目证据或实验结果

- `scripts/09_run_pipeline.py` 被报告描述为主线全链路编排脚本。
- 主线结果包括 186 条 fusion 动作、96 个 align candidates、12 条 verified events。
- `check_pipeline_contract_v2.py` 检查文件存在、非空、字段完整和候选非空。

### 需要插入的图、表或公式标注

【建议插入图 2：系统总流程图】  
放置章节：Method 4.1。  
主要内容：视频输入到 verified events 和 timeline 的完整流程。  
数据来源：`full_research_report.md` 中 E 节 Mermaid 总流程图、`paper_package_20260426/README.md` 目录结构。  
服务论点：系统为多阶段可审计 pipeline。

【建议插入算法 1：Dual-Verification Pipeline】  
放置章节：Method 4.1 或 4.7。  
主要内容：pose 导出、tracking、UQ、行为检测、semanticize、fusion、ASR gate、fallback、alignment、verification、timeline。  
数据来源：`full_research_report.md` 中伪代码。  
服务论点：展示方法可复现流程。

---

## 4.2 Visual Behavior Sequence Construction

### 本段写作目标

说明如何从视频构建视觉行为序列，包括 YOLO11 行为检测、pose keypoints、tracking、smooth、规则动作和行为片段。

### 应展开的核心内容

视觉序列由两类证据组成。

第一类是 YOLO11s 行为检测。模型识别 8 类课堂行为：听课、写字、看书、转头、小组讨论、举手、站立、教师互动，对应短码 `tt/dx/dk/zt/xt/js/zl/jz`。检测输出包括 bbox、类别、置信度和时间帧。

第二类是 pose tracking。系统从 YOLO pose 提取关键点，经过 track and smooth 形成稳定 `track_id`，然后映射为 `S01/S02/...` 学生编号。需要明确学生 ID 不进入 YOLO 检测头，而是后处理生成。

视觉动作片段可以由行为框聚合而成，也可以由 pose rule、object evidence 和 behavior detection 共同构成 `actions.fusion_v2.jsonl`。

### 未来正文可以扩写的段落要点

- 说明 8 类行为的课堂意义和类别短码。
- 说明行为框到动作片段的聚合逻辑，包括时间连续性和置信度聚合。
- 说明 tracking 的目的不是身份识别，而是同一视频内的临时学生编号。
- 说明物品证据是辅助，不应夸大为独立多模态贡献。

### 需要引用的项目证据或实验结果

- 数据集 train 7416 张、val 1467 张、test 0 张。
- 行为类别为 `tt/dx/dk/zt/xt/js/zl/jz`。
- YOLO11s e150 指标为 mAP50 0.933、mAP50-95 0.804。
- 当前主线学生 ID 数为 11。

### 需要插入的图、表或公式标注

【建议插入三线表 2：8 类课堂行为 taxonomy】  
放置章节：Method 4.2 或 4.3。  
主要内容：短码、中文标签、英文标签、semantic id、典型视觉表现、可能混淆类别。  
数据来源：`action_semantics_8class.yaml`、`dataset.yaml`、`metrics_summary.md`。  
服务论点：语义桥接的基础。

【建议插入图 3：视觉行为序列构建示意图】  
放置章节：Method 4.2。  
主要内容：帧级检测到 track 级动作片段。  
数据来源：检测样例图、timeline 输出和行为动作片段文件。  
服务论点：说明视觉检测如何转为学生级时间序列。

---

## 4.3 Semantic Bridging and Fusion Contract

### 本段写作目标

说明如何把 `tt/dx/dk/zt/xt/js/zl/jz` 短码转为中英双语语义标签，以及为什么这个协议对后续验证重要。

### 应展开的核心内容

视觉检测模型通常输出数据集内类别短码。短码对检测训练足够，但对 ASR 事件、文本语义匹配、timeline 展示和论文可解释性不够。因此系统设计 fusion contract v2，将每条动作标准化为带有 `behavior_code`、`semantic_id`、`semantic_label_zh`、`semantic_label_en`、`taxonomy_version` 等字段的语义动作。

该模块的价值有三点。

1. 保证下游 align 和 verifier 使用稳定语义字段。
2. 保证 timeline 可读，避免图表中只出现 `tt`、`dx` 等短码。
3. 保证 contract 可以检查缺字段，并在语义缺失时失败，避免伪结果。

### 未来正文可以扩写的段落要点

- 给出字段定义表。
- 说明 semantic id 是跨模态对齐的最小语义单位。
- 说明中英双语标签服务中文论文和英文稿。
- 说明该协议是工程和方法之间的桥，不是简单重命名。

### 需要引用的项目证据或实验结果

- `semanticize_behavior_det.py` 将行为框从短码转为标准语义字段。
- `action_semantics_8class.yaml` 中固定 8 类 taxonomy。
- `test_fusion_contract_v2.py` 已覆盖缺少 `semantic_id` 时 contract 失败。
- 当前 `actions_fusion_v2_semantic_valid` 为 186，等于 fusion 动作总数 186。

### 需要插入的图、表或公式标注

【建议插入三线表 3：fusion contract v2 字段规范】  
放置章节：Method 4.3。  
主要内容：字段名、类型、来源、是否必需、下游用途。  
数据来源：fusion contract 脚本和 contract report。  
服务论点：语义桥接不是后处理小修补，而是保障下游验证的协议层。

---

## 4.4 Pose Uncertainty Estimation

### 本段写作目标

说明 pose tracking 不确定性如何产生，以及它如何影响后续对齐和验证。

### 应展开的核心内容

pose UQ 可综合关键点置信度、缺失比例、bbox 抖动、轨迹运动变化和时序稳定性。课堂中遮挡、多学生重叠和坐姿会降低关键点可靠性，因此 UQ 应作为对齐窗口和 verifier 的输入，而不是被忽略。

### 未来正文可以扩写的段落要点

- 定义 `u_pose` 或 `U_i`。
- 说明高 UQ 的片段不一定丢弃，而是扩大时间窗口或降低 reliability。
- 说明 UQ 是轻量估计，不是完整 Bayesian deep learning。
- 若 variance-head 尚未训练，不要写成已实现模型，只写为未来补强或扩展。

### 需要引用的项目证据或实验结果

- `03c_estimate_track_uncertainty.py` 已估计 pose tracking UQ。
- `xx_align_multimodal.py` 使用 UQ 参与自适应窗口。
- `07_dual_verification.py` 输出 uncertainty 字段。

### 需要插入的图、表或公式标注

【建议插入公式 2：pose tracking 不确定性定义】  
放置章节：Method 4.4。  
主要内容：可写为关键点缺失、关键点低置信和轨迹抖动的加权和。  
数据来源：`03c_estimate_track_uncertainty.py` 逻辑。  
服务论点：解释 UQ 如何进入后续对齐。

示例公式：

\[
U_i=\omega_1(1-\bar{p}_{kp})+\omega_2 r_{miss}+\omega_3 J_{track}
\]

其中 \(\bar{p}_{kp}\) 表示平均关键点置信度，\(r_{miss}\) 表示关键点缺失比例，\(J_{track}\) 表示轨迹抖动。

---

## 4.5 ASR Quality Control and Event Query Construction

### 本段写作目标

说明 Whisper 或其他 ASR 输出如何被质量门控，如何从 transcript 构建 event query。如果 ASR 为空或低质量，说明视觉事件 fallback 的作用。

### 应展开的核心内容

ASR 模块输出文本片段和质量指标。质量指标可包括 `avg_logprob`、`no_speech_prob`、`compression_ratio` 等。系统根据阈值决定片段是否 accepted。通过门控的文本可以抽取事件查询，例如“学生举手”“学生讨论”“教师互动”等。若 ASR 为空或低质量，系统不强行使用文本，而是从高置信视觉动作生成 fallback event query。

### 未来正文可以扩写的段落要点

- 解释 ASR 质量门控的必要性。
- 给出 accepted、rejected、fallback 的状态定义。
- 说明 fallback 不是证明文本模态有效，而是保证 pipeline 在文本缺失时仍能审计视觉事件。
- 当前样例 ASR accepted 为 0，应作为风险和鲁棒机制，而不是当作文本贡献。

### 需要引用的项目证据或实验结果

- `06_asr_whisper_to_jsonl.py` 输出 ASR 质量报告。
- 当前 ASR report 中 model 为 medium，device 为 cuda，compute_type 为 float16，segments_raw 为 3，accepted 为 0，rejected 为 3，status 为 placeholder。
- `build_event_queries_fusion_v2.py` 在 ASR 为空或低质时生成 visual fallback。

### 需要插入的图、表或公式标注

【建议插入公式 3：文本置信度定义】  
放置章节：Method 4.5。  
主要内容：根据 ASR 质量指标定义文本置信度。  
数据来源：`asr_quality_report.json` 字段。  
服务论点：低质 ASR 不应直接进入 verifier。

示例公式：

\[
C_t(q)=\sigma(w_1\cdot avg\_logprob-w_2\cdot no\_speech\_prob-w_3\cdot compression\_ratio)\cdot C_{keyword}
\]

【建议插入三线表 4：ASR 质量门控状态表】  
放置章节：Method 4.5 或 Results。  
主要内容：raw segments、accepted、rejected、fallback、status。  
数据来源：`asr_quality_report.json`。  
服务论点：系统能识别低可信文本并避免误导下游验证。

---

## 4.6 UQ-guided Adaptive Multimodal Alignment

### 本段写作目标

说明自适应窗口公式，候选视觉动作如何与文本事件对齐。

### 应展开的核心内容

固定时间窗口在课堂场景中不稳定。动作发生时间、ASR 时间戳和视觉片段时间可能存在偏移。若 pose UQ 高或轨迹运动不稳定，系统应适当扩大窗口，以提高候选召回；但窗口不能无限扩大，否则会引入无关候选。

### 未来正文可以扩写的段落要点

- 定义 query time \(t_q\)。
- 定义窗口半径 \(\Delta_q\)。
- 说明 `base_window`、`alpha_motion`、`beta_uq`、`min_window`、`max_window`。
- 说明候选排序依据可以包括时间重叠、动作置信度、语义一致性和 UQ。
- 说明该模块需要固定窗口对比和参数消融支撑。

### 需要引用的项目证据或实验结果

- `xx_align_multimodal.py` 定义窗口参数并计算 UQ 驱动窗口。
- 当前主线 `align_total_candidates` 为 96。
- `event_queries_fusion_v2` 为 12。

### 需要插入的图、表或公式标注

【建议插入公式 4：UQ 自适应时间窗口】  
放置章节：Method 4.6。  
主要内容：窗口半径由 base、motion 和 UQ 决定。  
数据来源：`xx_align_multimodal.py`。  
服务论点：本文不是固定窗口后融合，而是可靠性感知对齐。

公式：

\[
\Delta_q=clip(\Delta_0+\alpha M_q+\beta U_q,\Delta_{min},\Delta_{max})
\]

\[
W_q=[t_q-\Delta_q,t_q+\Delta_q]
\]

【建议插入算法 2：UQ-guided Event-Action Alignment】  
放置章节：Method 4.6。  
主要内容：输入 event query、action segments、pose UQ，输出 top-k candidates。  
数据来源：`xx_align_multimodal.py`。  
服务论点：说明候选生成过程可复现。

---

## 4.7 Dual Verification and Reliability Scoring

### 本段写作目标

说明如何计算 `p_match`、`p_mismatch`、`reliability`、`uncertainty` 和最终 label。

### 应展开的核心内容

Dual verification 不是简单的类别投票。它将视觉动作候选、文本事件、时间重叠、语义一致性和 UQ 共同用于判断。输出不只有 match 或 mismatch，还应包括 uncertain，用于表达低置信或冲突情况。

### 未来正文可以扩写的段落要点

- 定义视觉置信度 \(C_v\)。
- 定义文本置信度 \(C_t\)。
- 定义跨模态一致性分数 \(S\)。
- 定义最终决策函数。
- 说明 verifier 当前包括规则/MLP 部分。MLP verifier 属于部分实现，正式论文需要人工 gold label 训练和验证，不能夸大。

### 需要引用的项目证据或实验结果

- `07_dual_verification.py` 输出 `p_match/p_mismatch/reliability/uncertainty/label`。
- `verifier/model.py` 中存在 MLP verifier 和校准相关指标函数，但报告标注为部分实现。
- 当前 `verified_events` 为 12，样本过少，不能做显著性结论。

### 需要插入的图、表或公式标注

【建议插入公式 5：视觉置信度定义】  
放置章节：Method 4.7。  
主要内容：检测置信度、pose 不确定性、物品证据支持度融合。  
数据来源：行为检测输出、pose UQ、object evidence。  
服务论点：视觉置信度不是单一 bbox score。

\[
C_v(i,a,t_s,t_e)=\lambda_d \bar{p}_{det}+\lambda_p(1-\bar{u}_{pose})+\lambda_o p_{obj}
\]

【建议插入公式 6：跨模态一致性分数】  
放置章节：Method 4.7。  
主要内容：视觉置信度、文本置信度、语义相似度、时间重叠和 UQ。  
数据来源：align candidates、event queries、verified events。  
服务论点：说明 dual verification 的判定依据。

\[
S(i,a,q)=\alpha C_v(i,a)+\beta C_t(q)+\gamma Sim(\phi(a),\psi(q))+\delta O([t_s,t_e],W_q)-\eta UQ_i
\]

【建议插入公式 7：最终验证标签决策】  
放置章节：Method 4.7。  
主要内容：match、mismatch、uncertain 三状态。  
数据来源：`07_dual_verification.py` 输出字段。  
服务论点：支持可靠性判断，而不是强制二分类。

\[
D(S)=
\begin{cases}
\text{match}, & S \ge \tau_m\\
\text{mismatch}, & S < \tau_u\\
\text{uncertain}, & \tau_u \le S < \tau_m
\end{cases}
\]

---

## 4.8 Output Representation and Timeline Visualization

### 本段写作目标

说明 `verified_events`、`timeline_chart`、`timeline_students`、`student_id_map` 如何支撑论文图表和教学解释。

### 应展开的核心内容

系统最终输出两类结果。

第一类是验证结果，包括事件查询、候选动作、验证标签、概率、可靠性和不确定性。  
第二类是可视化结果，包括学生 ID 映射、学生级 timeline、timeline chart 和 case evidence。

### 未来正文可以扩写的段落要点

- 说明 `S01/S02/...` 是视频内匿名学生编号，不涉及真实身份识别。
- 说明 timeline 支持教师查看课堂行为分布和研究者审计模型输出。
- 说明 GitHub Pages 或前端界面是展示载体，不是核心方法。

### 需要引用的项目证据或实验结果

- `10_visualize_timeline.py` 输出 `timeline_chart.png/json`、`timeline_students.csv`、`student_id_map.json`。
- 当前主线有 11 个学生 ID 和 30 条 timeline 学生动作片段。
- `paper_package_20260426/README.md` 建议将 `mainline_timeline_chart.png` 作为系统效果图。

### 需要插入的图、表或公式标注

【建议插入图 4：学生级 timeline 可视化图】  
放置章节：Results and Analysis 或 Method 4.8。  
主要内容：每个学生一行，显示行为片段和时间范围。  
数据来源：`02_figures_pipeline/mainline_timeline_chart.png`、`timeline_students.csv`。  
服务论点：本文输出可解释课堂行为序列。

【建议插入案例图 1：verified event 证据链案例】  
放置章节：Results and Analysis。  
主要内容：事件 query、候选视觉动作、p_match、reliability、uncertainty、最终 label、对应视频帧。  
数据来源：`verified_events.jsonl`、`aligned`、timeline JSON、检测样例。  
服务论点：展示系统如何判断一个事件是否可靠。

---

# 5. Experiments

## 本段写作目标

设计正式论文实验。需要把当前已有结果和待补强实验分开写。

---

## 5.1 Dataset and Preprocessing

### 本段写作目标

说明当前数据集、8 类课堂行为、train/val/test 情况。如果 test split 缺失，明确写成论文风险和后续补强点。

### 应展开的核心内容

当前数据集已处理为 YOLO detect 格式，类别为 8 类课堂行为。已有 train 7416 张、val 1467 张、test 0 张。正式论文必须按视频或 case 重切 train/val/test，建议 70/15/15 或按班级/视频切分，避免相邻帧泄漏。

### 未来正文可以扩写的段落要点

- 描述视频来源、帧抽样、标注格式、8 类行为。
- 报告每类样本数量和长尾类别。
- 明确 `jz` 教师互动类别偏少，可能影响 recall。
- 说明数据隐私和匿名化，如果涉及真实课堂视频，要写处理方式。

### 需要引用的项目证据或实验结果

- train 7416 张，val 1467 张，test 0 张。
- `jz` 类 train 563，val 117，属于较少类别。
- `dataset.yaml` 中类别定义为 8 类。

### 需要插入的图、表或公式标注

【建议插入三线表 5：数据集划分与类别统计】  
放置章节：Experiments 5.1。  
主要内容：split、图片数、标注框数、8 类类别数量。  
数据来源：`dataset_classroom_yolo_summary.json`、`dataset.yaml`。  
服务论点：说明当前数据基础和长尾风险。

---

## 5.2 Implementation Details

### 本段写作目标

说明 YOLO11s 行为检测、pose 模型、Whisper ASR、pipeline 运行方式和输出文件。

### 应展开的核心内容

- 行为检测模型：YOLO11s，150 epoch，当前主线权重 `official_yolo11s_detect_e150_v1/weights/best.pt`。
- 输入尺寸：资料中已有 832 的写法，可在代码和训练日志中再次核对后写入正式稿。
- Pose 模型：YOLO pose，用于关键点导出和 tracking。
- ASR：Whisper medium，cuda，float16。当前样例低可信。
- Pipeline：`09_run_pipeline.py` 统一编排，输出 actions、event queries、align candidates、verified events 和 timeline。
- Contract：检查关键产物存在、非空、字段完整、候选非空。

### 未来正文可以扩写的段落要点

- 说明硬件环境，本地 RTX 4060 8GB 和内存条件可放在实现细节中，但不要让论文显得像课程项目。
- 说明参数：base window、alpha_motion、beta_uq、阈值等。
- 说明每个模块的依赖版本，可放附录。

### 需要引用的项目证据或实验结果

- `metrics_summary.md` 说明 e150 检测指标和主线模型。
- `asr_quality_report.json` 说明 ASR 配置。
- `pipeline_contract_v2_report.json` 说明主线结果。

### 需要插入的图、表或公式标注

【建议插入三线表 6：实现配置表】  
放置章节：Experiments 5.2。  
主要内容：模块、模型、关键参数、输出文件、实现状态。  
数据来源：pipeline 脚本、metrics summary、README。  
服务论点：增强复现性。

---

## 5.3 Evaluation Metrics

### 本段写作目标

列出视觉检测、跨模态验证、校准、ASR 和 timeline 的指标。

### 应展开的核心内容

视觉检测指标：

- mAP50
- mAP50-95
- Precision
- Recall
- F1
- class-wise AP
- confusion matrix
- FPS 或 latency

事件级验证指标：

- event-level Accuracy
- event-level Precision
- event-level Recall
- event-level F1
- candidate recall
- Top-k candidate recall
- mismatch false positive rate
- uncertain ratio

校准指标：

- ECE
- Brier score
- reliability diagram

ASR 质量指标：

- accepted rate
- rejected rate
- fallback rate
- raw segments
- accepted segments
- rejected segments
- 可选 WER 或人工转写相似度

Timeline 和 tracking 指标：

- timeline coverage
- student count
- action segment count
- ID switch
- 人工抽样 track 准确率

### 未来正文可以扩写的段落要点

正式论文中至少要报告视觉检测指标、事件级 F1、candidate recall、ECE、Brier、ASR accepted/rejected/fallback rate、timeline coverage。若某些指标暂时没有数据，应在大纲和实验计划中列为待补强。

### 需要引用的项目证据或实验结果

- 视觉检测指标已有。
- ECE/Brier/reliability diagram 相关代码已有，但 verifier 样本太少。
- ASR accepted/rejected 已有。
- timeline count 已有。

### 需要插入的图、表或公式标注

【建议插入三线表 7：评测指标定义表】  
放置章节：Experiments 5.3。  
主要内容：指标名称、计算对象、公式或定义、用途。  
数据来源：实验设计和当前输出文件。  
服务论点：说明本文评估不只看 mAP。

---

## 5.4 Baselines

### 本段写作目标

设置对比方法，避免只报告自己的 pipeline。

### 应展开的核心内容

至少包含：

1. Visual-only YOLO behavior detection  
   只使用 YOLO11s 行为检测结果，不进行文本验证。
2. Pose-rule baseline  
   使用 pose rules 生成动作，不使用行为检测和文本。
3. ASR/text-only baseline  
   只根据文本事件判断课堂事件。若当前 ASR 太差，应说明这组可能不可用或只在清晰音频样例上做。
4. Fixed-window alignment  
   使用固定时间窗口进行 event-action 对齐。
5. Late fusion baseline  
   先分别得到视觉和文本结果，再简单投票或加权。
6. Fusion v2 baseline  
   使用语义桥接和动作融合，但不使用 dual verification。
7. Ours  
   fusion contract v2 + ASR quality gate + visual fallback + UQ adaptive alignment + dual verification。

### 未来正文可以扩写的段落要点

- 对每个 baseline 说明输入、输出和区别。
- 不要把未实现的 SOTA 大模型作为必须 baseline。可在后续有资源时加入 CLIP/Video-LLaMA 或 temporal grounding 模型，但当前不要硬写。
- 如果 ChinaVis 方向更强调系统和可视分析，可以把可视分析用户任务或 case study 作为补充评估。

### 需要引用的项目证据或实验结果

- `actions.behavior.semantic`、`actions.fusion_v2` 可以支持 visual-only 和 fusion v2 对比。
- `xx_align_multimodal.py` 可以通过参数构造 fixed-window 和 UQ-window 对比。
- `build_event_queries_fusion_v2.py` 支持 ASR fallback 对比。

### 需要插入的图、表或公式标注

【建议插入三线表 8：Baseline 设置表】  
放置章节：Experiments 5.4。  
主要内容：baseline 名称、使用视觉、使用文本、使用 UQ、使用 ASR gate、使用 fallback、输出指标。  
数据来源：当前 pipeline 模块。  
服务论点：确保实验对比清晰。

---

## 5.5 Robustness Settings

### 本段写作目标

设计遮挡、低光、运动模糊、多人重叠、关键点缺失、ASR 错字漏字、事件时间偏移等鲁棒性实验。

### 应展开的核心内容

视觉噪声：

- 遮挡 patch
- 低光 gamma
- 运动模糊
- 多人重叠区域
- 随机丢关键点
- bbox 抖动

文本噪声：

- ASR 错字
- 漏字
- 空文本
- 时间戳偏移
- 幻觉文本
- 近义词替换

时间错位：

- 事件时间偏移 ±0.5s、±1s、±2s、±4s。
- 对比 fixed-window 和 UQ-window 的 candidate recall 和 event F1。

### 未来正文可以扩写的段落要点

- 说明噪声构造只用于鲁棒性测试，不改变训练集。
- 说明每类噪声对哪个模块产生影响。
- 说明鲁棒实验的预期不是绝对提升，而是验证 quality gate、UQ window 和 uncertain 机制是否减少错误确认。

### 需要引用的项目证据或实验结果

- 当前报告已经建议遮挡、低光、运动模糊、多人重叠、关键点缺失、ASR 错字漏字和事件时间偏移等鲁棒设置。
- UQ 和 ASR gate 已具备运行基础。

### 需要插入的图、表或公式标注

【建议插入图 5：鲁棒性曲线图】  
放置章节：Results and Analysis 或 Ablation Study。  
主要内容：不同噪声强度下 candidate recall、event F1、uncertain ratio。  
数据来源：后续批量实验。  
服务论点：证明 dual verification 不是只在干净 demo 上成立。

---

# 6. Results and Analysis

## 本段写作目标

设计应展示哪些表格和图片，并说明当前已有结果与后续待补结果。

## 应展开的核心内容

### 6.1 Behavior Detection Main Results

展示 YOLO11s e150 的主结果，包括 mAP50、mAP50-95、Precision、Recall。当前已有数值，可以作为阶段性主结果。正式论文需要在独立 test split 上重跑。

【建议插入三线表 9：行为检测主结果表】  
放置章节：Results and Analysis 6.1。  
主要内容：模型、epoch、mAP50、mAP50-95、Precision、Recall、F1、split。  
数据来源：`metrics_summary.md`、`e150_effect_summary.md`、`e150_results.csv`。  
服务论点：YOLO11 行为检测为后续验证提供可用视觉基础。

### 6.2 Class-wise AP and Confusion Matrix

展示 8 类行为 AP 和混淆矩阵。尤其关注听课、看书、写字、转头等相似类别，以及 `jz` 教师互动长尾类别。

【建议插入三线表 10：8 类行为类别 AP 表】  
放置章节：Results and Analysis 6.2。  
主要内容：类别短码、中文名、英文名、AP50、AP50-95、样本数。  
数据来源：训练日志、验证结果、类别统计。  
服务论点：分析哪些课堂动作可靠，哪些类别需要增强。

【建议插入图 6：混淆矩阵图】  
放置章节：Results and Analysis 6.2。  
主要内容：8 类行为混淆情况。  
数据来源：`01_figures_detection/e150_confusion_matrix.png` 或 normalized 版本。  
服务论点：证明动作相似和长尾类别是实际问题。

### 6.3 Pipeline Contract Results

展示主线 pipeline 结果，包括 tracked_students、actions_fusion_v2、semantic_valid、event_queries、align candidates、verified events、timeline rows。

【建议插入三线表 11：pipeline contract 结果表】  
放置章节：Results and Analysis 6.3。  
主要内容：文件或阶段、数量、字段完整性、是否通过 contract。  
数据来源：`pipeline_contract_v2_report.json`、`fusion_contract_report.json`、`metrics_summary.md`。  
服务论点：证明 pipeline 可审计，不是只生成最终图。

### 6.4 ASR Quality Gate Statistics

展示 ASR 原始片段、accepted、rejected 和 fallback 状态。当前结果应谨慎解释。

【建议插入三线表 12：ASR 质量门控统计表】  
放置章节：Results and Analysis 6.4。  
主要内容：model、device、compute_type、segments_raw、segments_accepted、segments_rejected、fallback status。  
数据来源：`asr_quality_report.json`。  
服务论点：说明系统不会盲信低质量 ASR。

### 6.5 Dual Verification Results

展示 dual verification 输出，包括 event count、p_match、p_mismatch、reliability、uncertainty、label 分布。当前只有 12 条 verified events，不足以做强结论。正式论文需要扩大 case 数和人工标注。

【建议插入三线表 13：双重验证结果表】  
放置章节：Results and Analysis 6.5。  
主要内容：event_id、query_text、top_candidate、p_match、p_mismatch、reliability、uncertainty、verified label。  
数据来源：`verified_events.jsonl`。  
服务论点：展示本文如何从候选动作转为可靠性判断。

### 6.6 Timeline Visualization

展示学生级 timeline。说明学生 ID 映射、行为片段数量和解释价值。

【建议插入图 7：学生级 timeline 可视化图】  
放置章节：Results and Analysis 6.6。  
主要内容：S01 到 S11 的行为时间线。  
数据来源：`02_figures_pipeline/mainline_timeline_chart.png`、`timeline_students.csv`、`student_id_map.json`。  
服务论点：证明输出可以服务教学可视分析。

### 6.7 Successful Cases

选 2 到 3 个成功案例。每个案例包括视频帧、视觉行为、event query、候选动作、reliability 和 verified label。

【建议插入案例图 2：典型成功案例】  
放置章节：Results and Analysis 6.7。  
主要内容：如“学生举手”“站立”“讨论”等高置信事件。  
数据来源：检测预测图、verified events、timeline。  
服务论点：展示多模态验证的可解释性。

### 6.8 Failure Cases

选 2 到 3 个失败案例。包括 ASR 空文本、遮挡导致关键点缺失、动作相似导致误判、时间偏移导致候选错误。

【建议插入案例图 3：典型失败案例】  
放置章节：Results and Analysis 6.8。  
主要内容：错误原因、系统输出、应改进方向。  
数据来源：低可靠 verified events、ASR report、confusion matrix、人工检查。  
服务论点：诚实呈现风险，避免夸大。

---

# 7. Ablation Study

## 本段写作目标

设计消融实验，证明每个模块不是随意堆叠。

## 应展开的核心内容

### 7.1 无语义桥接 vs 有语义桥接

对比原始短码输出与 fusion contract v2。指标可以是 semantic_valid rate、event extraction success、timeline readable rate、contract pass rate。  
预期：有语义桥接时下游字段完整，timeline 可读，contract 能检查语义缺失。

### 7.2 固定时间窗口 vs UQ 自适应窗口

对比 fixed window 和 UQ adaptive window。指标包括 candidate recall、Top-k candidate recall、event F1、候选数量、错误候选比例。  
预期：UQ window 在遮挡、tracking 抖动和时间偏移下更稳，但可能增加候选数量。

### 7.3 无 ASR 质量门控 vs 有 ASR 质量门控

对比直接使用所有 ASR 和使用 quality gate。指标包括错误文本进入 verifier 的比例、fallback rate、event F1、uncertain ratio。  
预期：质量门控减少低质文本造成的假匹配。

### 7.4 无视觉事件兜底 vs 有视觉事件兜底

对比 ASR 空文本时是否生成 visual fallback。指标包括 event query count、pipeline pass rate、verified events count、timeline coverage。  
预期：有 fallback 时系统可继续生成可审计事件，但不能证明文本模态增强。

### 7.5 visual-only vs fusion_v2 vs dual verification

核心对比。  
visual-only 只输出检测动作。  
fusion_v2 输出融合动作和语义字段。  
dual verification 输出可靠性、uncertainty 和 verified label。  
指标包括 event-level F1、candidate recall、ECE、Brier、人工审计一致性。

### 7.6 不同 alpha_motion 和 beta_uq 参数

网格搜索 `alpha_motion` 和 `beta_uq`。  
指标包括 candidate recall、平均候选数、event F1、uncertain ratio。  
目标是找到召回和噪声之间的平衡。

### 7.7 不同 ASR 模型或 ASR 配置

对比 small/cpu/int8、medium/cpu/int8、medium/cuda/float16 等。  
指标包括 accepted/rejected/fallback rate、推理速度、文本可用率、event F1。  
当前已有 medium/cuda/float16 的低质样例，需补更多音频样本。

### 7.8 长尾类别增强前后对比

对 `jz/zt/js/xt` 等类别做重采样、增强或类别权重。  
指标包括 class-wise AP、Recall、混淆矩阵变化。  
正式论文中可把它放在附录或补充实验。

## 未来正文可以扩写的段落要点

每个消融都要有一个明确被移除的模块。不能只写“我们的方法更好”，必须说明变量。

## 需要引用的项目证据或实验结果

- `test_fusion_contract_v2.py` 已覆盖空 ASR 和缺语义字段。
- `xx_align_multimodal.py` 支持 UQ 窗口参数。
- `metrics_summary.md` 已记录主线结果。
- ASR 当前样例可用于说明 quality gate，但还不能支撑完整消融结论。

## 需要插入的图、表或公式标注

【建议插入三线表 14：消融实验总表】  
放置章节：Ablation Study。  
主要内容：方法变体、语义桥接、UQ window、ASR gate、fallback、dual verifier、event F1、candidate recall、ECE、Brier。  
数据来源：后续批量实验。  
服务论点：证明每个模块的必要性。

【建议插入图 8：alpha_motion 与 beta_uq 参数敏感性图】  
放置章节：Ablation Study。  
主要内容：不同参数下 candidate recall 和平均候选数。  
数据来源：后续参数网格实验。  
服务论点：说明 UQ 窗口不是任意设置。

---

# 8. Discussion

## 本段写作目标

解释实验现象、系统价值和方法边界。

## 应展开的核心内容

### 8.1 为什么 dual verification 有必要

视觉检测提供动作候选，但在遮挡和动作相似时不稳定。文本语义提供事件线索，但 ASR 也可能失败。dual verification 的价值不是强行融合，而是在两个模态不确定时输出 uncertain 或 fallback。

### 8.2 为什么 contract 有研究价值

在多阶段 pipeline 中，空中间文件、缺字段和无候选都可能导致伪成功。contract 的作用是让系统可失败、可审计、可复现。这对论文和系统展示都重要。

### 8.3 可视分析价值

学生级 timeline 把底层检测框转为教学可解释信息。它可以支持课堂参与度观察、异常行为回放、模型错误审计和教师反馈。

### 8.4 与大模型方法的关系

Video-LLaMA、InternVideo2 等大模型可作为未来扩展，但当前项目重点是轻量、可审计、可在本地硬件运行的 pipeline。不要声称当前系统具备大模型语义推理能力。

## 未来正文可以扩写的段落要点

- 讨论为什么有时 fallback 比低质 ASR 更安全。
- 讨论 UQ 窗口扩大带来的召回和噪声权衡。
- 讨论 timeline 的教学解释价值和隐私边界。
- 讨论 ChinaVis 方向下的可视分析贡献。

## 需要引用的项目证据或实验结果

- 当前 ASR 被拒绝说明 quality gate 的必要性。
- Contract 检查说明 pipeline 可审计。
- 主线生成 11 个学生 ID 和 timeline 动作片段说明可视分析输出已具备基础。

## 需要插入的图、表或公式标注

【建议插入图 9：方法边界与失败模式示意图】  
放置章节：Discussion 或 Limitations。  
主要内容：视觉失败、ASR 失败、时间错位、长尾类别、ID drift。  
数据来源：失败案例和风险清单。  
服务论点：说明本文诚实处理系统边界。

---

# 9. Limitations

## 本段写作目标

明确不能夸大的内容，并列出正式投稿前必须补强的问题。

## 应展开的核心内容

1. 缺独立 test split。当前 test 为 0，验证集指标不能直接代表泛化性能。
2. ASR 当前 demo 质量不足。accepted 为 0，说明文本模态在当前样例中没有形成有效语义输入。
3. verifier 样本太少。当前 verified events 为 12，不足以支撑统计显著结论。
4. 长尾类别不足。`jz` 教师互动样本偏少，类别召回可能不稳定。
5. 自定义 YOLO 结构未接主线。ASPN、DySnake、GLIDE 等只能写成设想或待验证，不能写成本文核心贡献。
6. MLP verifier 和校准评估需要更多人工 gold label。
7. Timeline 的学生 ID 是视频内临时 track，不代表真实身份识别，且可能发生 ID switch。

## 未来正文可以扩写的段落要点

Limitations 应放在 Discussion 之后或 Conclusion 之前。语气要克制，但不要过度否定。强调这些限制是下一步实验计划。

## 需要引用的项目证据或实验结果

- train 7416、val 1467、test 0。
- ASR raw 3、accepted 0、rejected 3。
- verified events 12。
- `jz` 类 train 563、val 117。
- 自定义 YOLO 结构在审计报告中标记为部分实现或未接主线。

## 需要插入的图、表或公式标注

【建议插入三线表 15：风险与补强计划表】  
放置章节：Limitations。  
主要内容：风险、当前证据、影响、补强方式、优先级。  
数据来源：`full_research_report.md` K 节风险和短板。  
服务论点：说明论文不会夸大现有结果。

---

# 10. Conclusion

## 本段写作目标

总结本文完成了什么，避免重复实验细节。

## 应展开的核心内容

本文提出了面向智慧课堂的视觉行为序列与文本语义流双重验证框架。框架通过 YOLO11 行为检测和 pose tracking 构建视觉行为序列，通过 ASR 质量门控和事件查询构建文本语义流，通过 UQ 自适应窗口和 dual verification 判断事件可靠性，最后输出学生级 timeline 和 verified events。

## 未来正文可以扩写的段落要点

- 重申本文解决的是“可靠性验证”而不只是“检测准确率”。
- 强调 fusion contract 和 pipeline contract 的可审计性。
- 强调 timeline 的可解释价值。
- 用一句话指出下一步会在更大数据、独立 test split 和人工 gold label 上验证。

## 需要引用的项目证据或实验结果

- YOLO11s e150 阶段性结果。
- 主线 pipeline 输出和 timeline 输出。
- ASR quality gate 和 fallback 机制。

## 需要插入的图、表或公式标注

结论通常不插图表。

---

# 11. Future Work

## 本段写作目标

说明后续工作，不要把未完成的工作写成当前贡献。

## 应展开的核心内容

1. 按视频或 case 重划 train/val/test，形成独立测试集。
2. 构建 20 到 50 个以上事件级人工 gold label，用于 verifier F1、ECE、Brier 评估。
3. 扩展 ASR 样例，比较不同 Whisper 模型、降噪和设备配置。
4. 增补 `jz/zt/js/xt` 等长尾类别，做类别增强和 hard negative。
5. 引入 temporal transformer 或轻量 action localization 作为动作时序建模模块。
6. 尝试 CLIP、Video-LLaMA 或其他 video-language 模型作为语义相似度 baseline，但只在实现后写入正文。
7. 完善可视化系统，支持事件回放、证据链查看、timeline 交互和教师分析任务。

## 未来正文可以扩写的段落要点

Future Work 可以放在 Conclusion 后单独成节，或与 Limitations 合并。面向 ChinaVis 时，可突出交互式可视分析和用户研究。

## 需要引用的项目证据或实验结果

- 当前风险清单。
- 当前 paper package 中已有 demo materials 和 timeline 资产。
- 后端和前端已有部分实现，但需要适配 fusion_v2 优先级。

## 需要插入的图、表或公式标注

【建议插入图 10：未来系统交互界面规划图】  
放置章节：Future Work 或 Appendix。  
主要内容：timeline、事件列表、视频回放、证据面板、reliability 面板。  
数据来源：前端现有页面和 paper package demo materials。  
服务论点：说明未来可视分析系统方向。

---

# 12. References

## 本段写作目标

按类别整理参考文献草案。只写已有材料中出现过或需要核对的真实论文，不编造结果。

## 应展开的核心内容

### 12.1 Classroom behavior detection

- Classroom Behavior Detection Based on Improved YOLOv5 Algorithm Combining Multi-Scale Feature Fusion and Attention Mechanism, 2022, Applied Sciences.
- MSTA-SlowFast: A Student Behavior Detector for Classroom Environments, 2023, Sensors.
- Student Behavior Detection in the Classroom Based on Improved YOLOv8, 2023, Sensors.
- A Spatio-Temporal Attention-Based Method for Detecting Student Classroom Behaviors, 2023/2024, arXiv.
- BiTNet: A Lightweight Object Detection Network for Real-time Classroom Behavior Recognition with Transformer and Bi-directional Pyramid Network, 2023.

### 12.2 YOLO 系列和实时目标检测

- YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors, CVPR 2023.
- YOLOv10: Real-Time End-to-End Object Detection, 2024.
- Ultralytics YOLO Evolution Overview, 待核对具体版本和年份。
- YOLO11 或 Ultralytics 官方文档，待核对引用格式。

### 12.3 Pose estimation and uncertainty modeling

- 可引用 YOLO pose 或 Ultralytics pose 文档，待核对正式引用。
- 可补充关键点检测、tracking 和 uncertainty calibration 相关论文，待检索。
- 如果使用 ECE 和 Brier score，需要引用校准相关经典工作，待核对。

### 12.4 Multimodal video-language understanding

- Learning Transferable Visual Models From Natural Language Supervision, CLIP, 2021.
- UniVL: A Unified Video and Language Pre-Training Model for Multimodal Understanding and Generation, 2020.
- Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding, 2023.
- InternVideo2: Scaling Foundation Models for Multimodal Video Understanding, 2024.
- MAViL: Masked Audio-Video Learners, NeurIPS 2023.
- Meerkat: Audio-Visual LLM for Grounding, ECCV 2024.
- CAT: Enhancing MLLM for Audio-Visual QA, ECCV 2024.

### 12.5 ASR and speech-based classroom analysis

- Robust Speech Recognition via Large-Scale Weak Supervision, Whisper, 2022.
- Audio-Visual ASR with Whisper, 2026，待核对是否已正式发表。
- Multimodal Audio-Visual Detection in Classroom, Scientific Reports 2025.

### 12.6 Temporal grounding and verification

- Retrieval from Counterfactually Augmented Data, ECCV 2024.
- 可补充 temporal grounding 和 video-text retrieval 相关论文，待检索。
- 可补充 event localization 和 moment retrieval 论文，待检索。

### 12.7 Reliability, calibration and uncertainty evaluation

- ECE、Brier score、reliability diagram 相关经典校准论文，待核对。
- 可引用 verifier calibration 相关实现时的理论来源。
- 若使用 temperature scaling，应引用 On Calibration of Modern Neural Networks，待核对。

### 12.8 Visualization and smart classroom analytics

- ChinaVis 2026 call for paper，待最终核对页面要求。
- 智能可视化与可视分析相关论文，待补充。
- MM-TBA: Teacher Behavior Dataset, Scientific Data 2025，可作为教师行为数据集相关参考。

## 未来正文可以扩写的段落要点

References 正文阶段必须用 BibTeX 或目标模板格式。当前只作为分类草案。所有 2026 年论文和网页要求必须在正式投稿前重新核对。

## 需要引用的项目证据或实验结果

- `论文准备.md` 中已有注释书目。
- `full_research_report.md` 中已有论文矩阵和差异分析。

## 需要插入的图、表或公式标注

不插入图表。可在附录放“相关工作对比表”。

---

# 13. Appendix Plan

## 本段写作目标

规划附录材料，避免主文超过 8 页，同时保留复现和审计信息。

## 应展开的核心内容

### Appendix A：Pipeline 运行命令和环境

包含 `09_run_pipeline.py` 的主要参数、模型路径、输入视频路径和输出目录说明。

### Appendix B：Fusion Contract v2 Schema

列出动作字段、事件字段、验证字段和 contract 检查条件。

### Appendix C：更多检测结果图

放 PR 曲线、F1 曲线、P/R 曲线、训练曲线、更多 val batch prediction。

### Appendix D：更多 timeline 和 case story

放多个 case 的学生级 timeline、成功案例和失败案例。

### Appendix E：ASR 质量报告

放不同 ASR 配置的 accepted/rejected/fallback rate。

### Appendix F：Verifier 校准结果

放 reliability diagram、ECE、Brier score 和 temperature scaling 结果。当前样本不足时标为 preliminary。

### Appendix G：风险清单和伦理说明

说明真实课堂数据的匿名化、学生 ID 仅为 track 编号、系统不用于自动惩罚学生。

## 未来正文可以扩写的段落要点

附录内容要和主文图表编号区分。图像和 CSV 可以从 `paper_package_20260426` 中选择，但不要把大型原始文件直接嵌入论文仓库。

## 需要引用的项目证据或实验结果

- `paper_package_20260426/README.md` 已列出 01 到 06 的资料包结构。
- `03_metrics_tables` 已包含指标表、contract report、ASR report、verifier report 和 timeline CSV。
- `06_demo_materials` 已包含 actions、verified events 和 timeline JSON。

## 需要插入的图、表或公式标注

【建议插入附录表 A1：产物文件清单】  
主要内容：文件名、作用、对应章节、是否进入主文。  
数据来源：`paper_package_20260426/README.md`。  
服务论点：提升复现性。

---

# 14. 图表公式清单

| 编号 | 类型 | 建议标题 | 放置章节 | 内容说明 | 数据来源 | 服务论点 | 优先级 |
|---|---|---|---|---|---|---|---|
| 图 1 | 图 | 整体方法框架图 | Introduction 或 Method 4.1 | 视频输入、YOLO11、pose tracking、UQ、ASR gate、fallback、alignment、dual verifier、timeline | `full_research_report.md`、`paper_package_20260426/README.md` | 说明本文是双重验证框架 | 高 |
| 表 1 | 三线表 | 相关工作对比表 | Related Work | 年份、方法、任务、视觉、文本、验证、UQ、课堂、timeline、差异 | `论文准备.md`、`full_research_report.md` | 说明研究空白 | 高 |
| 公式 1 | 公式 | 任务输入输出定义 | Problem Definition | 视觉动作集合、文本事件集合、验证标签 | pipeline 字段 | 形式化本文任务 | 高 |
| 图 2 | 图 | 系统总流程图 | Method 4.1 | 多阶段 pipeline 流程 | 主线脚本和报告 | 说明可复现流程 | 高 |
| 算法 1 | 算法 | Dual-Verification Pipeline | Method 4.1 | 从 pose 到 verified events 的完整伪代码 | `full_research_report.md` | 说明方法步骤 | 高 |
| 表 2 | 三线表 | 8 类课堂行为 taxonomy | Method 4.2 | 短码、中文、英文、semantic id、混淆类别 | `action_semantics_8class.yaml` | 支撑语义桥接 | 高 |
| 图 3 | 图 | 视觉行为序列构建示意图 | Method 4.2 | 帧级检测到 track 级动作片段 | 检测样例和 timeline | 说明序列构建 | 中 |
| 表 3 | 三线表 | fusion contract v2 字段规范 | Method 4.3 | 字段名、类型、来源、必需性、用途 | fusion contract 脚本 | 说明协议层贡献 | 高 |
| 公式 2 | 公式 | Pose tracking 不确定性定义 | Method 4.4 | 关键点置信、缺失比例、轨迹抖动 | `03c_estimate_track_uncertainty.py` | 支撑 UQ | 高 |
| 公式 3 | 公式 | 文本置信度定义 | Method 4.5 | ASR 质量指标与关键词置信 | `asr_quality_report.json` | 支撑 ASR gate | 中 |
| 表 4 | 三线表 | ASR 质量门控状态表 | Method 4.5 或 Results | raw、accepted、rejected、fallback | ASR report | 说明低质文本处理 | 高 |
| 公式 4 | 公式 | UQ 自适应时间窗口 | Method 4.6 | base、motion、UQ、clip | `xx_align_multimodal.py` | 核心方法贡献 | 高 |
| 算法 2 | 算法 | UQ-guided Event-Action Alignment | Method 4.6 | query 到 action candidates | alignment 脚本 | 说明候选生成 | 中 |
| 公式 5 | 公式 | 视觉置信度定义 | Method 4.7 | 检测置信、pose UQ、物品证据 | behavior、pose、object evidence | 解释视觉可靠性 | 高 |
| 公式 6 | 公式 | 跨模态一致性分数 | Method 4.7 | 视觉、文本、语义、时间、UQ | verified events | 解释 verifier | 高 |
| 公式 7 | 公式 | 验证标签决策函数 | Method 4.7 | match、mismatch、uncertain | `07_dual_verification.py` | 支撑三状态输出 | 高 |
| 图 4 | 图 | 学生级 timeline 可视化图 | Method 4.8 或 Results | S01 到 S11 行为片段 | `mainline_timeline_chart.png` | 可解释输出 | 高 |
| 案例图 1 | 案例图 | verified event 证据链案例 | Method 4.8 或 Results | query、candidate、probability、label、frame | verified events、timeline | 展示可审计性 | 高 |
| 表 5 | 三线表 | 数据集划分与类别统计 | Experiments 5.1 | train、val、test、8 类数量 | dataset summary | 说明数据基础与风险 | 高 |
| 表 6 | 三线表 | 实现配置表 | Experiments 5.2 | 模型、参数、输出文件、状态 | pipeline 和 metrics | 提升复现性 | 中 |
| 表 7 | 三线表 | 评测指标定义表 | Experiments 5.3 | mAP、F1、ECE、Brier、fallback rate、coverage | 实验设计 | 说明评价体系 | 中 |
| 表 8 | 三线表 | Baseline 设置表 | Experiments 5.4 | baseline 组件开关 | pipeline 变体 | 说明对比公平 | 高 |
| 图 5 | 图 | 鲁棒性曲线图 | Results 或 Ablation | 噪声强度与 F1/candidate recall | 后续实验 | 说明鲁棒性 | 中 |
| 表 9 | 三线表 | 行为检测主结果表 | Results 6.1 | mAP50、mAP50-95、Precision、Recall | metrics summary | 证明视觉基础 | 高 |
| 表 10 | 三线表 | 8 类行为类别 AP 表 | Results 6.2 | class-wise AP 和样本数 | 训练日志和类别统计 | 分析类别差异 | 高 |
| 图 6 | 图 | 混淆矩阵图 | Results 6.2 | 8 类混淆矩阵 | `e150_confusion_matrix.png` | 说明混淆问题 | 高 |
| 表 11 | 三线表 | pipeline contract 结果表 | Results 6.3 | 学生数、动作数、事件数、候选数 | contract report | 说明可审计链路 | 高 |
| 表 12 | 三线表 | ASR 质量门控统计表 | Results 6.4 | ASR accepted/rejected/fallback | ASR report | 说明文本风险 | 高 |
| 表 13 | 三线表 | 双重验证结果表 | Results 6.5 | p_match、reliability、uncertainty、label | verified events | 展示验证输出 | 高 |
| 图 7 | 图 | 学生级 timeline 可视化图 | Results 6.6 | 学生动作时间线 | timeline chart | 展示教学解释 | 高 |
| 案例图 2 | 案例图 | 典型成功案例 | Results 6.7 | 成功事件证据链 | verified events 和帧图 | 说明方法有效 | 中 |
| 案例图 3 | 案例图 | 典型失败案例 | Results 6.8 | ASR 空、遮挡、错位等 | 失败样例 | 说明限制 | 中 |
| 表 14 | 三线表 | 消融实验总表 | Ablation | 模块开关与指标变化 | 后续实验 | 验证模块必要性 | 高 |
| 图 8 | 图 | UQ 参数敏感性图 | Ablation | alpha/beta 与召回、候选数 | 后续实验 | 解释参数 | 中 |
| 表 15 | 三线表 | 风险与补强计划表 | Limitations | 风险、证据、影响、补强 | 风险清单 | 防止夸大 | 高 |
| 图 10 | 图 | 未来交互界面规划图 | Future Work | timeline、视频、证据、可靠性面板 | 前端 demo 规划 | 服务 ChinaVis | 低 |
| 表 A1 | 附录表 | 产物文件清单 | Appendix | 文件名、作用、章节、是否主文 | paper package README | 提升复现性 | 中 |

---

# 15. 实验补强清单

## 15.1 必须补强

1. 独立 test split  
   按视频或 case 重划，禁止相邻帧同时出现在 train 和 test。
2. 事件级人工 gold label  
   至少 20 到 50 个事件，最好更多。字段包括 event_id、track_id、action、start、end、match label、噪声类型、遮挡等级、ASR 质量。
3. Dual verification 指标  
   计算 event-level Accuracy、Precision、Recall、F1、candidate recall、ECE、Brier。
4. 固定窗口 vs UQ 自适应窗口  
   必须做，否则 UQ-window 只能算设计，不能算充分验证。
5. ASR 质量门控消融  
   比较 gate 前后，以及 fallback 是否触发。
6. 长尾类别分析  
   对 `jz/zt/js/xt` 给出类别数量、AP、Recall 和失败案例。

## 15.2 建议补强

1. 更多课堂视频 case，覆盖低光、遮挡、多人重叠和不同视角。
2. ASR 模型配置对比，包括 small、medium、cpu、cuda、int8、float16。
3. Timeline coverage 和 ID switch 人工抽样评估。
4. 速度和资源占用，包括 FPS、latency、显存。
5. 可视分析用户任务，例如根据 timeline 找到课堂互动片段，作为 ChinaVis 投稿亮点。

---

# 16. 写作风险清单

| 风险 | 当前证据 | 不能怎么写 | 建议写法 |
|---|---|---|---|
| 缺独立 test split | test 为 0 | 不能说模型已充分验证泛化能力 | 写成阶段性验证集结果，正式版补 test |
| ASR 当前样例不可用 | accepted 0，rejected 3 | 不能说文本模态显著提升准确率 | 写成 ASR 质量门控和 fallback 机制有效运行 |
| verifier 样本少 | verified events 12 | 不能做显著性提升结论 | 写成 pipeline demo，补人工 gold label |
| 自定义 YOLO 结构未接主线 | 审计报告标注部分实现或未接主线 | 不能写成本文核心网络创新 | 写成未来工作或待验证增强 |
| 大模型 verifier 未完成 | 没有明确代码和实验 | 不能写成 LLM verifier 已实现 | 只写可作为未来扩展 |
| Timeline ID 不是身份识别 | 学生 ID 来自 track 映射 | 不能暗示真实身份识别 | 写成视频内匿名 track 编号 |
| 前端 demo 不是算法贡献 | 后端和前端部分实现 | 不能把 GitHub Pages 写成核心算法 | 放在系统展示和附录 |
| 长尾类别不足 | `jz` 样本少 | 不能说 8 类全部稳定 | 分析长尾风险并补实验 |
| CSV/PNG 未逐个读取 | 按用户要求跳过大文件 | 不能声称逐项检查所有大图和大 CSV 内容 | 说明依据 README、报告和指标摘要规划 |

---

# 17. 可直接扩写的章节篇幅建议

| 章节 | 建议篇幅 | 写作重点 |
|---|---:|---|
| Abstract | 0.3 页 | 问题、方法、阶段性结果、限制 |
| Introduction | 1.0 页 | 课堂挑战、双重验证动机、贡献 |
| Related Work | 1.2 页 | 课堂检测、多模态、ASR、UQ、可视分析 |
| Problem Definition | 0.5 页 | 输入、输出、事件验证任务 |
| Method | 2.0 页 | 3 条主线，重点写 UQ window 和 verifier |
| Experiments | 1.2 页 | 数据、实现、指标、baseline |
| Results | 1.2 页 | 检测结果、contract、ASR gate、timeline |
| Ablation | 0.8 页 | 模块消融和参数敏感性 |
| Discussion / Limitations | 0.8 页 | 风险、边界、ChinaVis 可视分析价值 |
| Conclusion / Future Work | 0.3 页 | 总结和补强方向 |

若目标是 8 页以内，图表应控制在 5 到 7 个主文图表，其余放附录。

---

# 18. 最终自查

- 已包含 Title、Abstract、Keywords、Introduction、Related Work、Problem Definition、Method、Experiments、Results and Analysis、Ablation Study、Discussion、Limitations、Conclusion、Future Work、References、Appendix Plan。
- 每个大段都包含写作目标、核心内容、扩写要点、项目证据和图表建议。
- Method 已细分为 4.1 到 4.8，覆盖 Overall Framework、Visual Behavior Sequence Construction、Semantic Bridging and Fusion Contract、Pose UQ、ASR Quality Control、UQ-guided Alignment、Dual Verification、Timeline Visualization。
- 创新点控制在 5 个以内，并区分已实现、部分实现和待补强。
- 未虚构实验结果。所有已写数值均来自已有报告或指标摘要。
- 明确标注 test split 缺失、ASR 质量不足、verifier 样本少、长尾类别不足等风险。
- 适合先写中文论文，再翻译英文论文。
- 面向 ChinaVis 2026 时，需要进一步突出可视分析系统、timeline 交互和教学解释价值。
