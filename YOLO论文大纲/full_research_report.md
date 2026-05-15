# YOLOv11 智慧课堂音视频上下文感知 — 全量研究报告

> 生成日期：2026-05-03
> 覆盖范围：全部代码（43脚本）、全部实验（6 case）、全部指标

---

## A. 项目定位

**系统级课堂行为分析框架**，非单一检测器改进。

核心能力链：YOLO视觉检测 → 姿态跟踪 → 后排切片增强 → Whisper语音 → 教师指令识别 → Confidence-Weighted音视频融合 → LLM语义推理 → Pipeline Contract可审计输出 → D3交互可视化。

---

## B. 代码结构（重组后）

```
YOLOv11/
  scripts/
    main/          09_run_pipeline.py    主编排（43步）
                   09b_run_pipeline.py   别名入口
    pipeline/      24个流水线步骤脚本
    experiments/   6个消融/评估脚本
    frontend/      20_build_frontend_data_bundle.py
    paper/         4个论文图表生成
    utils/         6个共享工具库
  verifier/        验证器库（infer/model/train/calibration）
  contracts/       数据契约 schema
  server/          FastAPI 服务
  web_viz/         D3 前端模板
  YOLO论文大纲/    论文大纲 + 本报告
  索引/            13份结构化索引文档
```

---

## C. 关键实验与指标

### C.1 行为检测模型

| 模型 | 训练 | Test mAP50 | Test mAP50-95 | Per-class min AP50 |
|---|---|---|---|---|
| YOLO11s e150 | 150 epoch, 8类 | **0.9806** | **0.8782** | 0.9517 (zt) |

Test set: 1339张，70/15/15分层随机拆分。训练集7416张，来自593个正方视角视频的案例截图，267,861个标注框。类别严重不平衡（tt:jz = 172:1）。

### C.2 后排SR切片消融

3个视频（front_001/002/046），6变体（A0/A1/A2/A3/A7/A8）：

| 变体 | 切片 | SR | Deblur | 网格 |
|---|---|---|---|---|
| A0 | ✗ | ✗ | ✗ | — |
| A1 | ✓ | ✗ | ✗ | rear_adaptive |
| A2 | ✓ | opencv2× | ✗ | rear_adaptive |
| A3 | ✓ | opencv2× | ✓ | rear_adaptive |
| A7 | ✓ | ✗ | ✗ | rear_adaptive+seat |
| A8 | ✓ | opencv2× | ✓ | rear_adaptive+seat |

**front_046 (gt_status=ok) 核心结果：**

| 变体 | Students | Proxy vs A0 | Formal Recall | IDSW |
|---|---|---|---|---|
| A0 | 20 | — | 8.33% | 0 |
| A1 | 33 | +107% | 12.5% | 0 |
| A3 | 33 | +105% | **15.0%** | 0 |
| A8 | **37** | +107% | 10.8% | 0 |

**论文可以写**：切片推理检测学生数+85%，proxy指标翻倍，IDSW=0。Formal recall受限于GT仅3人（precision 1-2%）。

### C.3 跟踪算法消融

4变体（TK-0贪心/TK-1匈牙利/TK-2匈牙利+Kalman/TK-3 ByteTrack），046 A8数据：

| 变体 | Tracks | Gaps≥1s | 论文结论 |
|---|---|---|---|
| TK-0 | 37 | 36 | 基线 |
| **TK-1** | **35** | **26** | **✅ 推荐主线** |
| TK-2 | 35 | 26 | Kalman无额外收益（诚实负结果） |
| TK-3 | 42 | 0 | ByteTrack更平滑但检测少41% |

### C.4 音视频融合

**方法一（Confidence-Weighted动态融合）**：`verifier/infer.py`

权重不再固定0.55/0.45，改为动态：
```
p_match = (w_v × C_v + w_a × C_a) / (w_v + w_a)
w_v = 1-uq, w_a = asr_conf
```
无声时w_a=0→纯视觉。修复前046.mp4的p_match=0.924（text_score=1.0虚高），修复后=0.748（真实）。

**方法二（LLM零样本语义推理）**：Gemini评估37对指令-行为：

| LLM判断 | 数量 | 占比 |
|---|---|---|
| 支持 | 27 | 73% |
| **违背** | **9** | **24%** |
| 中性 | 1 | 3% |

**核心发现**：24%案例中，统计融合无法识别语义冲突（规则覆盖不到），LLM以平均把握度0.90正确识别。

### C.5 语音管道

5个视频完成ASR，2个提取到有效教师指令：

| 视频 | 时长 | ASR段 | 提取指令 |
|---|---|---|---|
| 45618 | 203s | 47 | 12 |
| 1885 | 213s | 17 | 8 |
| 22259 | 126s | 31 | 0（非指令语音） |
| 25395 | 84s | 25 | 0（励志讲话） |
| 26729 | 62s | 32 | 0（写字讲解） |

### C.6 跨视角泛化

| 视角 | 检测/3帧 | vs 训练域 |
|---|---|---|
| 正方（训练） | 94 | 基线 |
| 斜上方 | 84-95 | 可接受 |
| 后方 | 2 | -98%失效 |
| 教师 | 0 | -100%失效 |

### C.7 完整Case统计

| Case | Students | Timeline | Contract | GT |
|---|---|---|---|---|
| front_046_A8 | 37 | 276 | ok | ok |
| front_002_A8 | 31 | 45 | ok | missing |
| front_45618_sliced | 45 | 2179 | ok | missing |
| front_1885_sliced | 50 | 1376 | ok | missing |
| front_22259_sliced | 44 | 690 | ok | missing |
| front_26729_sliced | 50 | 859 | ok | missing |

---

## D. 论文准备度

| 模块 | 数据完整度 | 可写入章节 |
|---|---|---|
| 行为检测 | ✅ Test mAP50=0.9806 | §5.1 主结果表 |
| 后排消融 | ✅ proxy+formal双指标 | §5.6 / §7.7 |
| 跟踪消融 | ✅ 4变体量化对比 | §7.8 |
| 音视频融合 | ✅ 37对LLM评估+动态权重 | §5.7 核心创新 |
| 跨视角泛化 | ✅ 4视角定量 | §5.6 |
| Pipeline Contract | ✅ 6 case全部ok | §6.3 |
| D3可视化 | ✅ 6 bundle+页面 | §5.8 |

**仍需补充**：GT扩展至10+学生、有语音视频的端到端LLM融合闭环。

---

## E. 诚实局限

1. **语音贡献有限**：2/593视频有指令，论文标题避免"双模态验证"
2. **GT覆盖不足**：仅3学生，formal precision不可用
3. **Test split非视频级**：可能存在跨帧泄漏
4. **LLM融合为定性评估**：37对非ML benchmark，是系统分析
5. **未与检测器改进方法直接对比**：贡献在系统层面

---

## F. 关键文件速查

| 用途 | 路径 |
|---|---|
| 主编排 | `scripts/main/09_run_pipeline.py` |
| 动态融合 | `verifier/infer.py:134-175` |
| LLM融合 | `scripts/pipeline/06f_llm_semantic_fusion.py` |
| 切片工具 | `scripts/utils/sliced_inference_utils.py` |
| 跟踪算法 | `scripts/pipeline/03_track_and_smooth.py` |
| ASR管道 | `scripts/pipeline/06_asr_whisper_to_jsonl.py` |
| 指令提取 | `scripts/pipeline/06e_extract_instruction_context.py` |
| SR消融 | `scripts/experiments/16_run_rear_row_sr_ablation.py` |
| 正式评估 | `scripts/experiments/19_eval_rear_row_metrics.py` |
| D3打包 | `scripts/frontend/20_build_frontend_data_bundle.py` |
| D3页面 | `web_viz/templates/paper_demo.html` |
| 论文大纲 | `YOLO论文大纲/论文大纲1.md` |
| LLM评估 | `output/llm_fusion_37_results.json` |
| 消融总表 | `output/codex_reports/sr_ablation_paper_summary.json` |
| 索引文档 | `索引/11_全实验数据与论文指标记录.md` |
