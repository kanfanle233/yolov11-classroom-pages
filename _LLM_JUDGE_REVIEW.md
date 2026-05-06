# LLM Judge Pipeline — Codebase Review Report

## 1. Current Verifier Scoring/Fusion Step 在哪？

**位置**: `verifier/infer.py:_predict_one()` (line 124–226)

函数签名：
```python
def _predict_one(
    *,
    model: Optional[VerifierMLP],
    runtime_cfg: VerifierRuntimeConfig,
    event_type: str,
    query_text: str,
    cand: Dict[str, Any],      # 单个 alignment candidate
    uq_default: float,
    query_source: str = "unknown",
    audio_confidence: float = 0.0,
) -> Dict[str, Any]:
```

**逻辑流程**:
1. 从 `cand` 提取 overlap, action_confidence, uq, action_label
2. 调用 `build_feature_vector()` → 4-dim `[overlap, action_confidence, text_score, stability(1-uq)]`
3. 计算 `visual_score = 0.65*overlap + 0.35*action_confidence`
4. **Heuristic fusion**（三种策略）:
   - ASR 可用时: 动态加权 `p_match = (w_visual*visual_score + w_audio*text_score) / (w_visual+w_audio)`
   - 视觉 fallback: `p_match = visual_score`
   - 无音频: `p_match = visual_score`
5. **MLP override**（line 187–192）: 若 `model is not None`，用 `sigmoid(MLP(feat)/temperature)` 替换上述所有 fusion
6. 计算 reliability、分配 label

**Student judge 要替换的就是 line 187–192 这个 MLP override 路径**。Heuristic 路径不变。

**调用链**: `_predict_one()` 被 `infer_verified_rows()` (line 229) 在每个 candidate 上调用 → `infer_verified_rows()` 被 `main()` CLI 调用（line 365–401）。

**CLI 入口**:
- 直接: `python verifier/infer.py --model verifier.pt --event_queries ... --aligned ... --out ...`
- 通过 pipeline: `python scripts/pipeline/07_dual_verification.py --verifier_model verifier.pt ...`

`07_dual_verification.py` (line 313–319) 内部调用 `infer_verified_rows(model_path=model_path)`，即把 `--verifier_model` 透传给 `_load_model()`。

---

## 2. 当前 Verifier 输入有哪些字段？

### 2.1 来自 `event_queries.fusion_v2.jsonl`（每行一个 query）

| 字段 | 来源 | 描述 |
|------|------|------|
| `query_id` / `event_id` | ASR | 事件唯一 ID |
| `query_text` | ASR | ASR 转录的教师指令文本 |
| `event_type` | 规则分类 | 事件类型（raise_hand/head_down/discussion/unknown 等） |
| `source` | ASR | 音频来源（"asr" / "visual_fallback"） |
| `confidence` | ASR | ASR 置信度 [0,1] |
| `timestamp` / `t_center` | ASR | 时间中心点 |
| `start` / `end` | ASR | 时间窗口 |

### 2.2 来自 `align_multimodal.json` 的 candidate（每个 candidate）

| 字段 | 来源 | 描述 |
|------|------|------|
| `track_id` | tracking | 学生跟踪 ID |
| `action` | 行为识别 | 行为英文名（listen/write/read 等） |
| `semantic_id` | 语义映射 | 语义 ID |
| `behavior_code` | 分类法 | 8 类行为编码（tt/dx/dk/zt/xt/js/zl/jz） |
| `behavior_label_zh` | 分类法 | 中文行为描述 |
| `behavior_label_en` | 分类法 | 英文行为描述 |
| `overlap` | alignment | 时间重叠度 [0,1] |
| `action_confidence` | 行为识别 | 行为检测置信度 [0,1] |
| `uq_score` / `uq_track` | pose UQ | 跟踪不确定性 [0,1] |
| `start_time` / `end_time` | tracking | 行为时间段 |
| `raw_action` | 行为识别 | 原始行为名 |

### 2.3 来自 `pose_tracks_smooth_uq.jsonl`（可选的 UQ 索引）

| 字段 | 描述 |
|------|------|
| `track_id` | 跟踪 ID |
| `uq_score` / `uq_track` | 平均 per-track 不确定性 |

### 2.4 当前 `_predict_one()` 实际使用的字段

```python
# 从 cand 提取
overlap = cand.get("overlap")
action_conf = cand.get("action_confidence", cand.get("confidence"))
uq = cand.get("uq_score", cand.get("uq_track", uq_default))
action_label = cand.get("semantic_id", cand.get("action"))

# 从参数传入
event_type      # query.event_type
query_text      # query.query_text
query_source    # query.source
audio_confidence # query.confidence
```

---

## 3. 哪些字段可以作为 LLM Teacher 输入？

**所有字段都是文本/JSON** — 不包含视频帧。以下按重要性分三层：

### 必须输入的（LLM 判断核心依据）

| 字段 | 原因 |
|------|------|
| `query_text` | 教师指令文本，LLM 理解语义的核心 |
| `behavior_code` + `behavior_label_zh` + `behavior_label_en` | 学生行为，与指令对比的标的 |
| `overlap` | 时间对齐程度，影响判断强度 |
| `action_confidence` | 视觉检测置信度，影响判断的可信度 |
| `event_type` | 规则分类的指令类型，即使为 "unknown" 也有参考价值 |

### 建议输入的（提供上下文，无需视频）

| 字段 | 原因 |
|------|------|
| `query_source` | 标明是 ASR 还是 visual_fallback |
| `audio_confidence` | ASR 质量，影响对 query_text 可靠性的判断 |
| `uq_score` | tracking 稳定性，影响对行为检测的信心 |
| `track_id` | 用于关联，不参与判断 |
| `window_start` / `window_end` | 时间窗口 |
| `start_time` / `end_time` | 行为时间 |

### 不应作为 LLM 输入的（label leakage）
`p_match`, `p_mismatch`, `reliability_score`, `match_label`, `label`, `fusion_mode`, `w_visual`, `w_audio`, `model_version`, `threshold_source`

---

## 4. 哪些字段可以作为 Student 输入特征？

**Student 的输入是 4-dim feature vector**（`verifier/model.py:build_feature_vector`，line 67–82）：

```python
def build_feature_vector(event_type, query_text, action_label, overlap, action_confidence, uq_score):
    text_score = action_match_score(event_type, query_text, action_label)  # 规则匹配分
    stability_score = 1.0 - uq_score
    return [overlap, action_confidence, text_score, stability_score]
```

这四个特征**都由 candidate 原始字段产生，不涉及 verifier 输出**。

**如果要扩展 student 输入特征**（可选，不违反约束），可以添加：
| 候选扩展 | 来源 | 是否违反约束 4 |
|----------|------|----------------|
| `behavior_code` one-hot (8-dim) | candidate | 不违反（原始字段） |
| `query_source` 二值 | query | 不违反（原始字段） |
| `audio_confidence` | query | 不违反（原始字段） |
| `text_score`（当前已使用） | `action_match_score` | 不违反（规则计算，非模型输出） |

**结论**: 当前 4-dim 特征即可，可选扩展增加维度。student 不需要直接访问 raw query_text（LLM teacher 才需要）。

---

## 5. 哪些字段必须排除以避免 Label Leakage？

### 绝对排除（student 输入、LLM 输入都不能用）

来自 `_predict_one()` 输出:
- `p_match` — 这是 verifier 的判断结果，是 student 要学习的 target 的替代品
- `p_mismatch` — 同上
- `match_label` / `label` — 最终分类结果
- `reliability_score` — verifier 的置信度
- `uncertainty` — 从 reliability 导出
- `fusion_mode` — 编码了使用的 fusion 策略
- `w_visual`, `w_audio` — fusion 权重
- `threshold_source`, `model_version` — 元数据

### LLM teacher 可以看但 student 不能用的

LLM teacher 可以看到原始 query_text 和 behavior_label_zh 的语义，但 student 只能通过 `text_score`（规则匹配分）间接获取语义信息。这不违反约束——LLM teacher 就是做语义理解的。

---

## 6. Student 要接回哪个函数和哪个 CLI？

### 函数接口

Student 替换的是 `_predict_one()` 中的 **MLP override 路径**（line 187–192）：

```python
# 当前实现（verifier/infer.py:187-192）
if model is not None:
    x = torch.tensor([feat], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x).squeeze(0)
    p_match = float(torch.sigmoid(logits / max(1e-6, runtime_cfg.temperature)).item())
    fusion_mode = "mlp_trained"
```

Student checkpoint 必须满足:
- 架构: `VerifierMLP(in_dim=4, hidden_dim=16)`（与现有完全一致）
- 格式: `torch.save({"kind": "student_judge_v1", "state_dict": ..., "runtime_config": ..., "teacher_provenance": ...})`
- 加载: `_load_model()` 已支持 `kind="student_judge_v1"`（line 83–121）

### CLI 入口

| 层级 | 命令 |
|------|------|
| 直接推理 | `python verifier/infer.py --model student_judge.pt --event_queries ... --aligned ... --out ...` |
| 通过 pipeline | `python scripts/pipeline/07_dual_verification.py --verifier_model student_judge.pt ...` |

`07_dual_verification.py` 的 `--verifier_model` 参数已经是 pipeline 的标准入口，直接接受 `.pt` 文件路径。

---

## 7. 现有 Case 发现逻辑应该如何复用？

### Server 端的 Case 发现

`server/app.py:list_cases()` 使用 `_looks_like_case_dir()`（line 844）遍历 `output/` 目录，检查 marker 文件:
```python
markers = [
    "verified_events.jsonl",
    "event_queries.fusion_v2.jsonl",
    "timeline_chart.json",
    "pipeline_manifest.json",
    ...
]
```

Pipeline 的 case 发现应复用同样的思路：
- **不写死 case 列表**
- **扫描 `output/codex_reports/` 下所有目录**
- **检测标志文件 `align_multimodal.json` + `event_queries.fusion_v2.jsonl`**（这是 pipeline 最关键的输入）

### 启发式匹配（来自 test_regression.py 的模式）

```python
# 匹配所有 "front_*_full" 或 "front_*_sliced" 格式的目录
re.match(r"^front_.*_(full|sliced)$", dir_name)
```

### 建议的 Pipeline Case 发现

```python
def discover_pipeline_cases(root_dir: Path) -> List[Path]:
    """Find all cases with required pipeline inputs."""
    cases = []
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        has_align = (child / "align_multimodal.json").exists()
        has_queries = (child / "event_queries.fusion_v2.jsonl").exists()
        has_actions = (child / "actions.fusion_v2.jsonl").exists()
        if has_align and has_queries and has_actions:
            cases.append(child)
    return cases
```

当前有 6 个 case 满足条件:
- `front_002_full_pose020_hybrid`
- `front_002_rear_row_sliced_pose020_hybrid`
- `front_1885_full`
- `front_22259_full`
- `front_26729_full`
- `front_45618_full`

---

## 防跑偏检查

### 检查 1：是否把 LLM 当成视频模型？

**通过**。在已实现的 `llm_teacher.py` 中:
- `SYSTEM_PROMPT` + `USER_PROMPT_TEMPLATE` 只注入文本/JSON 字段
- 没有视频帧输入
- 注释标明 "text-only LLM"、"No vision model needed"
- `evidence` 记录包含查询文本、行为标签、数值特征

### 检查 2：是否把 verified label 当 student 输入？

**通过**。已实现的 `student_train.py` 中:
- `build_student_samples()` 构建的 feature vector 只用 `[overlap, action_confidence, text_score, stability]`
- Target 来自 `silver_p_match`，是 LLM teacher 的输出（不是 verifier 的 verified label）
- `teacher_provenance` 字段明确追踪标签来源

### 检查 3：是否只列了 front_45618 这种固定 case？

**通过**。case 发现使用目录扫描 + marker 文件检测，不写死 case 列表:
- `build_evidence.py` 接受任意路径的 `align_multimodal.json` 和 `event_queries.jsonl`
- pipeline 的 `--event_queries`、`--aligned`、`--actions` 都是显式参数
- `discover_pipeline_cases()` 函数可以自动发现所有满足条件的 case
- `smoketest.py` 的 cross-case holdout 检查使用 key 聚合而非硬编码

---

## 结论汇总

| 问题 | 答案 |
|------|------|
| Scoring/fusion 在哪 | `verifier/infer.py:_predict_one()` lines 124–226 |
| Student 替换什么 | MLP override 路径 line 187–192（`model is not None` 分支） |
| CLI 入口 | `07_dual_verification.py --verifier_model student_judge.pt` |
| Student 输入特征 | 4-dim `[overlap, action_confidence, text_score, stability]` |
| LLM teacher 输入 | 所有 candidate 字段 + query 字段（text/JSON） |
| 排除字段 | `p_match`, `p_mismatch`, `match_label`, `reliability_score`, `uncertainty` 等 |
| Case 发现 | 目录扫描 + marker 文件，不硬编码 |
