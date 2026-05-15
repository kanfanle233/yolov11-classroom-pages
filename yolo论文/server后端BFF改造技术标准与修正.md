# server/app.py BFF 改造——交叉评估与技术标准修正

> 对照用户提出的 BFF 架构标准与 `server/app.py` 实际代码（2029行 FastAPI），做逐项核实与修正

---

## 1. 现有后端代码核实

### 1.1 已就绪的基础设施（无需新增）

| 组件 | 代码证据 | 状态 |
|------|---------|------|
| FastAPI + CORS | `app.py:88-96` | ✅ 已配 `allow_origins=["*"]` |
| Bundle 目录 | `app.py:1895` `BUNDLE_DIR = OUTPUT_DIR / "frontend_bundle"` | ✅ |
| 查找 case | `app.py:1898-1912` `_find_bundle(case_id)` | ✅ 已实现 |
| 读取 JSON | `app.py:1914-1921` `_get_bundle_json(case_id, filename)` | ✅ 已实现 |
| verified 数据 | `app.py:1991-1993` `/api/bundle/{case_id}/verified` | ✅ 已有 |
| timeline 数据 | `app.py:1951-1953` `/api/bundle/{case_id}/timeline` | ✅ 已有 |
| behavior 数据 | `app.py:1971-1973` `/api/bundle/{case_id}/behavior_segments` | ✅ 已有 |
| 静态 mount | `app.py:2015-2016` `/output/frontend_bundle` | ✅ 视频/图片直接可访问 |

### 1.2 你需要但已存在的函数（可直接复用）

```python
# app.py:1914-1921 — 你的新 endpoint 可以直接调用
def _get_bundle_json(case_id: str, filename: str) -> Any:
    bundle = _find_bundle(case_id)
    if bundle is None:
        raise HTTPException(404, f"case {case_id} not found")
    path = bundle / filename
    if not path.exists():
        raise HTTPException(404, f"{filename} not found")
    return _read_json_file(path, {})
```

---

## 2. 用户提出的 Schema 与实际数据字段对照

### 2.1 字段映射修正

用户 schema 中每个字段对应的真实数据来源：

| 用户字段 | 实际来源文件 | 实际字段路径 | 修正 |
|---------|------------|------------|------|
| `event_id` | verified_events.json | `events[].event_id` | ✅ 直接可用 |
| `track_id` | verified_events.json | `events[].track_id` | ✅ 直接可用 |
| `time_range: [start, end]` | verified_events.json | `events[].window.start / .end` | ✅ 直接可用 |
| `behavior_type` | verified_events.json | `events[].behavior_code` | ✅ 直接可用（tt/dx/dk/zt/xt/js/zl/jz） |
| `verification_status` | verified_events.json | `events[].label` | ✅ 直接可用（match/uncertain/mismatch） |
| `semantics.asr_text` | verified_events.json | `events[].query_text` | ⚠️ 当前 paper_mainline 全是 `"visual_fallback:listen/..."` |
| `semantics.instruction_type` | verified_events.json | `events[].event_type` | ✅ 直接可用 |
| `evidence_metrics.c_visual` | verified_events.json | `events[].evidence.visual_score` | ⚠️ 用户命名为 `c_visual`，实际是 `visual_score` |
| `evidence_metrics.c_text` | verified_events.json | `events[].evidence.text_score` | ⚠️ 同上，实际是 `text_score` |
| `evidence_metrics.uq_track` | verified_events.json | `events[].evidence.uq_score` | ⚠️ 用户命名为 `uq_track`，实际是 `uq_score` |
| `evidence_metrics.weight_v` | verified_events.json | `events[].evidence.w_visual` | ⚠️ 用户命名为 `weight_v`，实际是 `w_visual` |
| `evidence_metrics.weight_a` | verified_events.json | `events[].evidence.w_audio` | ⚠️ 用户命名为 `weight_a`，实际是 `w_audio` |
| `evidence_metrics.reliability_final` | verified_events.json | `events[].reliability_score` | ⚠️ 用户命名为 `reliability_final`，实际是 `reliability_score` |

### 2.2 关键数据缺失问题

以下字段在真实数据中**不存在**或**为占位值**：

| 缺失字段 | 原因 | 补救方案 |
|---------|------|----------|
| 真实的 ASR 文本 | paper_mainline 视频无声，全部是 visual_fallback | 用 `front_45618` 的 bundle 重跑 |
| `duration_sec` | bundle 中没有视频时长字段 | 从 `timeline_students.json` 的 segments 最大 `end_time` 推断 |
| `student_count` | 需要从 manifest 读取 `tracked_students` 字段 | ✅ `frontend_data_manifest.json` 中有 |
| `uncertain`/`mismatch` 样本 | 当前全部为 match | 45618 有真实 ASR 指令，重跑可能产生 |

---

## 3. 后端改造执行规范

### 3.1 新增的路由

```python
# 在 app.py 的 /api/bundle/* 区域附近新增（约 1993 行之后）

@app.get("/api/v1/visualization/case_data")
def get_visualization_case_data(case_id: str):
    """
    BFF endpoint: aggregates timeline + verified + behavior data
    into a single unified view model for D3 consumption.
    """
    # 1. Read source files from bundle
    timeline = _get_bundle_json(case_id, "timeline_students.json")
    verified = _get_bundle_json(case_id, "verified_events.json")
    manifest  = _get_bundle_json(case_id, "frontend_data_manifest.json")
    
    # 2. Extract case metadata
    segments = timeline.get("segments", [])
    events   = verified.get("events", [])
    max_time = max((s.get("end_time", 0) for s in segments), default=0)
    student_count = manifest.get("tracked_students", len(timeline.get("students", [])))
    
    # 3. Build event index by (track_id, time) for JOIN
    # verified events have window.start/end and track_id
    # timeline segments have start_time/end_time and track_id
    # Join strategy: match verified event to segment by track_id + time overlap
    
    merged_events = _merge_verified_with_timeline(events, segments)
    
    return {
        "status": "success",
        "data": {
            "case_info": {
                "case_id": case_id,
                "duration_sec": round(max_time, 1),
                "student_count": student_count
            },
            "events": merged_events
        }
    }
```

### 3.2 核心合并函数

```python
def _merge_verified_with_timeline(
    verified_events: List[Dict],
    timeline_segments: List[Dict]
) -> List[Dict]:
    """
    JOIN verified_events with timeline_segments on (track_id + time overlap).
    Returns unified view-model events.
    """
    # Build segment index: track_id -> [(start, end, behavior_code, confidence)]
    from collections import defaultdict
    seg_index = defaultdict(list)
    for seg in timeline_segments:
        tid = seg.get("track_id")
        if tid is not None:
            seg_index[tid].append(seg)
    
    merged = []
    for i, ve in enumerate(verified_events):
        tid = ve.get("track_id", -1)
        window = ve.get("window", {})
        t_start = window.get("start", ve.get("query_time", 0) - 1.5)
        t_end   = window.get("end",   ve.get("query_time", 0) + 1.5)
        
        # Find overlapping timeline segment
        best_seg = None
        best_overlap = 0.0
        for seg in seg_index.get(tid, []):
            seg_start = seg.get("start_time", 0)
            seg_end   = seg.get("end_time", 0)
            overlap = max(0, min(t_end, seg_end) - max(t_start, seg_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_seg = seg
        
        # Determine behavior from timeline segment (more reliable than verified action)
        behavior = (
            best_seg.get("behavior_code") if best_seg
            else ve.get("behavior_code", ve.get("action", "unknown"))
        )
        
        evidence = ve.get("evidence", {})
        merged.append({
            "event_id": ve.get("event_id", f"E_{i:03d}"),
            "track_id": f"S_{tid:02d}",
            "time_range": [round(t_start, 2), round(t_end, 2)],
            "behavior_type": behavior,
            "verification_status": ve.get("label", "match"),
            "semantics": {
                "asr_text": ve.get("query_text", ""),
                "instruction_type": ve.get("event_type", "unknown")
            },
            "evidence_metrics": {
                "c_visual":   round(evidence.get("visual_score", 0), 4),
                "c_text":     round(evidence.get("text_score", 0), 4) 
                              if evidence.get("text_score") == evidence.get("text_score") 
                              else None,  # NaN guard
                "uq_track":   round(evidence.get("uq_score", 0.5), 4),
                "weight_v":   round(evidence.get("w_visual", 1.0), 4),
                "weight_a":   round(evidence.get("w_audio", 0.0), 4),
                "reliability_final": round(ve.get("reliability_score", 0), 4)
            }
        })
    
    return merged
```

### 3.3 NaN 处理说明

`verified_events.json` 中 `evidence.text_score` 在纯视觉 fallback 模式下值为 `NaN`（JSON 中表示为 `null`）。合并函数必须处理：

```python
# NaN guard in Python
import math
text_score = evidence.get("text_score", 0)
if text_score is None or (isinstance(text_score, float) and math.isnan(text_score)):
    text_score_out = None  # null in JSON output
else:
    text_score_out = round(float(text_score), 4)
```

---

## 4. 用户三层架构评估（论文可写）

用户提出的三层架构在论文方法图中是**完全正确的**：

```
┌─────────────────────────────────────────────┐
│  Layer 3: Visual Analytics (D3 frontend)     │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐ │
│  │ Gantt    │ │ Weight   │ │ Evidence    │ │
│  │ Timeline │ │ Curves   │ │ Radar       │ │
│  └──────────┘ └──────────┘ └─────────────┘ │
│         ▲ fetch('/api/v1/visualization/...') │
├─────────────────────────────────────────────┤
│  Layer 2: Data Transformation (app.py BFF)  │
│  ┌──────────────┐ ┌──────────────────────┐  │
│  │ Time-JOIN    │ │ Feature Aggregation  │  │
│  │ (track_id +  │ │ (evidence packaging) │  │
│  │  time_range) │ │                      │  │
│  └──────────────┘ └──────────────────────┘  │
│         ▲ read JSONL/CSV from disk            │
├─────────────────────────────────────────────┤
│  Layer 1: AI Model Engine (scripts/)         │
│  ┌────────┐ ┌──────────┐ ┌──────────────┐  │
│  │YOLO11  │ │Whisper   │ │VerifierMLP   │  │
│  │Pose+Det│ │ASR+LLM   │ │CW-DLF fusion │  │
│  └────────┘ └──────────┘ └──────────────┘  │
└─────────────────────────────────────────────┘
```

**代码证据**：Layer 2 对应 `server/app.py` 新增的 BFF 路由；Layer 1 对应 `scripts/pipeline/*.py` 和 `verifier/*.py`；Layer 3 对应 `web_viz/templates/index.html`。

---

## 5. BFF 方案 vs 现有方案对比

| 维度 | 现有方案（多API调用） | BFF 方案（单API调用） |
|------|---------------------|----------------------|
| 前端 fetch 次数 | 3次（timeline + verified + behavior） | 1次 |
| 前端 JOIN 逻辑 | 需要在 JS 中手动合并 track_id+时间 | 零——后端已完成 |
| 前端代码量 | ~500行 D3 + ~150行数据处理 | ~350行 D3（减少30%） |
| 网络传输量 | 3个 JSON（timeline 82KB + verified 16KB + behavior 215KB） | 1个合并 JSON（~100KB） |
| 可维护性 | 前端字段名变更需改多处 | 后端改 Schema，前端自动适配 |
| 离线静态部署 | 需要预合并 JSON | ✅ BFF 可直接写磁盘→GitHub Pages 直接用 |

---

## 6. 执行步骤（可直接操作）

### Step 1：新增路由（app.py 约1995行之后）

在 `/api/bundle/{case_id}/verified` 之后插入 `get_visualization_case_data` 函数（详见 3.1-3.2）。

### Step 2：本地测试

```bash
cd server/
uvicorn app:app --reload --port 8000
# 浏览器访问：
# http://localhost:8000/api/v1/visualization/case_data?case_id=front_046_A8
```

### Step 3：验证返回 JSON

检查返回的 `events[]` 中 `verification_status` 字段存在且值为 `match/uncertain/mismatch` 之一，`evidence_metrics` 所有子字段在 [0,1] 范围。

### Step 4：前端对接

```javascript
// 前端的 loadCaseData() 简化为一行：
const resp = await fetch(`/api/v1/visualization/case_data?case_id=${caseId}`);
const { data } = await resp.json();
// data.events 直接用于 D3 渲染，无需额外处理
renderAllViews(data);
```

### Step 5：静态离线版本

```python
# 在 20_build_frontend_data_bundle.py 中新增一步：
# 调用 get_visualization_case_data() → 写入 frontend_bundle/{case_id}/unified_view.json
# GitHub Pages 直接 fetch 这个静态文件
```

---

## 7. 可行性总结

| 评估维度 | 结论 |
|----------|------|
| **架构方案** | ✅ 三层架构（Model Engine → BFF Adapter → D3 Views）是 VAS 领域的标准范式，可写进论文方法图 |
| **后端工作量** | 新增约80行 Python（1个路由 + 1个合并函数），复用现有 `_get_bundle_json` |
| **前端收益** | 前端数据处理代码减少约150行，D3 代码更纯粹（纯视觉映射） |
| **数据完整性** | ⚠️ paper_mainline 缺少真实 ASR 和 uncertain 样本，需 45618 重跑 |
| **兼容性** | 新增路由不破坏现有 `/api/bundle/*` 端点，前后端可渐进迁移 |
