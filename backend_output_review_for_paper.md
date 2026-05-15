# 后端输出与论文可写性审计报告

## 0. Executive Summary

本次审计的核心判断是，当前后端输出已经能支撑 IEEE VGTC 风格论文的系统设计、方法流程、展示界面和部分案例分析。当前仓库还不能直接支撑强实验结论。主要原因是单事件证据接口未实现，`verified_events` 和 `align_multimodal` 没有被统一成可回溯的 evidence API，ASR 与视觉 fallback 的语义仍然混在同一分数体系中，gold label 规模仍然偏小。

当前最有论文价值的证据来自以下文件。

| 证据 | 路径 | 判断 |
|---|---|---|
| 完整主线输出目录 | `output/codex_reports/front_1885_full`、`output/codex_reports/front_22259_full`、`output/codex_reports/front_26729_full`、`output/codex_reports/front_45618_full` | 可以支撑系统展示和案例分析 |
| 事件验证输出 | `output/codex_reports/front_45618_full/verified_events.jsonl` | 可以展示 `p_match`、`p_mismatch`、`uncertainty` 和 `label` |
| 候选对齐输出 | `output/codex_reports/front_45618_full/align_multimodal.json` | 可以展示自适应窗口和 top-k 视觉候选 |
| pipeline contract | `output/codex_reports/front_45618_full/pipeline_contract_v2_report.json` | 可以支撑工程完整性和可复现性描述 |
| 前端 bundle | `output/frontend_bundle/front_45618_sliced/frontend_data_manifest.json` | 可以支撑静态展示，但字段不完整 |
| Gold 标注 | `paper_experiments/gold/gold_events.real.jsonl` | 可以做 pilot reference，但规模只有 19 条 |
| 论文图表 | `docs/assets/charts/paper_d3_selected/*.png` | 可以作为论文图表草稿 |

本次实际检查发现 35 个候选输出目录含有至少两个关键产物。其中 22 个目录具备完整的 fusion v2、timeline、verified events 和 contract 组合。`front_1885_full` 通过了真实 schema 检查。检查命令覆盖了 `event_queries.fusion_v2.jsonl`、`pose_tracks_smooth_uq.jsonl`、`align_multimodal.json`、`verified_events.jsonl`、`verifier_eval_report.json`、`verifier_calibration_report.json` 和 `pipeline_manifest.json`。

当前最主要的问题是后端还没有论文需要的单事件证据接口。审稿人点击一个事件时，系统应该能返回 query、ASR 质量、对齐候选、视觉分数、文本分数、不确定性、最终标签、源文件和可定位视频帧。现在这些信息分散在多个文件和多个 API 中。前端能够展示一部分聚合字段，但后端没有一个权威的 evidence 返回对象。

最终判断见第 13 节。本报告选择等级 1。当前项目可以开始写正文，但必须补充后端证据接口和 case evidence，才能进入投稿级实验叙事。

## 1. Audit Scope

本次审计基于本地仓库真实文件和真实接口完成。工作区中 `server/app.py` 在审计开始时已经处于修改状态。本报告没有回退任何已有改动。

已重点检查以下文件和目录。

| 范围 | 实际检查路径 | 结果 |
|---|---|---|
| 后端核心 | `server/app.py`、`server/` | 已读取 API、路径解析、文件读取、BFF 和 VSumVis 接口 |
| 前端模板 | `web_viz/templates/index.html`、`web_viz/templates/paper_demo.html`、`web_viz/templates/front_vsumvis.html` | 已检查 API 调用和 evidence 展示 |
| 前端 bundle 构建 | `scripts/frontend/20_build_frontend_data_bundle.py` | 已检查输出字段、manifest 和缺失对象 |
| 静态展示 | `docs/index.html`、`docs/js/data_source.js`、`docs/js/paper_v2_data_source.js`、`docs/data/` | 已检查 GitHub Pages 模式 |
| 前端 bundle | `output/frontend_bundle/` | 已检查 6 个 bundle |
| pipeline 输出 | `output/codex_reports/` | 已扫描 35 个候选输出目录 |
| contract 和 schema | `contracts/schemas.py`、`scripts/utils/02b_check_jsonl_schema.py`、`codex_reports/smart_classroom_yolo_feasibility/scripts/50_fusion_contract/` | 已检查并实际运行一次 schema 检查 |
| 论文实验 | `paper_experiments/`、`scripts/paper/`、`docs/assets/charts/`、`docs/assets/tables/` | 已检查 gold、run logs、图表和表格 |
| 论文材料 | `YOLO论文大纲/`、`YOLO论文/`、`official_yolo_finetune_compare/`、`runs/` | 已抽查大纲、报告和训练对比输出 |

本次实际运行的关键检查包括。

| 检查 | 结果 |
|---|---|
| FastAPI 路由枚举 | 发现 `/api/list_cases`、`/api/v2/vsumvis/*`、`/api/bundle/*`、`/api/v1/visualization/case_data` 等接口 |
| API 样例调用 | `/api/v2/vsumvis/case/front_45618_full` 返回 2179 条 timeline segment、24 条 verified event、24 条 event query |
| bundle 样例调用 | `/api/v1/visualization/case_data?case_id=front_45618_sliced` 返回 2179 条 event、24 条 verified event、48 条 event query |
| schema 检查 | `front_1885_full` 的 7 类 contract 输出全部 PASS |
| ASR 质量扫描 | 多个 SR ablation 和 paper mainline 目录为 `placeholder` 且 accepted 为 0 |

## 2. Backend Architecture Review

`server/app.py` 是一个单文件 FastAPI 后端。它同时承担静态文件服务、视频流服务、旧版 API、bundle API、论文 dashboard API 和 VSumVis API。这个设计便于快速演示，但对论文级证据管理不够清晰。

### 2.1 路径与静态文件挂载

后端在 `server/app.py` 中定义了以下核心路径。

| 路径变量 | 指向 | 证据 |
|---|---|---|
| `PROJECT_ROOT` | 仓库根目录 | `server/app.py` |
| `OUTPUT_DIR` | `output/` | `server/app.py` |
| `DATA_DIR` | `data/` | `server/app.py` |
| `DOCS_DIR` | `docs/` | `server/app.py` |
| `ASSETS_DIR` | `docs/assets/` | `server/app.py` |
| `BUNDLE_DIR` | `output/frontend_bundle/` | `server/app.py` |
| `YOLO_PAPER_DIR` | `YOLO论文/` | `server/app.py` |

后端挂载了 `/output`、`/outputs`、`/data`、`/videos`、`/static`、`/docs`、`/assets`、`/yolo_paper` 和 `/output/frontend_bundle`。这对本地演示友好。它也让前端可以直接读取产物文件。

主要问题是，路径策略混合了本地演示、论文包和静态网站三种目标。`PAPER_MAINLINE_OUTPUT_DIR` 固定指向 `output/codex_reports/run_full_paper_mainline_001/full_integration_001`。这个硬编码可以支持当前演示，但不适合长期复现实验。

### 2.2 Case 查找逻辑

后端有多套 case 查找逻辑。

| 函数 | 作用 | 审计判断 |
|---|---|---|
| `_find_case_dir(video_id)` | 在 `output/`、论文目录和嵌套目录中查找 case | 部分实现。它能找到丰富产物目录，但对同名目录依赖质量分排序 |
| `_find_case_dir_for_analysis(video_id)` | 优先查找 `analysis/bundle.json` | 部分实现。适合旧分析 bundle |
| `_find_bundle(case_id)` | 查找 `output/frontend_bundle/<case_id>` | 实现较稳，限制了 case_id 字符集 |
| `_iter_vsumvis_dirs()` | 从 `output/codex_reports` 和 `output/frontend_bundle` 收集 VSumVis case | 部分实现。会按目录名去重，但 full case 和 sliced bundle 是不同 case |
| `_find_front_case_dir(case_id)` | 查找 VSumVis case | 部分实现。只能按目录名匹配 |

这些函数可以支撑 demo。它们还不能完全支撑论文中的可复现实验，因为同一 case 可能存在 raw output、codex report、frontend bundle 和 paper package 多份副本。后端没有返回统一的 `source_case_dir`、`raw_case_dir`、`bundle_case_dir` 和 `paper_case_dir` 对照表。

### 2.3 缺失文件和错误处理

后端对缺失文件通常使用 fallback 或空返回。

| 行为 | 代码位置 | 判断 |
|---|---|---|
| JSON 文件读取失败返回 default | `server/app.py` 的 `_read_json_file` | 适合 demo，不适合 contract 审计。它会吞掉 JSON 格式错误 |
| JSONL 行解析失败跳过该行 | `_read_jsonl`、`_load_jsonl_rows` | 适合容错，不适合论文复现。应返回 warning |
| `/api/timeline/{video_id}` 缺失时返回 `{"items":[]}` | `get_timeline` | 对前端友好，但会掩盖产物缺失 |
| `/api/transcript/{video_id}` 缺失时返回空数组 | `get_transcript` | 对前端友好，但不利于 ASR 质量解释 |
| `/api/tracks/{video_id}` 缺失时返回空对象 | `get_tracks` | 对前端友好，但不利于追踪失败案例 |
| `/api/case/{video_id}/manifest` 返回文件存在性 | `get_case_manifest` | 较好，适合展示 contract status |

论文演示需要容错。论文审计需要显式错误。建议保留 demo fallback，但新增 `strict=true` 参数或独立 contract API。

### 2.4 JSON 结构清晰度

现有 JSON 可以分为四类。

| 类型 | 代表接口 | 清晰度 | 问题 |
|---|---|---|---|
| 文件清单 | `/api/case/{video_id}/manifest` | 较清晰 | 缺少分组和 schema version 统一说明 |
| bundle 文件 | `/api/bundle/{case_id}/*` | 较清晰 | 缺少 `event_queries` 和 `align_multimodal` |
| 聚合视图 | `/api/v1/visualization/case_data` | 较清晰 | 将 evidence 字段改名为 `c_visual`、`c_text`、`reliability_final` |
| VSumVis 视图 | `/api/v2/vsumvis/case/{case_id}` | 适合展示 | 对 bundle case 不能读 manifest 的 `source_case_dir` 中的 event queries |

最明显的问题是字段命名不统一。后端同时使用 `reliability_score`、`reliability`、`reliability_final`，同时使用 `label`、`match_label`、`verification_status`，同时使用 `visual_score` 和 `c_visual`。这些名字可以在 UI 中映射，但会让论文审稿人难以追溯指标来源。

### 2.5 是否适合论文展示和录屏

当前后端适合做系统录屏。证据如下。

| 展示对象 | 是否已有 | 证据 |
|---|---|---|
| case 列表 | 已有 | `/api/v2/vsumvis/cases`、`/api/list_cases` |
| 时间线 | 已有 | `/api/v2/vsumvis/timeline/{case_id}`、`timeline_chart.json` |
| 事件查询 | 部分实现 | raw case 有 `event_queries.fusion_v2.jsonl`，bundle detail 缺失 |
| top-k 候选 | 部分实现 | `align_multimodal.json` 存在，但没有独立 API |
| 分数展示 | 部分实现 | `front_vsumvis.html` 展示 `P(Match)`、`Reliability`、`Uncertainty` |
| contract 状态 | 已有 | `front_vsumvis.html` 和 `/api/v2/vsumvis/cases` 展示 `contract_status` |
| case comparison | 部分实现 | `/api/v2/vsumvis/ablation/sr` 和 `/api/paper/v2/overview` |
| failure cases | 未实现为真实数据 | bundle 中 `failure_cases.json` 为空 |

当前展示仍然有被审稿人认为是“高级目标检测 demo”的风险。原因是事件证据没有成为后端第一等对象。系统应该把“文本事件 query 触发候选查找，系统给出 match、mismatch 或 uncertain”的过程作为主视图和 API 主线。

## 3. API Inventory

下表基于 `server/app.py` 和 FastAPI TestClient 实际枚举结果整理。

| 接口路径 | 方法 | 输入参数 | 返回字段 | 数据来源文件 | 是否适合论文展示 | 缺字段 | 需要优化 |
|---|---|---|---|---|---|---|---|
| `/` | GET | 无 | HTML | `web_viz/templates/index.html` | 部分适合 | 无事件证据 API | 应突出事件验证 |
| `/paper/v2` | GET | 无 | HTML | `docs/paper_v2_dashboard.html` | 适合图表展示 | 单 case 媒体常为空 | 应和 case bundle 对齐 |
| `/api/paper/v2/overview` | GET | 无 | `run_metric_ci`、`case_pairs`、`figures`、`videos` | `docs/assets/tables/paper_d3_selected`、`docs/assets/charts/paper_d3_selected` | 适合论文图表草稿 | 缺字段来源说明 | 建议拆出 `/api/paper/figures` 和 `/api/paper/metrics` |
| `/api/paper/v2/case/{case_id}` | GET | `case_id` | `case_pair`、`runs`、`media` | paper v2 表格和 `get_media` | 部分适合 | 常见 case 不匹配时 404，`media` 可为 null | 应返回数据来源和可用性 |
| `/api/paper/v2/refresh` | POST | 无 | `ok`、`message` | 缓存 | 工程用途 | 无 | 保留即可 |
| `/api/list_cases` | GET | 无 | `video_id`、`case_id`、`path`、`pipeline_status`、counts | `output/`、paper package | 适合 case 选择 | 没有 canonical case id | 建议新增 `/api/cases` |
| `/api/analysis/list_cases` | GET | 无 | analysis bundle 列表 | `analysis/bundle.json` | 附录用途 | 与主 case 不统一 | 可降级为 legacy |
| `/api/analysis/case/{case_id}` | GET | `case_id` | bundle 原文 | `analysis/bundle.json` | 附录用途 | 无 contract 状态 | 可保留 |
| `/api/analysis/slices/{case_id}` | GET | `case_id` | `meta`、`video_meta`、`slices` | `analysis/bundle.json` | 附录用途 | 无事件验证字段 | 可保留 |
| `/api/media/{video_id}` | GET | `video_id` | `original`、`overlay`、`timeline_png`、`verified_events`、`event_queries`、`actions` | case 输出视频和产物 | 适合录屏 | 缺 frame seek 元数据 | 应加入 case source 和 codec 状态 |
| `/api/media/{video_id}/original` | GET | `video_id` | 文件流 | 原始视频 | 适合录屏 | 无 | 保留 |
| `/api/media/stream/{case_id}` | GET、HEAD | `case_id`、Range header | 视频流或 debug JSON | `docs/assets/videos`、`output/codex_reports`、`output/frontend_bundle` | 适合录屏 | 无事件定位 | 应和 evidence API 返回帧时间对齐 |
| `/api/debug/routes` | GET | 无 | route 列表 | FastAPI app | 工程用途 | 无 | 不进论文展示 |
| `/api/case/{video_id}/manifest` | GET | `video_id` | `files`、`pipeline_status`、`ready_for_frontend` | case 输出目录 | 适合 contract 展示 | 缺分组和 source path 正规化 | 建议改成 `/api/case/{case_id}/contract` |
| `/api/config` | GET | 无 | 路径配置 | 后端变量 | 工程用途 | 暴露本地路径 | 论文 demo 不应公开本机路径 |
| `/api/timeline/{video_id}` | GET | `video_id` | `items` | `timeline_chart.json`、`timeline_viz.json`、`actions*.jsonl`、`verified_events.jsonl` | 部分适合 | 缺 event query、分数、source | 建议新增 canonical timeline API |
| `/api/stats/{video_id}` | GET | `video_id` | `class_pie_chart` | `timeline_chart_stats.json` | 一般 | 输出过弱 | 可放 legacy |
| `/api/transcript/{video_id}` | GET | `video_id` | transcript rows with linked top verified | `transcript.jsonl`、`verified_events.jsonl` | 部分适合 | 没有 ASR quality 状态 | 应合并 `/asr-quality` |
| `/api/tracks/{video_id}` | GET | `video_id` | frame 到 bbox map | `pose_tracks_smooth*.jsonl` | 适合视频 overlay | 缺 UQ 和 student_id | 应支持 frame range |
| `/api/projection/{video_id}` | GET | `method`、`metric` | projection points | timeline 和 tracks | 系统展示可用 | 与论文指标无关 | 保留 |
| `/api/bundle/list` | GET | 无 | bundle 列表 | `output/frontend_bundle/*/frontend_data_manifest.json` | 适合静态 bundle 选择 | 只返回文件名，不返回 schema detail | 增加 schema 和 source summary |
| `/api/bundle/{case_id}/manifest` | GET | `case_id` | manifest | bundle manifest | 适合展示 | 无 raw query 和 align | 补 `source_files` 检查状态 |
| `/api/bundle/{case_id}/timeline` | GET | `case_id` | `students`、`segments` | `timeline_students.json` | 适合静态时间线 | 缺 verified linkage | 补 event_id 或 evidence id |
| `/api/bundle/{case_id}/tracks_sampled` | GET | `case_id` | sampled tracks | `tracks_sampled.json` | 附录和录屏可用 | 缺 source file | 可保留 |
| `/api/bundle/{case_id}/metrics` | GET | `case_id` | `gt_status`、groups | `metrics_summary.json` | 部分适合 | 多数 bundle 为 `gt_status=missing` | 指标不可写成 accuracy 结论 |
| `/api/bundle/{case_id}/ablation` | GET | `case_id` | variants | `ablation_summary.json` | 适合 SR 消融 | 只覆盖部分 case | 需要统一实验来源 |
| `/api/bundle/{case_id}/behavior_segments` | GET | `case_id` | segments | `behavior_segments.json` | 附录可用 | 缺 frame source | 可保留 |
| `/api/bundle/{case_id}/fusion_segments` | GET | `case_id` | segments | `fusion_segments.json` | 附录可用 | 缺 query linkage | 可保留 |
| `/api/bundle/{case_id}/student_id_map` | GET | `case_id` | map | `student_id_map.json` | 适合说明 ID | 无稳定性统计 | 增加 track continuity |
| `/api/bundle/{case_id}/failure_cases` | GET | `case_id` | `items` | `failure_cases.json` | 当前不适合 | 文件为空 | 必须补真实 mismatch 和 uncertain |
| `/api/bundle/{case_id}/verified` | GET | `case_id` | verified events | `verified_events.json` | 部分适合 | 缺 align candidates | 建议改名 `/verified-events` |
| `/api/v1/visualization/case_data` | GET | `case_id` | `case_info`、`events`、`timeline_segments`、`verified_events`、`event_queries` | bundle 和 raw case fallback | 适合 D3 展示 | 没有 per-event source file 和 top-k candidates | 应成为 v2 evidence API 的基础 |
| `/paper/bundle/{case_id}` | GET | `case_id` | HTML | `paper_demo.html`、bundle manifest | 适合静态录屏 | 证据指标弱 | 页面应读取 evidence API |
| `/api/v2/vsumvis/cases` | GET | 无 | cases、total | `output/codex_reports`、`output/frontend_bundle` | 适合展示 | 缺 raw 和 bundle 对照 | 加 source summary |
| `/api/v2/vsumvis/case/{case_id}` | GET | `case_id` | case_info、timeline、verified、queries、feature_rows、projection | raw case 或 bundle | 适合主展示 | bundle case 的 event_queries 可能为 0 | 必须读取 manifest source |
| `/api/v2/vsumvis/clusters/{case_id}` | GET | `case_id` | cluster stats | feature rows | 适合可视分析 | 聚类方法说明不足 | 加参数和 provenance |
| `/api/v2/vsumvis/timeline/{case_id}` | GET | `case_id` | tracks、transcripts、alignments、frame_series | timeline、verified、transcript | 适合时间线 overlay | alignments 是合成的，不是 raw align_multimodal | 应暴露 raw alignment |
| `/api/v2/vsumvis/projection/{case_id}` | GET | `case_id`、`unit`、`method` | points | feature rows | 适合分析视图 | 不返回输入特征说明 | 加 provenance |
| `/api/v2/vsumvis/ablation/sr` | GET | 无 | SR ablations | `sr_ablation_metrics.json` | 适合 SR 消融 | 只支持 SR | 需新增 tracking、fusion、alignment |
| `/api/v2/vsumvis/compare/sr` | GET | `case_id`、`a`、`b` | variant delta、assets | SR ablation case | 适合对比展示 | 只支持 A0 和 A8 语义 | 可泛化 |
| `/api/v2/front/*` | GET | 同上 | VSumVis alias | 同上 | 兼容用途 | 命名重复 | 保留但文档标注 legacy |
| `/paper/front-vsumvis` | GET | 无 | HTML | `front_vsumvis.html` | 当前最适合论文演示 | 依赖聚合字段 | 补 evidence API 后更稳 |

### 建议新增接口清单

| 建议接口 | 当前状态 | 是否建议新增 | 理由 |
|---|---|---|---|
| `GET /api/cases` | 未实现 | 是 | 统一替代 `/api/list_cases`、`/api/bundle/list`、`/api/v2/vsumvis/cases` |
| `GET /api/case/{case_id}/summary` | 未实现 | 是 | 返回 case counts、contract、ASR、label distribution、assets |
| `GET /api/case/{case_id}/timeline` | 部分实现 | 是 | 规范化现有 `/api/timeline` 和 VSumVis timeline |
| `GET /api/case/{case_id}/events` | 未实现 | 是 | 返回事件级列表，不混入全部 timeline segment |
| `GET /api/case/{case_id}/evidence/{event_id}` | 未实现 | 是，P0 | 论文展示最缺的接口 |
| `GET /api/case/{case_id}/alignment/{event_id}` | 未实现 | 是，P0 | 返回 raw `align_multimodal` 的候选和窗口 |
| `GET /api/case/{case_id}/verified-events` | 部分实现 | 是 | 当前只有 bundle verified 和 raw manifest URL |
| `GET /api/case/{case_id}/asr-quality` | 未实现 | 是 | 明确 ASR accepted、placeholder 和 visual fallback |
| `GET /api/case/{case_id}/contract` | 部分实现 | 是 | 从 manifest 中拆出 contract 专用返回 |
| `GET /api/ablation/rear-row-sr` | 部分实现 | 是 | 可映射现有 `/api/v2/vsumvis/ablation/sr` |
| `GET /api/ablation/tracking` | 未实现 | 是 | tracking 是论文实验重点之一 |
| `GET /api/ablation/fusion` | 未实现 | 是 | fusion v2 需要独立消融 |
| `GET /api/ablation/alignment` | 未实现 | 是 | 自适应窗口需要独立实验证据 |
| `GET /api/paper/figures` | 部分实现 | 是 | 从 `/api/paper/v2/overview` 拆出更清晰 |
| `GET /api/paper/metrics` | 部分实现 | 是 | 返回指标表和 provenance |

## 4. Output File Inventory

本次扫描发现 35 个候选输出目录。其中多个目录具备完整主线输出。代表目录包括：

- `output/codex_reports/front_1885_full`
- `output/codex_reports/front_22259_full`
- `output/codex_reports/front_26729_full`
- `output/codex_reports/front_45618_full`
- `output/codex_reports/front_002_full_pose020_hybrid`
- `output/codex_reports/front_002_rear_row_sliced_pose020_hybrid`
- `output/codex_reports/run_full_paper_mainline_001/full_integration_001`

下表中的数量来自本次扫描。`keypoints.jsonl` 和 `tracks.jsonl` 以项目内部别名存在。

| 文件 | 扫描结果 | 证据路径 | 论文价值 | 问题 |
|---|---|---|---|---|
| `keypoints.jsonl` | exact 名称未确认，别名出现 24 个 | `output/codex_reports/front_1885_full/pose_keypoints_v2.jsonl` | 可支撑视觉输入 | 命名与论文清单不一致 |
| `tracks.jsonl` | exact 名称未确认，别名出现 24 个 | `output/codex_reports/front_1885_full/pose_tracks_smooth.jsonl` | 可支撑学生轨迹 | 命名与论文清单不一致 |
| `behavior_det.jsonl` | 24 个 | `output/codex_reports/front_1885_full/behavior_det.jsonl` | 可说明行为检测输入 | 应说明 detector 与 semantic 投影关系 |
| `behavior_det.semantic.jsonl` | 24 个 | `output/codex_reports/front_1885_full/behavior_det.semantic.jsonl` | 可支撑语义行为标签 | 嵌套在 `behaviors` 中 |
| `actions.jsonl` | 24 个 | `output/codex_reports/front_1885_full/actions.jsonl` | 可作为基础动作流 | 论文主文应优先引用 fusion v2 |
| `actions.fusion_v2.jsonl` | 22 个 | `output/codex_reports/front_45618_full/actions.fusion_v2.jsonl` | 可支撑视觉行为候选 | 字段较多，需 schema 摘要 |
| `transcript.jsonl` | 29 个 | `output/codex_reports/front_45618_full/transcript.jsonl` | 可支撑 ASR 文本流 | 有些 case 为 placeholder 或空 |
| `asr_quality_report.json` | 26 个 | `output/codex_reports/front_45618_full/asr_quality_report.json` | 可支撑 ASR gate | 多个 case accepted 为 0 |
| `event_queries.jsonl` | 24 个 | `output/codex_reports/front_45618_full/event_queries.jsonl` | 可支撑文本事件 query | 主文应引用 fusion v2 |
| `event_queries.fusion_v2.jsonl` | 22 个 | `output/codex_reports/front_45618_full/event_queries.fusion_v2.jsonl` | 可支撑事件抽取 | bundle 未复制该文件 |
| `align_multimodal.json` | 24 个 | `output/codex_reports/front_45618_full/align_multimodal.json` | 可支撑自适应窗口和 top-k 候选 | 无独立 API |
| `verified_events.jsonl` | 24 个 | `output/codex_reports/front_45618_full/verified_events.jsonl` | 可支撑三态验证结果 | 和 align candidates 未统一返回 |
| `timeline_chart.json` | 22 个 | `output/codex_reports/front_45618_full/timeline_chart.json` | 可支撑时间线图 | 缺事件证据字段 |
| `timeline_chart.png` | 22 个 | `output/codex_reports/front_45618_full/timeline_chart.png` | 可作附录图 | 主文更适合用可交互 timeline |
| `timeline_students.csv` | 22 个 | `output/codex_reports/front_45618_full/timeline_students.csv` | 可写学生行为时间线 | CSV 缺 `event_id` |
| `student_id_map.json` | 28 个 | `output/codex_reports/front_45618_full/student_id_map.json` | 可说明 track 到 student 映射 | 缺 ID 稳定性统计 |
| `pipeline_contract_v2_report.json` | 22 个 | `output/codex_reports/front_45618_full/pipeline_contract_v2_report.json` | 可支撑工程完整性 | 需要前端可视化 |
| `fusion_contract_report.json` | 22 个 | `output/codex_reports/front_45618_full/fusion_contract_report.json` | 可支撑融合输出完整性 | 与 pipeline contract 重叠 |
| `verifier_eval_report.json` | 24 个 | `output/codex_reports/front_1885_full/verifier_eval_report.json` | 可支撑 verifier 评估 | 必须说明是否基于人工 gold |
| `verifier_calibration_report.json` | 24 个 | `output/codex_reports/front_1885_full/verifier_calibration_report.json` | 可支撑可靠性章节 | reliability bins 样本数偏小 |
| `verifier_reliability_diagram.svg` | 24 个 | `output/codex_reports/front_1885_full/verifier_reliability_diagram.svg` | 可作附录或系统图 | 主文需配样本数说明 |

### 关键输出样例

`output/codex_reports/front_45618_full/align_multimodal.json` 的第一条事件包含 `event_id`、`query_text`、`window`、`basis_motion`、`basis_uq` 和 `candidates`。候选字段包含 `track_id`、`action`、`start_time`、`end_time`、`overlap`、`action_confidence`、`uq_track` 和语义标签。

`output/codex_reports/front_45618_full/verified_events.jsonl` 的第一条事件包含 `event_id=e_000000_00`、`query_text=我们可以再多放上直线`、`track_id=10`、`p_match=0.7132`、`p_mismatch=0.2868`、`reliability_score=0.6603`、`uncertainty=0.3397`、`label=match`。它的 `evidence` 包含 `visual_score=0.9167`、`text_score=0.0`、`uq_score=0.1236`。

这个样例说明后端产物已经有事件级信息。它也暴露了一个风险。真实 ASR 文本存在时，`text_score` 仍然为 0.0。另一个目录 `output/codex_reports/run_full_paper_mainline_001/full_integration_001` 的 ASR quality 是 `placeholder`，但 visual fallback 的 `text_score` 为 1.0。这两种情况都需要在论文中谨慎解释。

## 5. Field Consistency Review

下表检查任务要求的 22 个字段。结论基于 `front_45618_full`、`front_1885_full` 和 `front_45618_sliced` 的真实产物。

| 字段 | 字段来源 | 被哪些模块使用 | 是否缺失 | 是否命名不一致 | 是否需要统一 |
|---|---|---|---|---|---|
| `case_id` | `pipeline_manifest.json`、`frontend_data_manifest.json`、API path | case list、bundle、VSumVis | 不缺 | raw case 和 bundle case 不同名 | 需要。应提供 canonical id 和 display id |
| `video_path` | `pipeline_manifest.json`、`asr_quality_report.json.video`、`/api/media` | media、ASR、录屏 | 部分缺 | `video`、`original`、`overlay` 混用 | 需要。应拆成 original、overlay、browser |
| `frame_id` | `pose_keypoints_v2.jsonl.frame`、`pose_tracks_smooth.jsonl.frame`、`actions.fusion_v2.jsonl.start_frame/end_frame` | tracks、overlay、evidence | 事件层缺 | `frame`、`start_frame`、`end_frame` 混用 | 需要。event evidence 应返回 frame range |
| `timestamp` | `t`、`start_time`、`end_time`、`timestamp`、`t_center`、`query_time` | timeline、queries、verified | 不缺 | 命名明显不统一 | 需要。建议统一 `time_sec`、`start_sec`、`end_sec` |
| `student_id` | `timeline_students.csv`、`student_id_map.json`、bundle timeline | timeline、UI label | 部分缺 | verified event 只有 `track_id` | 需要。verified event 应补 `student_id` |
| `track_id` | tracks、actions、align candidates、verified events、timeline CSV | 全部核心模块 | 不缺 | 类型有 int 和 `S_02` 字符串 | 需要。内部 int，UI label 单独字段 |
| `behavior_code` | behavior semantic、actions、verified、timeline | timeline、legend、feature rows | 不缺 | 无严重问题 | 可统一字典 |
| `semantic_id` | actions、event queries 部分 case、verified events | verifier、UI | 部分缺 | query 有时没有 semantic 字段 | 需要。event query 应补语义字段 |
| `semantic_label_zh` | behavior semantic、actions、verified、timeline | UI、论文案例 | 不缺 | 一些页面显示编码乱码 | 需要修复编码和来源 |
| `semantic_label_en` | behavior semantic、actions、verified、timeline | UI、论文方法 | 不缺 | 无严重问题 | 需要固定 taxonomy |
| `event_id` | event queries、align、verified | evidence trace | 不缺 | 同时有 `query_id` | 需要。`event_id` 为主，`query_id` 为 alias |
| `query_text` | event queries、align、verified、transcript | ASR 展示、verification | 不缺 | visual fallback 和 ASR 文本混用 | 需要。加 `query_source` 和 `is_visual_fallback` |
| `align_candidates` | `align_multimodal.json.candidates` | alignment、top-k 展示 | API 缺 | raw 字段叫 `candidates` | 需要。evidence API 返回 `align_candidates` |
| `visual_score` | `verified_events.evidence.visual_score`、BFF `c_visual` | verifier、parallel coordinates | 不缺 | `visual_score` 和 `c_visual` 混用 | 需要 |
| `text_score` | `verified_events.evidence.text_score`、BFF `c_text` | verifier、parallel coordinates | 不缺但质量存疑 | `text_score` 和 `c_text` 混用 | 需要。还要解释 0 和 1 的来源 |
| `uq_score` | `verified_events.evidence.uq_score`、align candidate `uq_track`、BFF `uq_track` | UQ、alignment | 不缺 | `uq_score`、`uq_track`、`uncertainty` 混用 | 需要拆分 track UQ 和 decision uncertainty |
| `reliability_score` | verified events、BFF `reliability_final` | verifier、UI | 不缺 | `reliability`、`reliability_score`、`reliability_final` | 需要 |
| `p_match` | verified events、BFF feature rows | verifier、UI | 不缺 | 无严重问题 | 保留 |
| `p_mismatch` | verified events | verifier | 不缺 | 前端展示较少 | 需要在 evidence panel 展示 |
| `label` | verified events `label`、`match_label`、BFF `verification_status` | UI、论文案例 | 不缺 | 三种字段名混用 | 需要。主字段用 `label` |
| `evidence` | verified events | verifier evidence | 不缺 | BFF 会拆成 metrics | 需要保留原文和规范化字段 |
| `source file path` | `frontend_data_manifest.source_files`、contract `files.*.path` | 可复现、审计 | 事件层缺 | raw path 和 URL 混用 | 需要。evidence API 返回 source file 和 line index |

字段一致性结论是，核心字段基本存在，但事件层缺少统一 schema。论文需要一个 `event_evidence_v1` 结构。它应把 query、alignment、verified result、timeline segment、track sample、media URL 和 source files 放在一个对象里。

## 6. Evidence Trace Review

当前仓库可以部分回答“哪个学生在什么时候做了什么，以及系统为什么给出 match、mismatch 或 uncertain”。它还不能用一个后端接口完整回答。

### 6.1 已经能追溯的对象

| 对象 | 可追溯情况 | 证据 |
|---|---|---|
| 文本事件 query | 可追溯 | `output/codex_reports/front_45618_full/event_queries.fusion_v2.jsonl` |
| 自适应窗口 | 可追溯 | `output/codex_reports/front_45618_full/align_multimodal.json` |
| top-k 候选 | 可追溯 | `align_multimodal.json.candidates` |
| visual score | 可追溯 | `verified_events.jsonl.evidence.visual_score` |
| text score | 可追溯但含义不稳 | `verified_events.jsonl.evidence.text_score` |
| UQ | 可追溯 | `align_multimodal.json.candidates.uq_track`、`verified_events.jsonl.evidence.uq_score` |
| final label | 可追溯 | `verified_events.jsonl.label` 和 `match_label` |
| p_match 和 p_mismatch | 可追溯 | `verified_events.jsonl` |
| timeline overlay | 可追溯 | `timeline_chart.json`、`timeline_students.csv` |
| source files | 部分可追溯 | `frontend_data_manifest.json.source_files`、`pipeline_contract_v2_report.json.files` |

### 6.2 断点和风险

第一，`verified_events.jsonl` 没有直接包含所有 align candidates。审稿人需要看到系统为什么从 8 个候选中选了某个 track。现在必须额外读取 `align_multimodal.json` 并按 `event_id` 手动关联。

第二，事件层没有 frame range。`align_multimodal.json` 候选只有时间段和 track_id。它没有直接给出原始视频帧、关键帧 URL 或 contact sheet URL。`paper_experiments/gold/annotation_workbench/evidence/*/evidence.json` 具备人工标注证据包，但后端没有把这类证据包接入 case evidence API。

第三，bundle 会丢失 query 和 alignment。`output/frontend_bundle/front_45618_sliced/frontend_data_manifest.json` 没有把 `event_queries`、`align_multimodal` 和 `asr_quality_report` 加入 `files`。`/api/v2/vsumvis/case/front_45618_sliced` 实际返回 `event_queries` 长度为 0。`/api/v1/visualization/case_data?case_id=front_45618_sliced` 能通过 `source_case_dir` fallback 返回 48 条 event query。这说明两个后端展示路径不一致。

第四，`failure_cases.json` 当前为空。路径是 `output/frontend_bundle/front_45618_sliced/failure_cases.json`。这会削弱 Discussion 和 Limitations 中的失败案例展示。

第五，ASR 和 visual fallback 没有被显式分离。`output/codex_reports/run_full_paper_mainline_001/full_integration_001/asr_quality_report.json` 的 `status` 是 `placeholder`，`segments_accepted` 是 0。但同目录 `verified_events.jsonl` 中 visual fallback query 的 `text_score` 是 1.0。另一些真实 ASR case 的 `text_score` 又是 0.0。这个现象必须在论文前修正或解释。

### 6.3 论文级 evidence API 建议结构

建议新增 `GET /api/case/{case_id}/evidence/{event_id}`。最小返回结构如下。

```json
{
  "case_id": "front_45618_full",
  "event_id": "e_000000_00",
  "query": {
    "query_text": "我们可以再多放上直线",
    "query_source": "asr",
    "query_time": 12.5,
    "window_start": 11.34,
    "window_end": 13.66
  },
  "selected": {
    "student_id": "S10",
    "track_id": 10,
    "label": "match",
    "p_match": 0.7132,
    "p_mismatch": 0.2868,
    "reliability_score": 0.6603,
    "uncertainty": 0.3397,
    "visual_score": 0.9167,
    "text_score": 0.0,
    "uq_score": 0.1236
  },
  "align_candidates": [],
  "media": {
    "video_url": "/api/media/stream/front_45618_full",
    "start_sec": 11.34,
    "end_sec": 13.66
  },
  "source_files": {}
}
```

## 7. Pipeline Contract Review

当前 contract 体系是本仓库最强的论文支撑之一。

### 7.1 Schema 和检查脚本

| 文件 | 作用 | 判断 |
|---|---|---|
| `contracts/schemas.py` | 定义 event query、UQ pose、align、verified event、eval report、calibration report、pipeline manifest 的 validator | 适合论文附录 |
| `scripts/utils/02b_check_jsonl_schema.py` | 检查正式 verifier contract artifact | 可用于复现实验 |
| `codex_reports/smart_classroom_yolo_feasibility/scripts/50_fusion_contract/check_pipeline_contract_v2.py` | 检查 pose 到 timeline 的 pipeline contract | 可支撑工程完整性 |
| `codex_reports/smart_classroom_yolo_feasibility/scripts/50_fusion_contract/check_fusion_contract.py` | 检查 fusion v2 核心文件 | 可支撑融合输出完整性 |

`contracts/schemas.py` 明确要求 `verified_events` 包含 `event_id`、`track_id`、`event_type`、`query_text`、`query_time`、`window_start`、`window_end`、`p_match`、`p_mismatch`、`reliability_score`、`uncertainty`、`label`、`threshold_source` 和 `model_version`。这对论文复现非常有价值。

### 7.2 实际 schema 检查结果

本次对 `output/codex_reports/front_1885_full` 运行了 `scripts/utils/02b_check_jsonl_schema.py`。结果如下。

| Artifact | 结果 |
|---|---|
| `event_queries.fusion_v2.jsonl` | PASS，12 rows |
| `pose_tracks_smooth_uq.jsonl` | PASS，6392 rows |
| `align_multimodal.json` | PASS |
| `verifier_samples_train.jsonl` | PASS，48 rows |
| `verified_events.jsonl` | PASS，12 rows |
| `verifier_eval_report.json` | PASS |
| `verifier_calibration_report.json` | PASS |
| `pipeline_manifest.json` | PASS |
| `contracts/examples/*` | PASS |

这个结果说明 contract 输出不是空壳。它可以作为论文 supplementary material 的核心证据。

### 7.3 Contract 风险

| 风险 | 证据 | 影响 |
|---|---|---|
| contract 通过不等于实验有效 | `pipeline_contract_v2_report.json` 检查文件完整性和字段覆盖 | 论文不能把 contract pass 写成准确率结论 |
| ASR placeholder 仍可通过部分 pipeline 展示 | `output/codex_reports/run_full_paper_mainline_001/full_integration_001/asr_quality_report.json` | 需要明确 visual fallback |
| `verifier_eval_report.json` 的 reference 来源需要澄清 | `paper_experiments/gold/gold_events.real.jsonl` 只有 19 条 | 不能夸大 verifier accuracy |
| VSumVis contract status 简化为 OK/WARN/FAIL | `server/app.py` 的 `_front_contract_status` | 非严格报告可能被显示为 OK |

## 8. Frontend Bundle Review

### 8.1 已有 bundle

`output/frontend_bundle/` 下有 6 个可用 bundle。

| bundle | schema | contract | students | source |
|---|---|---|---|---|
| `front_002_A8` | `2026-04-01+frontend_bundle_v1` | ok | 31 | `output/codex_reports/front_002_sr_ablation/A8_adaptive_sliced_artifact_deblur_opencv` |
| `front_046_A8` | `2026-04-01+frontend_bundle_v1` | ok | 37 | `output/codex_reports/front_046_sr_ablation/A8_adaptive_sliced_artifact_deblur_opencv` |
| `front_1885_sliced` | `2026-04-01+frontend_bundle_v1` | ok | 50 | `output/codex_reports/front_1885_full` |
| `front_22259_sliced` | `2026-04-01+frontend_bundle_v1` | ok | 44 | `output/codex_reports/front_22259_full` |
| `front_26729_sliced` | `2026-04-01+frontend_bundle_v1` | ok | 50 | `output/codex_reports/front_26729_full` |
| `front_45618_sliced` | `2026-04-01+frontend_bundle_v1` | ok | 45 | `output/codex_reports/front_45618_full` |

### 8.2 Bundle 优点

`frontend_data_manifest.json` 保存了 `source_case_dir`、`counts`、`files`、`assets`、`students` 和 `source_files`。这对静态展示和复现很有帮助。

`timeline_students.json`、`verified_events.json`、`metrics_summary.json` 和 `ablation_summary.json` 可以直接支撑前端展示。`paper_demo.html` 可以读取 bundle 并画 timeline、metrics、contract 和 SR ablation。

### 8.3 Bundle 缺口

| 问题 | 证据 | 影响 |
|---|---|---|
| bundle 未包含 `event_queries` | `scripts/frontend/20_build_frontend_data_bundle.py` 的 manifest `files` 不包含 event query | 静态展示不能完整解释文本事件 |
| bundle 未包含 `align_multimodal` | 同上 | 静态展示不能展示 top-k 候选 |
| bundle 未包含 `asr_quality_report` | 同上 | 静态展示不能说明 ASR accepted 或 fallback |
| `failure_cases.json` 为空 | `scripts/frontend/20_build_frontend_data_bundle.py` 写入空 `items` | 不能展示失败案例 |
| `tracks_sampled` 在脚本中先被引用后赋值 | `scripts/frontend/20_build_frontend_data_bundle.py` 第 303 行和第 329 行 | 如果 timeline CSV 为空，脚本会出错 |
| `multi_case_data.json` 有乱码字段 | `output/frontend_bundle/multi_case_data.json` 中出现 `鎶婃彙` | 静态展示质量会被审稿人质疑 |
| `front_vsumvis` 的 bundle detail 读不到 query | `/api/v2/vsumvis/case/front_45618_sliced` 返回 `event_queries` 长度为 0 | 同一数据在不同接口不一致 |

### 8.4 GitHub Pages 适配

`docs/js/data_source.js` 和 `docs/data/` 已经支持静态展示。`docs/data/manifest.json`、`docs/data/list_cases.json` 和 `docs/data/cases/*` 存在。这个模式适合 GitHub Pages。

当前 Pages 数据仍然偏旧。它主要包含 timeline、projection、transcript 和 tracks。它没有完整的 `verified_events`、`align_candidates`、`p_match`、`p_mismatch` 和 `contract` 展示。若论文 supplementary 要提供静态站点，需要用新的 `frontend_bundle_v2` 重新导出。

## 9. Paper Readiness Review

### 9.1 可以直接写进论文主文

| 内容 | 可以写什么 | 证据文件路径 | 注意 |
|---|---|---|---|
| 事件级三态验证输出 | 系统输出 `match`、`mismatch`、`uncertain` | `output/codex_reports/front_002_rear_row_sliced_pose020_hybrid/verified_events.jsonl` | 可写机制和案例，不可直接写大规模准确率 |
| 自适应窗口和候选生成 | query 触发时间窗口，系统生成 top-k 候选 | `output/codex_reports/front_45618_full/align_multimodal.json` | 需要补可视化 API |
| 学生行为 timeline | 每个学生的行为片段 | `output/codex_reports/front_45618_full/timeline_students.csv`、`timeline_chart.json` | 主文可放系统截图 |
| Pipeline contract | 输出完整性和字段一致性 | `output/codex_reports/front_45618_full/pipeline_contract_v2_report.json` | 只能证明产物完整，不证明算法有效 |
| Frontend VSumVis 展示 | 多视图系统和 evidence panel 原型 | `web_viz/templates/front_vsumvis.html` | 需要修正 bundle query 缺失 |
| Paper dashboard 图表 | run-level 对比和 figure manifest | `docs/assets/tables/paper_d3_selected/tbl01_run_metric_ci_enhanced.csv`、`docs/assets/charts/paper_d3_selected/*.png` | 应标注数据来源 |
| Gold 标注 protocol | 人工标注流程和证据包 | `paper_experiments/gold/annotation_guideline.real.md`、`paper_experiments/gold/annotation_workbench/tasks.jsonl` | 样本量小，适合 pilot |

### 9.2 可以放在附录或 supplementary material

| 内容 | 证据文件路径 | 适合位置 |
|---|---|---|
| 完整 JSONL 输出 | `output/codex_reports/front_1885_full/*.jsonl` | Supplementary data |
| 完整 frontend bundle | `output/frontend_bundle/front_45618_sliced/` | Demo package |
| Schema 定义 | `contracts/schemas.py`、`docs/formal_verifier_contracts.md` | Appendix |
| Schema 检查脚本 | `scripts/utils/02b_check_jsonl_schema.py` | Reproducibility appendix |
| Contract 检查脚本 | `codex_reports/smart_classroom_yolo_feasibility/scripts/50_fusion_contract/check_pipeline_contract_v2.py` | Supplementary |
| 录屏脚本 | `docs/recording_script.md` | Supplementary |
| 论文图表 manifest | `docs/assets/tables/paper_d3_selected/tbl05_figure_manifest.csv` | Supplementary |
| Gold evidence 图片包 | `paper_experiments/gold/annotation_workbench/evidence/*` | Supplementary |

### 9.3 只能作为工程实现说明

| 内容 | 证据文件路径 | 原因 |
|---|---|---|
| 静态文件挂载 | `server/app.py` | 这是工程服务能力，不是论文贡献 |
| CORS 配置 | `server/app.py` | 只能说明演示系统可访问 |
| 视频 Range streaming | `server/app.py` 的 `/api/media/stream/{case_id}` | 适合系统实现，不适合作算法贡献 |
| 缓存和 fallback | `server/app.py` 的 `lru_cache` 和 `_read_json_file` | 可写实现细节，不能当实验结论 |
| GitHub Pages 旧数据包 | `docs/data/`、`docs/js/data_source.js` | 可作为部署说明，不足以支撑事件验证贡献 |
| `paper_demo.html` 的 timeline demo | `web_viz/templates/paper_demo.html` | 目前更像展示页，不是完整验证系统 |

### 9.4 当前不能写成论文结论

| 不能写的结论 | 证据文件路径 | 原因 |
|---|---|---|
| verifier 在真实课堂上有稳定高 accuracy | `paper_experiments/gold/gold_events.real.jsonl` | 真实 gold 只有 19 条，无法支撑强结论 |
| ASR 是系统主要贡献并显著提升效果 | `output/codex_reports/run_full_paper_mainline_001/full_integration_001/asr_quality_report.json` | paper mainline accepted 为 0，多个 SR case 为 placeholder |
| `p_match` 等同人工标注准确率 | `output/codex_reports/front_45618_full/verified_events.jsonl` | `p_match` 是模型输出或启发式分数，不是 reference metric |
| 当前系统已具备实时部署稳定性 | `paper_experiments/run_logs/` | 有 runtime log，但没有端到端在线稳定性实验 |
| OCR 或板书理解贡献 | 仓库中未见正式 OCR 输出 | 无法确认 |
| mismatch 和 uncertain 覆盖充分 | `front_002_rear_row_sliced_pose020_hybrid/verified_events.jsonl` | 有样例，但数量少 |
| 所有前端图表都来自同一版本数据 | `docs/assets/tables/paper_*`、`output/frontend_bundle/` | 多套数据包并存，来源未统一 |

### 9.5 论文各章节可引用的产物

| 章节 | 可引用产物 | 不应写过头的点 |
|---|---|---|
| Method | `scripts/pipeline/xx_align_multimodal.py`、`scripts/pipeline/07_dual_verification.py`、`contracts/schemas.py` | 不要把启发式 UQ 写成严格贝叶斯不确定性 |
| System | `server/app.py`、`web_viz/templates/front_vsumvis.html`、`output/frontend_bundle/front_45618_sliced/frontend_data_manifest.json` | 不要把静态 demo 写成完整产品 |
| Experiments | `docs/assets/tables/paper_d3_selected/*.csv`、`paper_experiments/real_cases/*.json` | 不要把 proxy 指标写成 gold accuracy |
| Case Study | `output/codex_reports/front_45618_full/verified_events.jsonl`、`align_multimodal.json` | 需要补单事件 evidence API |
| Discussion | `paper_experiments/gold/gold_validation_report.json`、`asr_quality_report.json` | 要主动承认 ASR 和 gold 规模限制 |

## 10. Reviewer Critical Questions

| 问题 | 当前仓库能否回答 | 需要检查哪个文件 | 如果回答不了，应该补什么 |
|---|---|---|---|
| 1. `verified_events` 是否有人工标注作为 reference？ | 部分能回答 | `paper_experiments/gold/gold_events.real.jsonl` | 补更多 peer-reviewed gold，并把 case id 对齐到输出目录 |
| 2. `p_match` 的高分是否只是启发式规则造成的自洽分数？ | 无法确认 | `verifier/infer.py`、`verified_events.jsonl`、`verifier_eval_report.json` | 补基于 gold 的外部评估和 ablation |
| 3. ASR accepted 为 0 时，为什么还能称为文本语义流验证？ | 部分能回答 | `asr_quality_report.json`、`event_queries.fusion_v2.jsonl` | 显式标注 `query_source=visual_fallback`，并分开报告 ASR case |
| 4. timeline 中 `student_id` 是否稳定？ | 部分能回答 | `student_id_map.json`、`pose_tracks_smooth.jsonl`、`student_tracks.jsonl` | 补 track continuity、IDSW 和跨时间稳定性统计 |
| 5. 后端如何证明一个事件的证据可以回溯到原始视频？ | 目前不能完整回答 | `align_multimodal.json`、`verified_events.jsonl`、`/api/media/*` | 新增 evidence API 返回 media URL、frame range 和 source file |
| 6. 如果缺少 `align_multimodal.json`，fallback 是否会掩盖错误？ | 部分能回答 | `server/app.py`、`pipeline_contract_v2_report.json` | 增加 strict mode 和 warning 字段 |
| 7. case bundle 是否只覆盖成功案例？ | 部分能回答 | `output/frontend_bundle/*/frontend_data_manifest.json`、`failure_cases.json` | 补 failure case 选择策略和样例 |
| 8. uncertain 样例是否足够？ | 不能充分回答 | `front_002_rear_row_sliced_pose020_hybrid/verified_events.jsonl` | 补更多 uncertain case 和 coverage 统计 |
| 9. 系统是否能展示 mismatch，而不是只展示 match？ | 部分能回答 | `front_002_rear_row_sliced_pose020_hybrid/verified_events.jsonl` | 在 demo 默认 case 中加入 mismatch evidence |
| 10. 后端输出是否能支持复现实验？ | 部分能回答 | `contracts/schemas.py`、contract reports、paper tables | 增加 canonical case registry 和版本化数据清单 |
| 11. 输出文件是否有 schema version？ | 大部分能回答 | `verified_events.jsonl`、`event_queries.fusion_v2.jsonl`、`frontend_data_manifest.json` | 补 timeline 和 bundle v2 schema |
| 12. 前端展示是否和论文指标来自同一份数据？ | 无法完全确认 | `docs/assets/tables/paper_d3_selected`、`output/frontend_bundle` | 建立 paper artifact manifest，将图表和 demo case 绑定 |
| 13. `text_score=0` 的真实 ASR case 如何解释？ | 不能充分回答 | `output/codex_reports/front_45618_full/verified_events.jsonl` | 说明文本匹配算法，修正或重命名 text score |
| 14. visual fallback 的 `text_score=1` 是否合理？ | 不能充分回答 | `run_full_paper_mainline_001/full_integration_001/verified_events.jsonl` | 将 fallback score 和 ASR score 分开 |
| 15. top-k candidate 的排序依据是什么？ | 部分能回答 | `align_multimodal.json`、`xx_align_multimodal.py` | API 返回 rank、score decomposition 和 selection reason |
| 16. verified event 选择了哪个候选？ | 部分能回答 | `verified_events.jsonl`、`align_multimodal.json` | 在 verified event 中增加 `selected_candidate_rank` |
| 17. contract OK 是否代表结果可信？ | 能回答但需强调 | `pipeline_contract_v2_report.json` | UI 中区分 contract pass 和 metric quality |
| 18. SR ablation 是否影响最终 verification，而不是只影响检测数量？ | 部分能回答 | `sr_ablation_metrics.json`、`ablation_summary.json` | 增加 SR 对 `p_match`、uncertain rate、IDSW 的影响 |
| 19. 后端是否能导出审稿人可复查的单事件包？ | 未实现 | `paper_experiments/gold/annotation_workbench/evidence/*` | 新增 evidence package export |
| 20. 当前图表是否有最小样本量标注？ | 部分能回答 | `docs/assets/tables/paper_curated/tbl04_data_quality_gate.csv` | 所有 figure API 返回 `n` 和 quality tier |

## 11. Optimization Suggestions

| 优先级 | 问题 | 为什么影响论文 | 需要修改的文件 | 建议新增或修改的接口 | 建议新增或修改的输出字段 | 最小可行实现 | 验收标准 |
|---|---|---|---|---|---|---|---|
| P0 | 缺少单事件 evidence API | 审稿人无法看到一次决策的完整证据 | `server/app.py` | 新增 `/api/case/{case_id}/evidence/{event_id}` | `query`、`selected`、`align_candidates`、`media`、`source_files` | 按 `event_id` join query、align、verified、timeline、media | 对 `front_45618_full/e_000000_00` 返回完整证据 |
| P0 | verified events 与 align candidates 没有联动 | 系统无法解释为什么选中某个学生 | `server/app.py`、`scripts/frontend/20_build_frontend_data_bundle.py` | 新增 `/api/case/{case_id}/alignment/{event_id}` | `selected_candidate_rank`、`candidate_count`、`candidate_scores` | 在 API 中按 `event_id` 返回 raw candidates | 前端 evidence panel 显示 top-k 和选中候选 |
| P0 | ASR quality 与 visual fallback 混用 | 文本语义贡献会被审稿人质疑 | `scripts/pipeline/06b_event_query_extraction.py`、`07_dual_verification.py`、`server/app.py` | 新增 `/api/case/{case_id}/asr-quality` | `query_source`、`is_visual_fallback`、`asr_accepted`、`text_score_source` | 输出和 API 分开 ASR query 与 visual fallback query | placeholder case 不再显示为文本证据成功 |
| P0 | mismatch 和 uncertain 展示不足 | 论文会像只展示成功案例 | `scripts/frontend/20_build_frontend_data_bundle.py`、`front_vsumvis.html` | `/api/case/{case_id}/events?label=mismatch` | `failure_reason`、`review_note` | 从 `verified_events` 自动生成 `failure_cases.json` | demo 默认能打开 1 个 mismatch 和 1 个 uncertain |
| P0 | 指标来源不统一 | 论文实验结论可能无法复现 | `scripts/paper/*`、`server/app.py` | 新增 `/api/paper/metrics` | `metric_name`、`source_file`、`sample_size`、`quality_tier` | 读取 figure/table manifest 并返回 provenance | 每个图表和表格都能追溯到 CSV/JSON |
| P1 | case summary 缺失 | 前端多处重复拼装 counts 和 status | `server/app.py` | 新增 `/api/case/{case_id}/summary` | `label_distribution`、`asr_status`、`contract_status` | 从 manifest、contract、ASR 和 verified 聚合 | 一个接口能驱动 case card |
| P1 | canonical API 命名缺失 | 当前接口族太多，论文难描述 | `server/app.py` | 新增 `/api/cases`，保留旧接口 alias | `api_version`、`case_kind` | 包装现有 list API | 文档中只介绍 `/api/cases` |
| P1 | 字段命名不统一 | 论文图和接口样例难对齐 | `server/app.py`、`contracts/schemas.py` | 所有新接口用 `event_evidence_v1` | `visual_score`、`text_score`、`uq_score`、`reliability_score`、`label` | 新增 normalized 层，不改旧文件 | 新 API 不出现 `reliability_final` 和 `verification_status` |
| P1 | bundle schema 过轻 | GitHub Pages 不能展示事件验证核心贡献 | `scripts/frontend/20_build_frontend_data_bundle.py` | 修改 `/api/bundle/*` 或导出 v2 | `event_queries`、`align_multimodal`、`asr_quality`、`schema_version` | 复制或压缩导出 query、align、ASR | `front_45618_sliced` 静态模式也有 24 条 query |
| P1 | 后端错误处理会吞掉损坏文件 | 复现时难发现数据坏行 | `server/app.py` | `strict=true` 参数或 `/api/case/{id}/contract` | `warnings`、`parse_errors` | 读取 JSON/JSONL 时记录错误数量 | 损坏 JSONL 不再静默成功 |
| P1 | contract status 可视化过粗 | OK/WARN 不能说明具体缺失 | `front_vsumvis.html`、`server/app.py` | `/api/case/{id}/contract` | `missing_files`、`error_count`、`warning_count` | 将 contract report 原文摘要返回前端 | UI 能显示缺失文件名 |
| P2 | `paper_demo.html` 展示仍偏普通 timeline demo | 审稿人可能认为只是检测可视化 | `web_viz/templates/paper_demo.html` | 读取 evidence API | `p_match`、`p_mismatch`、`top_k` | 在 timeline 点击后显示证据面板 | 静态 demo 有事件验证解释 |
| P2 | 文档和部分字段有编码乱码 | 影响审稿人观感 | `docs/README_PAGES.md`、`paper_demo.html`、`multi_case_data.json` | 无 | 正常中文 label | 统一 UTF-8 重写受影响文件 | 页面不出现 `鎶婃彙` 或乱码 |
| P2 | ablation API 只覆盖 SR | 论文实验维度不完整 | `server/app.py`、`scripts/paper/*` | `/api/ablation/tracking`、`/api/ablation/fusion`、`/api/ablation/alignment` | `baseline`、`variant`、`delta`、`source_file` | 读取已有 paper tables | dashboard 可切换 4 类 ablation |
| P2 | OpenAPI 示例缺失 | 评审和复现人员不易使用 | `server/app.py`、`docs/` | `/api/docs` 默认即可 | response example | 为新 API 写 Pydantic schema | Swagger 中可见字段说明 |

## 12. Minimal Backend Upgrade Plan

建议用一个小版本完成论文前必须补的后端升级。目标不是重写后端，而是建立事件证据对象。

1. 新增 case registry。

   在 `server/app.py` 中增加 `_resolve_case_context(case_id)`。它返回 `case_id`、`case_dir`、`bundle_dir`、`source_case_dir`、`data_source` 和 `available_files`。它应复用 `_find_case_dir`、`_find_bundle` 和 `_find_front_case_dir`。

2. 新增 normalized readers。

   增加 `_read_event_queries_any(ctx)`、`_read_alignments_any(ctx)`、`_read_verified_events_any(ctx)`、`_read_timeline_any(ctx)` 和 `_read_asr_quality_any(ctx)`。这些 reader 应优先读取 raw case，再读取 bundle manifest 的 `source_case_dir`。

3. 新增 summary API。

   实现 `GET /api/cases` 和 `GET /api/case/{case_id}/summary`。summary 至少包含 `student_count`、`timeline_event_count`、`query_event_count`、`verified_event_count`、`label_distribution`、`asr_status`、`contract_status`、`data_source`。

4. 新增 evidence API。

   实现 `GET /api/case/{case_id}/evidence/{event_id}`。它按 `event_id` join event query、alignment、verified event 和 timeline overlap。它返回 source file path 和 line index。它返回 media URL 和 time range。

5. 升级 frontend bundle。

   修改 `scripts/frontend/20_build_frontend_data_bundle.py`。把 `event_queries.fusion_v2.jsonl`、`align_multimodal.json`、`asr_quality_report.json` 和 contract summary 加入 bundle v2。修复第 303 行 `tracks_sampled` 先引用后赋值的问题。

6. 升级 VSumVis detail。

   修改 `_read_event_queries_front`。当 case_dir 是 bundle 时，它应读取 `frontend_data_manifest.source_case_dir`。同理应补 alignment 和 ASR quality。

7. 增加最小验收测试。

   为 `front_45618_full`、`front_45618_sliced` 和 `front_002_rear_row_sliced_pose020_hybrid` 写 3 个 smoke tests。测试应确认 summary、evidence、alignment、asr-quality 和 contract 都返回非空对象。

## 13. Final Decision

最终等级选择：

1. 可以开始写正文，但需要补充后端证据接口和 case evidence

### 支持写论文的证据

当前仓库已经有完整的多模态 pipeline 产物。`output/codex_reports/front_45618_full` 等目录包含 actions、event queries、alignment、verified events、timeline、student map、contract、verifier eval 和 calibration report。

当前仓库已经有形式化 schema 和 contract 检查。`contracts/schemas.py` 和 `scripts/utils/02b_check_jsonl_schema.py` 能支撑 supplementary material。`front_1885_full` 的实际 schema 检查已经通过。

当前仓库已经有可展示的 VSumVis 前端。`web_viz/templates/front_vsumvis.html` 展示了 case list、contract、timeline、projection、parallel coordinates、evidence panel 和 SR ablation。

当前仓库已经有论文图表和表格草稿。`docs/assets/charts/paper_d3_selected/*.png` 和 `docs/assets/tables/paper_d3_selected/*.csv` 可以支撑写作草稿。

### 阻碍投稿的风险

后端没有单事件 evidence API。审稿人无法通过一个接口看到 query、top-k candidates、selected track、分数、标签、source file 和视频帧。

ASR 和 visual fallback 的分数语义不稳定。`front_45618_full` 的真实 ASR 文本存在，但 `text_score=0.0`。`run_full_paper_mainline_001/full_integration_001` 的 ASR accepted 为 0，但 visual fallback 的 `text_score=1.0`。

人工 gold label 规模不足。`paper_experiments/gold/gold_events.real.jsonl` 只有 19 条，虽然包含 10 match、5 uncertain、4 mismatch，但不足以支撑强 verifier accuracy 结论。

frontend bundle 没有导出 event queries、alignment 和 ASR quality。静态展示无法完整解释事件验证系统。

失败案例展示不足。`failure_cases.json` 当前为空，mismatch 和 uncertain 没有成为默认展示内容。

### 最优先补的三项内容

1. 补 `GET /api/case/{case_id}/evidence/{event_id}`。这个接口必须能把 `event_queries`、`align_multimodal`、`verified_events`、timeline、media 和 source files 串起来。

2. 补 `frontend_bundle_v2`。新 bundle 必须包含 `event_queries`、`align_multimodal`、`asr_quality_report`、contract summary 和非空 `failure_cases`。

3. 明确 ASR 和 fallback 的评分规则。新字段必须区分 `query_source=asr` 和 `query_source=visual_fallback`，并且不能把 fallback 的 `text_score` 当作 ASR 语义证据。

## 14. Appendix: Checked Files

### 后端文件

- `server/app.py`
- `server/test_range.bin`

### 前端文件

- `web_viz/templates/index.html`
- `web_viz/templates/paper_demo.html`
- `web_viz/templates/front_vsumvis.html`
- `docs/index.html`
- `docs/paper_v2_dashboard.html`
- `docs/js/data_source.js`
- `docs/js/paper_v2_data_source.js`
- `docs/README_PAGES.md`

### Bundle 构建和输出

- `scripts/frontend/20_build_frontend_data_bundle.py`
- `output/frontend_bundle/multi_case_data.json`
- `output/frontend_bundle/front_002_A8/frontend_data_manifest.json`
- `output/frontend_bundle/front_046_A8/frontend_data_manifest.json`
- `output/frontend_bundle/front_1885_sliced/frontend_data_manifest.json`
- `output/frontend_bundle/front_22259_sliced/frontend_data_manifest.json`
- `output/frontend_bundle/front_26729_sliced/frontend_data_manifest.json`
- `output/frontend_bundle/front_45618_sliced/frontend_data_manifest.json`
- `output/frontend_bundle/front_45618_sliced/metrics_summary.json`
- `output/frontend_bundle/front_45618_sliced/failure_cases.json`

### Pipeline 输出样例

- `output/codex_reports/front_1885_full/pose_keypoints_v2.jsonl`
- `output/codex_reports/front_1885_full/pose_tracks_smooth.jsonl`
- `output/codex_reports/front_1885_full/pose_tracks_smooth_uq.jsonl`
- `output/codex_reports/front_1885_full/event_queries.fusion_v2.jsonl`
- `output/codex_reports/front_1885_full/align_multimodal.json`
- `output/codex_reports/front_1885_full/verified_events.jsonl`
- `output/codex_reports/front_1885_full/pipeline_contract_v2_report.json`
- `output/codex_reports/front_1885_full/verifier_eval_report.json`
- `output/codex_reports/front_1885_full/verifier_calibration_report.json`
- `output/codex_reports/front_45618_full/actions.fusion_v2.jsonl`
- `output/codex_reports/front_45618_full/event_queries.fusion_v2.jsonl`
- `output/codex_reports/front_45618_full/align_multimodal.json`
- `output/codex_reports/front_45618_full/verified_events.jsonl`
- `output/codex_reports/front_45618_full/asr_quality_report.json`
- `output/codex_reports/front_45618_full/timeline_students.csv`
- `output/codex_reports/front_45618_full/student_id_map.json`
- `output/codex_reports/front_002_rear_row_sliced_pose020_hybrid/verified_events.jsonl`
- `output/codex_reports/run_full_paper_mainline_001/full_integration_001/asr_quality_report.json`
- `output/codex_reports/run_full_paper_mainline_001/full_integration_001/verified_events.jsonl`

### Contract 和 schema

- `contracts/schemas.py`
- `scripts/utils/02b_check_jsonl_schema.py`
- `codex_reports/smart_classroom_yolo_feasibility/scripts/50_fusion_contract/check_pipeline_contract_v2.py`
- `codex_reports/smart_classroom_yolo_feasibility/scripts/50_fusion_contract/check_fusion_contract.py`
- `docs/formal_verifier_contracts.md`

### 论文实验和图表

- `paper_experiments/gold/gold_events.real.jsonl`
- `paper_experiments/gold/gold_validation_report.json`
- `paper_experiments/gold/splits.real.json`
- `paper_experiments/gold/annotation_packet_manifest.json`
- `paper_experiments/gold/annotation_workbench/tasks.jsonl`
- `paper_experiments/gold/annotation_workbench/evidence/`
- `docs/assets/tables/paper_d3_selected/tbl01_run_metric_ci_enhanced.csv`
- `docs/assets/tables/paper_d3_selected/tbl05_figure_manifest.csv`
- `docs/assets/tables/paper_curated/tbl04_data_quality_gate.csv`
- `docs/assets/charts/paper_d3_selected/fig01_batch_metric_ci.png`
- `docs/assets/charts/paper_d3_selected/fig02_case_slope_mainline_vs_behavior.png`
- `docs/assets/charts/paper_d3_selected/fig03_score_hist_density_runs.png`
- `docs/assets/charts/paper_d3_selected/fig04_case_delta_heatmap_top20.png`
- `docs/assets/charts/paper_d3_selected/fig05_epoch_three_lines.png`
- `docs/assets/charts/paper_d3_selected/fig06_noise_robustness_curve.png`
- `docs/assets/charts/paper_d3_selected/fig07_reliability_bins_ece.png`
- `docs/assets/charts/paper_d3_selected/fig08_latency_vs_duration_scatter.png`
- `official_yolo_finetune_compare/reports/run_comparison.json`
- `runs/detect/*/results.csv`

