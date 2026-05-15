# 目录说明：`codex_reports/smart_classroom_yolo_feasibility/scripts/50_fusion_contract`

- 代码文件数：`9`
- 代码行数（LOC）：`1584`

## 目录职责
- 入口脚本职责：Convert semantic behavior detections to semantic action rows.
- 入口脚本职责：Merge ASR event queries with visual fallback queries for fusion v2.
- 入口脚本职责：Validate fusion_contract_v2 outputs.
- 入口脚本职责：Validate the paper pipeline contract from pose to timeline.

## 入口脚本
- `codex_reports/smart_classroom_yolo_feasibility/scripts/50_fusion_contract/behavior_det_to_actions_v2.py` (line 261)
- `codex_reports/smart_classroom_yolo_feasibility/scripts/50_fusion_contract/build_event_queries_fusion_v2.py` (line 140)
- `codex_reports/smart_classroom_yolo_feasibility/scripts/50_fusion_contract/check_fusion_contract.py` (line 107)
- `codex_reports/smart_classroom_yolo_feasibility/scripts/50_fusion_contract/check_pipeline_contract_v2.py` (line 289)
- `codex_reports/smart_classroom_yolo_feasibility/scripts/50_fusion_contract/merge_fusion_actions_v2.py` (line 230)
- `codex_reports/smart_classroom_yolo_feasibility/scripts/50_fusion_contract/run_full_fusion_v2.py` (line 134)
- `codex_reports/smart_classroom_yolo_feasibility/scripts/50_fusion_contract/semanticize_behavior_det.py` (line 95)
- `codex_reports/smart_classroom_yolo_feasibility/scripts/50_fusion_contract/semanticize_objects.py` (line 61)

## 上游依赖
- 自动扫描未命中固定规则

## 下游产物
- 自动扫描未命中固定规则

## 文件清单
| file | LOC | entry | description |
| --- | --- | --- | --- |
| behavior_det_to_actions_v2.py | 262 | yes | Convert semantic behavior detections to semantic action rows. |
| build_event_queries_fusion_v2.py | 141 | yes | Merge ASR event queries with visual fallback queries for fusion v2. |
| check_fusion_contract.py | 108 | yes | Validate fusion_contract_v2 outputs. |
| check_pipeline_contract_v2.py | 290 | yes | Validate the paper pipeline contract from pose to timeline. |
| fusion_utils.py | 259 | no |  |
| merge_fusion_actions_v2.py | 231 | yes | Merge pose/rule, object, and semantic behavior actions into actions.fusion_v2.jsonl. |
| run_full_fusion_v2.py | 135 | yes | Run the full fusion_contract_v2 classroom pipeline. |
| semanticize_behavior_det.py | 96 | yes | Add canonical 8-class semantics to behavior_det.jsonl. |
| semanticize_objects.py | 62 | yes | Normalize classroom object detections for fusion evidence. |
