# 真实课堂人工金标标注指南（Pilot 5 Case）

## 0. 边界声明

- `annotation_candidates` 只是候选清单，不是 gold。
- `verified_events.jsonl` 是系统输出，不是 gold。
- 本标注包仅用于人工填写模板，不产生论文最终结论。

## 1. 标注输入与输出

- 输入：
  - `paper_experiments/real_cases/pilot_cases.json`
  - `output/智慧课堂学生行为数据集/real_pilot/<view_code>/<case_id>/`
  - `paper_experiments/gold/gold_events.real.template.jsonl`
- 输出（人工填写后）：
  - `paper_experiments/gold/gold_events.real.jsonl`（新文件，人工生成）
  - `paper_experiments/gold/splits.real.json`（新文件，人工确认）

## 2. 字段规则（必须遵守）

- `suggested_*` 字段是系统建议，仅供参考：
  - `suggested_event_id`
  - `suggested_track_id`
  - `suggested_action`
  - `suggested_time_window`
  - `suggested_label`
- 人工字段才是 gold：
  - `gold_track_id`
  - `gold_action`
  - `gold_start`
  - `gold_end`
  - `match_label`
  - `noise_type`
  - `occlusion_level`
  - `asr_quality`
  - `annotator`
  - `review_status`

## 3. match / mismatch / uncertain 判定

- `match`：事件文本/语义与可见行为在时间窗内一致，主体可定位。
- `mismatch`：文本与行为明显不一致，或主体错误、时间窗明显错位。
- `uncertain`：证据不足（遮挡、多人重叠、ASR 弱/空、时间边界不稳定）。

建议：无法稳定判断时优先标为 `uncertain`，并在 `noise_type` / 备注记录原因。

## 4. ASR 为空时怎么标

- 当 `transcript.jsonl` 为空或事件由占位文本触发时：
  - `asr_quality` 设为 `empty`
  - 不要因为系统建议强行标成 `match`
  - 若视觉证据也不足，`match_label` 设为 `uncertain`
  - 在 `noise_type` 标记 `asr_empty`（或 `other`+备注）

## 5. 遮挡、噪声、多人歧义记录

- `occlusion_level`：
  - `none` / `low` / `medium` / `high`
- `noise_type` 常用值：
  - `none`
  - `asr_empty`
  - `asr_noise`
  - `occlusion`
  - `multi_person_ambiguity`
  - `camera_angle`
  - `motion_blur`
  - `other`
- 多人歧义时优先保证“不误标”：
  - 无法唯一定位主体时，`match_label=uncertain`
  - `noise_type=multi_person_ambiguity`

## 6. 复核与状态

- `annotator` 填写标注人 ID。
- `review_status` 流转：
  - `draft` -> `peer_reviewed` -> `adjudicated`
- 未通过复核可设 `rejected` 并保留原因。

## 7. 禁止事项

- 禁止把系统预测直接复制为 gold。
- 禁止把 `verified_events.jsonl` 当作人工真值。
- 禁止在证据不足时伪造明确标签。
