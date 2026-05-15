# 人工标注执行清单（Gold Label）

> 用途：确保 A/B/C/E 主实验与 demo 案例使用同一套可追溯金标流程。

## A. 标注前检查

- [ ] 已确认本批次为真实 held-out 样本，不含 sample/smoke 测试数据。
- [ ] 已加载最新 `splits.sample.json` 对应的正式 split 配置。
- [ ] 已确认当前样本所属 split（train/dev/test/demo）。
- [ ] 已确认 train 与 test 的 `case_id/video_id` 无重叠。

## B. 单条样本标注

- [ ] `case_id/video_id/query_id/track_id` 填写完整。
- [ ] `event_start/event_end` 与视频时轴一致。
- [ ] `query_text` 与标注任务一致，无语义歧义。
- [ ] 填写 `gold_action`（若无法判定需备注原因）。
- [ ] 填写 `gold_match`（匹配/不匹配）。
- [ ] 填写 `gold_alignment_valid`。
- [ ] 填写 `gold_candidate_track_id`（无候选可设 null）。
- [ ] 评估并填写噪声等级：`visual_noise_level`、`occlusion_level`、`asr_noise_level`、`time_shift_level`。
- [ ] 填写 `fallback_allowed`（是否允许该样本在推理时触发 fallback）。
- [ ] 更新 `annotation_quality`。

## C. 双人复核与仲裁

- [ ] dev/test 至少双人复核。
- [ ] 冲突样本进入仲裁并记录原因。
- [ ] 仲裁后 `annotation_quality` 更新为 `adjudicated` 或 `final`。

## D. fallback 主结果规则核对

- [ ] 主结果统计仅使用 `fallback_allowed=false` 且未触发 fallback 的样本。
- [ ] fallback 触发样本单独汇总 auxiliary 指标。
- [ ] 报告中明确 primary 与 auxiliary 的区别。

## E. 隐私与发布前检查

- [ ] 样本文本与元数据完成脱敏（人名、学号、可识别身份信息）。
- [ ] demo 候选样本完成单独隐私复核。
- [ ] 对外发布版本移除不必要的内部备注与标注员隐私信息。

## F. 交付完成条件

- [ ] 目标 split 的 `gold_*` 字段在 dev/test 中无空值。
- [ ] 通过 schema 校验。
- [ ] 完成泄漏检查记录。
- [ ] 可复现实验配置（split + 标签版本）已冻结。
