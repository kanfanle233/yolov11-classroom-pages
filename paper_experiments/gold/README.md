# Gold Label 目录说明

本目录用于存放**真实 held-out 金标协议与样例模板**，不包含任何模型运行结果。

## 文件说明

- `gold_events.schema.json`
  - `gold_events` 单条标注记录的字段约束。
- `gold_events.sample.jsonl`
  - 标注模板样例，仅演示字段填写方式。
  - 样例中的 `gold_*` 可为空，表示待人工标注。
- `splits.sample.json`
  - case/video 级划分样例，展示防泄漏 split 结构。
- `annotation_checklist.md`
  - 人工标注与质检执行清单。

## 使用原则

1. 禁止把 sample/smoke 数据当作正式金标。
2. 必须按 `case_id/video_id` 级别划分 train/dev/test。
3. fallback 样本在主结果中需单独处理并披露。
4. 真实发布前必须完成脱敏与隐私审查。

## 交付边界

本目录仅提供协议与模板：
- 不生成模型输出
- 不自动补标签
- 不声明实验性能结论
