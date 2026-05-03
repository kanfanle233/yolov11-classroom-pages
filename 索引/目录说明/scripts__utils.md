# 目录说明：`scripts/utils`

- 代码文件数：`6`
- 代码行数（LOC）：`995`

## 目录职责
- 入口脚本职责：Validate formal verifier contract artifacts.
- 入口脚本职责：Build verifier training samples for exp-c negative sampling.

## 入口脚本
- `scripts/utils/02b_check_jsonl_schema.py` (line 156)
- `scripts/utils/12_export_features.py` (line 190)
- `scripts/utils/13_semantic_projection.py` (line 162)
- `scripts/utils/build_verifier_training_samples.py` (line 63)

## 上游依赖（静态线索）
- 自动扫描未命中固定规则

## 下游产物（静态线索）
- 自动扫描未命中固定规则

## 文件清单
| 文件 | LOC | 入口 | 说明 |
| --- | --- | --- | --- |
| 02b_check_jsonl_schema.py | 157 | yes | Validate formal verifier contract artifacts. |
| 12_export_features.py | 191 | yes |  |
| 13_semantic_projection.py | 163 | yes |  |
| build_verifier_training_samples.py | 64 | yes | Build verifier training samples for exp-c negative sampling. |
| object_evidence_mapping.py | 82 | no |  |
| sliced_inference_utils.py | 338 | no |  |
