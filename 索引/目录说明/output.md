# 目录说明：`output`

- 角色：运行产物、实验结果、GT、frontend bundle、ASR 测试和临时验证输出目录。
- frontend bundle case 数：`6`
- formal GT 进入汇总表的行数：`6`

## 顶层子目录文件数
| child | files |
| --- | --- |
| _verify_reorg | 3 |
| asr_test_1885 | 4 |
| asr_test_22259 | 4 |
| asr_test_25395 | 4 |
| asr_test_26729 | 4 |
| asr_test_45618 | 5 |
| codex_reports | 6546 |
| frontend_bundle | 78 |

## 顶层散落文件
- `_REPORT.md`
- `_test.jsonl`
- `_test_reorg_keypoints.jsonl`
- `llm_evaluation_batch.md`
- `llm_fusion_37_results.json`
- `llm_semantic_fusion_report.md`

## Frontend Bundle 摘要
| bundle case | students | timeline | verified | contract | gt_status |
| --- | --- | --- | --- | --- | --- |
| front_002_A8 | 31 | 45 | 7 | ok | missing |
| front_046_A8 | 37 | 276 | 12 | ok | ok |
| front_1885_sliced | 50 | 1376 | 12 | ok | missing |
| front_22259_sliced | 44 | 690 | 36 | ok | missing |
| front_26729_sliced | 50 | 859 | 22 | ok | missing |
| front_45618_sliced | 45 | 2179 | 24 | ok | missing |

## Rear GT 报告
| gt report | status | video | frames | fps | image_dir |
| --- | --- | --- | --- | --- | --- |
| front_002_rear_gt | ok | 002.mp4 | 8 | 25.0 | F:\PythonProject\pythonProject\YOLOv11\output\codex_reports\rear_gt\gt_frames |
| front_046_rear_gt | ok | 046.mp4 | 42 | 30.0 | F:\PythonProject\pythonProject\YOLOv11\output\codex_reports\rear_gt\front_046_rear_gt\gt_frames |

## ASR 测试目录
- `asr_test_1885`
- `asr_test_22259`
- `asr_test_25395`
- `asr_test_26729`
- `asr_test_45618`
