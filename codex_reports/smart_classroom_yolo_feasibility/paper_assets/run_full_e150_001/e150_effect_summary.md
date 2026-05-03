# E150 Effect Summary

## Detection Model

- model: `runs/detect/official_yolo11s_detect_e150_v1/weights/best.pt`
- validation summary from final validation:
  - `mAP50 = 0.933`
  - `mAP50-95 = 0.804`
  - `precision = 0.887`
  - `recall = 0.894`

Compared with `wisdom8_yolo11s_detect_v1` (`mAP50-95 = 0.79836`), the e150 detector improves the main detection metric by about `+0.0056`.

## Full Pipeline Output

- output_dir: `output/codex_reports/run_full_e150_001/full_integration_001`
- behavior detections: `408` frame rows
- behavior actions: `137`
- behavior-augmented actions: `212`
- verified events: `1`

Compared with `run_full_wisdom8_001`:

- behavior actions: `131 -> 137`
- behavior-augmented actions: `206 -> 212`
- behavior detections rows: `408 -> 408`
- verified events: `1 -> 1`

## Important Limitation

The current sample still has empty ASR:

- transcript text: `[ASR_EMPTY:empty_whisper_result]`
- event queries: placeholder-level event query
- verifier evaluation total: `1`

Therefore, the full multimodal verification result should be described as a chain-level feasibility result, not as a statistically meaningful text-vision improvement.

## Paper Figures Collected

All figure files are copied under:

`codex_reports/smart_classroom_yolo_feasibility/paper_assets/run_full_e150_001`

Recommended figures for the paper:

- `e150_results.png`: training loss and metric curves
- `e150_confusion_matrix.png`: class confusion
- `e150_confusion_matrix_normalized.png`: normalized class confusion
- `e150_BoxPR_curve.png`: precision-recall curve
- `e150_BoxF1_curve.png`: F1-confidence curve
- `e150_val_batch0_pred.jpg`, `e150_val_batch1_pred.jpg`, `e150_val_batch2_pred.jpg`: qualitative detection examples
- `e150_timeline_chart.png`: system timeline visualization
- `e150_verifier_reliability_diagram.svg`: verifier calibration visualization

Manifest:

`paper_image_manifest.csv`

