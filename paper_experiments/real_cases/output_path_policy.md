# Real Case Output Path Policy

This policy unifies real-case outputs for bridge, validation, demo packaging, and later gold annotation.

## 1) Raw video path

- Source dataset root: `data/智慧课堂学生行为数据集`
- Per-case raw video: `data/智慧课堂学生行为数据集/<view_name>/<stem>.mp4`
- Traceable case id rule: `<view_code>__<stem>`

## 2) Generated case output path

- Unified output root:
  - `output/智慧课堂学生行为数据集/real_pilot/<view_code>/<case_id>/`
- Examples:
  - `output/智慧课堂学生行为数据集/real_pilot/front/front__001/`
  - `output/智慧课堂学生行为数据集/real_pilot/rear/rear__0001/`

Do not scatter generated case folders directly under `output/智慧课堂学生行为数据集`.

## 3) Formal artifact path

Each case formal artifact should live in its own case output directory above, for example:

- `output/智慧课堂学生行为数据集/real_pilot/<view_code>/<case_id>/pipeline_manifest.json`
- `output/智慧课堂学生行为数据集/real_pilot/<view_code>/<case_id>/event_queries.jsonl`
- `output/智慧课堂学生行为数据集/real_pilot/<view_code>/<case_id>/verifier_eval_report.json`

## 4) Demo/package path

Demo/package assets should be generated from the case output directory and stored in dedicated demo/package targets, for example:

- Input side (case artifacts): `output/智慧课堂学生行为数据集/real_pilot/<view_code>/<case_id>/...`
- Package side (example convention): `docs/assets/cases/<case_id>/...`

The package path can evolve, but the upstream case output source must stay under `real_pilot/<view_code>/<case_id>`.

## 5) Gold label path

Gold labels are separate from system outputs:

- Planned gold file: `paper_experiments/gold/gold_events.real.jsonl`

Never treat `verified_events.jsonl` or any system output in case folders as gold labels.
