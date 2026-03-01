# GitHub Pages 静态离线数据包模式

## 1) 生成 demo 静态数据

在项目根目录执行（示例）：

```bash
python tools/build_pages_demo.py \
  --demo_root "output/智慧课堂学生行为数据集/_demo_web" \
  --docs_root "docs" \
  --views "后方视角,教师视角,正方视角" \
  --limit 8 \
  --clean
```

执行后会生成：

- `docs/data/manifest.json`
- `docs/data/list_cases.json`
- `docs/data/cases/<video_id>/*`
- `docs/assets/videos/<video_id>.mp4`

## 2) 前端读取方式

前端已支持 `DataSource` 自动切换：

- 如果存在 `docs/data/manifest.json`：使用静态模式
- 否则：回退后端 API 模式

## 3) 发布到 GitHub Pages

- 在 GitHub 仓库 Settings -> Pages
- Source 选择 `Deploy from a branch`
- Branch 选择 `main` / `docs`

> 建议只放少量 demo case，避免仓库体积过大。


## 4) 最小可发布清单（3~5 个 case）

目标：把仓库做成“演示仓”，只发布少量可稳定展示的样本。

### 推荐样本组合

- 后方视角（rear）1 个：优先包含 `*_overlay.mp4 + timeline_viz.json + student_projection.json`
- 正方视角（front）1 个：至少包含 `*_overlay.mp4 + timeline_viz.json`
- 教师视角（teacher）1 个：可用于验证“无视频兜底”
- 斜上视角（top1/top2）可选 1~2 个

### 打包命令（精确指定 case）

> 先从 `_demo_web` 中确认你的 `video_id`（例如 `rear__0001`、`front__0003`）。

```bash
python tools/build_pages_demo.py \
  --demo_root "output/智慧课堂学生行为数据集/_demo_web" \
  --docs_root "docs" \
  --case_ids "rear__0001,front__0003,teacher__0002,top1__0004" \
  --clean
```

如果你只想按视角自动取前 N 个，可继续用：

```bash
python tools/build_pages_demo.py \
  --demo_root "output/智慧课堂学生行为数据集/_demo_web" \
  --docs_root "docs" \
  --views "后方视角,教师视角,正方视角" \
  --limit 5 \
  --clean
```

### 验收（发布前必须过）

- `docs/data/manifest.json` 存在且 `cases` 数量 = 你期望的 3~5。
- `docs/data/list_cases.json` 可被页面读取。
- `docs/data/cases/<video_id>/timeline_viz.json` 存在。
- 有视频的 case：`docs/assets/videos/<video_id>.mp4` 存在。
- 本地静态服务验证：

```bash
cd docs
python -m http.server 8008
```

打开 `http://localhost:8008/`，确认不会依赖 `/api/*` 才可发布到 Pages。
