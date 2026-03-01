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
