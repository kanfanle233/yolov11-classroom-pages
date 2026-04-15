# GitHub Pages 部署说明（论文附加 Demo）

## 1) 生成静态数据包

在仓库根目录执行：

```bash
python tools/build_pages_demo.py --docs_root docs --output_root output --clean
```

生成的核心文件：

- `docs/index.html`
- `docs/js/data_source.js`
- `docs/data/demo_cases.json`
- `docs/data/metrics.json`
- `docs/data/timeline.json`
- `docs/data/verified_events.json`
- `docs/assets/cases/*.svg`
- `docs/assets/charts/*.svg`

说明：

- 默认只使用脱敏/合成帧素材（更安全，适合公开仓库）。
- 如需在本地临时挂接已有短视频，可增加 `--include_real_media`。

## 2) 本地预览

```bash
cd docs
python -m http.server 8008
```

打开：

- `http://localhost:8008/`（静态模式）
- `http://localhost:8008/?mode=live&api_base=http://127.0.0.1:8000`（Live API 预留模式）

## 3) GitHub Pages 设置

在 GitHub 仓库中：

1. 进入 **Settings → Pages**
2. Source 选择 **Deploy from a branch**
3. Branch 选择你的发布分支，目录选择 **`/docs`**
4. 保存后等待 Pages 构建完成

## 4) 与后端联动（可选）

如果运行 FastAPI：

```bash
python server/app.py
```

可直接访问：

- `http://127.0.0.1:8000/docs/index.html?mode=live`

该模式用于未来接入在线推理服务；论文录屏建议优先使用静态模式，保证可复现和稳定性。
