# server/ 与 tools/ 与 web_viz/ 与 索引/ 与 experiments/ 目录 — 完整维护报告

> **位置**：`F:\PythonProject\pythonProject\YOLOv11\{server,tools,web_viz,索引,experiments}\`
> **报告生成日期**：2026-05-03

---

## server/ — Web服务

### 大致背景
对外提供HTTP API的Flask/FastAPI Web服务，使流水线功能可以通过REST接口调用。

### 目录结构
```
server/
└── app.py          # Web应用主文件
```

### 是干什么的
- 启动Web服务，对外暴露流水线的HTTP API
- 可通过 `integration/run_server.py` 或直接 `python server/app.py` 启动
- 前端页面(`web_viz/`)通过此API获取数据

### 维护注意
- 依赖 `verifier/infer.py` 进行在线推理
- 需要确认端口配置和CORS设置
- 生产部署时需考虑并发处理和请求超时

---

## tools/ — 辅助工具

### 大致背景
项目构建和外部依赖管理的辅助工具。

### 目录结构
```
tools/
├── build_pages_demo.py    # Pages Demo构建脚本
└── fetch_d3_vendor.ps1    # D3.js vendor文件下载脚本
```

### 是干什么的
- `build_pages_demo.py` — 构建文档网站的Demo页面（Exp F实验产物）
- `fetch_d3_vendor.ps1` — 下载D3.js库文件到 `docs/vendor/d3/`

### 维护注意
- 这两个是低频使用的工具，不要在这里放核心业务逻辑
- `fetch_d3_vendor.ps1` 是PowerShell脚本，仅Windows环境可用

---

## web_viz/ — Web可视化模板

### 大致背景
课堂行为分析结果的Web前端可视化页面。

### 目录结构
```
web_viz/
└── templates/
    ├── index.html          # 主页面
    └── paper_demo.html     # 论文Demo页面
```

### 是干什么的
- 提供交互式网页界面展示分析结果
- 对接 `server/app.py` 提供的API
- `paper_demo.html` 是论文演示专用页面

### 维护注意
- HTML模板应与 `server/app.py` 的API路径保持一致
- 前端JS依赖（如D3.js）来自 `docs/vendor/d3/`
- 数据通过 `scripts/frontend/20_build_frontend_data_bundle.py` 生成

---

## 索引/ — 索引刷新

### 大致背景
项目索引刷新工具。

### 目录结构
```
索引/
└── refresh_index.py     # 索引刷新脚本
```

### 是干什么的
- 刷新项目的文件索引，可能是为了方便搜索和导航
- 具体功能需查看脚本源码确认

### 维护注意
- 这是一个单一工具脚本，功能相对独立
- 目录名是中文，在跨平台开发时需注意编码问题

---

## experiments/ — 独立评估实验

### 大致背景
与 `scripts/experiments/` 不同，这里的脚本是独立于流水线主线的评估实验。

### 目录结构
```
experiments/
├── eval_alignment.py                    # 对齐质量评估
└── eval_reliability_calibration.py     # 可靠性校准评估
```

### 是干什么的
- `eval_alignment.py` — 评估多模态对齐的质量
- `eval_reliability_calibration.py` — 评估可靠性校准效果（论文Exp B相关）

### 维护注意
- 与 `verifier/calibration.py` 和 `verifier/eval.py` 功能可能有重叠
- 需要确认这些脚本的输入数据来源（来自 `paper_experiments/` 目录）
