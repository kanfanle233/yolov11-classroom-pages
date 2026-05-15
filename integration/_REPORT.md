# integration/ 目录 — 完整维护报告

> **位置**：`F:\PythonProject\pythonProject\YOLOv11\integration\`
> **报告生成日期**：2026-05-03

## 一、大致背景

`integration/` 是项目的**工程化集成入口层**。当系统需要被部署为服务或以批处理模式运行时，通过本目录的脚本提供统一的对外接口。它与 `scripts/main/09_run_pipeline.py` 形成"算法入口 vs 工程入口"的双入口架构。

## 二、位置与目录结构

```
integration/
├── README.md             # 集成模块说明
├── run_pipeline.py       # 流水线集成运行器（主入口）
└── run_server.py         # Web服务启动器
```

## 三、是干什么的

| 文件 | 功能 |
|------|------|
| `run_pipeline.py` | **集成流水线入口**：包装了完整的端到端处理流程，提供更工程化的参数接口和错误处理，适合批量处理和生产环境部署 |
| `run_server.py` | **Web服务启动器**：启动 `server/app.py` 的Web服务，提供对外HTTP API接口 |

### 与 `scripts/main/09_run_pipeline.py` 的区别

| 维度 | `scripts/main/09_run_pipeline.py` | `integration/run_pipeline.py` |
|------|----------------------------------|-------------------------------|
| 定位 | 算法开发入口 | 工程化部署入口 |
| 调用方式 | 直接Python脚本调用 | 更完善的环境配置和错误处理 |
| 参数接口 | 面向算法实验 | 面向生产环境 |
| 下游 | 开发者直接调试 | 被调度系统/CI调用 |

## 四、有什么用

1. **生产部署**：作为对外统一入口，隐藏内部算法细节
2. **批处理**：适合多视频批量处理的场景
3. **服务化**：配合 `server/app.py` 提供Web API
4. **隔离层**：内部算法变更不影响外部调用方

## 五、维护注意事项

- 随着流水线步骤增加（如新增MLLM验证），需同步更新 `run_pipeline.py` 的参数和步骤编排
- 与 `scripts/main/09_run_pipeline.py` 保持功能同步但接口可以不同（工程化 vs 算法化）
- 当前 `报告.md` 建议"二选一保留统一入口"，但尚未实施，双入口暂时并存
