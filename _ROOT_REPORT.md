# YOLOv11项目根目录 — 完整维护报告

> **位置**：`F:\PythonProject\pythonProject\YOLOv11\`
> **项目名称**：YOLOv11 Classroom Behavior Analysis Pipeline（智慧课堂行为分析流水线）
> **报告生成日期**：2026-05-03

## 一、大致背景

这是一个**基于YOLOv11的课堂学生行为多模态分析系统**，用于从多视角课堂视频中自动识别学生行为、群体交互模式，并进行时间线可视化。项目的工程目标是搭建端到端流水线，学术目标是产出基于"视觉语义双重验证框架"的学术论文。

### 为什么存在这个项目

- 传统课堂行为分析依赖人工观察，效率低、主观性强
- 需要自动化手段从多视角视频中提取学生行为数据
- 支撑学术论文的实验数据生成和创新算法验证

## 二、位置与结构

位于 `F:\PythonProject\pythonProject\YOLOv11\`，包含以下一级子目录和文件：

### 直接子目录（21个）

| 目录 | 用途 | 重要性 |
|------|------|--------|
| `contracts/` | 数据契约定义（JSON Schema + 样例） | ⭐⭐ |
| `data/` | 原始数据集（视频、标注、图像） | ⭐⭐⭐ |
| `docs/` | 文档中心（报告、图表、案例数据） | ⭐⭐⭐ |
| `models/` | 自定义YOLO模型组件 | ⭐⭐⭐ |
| `scripts/` | 核心流水线脚本（最大目录） | ⭐⭐⭐ |
| `integration/` | 流水线集成入口 | ⭐⭐⭐ |
| `verifier/` | 双重验证系统（可训练） | ⭐⭐⭐ |
| `experiments/` | 独立评估实验脚本 | ⭐⭐ |
| `tools/` | 辅助工具 | ⭐ |
| `server/` | Web服务 | ⭐⭐ |
| `web_viz/` | Web可视化模板 | ⭐ |
| `codex_reports/` | Codex编排报告与脚本 | ⭐⭐ |
| `official_yolo_finetune_compare/` | 官方YOLO微调对比实验 | ⭐⭐ |
| `paper_experiments/` | 论文实验数据（配置、日志、Gold标注） | ⭐⭐⭐ |
| `runs/` | YOLO训练运行产物 | ⭐⭐ |
| `output/` | 流水线运行产物 | ⭐⭐ |
| `exp10-new/` | Exp10实验运行记录 | ⭐ |
| `YOLO论文大纲/` | 论文大纲和素材包 | ⭐⭐ |
| `yolo论文/` | 论文背景说明文档 | ⭐⭐ |
| `索引/` | 索引刷新工具 | ⭐ |

### 根目录重要文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `README.md` | 文档 | 项目简介（较简略，1行快速开始） |
| `paths.py` | Python | 项目路径配置 |
| `implementation_plan.md` | 文档 | **核心文档**：四大升级的逐文件执行计划 |
| `报告.md` | 文档 | **核心文档**：主线脚本全面审阅与清理建议（162个脚本分类） |
| `scripts_review.md` | 文档 | 脚本审阅记录 |
| `intelligence_class_review.md` | 文档 | 智慧课堂模块审阅 |
| `改造落地进度报告_2026-03-31.md` | 文档 | 改造进度报告 |
| `面向论文的仓库优化方案与可执行分支实验计划.md` | 文档 | 面向论文的仓库优化方案 |
| `yolo11n/s/m/l/x.pt` | 权重 | YOLOv11检测预训练权重（5个尺寸） |
| `yolo11s-pose.pt` / `yolo11x-pose.pt` | 权重 | YOLOv11姿态估计预训练权重（2个尺寸） |

## 三、是干什么的（功能总览）

### 核心功能

1. **姿态估计** — 从视频中提取17点人体关键点
2. **目标检测** — 检测人、书本、笔、电子设备等
3. **轨迹跟踪与平滑** — ByteTrack/Bot-SORT多人跟踪，含不确定性估计
4. **动作识别** — SlowFast R50识别学生行为（听讲、记笔记、讨论等）
5. **语音转写** — Whisper/OpenAI ASR课堂语音转文字
6. **多模态对齐** — 视觉动作 + 语音语义的时序融合
7. **双重验证** — 可学习验证器进行视觉语义交叉校验
8. **时间线可视化** — D3.js交互式课堂行为时间线

### 四大升级方向

详见 `implementation_plan.md`。涉及多人交互图网络(IGFormer)、同伴感知验证、MLLM多模态融合、YOLO底层增强(ASPN/DySnakeConv/GLIDE)。

## 四、有什么用（下游消费）

- **学术论文**：实验数据生成、消融实验、对比实验
- **课堂分析产品**：多视角批处理 → Web前端展示
- **数据集构建**：Gold标注工作台 → 高质量标注数据
- **模型训练**：自训练Case检测器、可学习验证器

## 五、维护注意事项

1. **双入口并存**：`scripts/main/09_run_pipeline.py`（算法入口）与 `integration/run_pipeline.py`（集成入口）功能重叠，需了解区别
2. **脚本数量大**：`scripts/`下有162个Python脚本（含vendored Ultralytics代码），建议参考`报告.md`的分类
3. **数据量庞大**：`data/`、`output/`、`runs/`、`paper_experiments/run_logs/`包含大量自动生成文件
4. **权重文件在根目录**：7个`.pt`文件共约500MB，不要误删
5. **实验命名约定**：`random6_*` = Random6测试集实验，`real_*` = 真实数据实验，`asr_*` = ASR消融实验
6. **重要文档入口**：
   - 技术架构 → `implementation_plan.md`
   - 脚本分类 → `报告.md`
   - 论文背景 → `yolo论文/`
   - 实验配置 → `paper_experiments/configs/`
