from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CODE_EXTENSIONS = {
    ".py",
    ".ps1",
    ".sh",
    ".bat",
    ".cmd",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".lua",
    ".r",
    ".m",
    ".kt",
    ".swift",
}


@dataclass
class FileRecord:
    relative_path: str
    root_name: str
    dir_rel: str
    name: str
    ext: str
    size_bytes: int
    is_code: bool
    is_cache: bool
    is_vendor_ultralytics: bool
    in_source_scope: bool
    loc: int


@dataclass
class PyMeta:
    has_main: bool
    main_line: int
    argparse_description: str
    syntax_ok: bool
    syntax_error: str
    syntax_error_line: int


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="ignore")


def posix_rel(path: Path, base: Path) -> str:
    return path.resolve().relative_to(base.resolve()).as_posix()


def count_lines(path: Path) -> int:
    return len(read_text(path).splitlines())


def esc_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def markdown_table(rows: List[List[Any]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(esc_cell(v) for v in rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(esc_cell(v) for v in row) + " |" for row in rows[1:]]
    return "\n".join([header, sep] + body)


def parse_argparse_description(text: str) -> str:
    match = re.search(r"ArgumentParser\(\s*description\s*=\s*([\"'])(.*?)\1", text, flags=re.S)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(2)).strip()


def parse_python_meta(path: Path) -> PyMeta:
    text = read_text(path)
    main_line = 0
    main_pattern = re.compile(r"if\s+__name__\s*==\s*['\"]__main__['\"]")
    for idx, line in enumerate(text.splitlines(), start=1):
        if main_pattern.search(line):
            main_line = idx
            break
    try:
        ast.parse(text.lstrip("\ufeff"), filename=str(path))
        syntax_ok = True
        syntax_error = ""
        syntax_error_line = 0
    except SyntaxError as exc:
        syntax_ok = False
        syntax_error = exc.msg or "SyntaxError"
        syntax_error_line = int(exc.lineno or 0)
    return PyMeta(
        has_main=main_line > 0,
        main_line=main_line,
        argparse_description=parse_argparse_description(text),
        syntax_ok=syntax_ok,
        syntax_error=syntax_error,
        syntax_error_line=syntax_error_line,
    )


def slug_for_dir(dir_rel: str) -> str:
    return (dir_rel or "project_root").replace("/", "__")


class IndexBuilder:
    def __init__(self, project_root: Path, out_dir: Path, roots: List[str], scope: str, loc_mode: str, dry_run: bool) -> None:
        self.project_root = project_root.resolve()
        self.out_dir = out_dir.resolve()
        self.roots = roots
        self.scope = scope
        self.loc_mode = loc_mode
        self.dry_run = dry_run
        self.root_paths: List[Tuple[str, Path]] = []
        self.inventory: List[FileRecord] = []
        self.py_meta: Dict[str, PyMeta] = {}

    def resolve_roots(self) -> None:
        resolved: List[Tuple[str, Path]] = []
        for root_name in self.roots:
            raw = Path(root_name)
            root_path = raw if raw.is_absolute() else self.project_root / raw
            root_path = root_path.resolve()
            if root_path.exists() and root_path.is_dir():
                resolved.append((root_name, root_path))
        if not resolved:
            raise FileNotFoundError("No valid roots found.")
        self.root_paths = resolved

    def is_cache(self, rel: str, ext: str) -> bool:
        parts = Path(rel).parts
        return "__pycache__" in parts or ext == ".pyc"

    def is_vendor_ultralytics(self, rel: str) -> bool:
        return rel.startswith("scripts/pipeline/ultralytics/")

    def in_source_scope(self, is_cache: bool, is_vendor_ultralytics: bool) -> bool:
        return (not is_cache) and (not is_vendor_ultralytics)

    def is_generated_run_artifact(self, rel: str) -> bool:
        return rel.startswith("codex_reports/smart_classroom_yolo_feasibility/runs/")

    def scan_inventory(self) -> None:
        records: List[FileRecord] = []
        for root_name, root_path in self.root_paths:
            for path in root_path.rglob("*"):
                if not path.is_file():
                    continue
                rel = posix_rel(path, self.project_root)
                ext = path.suffix.lower()
                is_code = ext in CODE_EXTENSIONS
                is_cache = self.is_cache(rel, ext)
                is_vendor = self.is_vendor_ultralytics(rel)
                in_scope = self.in_source_scope(is_cache, is_vendor) and (not self.is_generated_run_artifact(rel))
                loc = 0
                if is_code:
                    try:
                        loc = count_lines(path)
                    except Exception:
                        loc = 0
                records.append(
                    FileRecord(
                        relative_path=rel,
                        root_name=root_name,
                        dir_rel=Path(rel).parent.as_posix(),
                        name=path.name,
                        ext=ext,
                        size_bytes=path.stat().st_size,
                        is_code=is_code,
                        is_cache=is_cache,
                        is_vendor_ultralytics=is_vendor,
                        in_source_scope=in_scope,
                        loc=loc,
                    )
                )
        self.inventory = sorted(records, key=lambda r: r.relative_path)

    def scan_py_meta(self) -> None:
        meta: Dict[str, PyMeta] = {}
        for rec in self.inventory:
            if rec.ext != ".py" or not rec.in_source_scope:
                continue
            try:
                meta[rec.relative_path] = parse_python_meta(self.project_root / rec.relative_path)
            except Exception:
                meta[rec.relative_path] = PyMeta(False, 0, "", False, "Read/parse failed", 0)
        self.py_meta = meta

    def build_entrypoints(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for rel, meta in sorted(self.py_meta.items()):
            if meta.has_main:
                rows.append({"file": rel, "line": meta.main_line, "argparse_description": meta.argparse_description})
        return rows

    def extract_balanced_call(self, text: str, start_pos: int) -> str:
        open_pos = text.find("(", start_pos)
        if open_pos == -1:
            return text[start_pos:]
        depth = 0
        for idx in range(open_pos, len(text)):
            if text[idx] == "(":
                depth += 1
            elif text[idx] == ")":
                depth -= 1
                if depth == 0:
                    return text[start_pos : idx + 1]
        return text[start_pos:]

    def parse_orchestrator_stages(self) -> Dict[str, Any]:
        rel = "codex_reports/smart_classroom_yolo_feasibility/orchestrator.py"
        path = self.project_root / rel
        result: Dict[str, Any] = {
            "entry": rel,
            "stage_order": [],
            "wrapper_scripts": {
                "train_wisdom_s": "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/10_train/train_wisdom.py",
                "train_wisdom_m": "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/10_train/train_wisdom.py",
                "scb_clean": "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/20_scb/scb_tasks.py",
                "scb_train": "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/20_scb/scb_tasks.py",
                "infer_full_pipeline": "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/30_infer/run_full_integration.py",
                "semantic_bridge": "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/40_semantics/semantic_bridge.py",
                "validate_outputs": "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/90_tests/check_outputs.py",
            },
        }
        if not path.exists():
            return result
        text = read_text(path)
        match = re.search(r"STAGE_ORDER\s*=\s*\[(.*?)\]", text, flags=re.S)
        if match:
            result["stage_order"] = re.findall(r"[\"']([^\"']+)[\"']", match.group(1))
        return result

    def parse_pipeline_09_steps(self) -> Dict[str, Any]:
        rel = "scripts/pipeline/main/09_run_pipeline.py"
        path = self.project_root / rel
        output: Dict[str, Any] = {"entry": rel, "steps": []}
        if not path.exists():
            return output
        text = read_text(path)
        fallback_script_by_step = {
            1: "scripts/pipeline/01_pose_video_demo.py",
            15: "scripts/pipeline/02c_build_rear_roi_sr_cache.py",
            2: "scripts/pipeline/02_export_keypoints_jsonl.py",
            4: "scripts/pipeline/03_track_and_smooth.py",
            5: "scripts/pipeline/05_slowfast_actions.py",
            6: "scripts/pipeline/06_api_asr_realtime.py | scripts/pipeline/06_asr_whisper_to_jsonl.py | scripts/pipeline/06c_asr_openai_to_jsonl.py",
            35: "scripts/pipeline/03c_estimate_track_uncertainty.py",
            45: "scripts/pipeline/02b_export_objects_jsonl.py",
            46: "scripts/pipeline/02d_export_behavior_det_jsonl.py",
            47: "scripts/pipeline/05c_behavior_det_to_actions.py",
            55: "scripts/pipeline/05b_fuse_actions_with_objects.py",
            58: "scripts/pipeline/05d_merge_action_sources.py",
            59: "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/50_fusion_contract/merge_fusion_actions_v2.py",
            65: "scripts/pipeline/06b_event_query_extraction.py",
            66: "scripts/pipeline/xx_align_multimodal.py",
            67: "verifier/train.py",
            70: "scripts/pipeline/07_dual_verification.py",
            71: "verifier/eval.py",
            72: "verifier/calibration.py",
            80: "scripts/pipeline/10_visualize_timeline.py",
            90: "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/50_fusion_contract/check_fusion_contract.py",
            91: "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/50_fusion_contract/check_pipeline_contract_v2.py",
            456: "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/50_fusion_contract/semanticize_objects.py",
            471: "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/50_fusion_contract/semanticize_behavior_det.py",
            4711: "scripts/pipeline/03e_track_behavior_students.py",
            472: "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/50_fusion_contract/behavior_det_to_actions_v2.py",
            473: "scripts/pipeline/06_overlay_pose_behavior_video.py",
            474: "scripts/pipeline/06d_build_rear_row_contact_sheet.py",
            656: "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/50_fusion_contract/build_event_queries_fusion_v2.py",
        }
        steps: List[Dict[str, Any]] = []
        for order, match in enumerate(re.finditer(r"maybe_run\(", text), start=1):
            block = self.extract_balanced_call(text, match.start())
            head = re.search(r"maybe_run\(\s*(\d+)\s*,\s*[\"']([^\"']+)[\"']", block)
            if not head:
                continue
            step_id = int(head.group(1))
            step_name = head.group(2)
            script_rel = ""
            patterns = [
                (r"scripts_dir\s*/\s*[\"']([^\"']+\.py)[\"']", "scripts/pipeline/{}"),
                (r"fusion_scripts_dir\s*/\s*[\"']([^\"']+\.py)[\"']", "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/50_fusion_contract/{}"),
                (r"PROJECT_ROOT\s*/\s*[\"']verifier[\"']\s*/\s*[\"']([^\"']+\.py)[\"']", "verifier/{}"),
            ]
            for pattern, template in patterns:
                hit = re.search(pattern, block)
                if hit:
                    script_rel = template.format(hit.group(1))
            if not script_rel:
                script_rel = fallback_script_by_step.get(step_id, "")
            steps.append({"order": order, "step": step_id, "name": step_name, "script": script_rel})
        output["steps"] = steps
        return output

    def read_latest_sr_summary_rows(self) -> List[Dict[str, Any]]:
        path = self.project_root / "output/codex_reports/sr_ablation_paper_summary.json"
        if not path.exists():
            return []
        try:
            payload = json.loads(read_text(path))
        except Exception:
            return []
        rows = payload.get("rows", [])
        return [row for row in rows if isinstance(row, dict)]

    def build_pipeline_steps(self) -> Dict[str, Any]:
        return {
            "orchestrator": self.parse_orchestrator_stages(),
            "pipeline_09": self.parse_pipeline_09_steps(),
            "hybrid_pose_backbone": {
                "summary": "track_backend=hybrid 时，pose_tracks_smooth.jsonl 是学生身份主干；8 类行为检测只作为每个 pose track 的语义属性。",
                "student_tracks": "student_tracks.jsonl",
                "unmatched_debug": "behavior_unmatched.jsonl",
                "action_bridge": "actions.behavior.semantic.jsonl",
                "contract_rule": "hybrid 模式下 student_track_students == tracked_students，且 actions.fusion_v2.jsonl 不允许出现 track_id >= 200000。",
            },
            "rear_row_enhancement": {
                "summary": "后排低分辨率/遮挡增强采用 ROI 切片、可选 ROI SR/去噪去模糊、座位先验和消融评估，不改变下游 JSONL schema。",
                "main_flags": [
                    "--pose_infer_mode full_sliced",
                    "--behavior_infer_mode full_sliced",
                    "--pose_slice_grid 2x2|adaptive|rear_adaptive|rear_dense",
                    "--sr_backend off|opencv|realesrgan|basicvsrpp|realbasicvsr|nvidia_vsr|maxine_vfx",
                    "--sr_preprocess off|denoise|deblock|deblur|artifact_deblur|clahe",
                    "--pose_track_seat_prior_mode off|x_anchor",
                ],
                "ablation_script": "scripts/pipeline/16_run_rear_row_sr_ablation.py",
                "paper_summary_script": "scripts/pipeline/17_build_sr_ablation_paper_summary.py",
                "gt_template_script": "scripts/pipeline/18_build_rear_row_gt_template.py",
                "metrics_eval_script": "scripts/pipeline/19_eval_rear_row_metrics.py",
            },
        }

    def collect_risks(self, sr_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        risks: List[Dict[str, Any]] = []
        for rel, meta in sorted(self.py_meta.items()):
            if not meta.syntax_ok:
                risks.append(
                    {
                        "level": "high",
                        "type": "python_syntax_error",
                        "file": rel,
                        "line": meta.syntax_error_line,
                        "message": meta.syntax_error,
                    }
                )
        legacy_wrapper = "scripts/pipeline/09b_run_pipeline.py"
        legacy_path = self.project_root / legacy_wrapper
        if legacy_path.exists():
            try:
                text = read_text(legacy_path)
            except Exception:
                text = ""
            if "LEGACY_NOTICE" in text or "--allow_legacy" in text:
                risks.append(
                    {
                        "level": "medium",
                        "type": "legacy_wrapper",
                        "file": legacy_wrapper,
                        "line": 1,
                        "message": "Legacy wrapper exists; prefer scripts/pipeline/main/09_run_pipeline.py as the single entrypoint.",
                    }
                )
        if sr_rows and any(str(row.get("gt_status")) == "missing" for row in sr_rows):
            risks.append(
                {
                    "level": "medium",
                    "type": "missing_rear_row_gt",
                    "file": "output/codex_reports/sr_ablation_paper_summary.json",
                    "line": 1,
                    "message": "SR/切片消融目前仍缺后排人工 GT；只能写代理召回提升，不能写正式 precision/recall/F1 结论。",
                }
            )
        by_case_variant = {(row.get("case"), row.get("variant")): row for row in sr_rows}
        for case_id in sorted({row.get("case") for row in sr_rows}):
            base = by_case_variant.get((case_id, "A0_full_no_sr"), {})
            a8 = by_case_variant.get((case_id, "A8_adaptive_sliced_artifact_deblur_opencv"), {})
            if base and a8:
                try:
                    if float(a8.get("track_gap_count_proxy") or 0) > float(base.get("track_gap_count_proxy") or 0):
                        risks.append(
                            {
                                "level": "medium",
                                "type": "recall_identity_tradeoff",
                                "file": "scripts/pipeline/16_run_rear_row_sr_ablation.py",
                                "line": 1,
                                "message": f"{case_id} A8 后排召回更强，但 track gap 增加；需要 ReID/座位先验/GT 继续确认是否串 ID。",
                            }
                        )
                except Exception:
                    pass
        for metrics in (self.project_root / "output/codex_reports").glob("front_*_sr_ablation/sr_ablation_metrics.json"):
            try:
                payload = json.loads(read_text(metrics))
            except Exception:
                continue
            for row in payload.get("variants", []):
                if isinstance(row, dict) and row.get("variant_status") == "unavailable":
                    risks.append(
                        {
                            "level": "low",
                            "type": "external_backend_unavailable",
                            "file": metrics.as_posix(),
                            "line": 1,
                            "message": f"{row.get('variant')} 未执行：{row.get('sr_reason') or row.get('reason') or 'missing external dependency'}。",
                        }
                    )
        return risks

    def dependency_hints(self, dir_rel: str) -> Tuple[List[str], List[str]]:
        mapping: Dict[str, Tuple[List[str], List[str]]] = {
            "scripts": (
                ["课堂视频", "YOLO11x-pose / 8 类行为检测模型", "codex_reports/.../50_fusion_contract"],
                ["output/* 中的 pose、behavior、student_tracks、fusion、timeline、SR ablation 产物"],
            ),
            "scripts/pipeline/intelligence_class/pipeline": (
                ["scripts/pipeline/intelligence_class/training", "课堂数据集"],
                ["output/智慧课堂学生行为数据集/*"],
            ),
            "scripts/pipeline/intelligence_class/tools": (
                ["pipeline 输出目录", "output/*"],
                ["数据集报告、分析摘要、辅助检查产物"],
            ),
            "scripts/pipeline/intelligence_class/training": (
                ["data/processed/classroom_yolo/dataset.yaml"],
                ["runs/detect/*"],
            ),
            "scripts/pipeline/intelligence_class/web_ui": (
                ["output/*", "scripts/pipeline/intelligence_class/_utils"],
                ["本地可视化页面与接口"],
            ),
            "scripts/pipeline/models": (
                ["PyTorch", "scripts/pipeline/modules"],
                ["被主流程或实验脚本调用的模型组件"],
            ),
            "scripts/pipeline/modules": (
                ["scripts/pipeline/models", "pose/action features"],
                ["给主流程提供上下文特征"],
            ),
            "scripts/pipeline/training": (
                ["data/processed/classroom_yolo/*", "YOLO weights"],
                ["runs/detect/*"],
            ),
            "codex_reports/smart_classroom_yolo_feasibility": (
                ["profiles/*.yaml", "scripts/pipeline/00_common"],
                ["runs/* manifest/commands/logs"],
            ),
            "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/00_common": (
                ["orchestrator.py"],
                ["10_train, 20_scb, 30_infer, 40_semantics, 90_tests"],
            ),
            "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/10_train": (
                ["scripts/pipeline/intelligence_class/training/03_train_case_yolo.py"],
                ["runs/detect/*"],
            ),
            "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/20_scb": (
                ["SCB 数据目录或占位配置"],
                ["SCB clean/train 阶段产物"],
            ),
            "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/30_infer": (
                ["scripts/pipeline/main/09_run_pipeline.py"],
                ["output/codex_reports/*"],
            ),
            "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/40_semantics": (
                ["output/* 基础产物", "profiles/action_semantics_8class.yaml"],
                ["*.semantic.jsonl", "semantics_manifest.json"],
            ),
            "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/50_fusion_contract": (
                ["actions/objects/behavior/event_queries 基础文件"],
                ["actions.fusion_v2.jsonl", "event_queries.fusion_v2.jsonl", "contract reports"],
            ),
            "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/60_reports": (
                ["pipeline outputs", "paper assets"],
                ["paper_pipeline_report.md"],
            ),
            "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/90_tests": (
                ["output/*", "manifest.json"],
                ["validate_outputs.report.json", "contract checks"],
            ),
        }
        return mapping.get(dir_rel, (["自动扫描未命中固定规则"], ["自动扫描未命中固定规则"]))

    def responsibility_hints(self, dir_rel: str, files: List[FileRecord]) -> List[str]:
        explicit: Dict[str, List[str]] = {
            "scripts": [
                "项目主入口与实验脚本集合，核心入口是 scripts/pipeline/main/09_run_pipeline.py。",
                "负责 pose、ROI-SR/切片、行为检测、ASR、融合、验证、timeline 和论文消融。",
                "当前推荐路径为 hybrid：pose track 负责学生身份，8 类行为检测负责语义挂载。",
            ],
            "codex_reports/smart_classroom_yolo_feasibility": [
                "统一编排入口和 profile 驱动的实验任务目录。",
                "把阶段命令、输入输出和状态写入 manifest，支持 dry-run、emit-only 与验证。",
            ],
            "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/50_fusion_contract": [
                "维护 fusion contract v2：语义化、行为动作聚合、event queries 构建、合同校验。",
                "hybrid 模式要求 actions.fusion_v2.jsonl 中不再出现未链接的 200000+ 行为轨迹。",
            ],
        }
        if dir_rel in explicit:
            return explicit[dir_rel]
        descs: List[str] = []
        for rec in files:
            meta = self.py_meta.get(rec.relative_path)
            if rec.ext == ".py" and meta and meta.argparse_description:
                descs.append(meta.argparse_description)
        unique: List[str] = []
        for desc in descs:
            if desc not in unique:
                unique.append(desc)
        if unique:
            return [f"入口脚本职责：{desc}" for desc in unique[:4]]
        return ["该目录包含与主流程或自研工具相关的实现文件。"]

    def build_directory_docs(self) -> Dict[str, str]:
        by_dir: Dict[str, List[FileRecord]] = defaultdict(list)
        for rec in self.inventory:
            if rec.in_source_scope and rec.is_code:
                by_dir[rec.dir_rel].append(rec)
        docs: Dict[str, str] = {}
        for dir_rel in sorted(by_dir):
            files = sorted(by_dir[dir_rel], key=lambda rec: rec.name)
            total_loc = sum(rec.loc for rec in files)
            entry_files = [rec for rec in files if self.py_meta.get(rec.relative_path, PyMeta(False, 0, "", True, "", 0)).has_main]
            upstream, downstream = self.dependency_hints(dir_rel)
            rows: List[List[Any]] = [["文件", "LOC", "入口", "说明"]]
            for rec in files:
                meta = self.py_meta.get(rec.relative_path)
                rows.append(
                    [
                        rec.name,
                        rec.loc,
                        "yes" if meta and meta.has_main else "no",
                        meta.argparse_description if meta else "",
                    ]
                )
            lines: List[str] = [
                f"# 目录说明：`{dir_rel}`",
                "",
                f"- 代码文件数：`{len(files)}`",
                f"- 代码行数（LOC）：`{total_loc}`",
                "",
                "## 目录职责",
            ]
            for item in self.responsibility_hints(dir_rel, files):
                lines.append(f"- {item}")
            lines.extend(["", "## 入口脚本"])
            if entry_files:
                for rec in entry_files:
                    meta = self.py_meta.get(rec.relative_path)
                    lines.append(f"- `{rec.relative_path}` (line {meta.main_line if meta else 0})")
            else:
                lines.append("- 无 `__main__` 入口，主要作为库模块或配置脚本被调用。")
            lines.extend(["", "## 上游依赖（静态线索）"])
            for item in upstream:
                lines.append(f"- {item}")
            lines.extend(["", "## 下游产物（静态线索）"])
            for item in downstream:
                lines.append(f"- {item}")
            lines.extend(["", "## 文件清单", markdown_table(rows), ""])
            docs[f"目录说明/{slug_for_dir(dir_rel)}.md"] = "\n".join(lines)
        return docs

    def build_stats(self, entrypoints: List[Dict[str, Any]]) -> Dict[str, Any]:
        root_file_counts: Dict[str, int] = defaultdict(int)
        for rec in self.inventory:
            root_file_counts[rec.root_name] += 1
        code_all = [rec for rec in self.inventory if rec.is_code]
        code_scope = [rec for rec in self.inventory if rec.is_code and rec.in_source_scope]
        vendor_code = [rec for rec in self.inventory if rec.is_code and rec.is_vendor_ultralytics]
        dir_loc_scope: Dict[str, int] = defaultdict(int)
        for rec in code_scope:
            dir_loc_scope[rec.dir_rel] += rec.loc
        top_dirs = sorted(dir_loc_scope.items(), key=lambda item: item[1], reverse=True)[:14]
        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "roots": [name for name, _ in self.root_paths],
            "root_file_counts": dict(sorted(root_file_counts.items())),
            "total_files": len(self.inventory),
            "source_scope_files": len([rec for rec in self.inventory if rec.in_source_scope]),
            "code_files_full": len(code_all),
            "code_loc_full": sum(rec.loc for rec in code_all),
            "code_files_source_scope": len(code_scope),
            "code_loc_source_scope": sum(rec.loc for rec in code_scope),
            "vendor_code_files": len(vendor_code),
            "vendor_code_loc": sum(rec.loc for rec in vendor_code),
            "entrypoint_count": len(entrypoints),
            "top_source_dirs_by_loc": [{"dir": dir_rel, "loc": loc} for dir_rel, loc in top_dirs],
        }

    def render_sr_summary_table(self, sr_rows: List[Dict[str, Any]]) -> str:
        ok_rows = [row for row in sr_rows if row.get("pipeline_status") == "ok"]
        if not ok_rows:
            return "- 暂无可用 SR/切片消融汇总。"
        rows = [["case", "variant", "tracked", "rear_delta", "matched_delta", "gap", "unlinked", "GT"]]
        for row in ok_rows:
            rows.append(
                [
                    row.get("case", ""),
                    row.get("variant", ""),
                    row.get("tracked_students", ""),
                    row.get("rear_proxy_delta_pct_vs_A0", ""),
                    row.get("matched_delta_pct_vs_A0", ""),
                    row.get("track_gap_count_proxy", ""),
                    row.get("unlinked_rows_ge_200000", ""),
                    row.get("gt_status", ""),
                ]
            )
        return markdown_table(rows)

    def render_overview_md(self, stats: Dict[str, Any], pipeline: Dict[str, Any], risks: List[Dict[str, Any]], sr_rows: List[Dict[str, Any]]) -> str:
        lines: List[str] = [
            "# 00 全局总览",
            "",
            f"- 生成时间：`{stats['generated_at']}`",
            f"- 扫描根目录：`{', '.join(stats['roots'])}`",
            f"- 全部文件数：`{stats['total_files']}`",
            f"- Source-scope 文件数（self_first）：`{stats['source_scope_files']}`",
            "",
            "## 代码统计（全量口径）",
            f"- 代码文件数：`{stats['code_files_full']}`",
            f"- 代码总行数（LOC）：`{stats['code_loc_full']}`",
            f"- 自研范围代码（用于目录说明）：`{stats['code_files_source_scope']}` files / `{stats['code_loc_source_scope']}` LOC",
            f"- 第三方 `scripts/pipeline/ultralytics`：`{stats['vendor_code_files']}` files / `{stats['vendor_code_loc']}` LOC",
            f"- `__main__` 入口脚本数：`{stats['entrypoint_count']}`",
            "",
            "## 根目录文件分布",
        ]
        for root_name, count in stats["root_file_counts"].items():
            lines.append(f"- `{root_name}`: `{count}` files")
        lines.extend(["", "## Source-scope 目录 LOC Top"])
        for item in stats["top_source_dirs_by_loc"]:
            lines.append(f"- `{item['dir']}`: `{item['loc']}` LOC")
        lines.extend(
            [
                "",
                "## 主线入口",
                "- `scripts/pipeline/main/09_run_pipeline.py`：正式端到端主流程，当前覆盖 pose、ROI-SR/切片、行为检测、hybrid 绑定、融合视频、ASR、验证和 timeline。",
                "- `codex_reports/smart_classroom_yolo_feasibility/orchestrator.py`：profile 驱动的统一编排入口。",
                "- `scripts/pipeline/03e_track_behavior_students.py`：hybrid 行为检测桥接，以 pose track 作为学生身份主干。",
                "- `scripts/pipeline/16_run_rear_row_sr_ablation.py`：后排 ROI-SR / 切片 / 座位先验消融实验入口。",
                "- `scripts/pipeline/18_build_rear_row_gt_template.py`：生成后排 GT 标注模板，用于把代理指标升级成正式 precision/recall/F1。",
                "- `scripts/pipeline/19_eval_rear_row_metrics.py`：统一计算检测 AP、PCK/OKS、IDF1/HOTA/MOTA、行为 F1、SR 质量和工程性能指标。",
                "",
                "## 当前 hybrid 结论",
                "- `track_backend=hybrid`：下游身份只沿用 pose `track_id`。",
                "- 8 类行为检测只挂载到 pose 学生上，未匹配行为框写入 `behavior_unmatched.jsonl` 作为 debug，不进入 fusion。",
                "- contract 断言：`student_track_students == tracked_students`，且 `actions.fusion_v2.jsonl` 不允许 `track_id >= 200000`。",
                "",
                "## 后排增强最新结论",
                "- 当前可落地方案不是直接 DLSS，而是 `ROI 切片 + 可选 ROI SR/去噪去模糊 + pose 身份主干 + 行为语义融合`。",
                "- `A8_adaptive_sliced_artifact_deblur_opencv` 在 front_002 上显著提高后排代理召回和行为匹配，但 track gap 也上升，需要 GT/ReID/座位先验继续约束身份稳定性。",
                "- `nvidia_vsr` / `maxine_vfx` 已作为外部命令后端预留，未配置 SDK 命令前只会记录 unavailable。",
                "",
                self.render_sr_summary_table(sr_rows),
                "",
                "## 编排阶段数",
                f"- orchestrator stages: `{len(pipeline.get('orchestrator', {}).get('stage_order', []))}`",
                f"- 09_run_pipeline maybe_run stages: `{len(pipeline.get('pipeline_09', {}).get('steps', []))}`",
                "",
                "## 风险摘要",
            ]
        )
        if not risks:
            lines.append("- 当前未检测到已知异常。")
        else:
            for risk in risks[:12]:
                lines.append(f"- [{risk['level']}] `{risk['file']}` line `{risk['line']}` / `{risk['type']}`：{risk['message']}")
        lines.append("")
        return "\n".join(lines)

    def render_flow_md(self, pipeline: Dict[str, Any]) -> str:
        orchestrator = pipeline.get("orchestrator", {})
        steps = pipeline.get("pipeline_09", {}).get("steps", [])
        lines: List[str] = ["# 01 运行流程总图", ""]
        lines.append("## Orchestrator 主编排")
        stages = orchestrator.get("stage_order", [])
        if stages:
            lines.append("```mermaid")
            lines.append("flowchart LR")
            for idx, stage in enumerate(stages):
                lines.append(f'  S{idx}["{stage}"]')
                if idx > 0:
                    lines.append(f"  S{idx - 1} --> S{idx}")
            lines.append("```")
        else:
            lines.append("- 未解析到 orchestrator stage order。")
        lines.extend(["", "## 09_run_pipeline 主步骤", "```mermaid", "flowchart TB"])
        for idx, step in enumerate(steps):
            label = f"{step['step']} {step['name']}".replace('"', "'")
            lines.append(f'  P{idx}["{label}"]')
            if idx > 0:
                lines.append(f"  P{idx - 1} --> P{idx}")
        lines.append("```")
        lines.extend(
            [
                "",
                "## hybrid 身份与行为语义链路",
                "```mermaid",
                "flowchart LR",
                '  A["pose_tracks_smooth.jsonl"] --> B["pose track_id 身份主干"]',
                '  C["behavior_det.semantic.jsonl"] --> D["逐帧 IoU + center affinity 匹配"]',
                '  B --> D',
                '  D --> E["student_tracks.jsonl"]',
                '  D --> F["behavior_unmatched.jsonl (debug only)"]',
                '  E --> G["actions.behavior.semantic.jsonl"]',
                '  G --> H["actions.fusion_v2.jsonl"]',
                '  H --> I["align / dual verification / timeline"]',
                "```",
                "",
                "## 后排低分辨率与遮挡增强链路",
                "```mermaid",
                "flowchart LR",
                '  V["front video"] --> R["auto_rear ROI"]',
                '  R --> SR["step015 ROI SR cache: off/opencv/external"]',
                '  SR --> P["step002 pose full_sliced / rear_adaptive / rear_dense"]',
                '  SR --> B2["step046 behavior full_sliced"]',
                '  P --> T["step004 pose track + seat prior"]',
                '  B2 --> H2["step4711 hybrid pose-backbone binding"]',
                '  T --> H2',
                '  H2 --> F2["step473 fusion video"]',
                '  H2 --> C2["step090/091 contract"]',
                '  GT["rear GT template"] -. optional .-> M["SR ablation metrics"]',
                "```",
                "",
                "## 09_run_pipeline 步骤表（按代码执行顺序）",
            ]
        )
        rows = [["order", "step", "name", "script"]]
        for step in steps:
            rows.append([step["order"], step["step"], step["name"], step["script"] or "N/A"])
        lines.extend([markdown_table(rows), ""])
        return "\n".join(lines)

    def render_quick_locate_md(self) -> str:
        rows = [
            ["任务", "脚本", "关键产物", "入口命令（PowerShell 示例）"],
            [
                "刷新索引",
                "索引/refresh_index.py",
                "索引/*.md / 索引/data/*",
                "& F:/miniconda/envs/pytorch_env/python.exe ./索引/refresh_index.py --dry_run 0",
            ],
            [
                "直接跑正式主流程",
                "scripts/pipeline/main/09_run_pipeline.py",
                "actions.fusion_v2.jsonl / verified_events.jsonl / timeline_chart.png",
                "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/pipeline/main/09_run_pipeline.py --video data/智慧课堂学生行为数据集/正方视角/002.mp4 --out_dir output/demo --track_backend hybrid --pose_conf 0.20",
            ],
            [
                "只看 01 pose 视频效果",
                "scripts/pipeline/01_pose_video_demo.py",
                "pose_demo_yolo11x.mp4",
                "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/pipeline/01_pose_video_demo.py --video data/智慧课堂学生行为数据集/正方视角/002.mp4 --out output/pose_demo_yolo11x.mp4 --model yolo11x-pose.pt --conf 0.20 --imgsz 960",
            ],
            [
                "导出 pose keypoints（含切片）",
                "scripts/pipeline/02_export_keypoints_jsonl.py",
                "pose_keypoints_v2.jsonl / rear_row_pose_diagnostics.json",
                "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/pipeline/02_export_keypoints_jsonl.py --video data/智慧课堂学生行为数据集/正方视角/002.mp4 --out output/demo/pose_keypoints_v2.jsonl --model yolo11x-pose.pt --conf 0.20 --infer_mode full_sliced --slice_grid rear_adaptive --diagnostics_out output/demo/rear_row_pose_diagnostics.json",
            ],
            [
                "生成后排 ROI SR 缓存",
                "scripts/pipeline/02c_build_rear_roi_sr_cache.py",
                "rear_roi_sr/<backend>/frames / sr_cache.report.json",
                "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/pipeline/02c_build_rear_roi_sr_cache.py --video data/智慧课堂学生行为数据集/正方视角/002.mp4 --out_dir output/demo/rear_roi_sr/opencv --backend opencv --preprocess artifact_deblur --scale 2",
            ],
            [
                "导出 8 类行为检测（含切片）",
                "scripts/pipeline/02d_export_behavior_det_jsonl.py",
                "behavior_det.jsonl / rear_row_behavior_diagnostics.json",
                "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/pipeline/02d_export_behavior_det_jsonl.py --video data/智慧课堂学生行为数据集/正方视角/002.mp4 --out output/demo/behavior_det.jsonl --model runs/detect/official_yolo11s_detect_e150_v1/weights/best.pt --infer_mode full_sliced --slice_grid rear_adaptive",
            ],
            [
                "hybrid 行为绑定",
                "scripts/pipeline/03e_track_behavior_students.py",
                "student_tracks.jsonl / behavior_unmatched.jsonl",
                "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/pipeline/03e_track_behavior_students.py --in output/demo/behavior_det.semantic.jsonl --out output/demo/student_tracks.jsonl --track_backend hybrid --pose_tracks output/demo/pose_tracks_smooth.jsonl --behavior_unmatched_out output/demo/behavior_unmatched.jsonl",
            ],
            [
                "生成 pose+行为融合视频",
                "scripts/pipeline/06_overlay_pose_behavior_video.py",
                "pose_behavior_fusion_yolo11x.mp4 / preview.jpg / report.json",
                "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/pipeline/06_overlay_pose_behavior_video.py --video data/智慧课堂学生行为数据集/正方视角/002.mp4 --pose_tracks output/demo/pose_tracks_smooth.jsonl --student_tracks output/demo/student_tracks.jsonl --actions output/demo/actions.behavior.semantic.jsonl --out output/demo/pose_behavior_fusion_yolo11x.mp4",
            ],
            [
                "后排 SR/切片消融",
                "scripts/pipeline/16_run_rear_row_sr_ablation.py",
                "sr_ablation_metrics.csv/json / sr_ablation_contact_sheet.jpg",
                "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/pipeline/16_run_rear_row_sr_ablation.py --video data/智慧课堂学生行为数据集/正方视角/002.mp4 --out_root output/codex_reports/front_002_sr_ablation --variants A8_adaptive_sliced_artifact_deblur_opencv --force 1",
            ],
            [
                "生成后排 GT 标注模板",
                "scripts/pipeline/18_build_rear_row_gt_template.py",
                "rear_gt_template.jsonl / gt_frames/*.jpg",
                "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/pipeline/18_build_rear_row_gt_template.py --video data/智慧课堂学生行为数据集/正方视角/002.mp4 --out_jsonl output/codex_reports/rear_gt/front_002_rear_gt.jsonl --start_sec 0 --duration_sec 60 --every_sec 1",
            ],
            [
                "计算后排正式论文指标",
                "scripts/pipeline/19_eval_rear_row_metrics.py",
                "rear_row_metrics.json / rear_row_metrics.csv",
                "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/pipeline/19_eval_rear_row_metrics.py --case_dir output/codex_reports/front_002_sr_ablation/A8_adaptive_sliced_artifact_deblur_opencv --gt_jsonl output/codex_reports/rear_gt/front_002_rear_gt.jsonl --video data/智慧课堂学生行为数据集/正方视角/002.mp4 --roi auto_rear",
            ],
            [
                "汇总论文消融表",
                "scripts/pipeline/17_build_sr_ablation_paper_summary.py",
                "sr_ablation_paper_summary.md/csv/json",
                "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/pipeline/17_build_sr_ablation_paper_summary.py",
            ],
            [
                "fusion 合同校验",
                "codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/50_fusion_contract/check_pipeline_contract_v2.py",
                "pipeline_contract_v2_report.json",
                "& F:/miniconda/envs/pytorch_env/python.exe ./codex_reports/smart_classroom_yolo_feasibility/scripts/pipeline/50_fusion_contract/check_pipeline_contract_v2.py --output_dir output/demo --report output/demo/pipeline_contract_v2_report.json --track_backend hybrid --strict 1",
            ],
        ]
        return "\n".join(["# 02 快速定位", "", "## 任务 -> 脚本 -> 产物 -> 命令", "", markdown_table(rows), ""])

    def render_risks_md(self, risks: List[Dict[str, Any]]) -> str:
        lines = ["# 03 风险与异常", "", "## 已检测异常"]
        if risks:
            for risk in risks:
                lines.append(f"- [{risk['level']}] `{risk['file']}` line `{risk['line']}` / `{risk['type']}`: {risk['message']}")
        else:
            lines.append("- 当前未检测到已知异常。")
        lines.extend(
            [
                "",
                "## 已处理事项",
                "- `train_wisdom.py` 文件头语法错误已不再出现在扫描风险中。",
                "- `scripts/pipeline/09b_run_pipeline.py` 若保留为带保护提示的 legacy wrapper，只作为中风险提示，不阻断主流程。",
                "- hybrid 身份链路已改为 pose-backbone：未匹配行为框只进 `behavior_unmatched.jsonl`，不再污染 fusion。",
                "- 后排增强已形成可跑消融：A0/A1/A2/A3/A7/A8，A10/A11 作为 NVIDIA/Maxine 外部后端预留。",
                "",
                "## 当前需要继续确认",
                "- 后排 SR/切片消融缺少 30-60 秒人工 GT，论文中暂时只能使用代理召回、行为匹配数、track gap 等工程指标。",
                "- `A8_adaptive_sliced_artifact_deblur_opencv` 提高召回，但 track gap 增加；后续应接 ReID/更强座位先验或人工 IDF1/IDSW 验证。",
                "- Real-ESRGAN / BasicVSR++ / RealBasicVSR / NVIDIA VSR / Maxine VFX 都需要外部依赖或 SDK 命令，未配置时不应写成已完成实验。",
                "",
                "## 备注",
                "- 本文档由索引生成器扫描当前工作区得到，不自动修复源码。",
                "- 修改主流程后应运行 `refresh_index.py --dry_run 0` 重新生成。",
                "",
            ]
        )
        return "\n".join(lines)

    def build_readme(self) -> str:
        return "\n".join(
            [
                "# 索引文档体系",
                "",
                "本目录由 `refresh_index.py` 自动生成或刷新，用于让后续 Codex 快速理解项目结构、主流程、脚本定位、后排增强实验和当前风险。",
                "",
                "## 一键刷新",
                "",
                "```powershell",
                "& F:/miniconda/envs/pytorch_env/python.exe ./索引/refresh_index.py --dry_run 1",
                "& F:/miniconda/envs/pytorch_env/python.exe ./索引/refresh_index.py --dry_run 0",
                "```",
                "",
                "## 关键输出",
                "- `00_全局总览.md`",
                "- `01_运行流程总图.md`",
                "- `02_快速定位.md`",
                "- `03_风险与异常.md`",
                "- `目录说明/*.md`",
                "- `data/file_inventory_all.csv`",
                "- `data/source_scope_inventory.csv`",
                "- `data/entrypoints.json`",
                "- `data/pipeline_steps.json`",
                "- `data/index_stats.json`",
                "- `data/risks.json`",
                "",
                "## 口径说明",
                "- `roots` 默认只扫描 `codex_reports scripts`。",
                "- `scope=self_first`：目录说明跳过 `scripts/pipeline/ultralytics` 和缓存目录。",
                "- `loc_mode=full`：全量 LOC 只输出一个主指标；自研 LOC 仅作为目录说明辅助。",
                "- 当前推荐主线：`track_backend=hybrid`，pose track 负责学生身份，8 类行为检测负责行为语义挂载。",
                "- 当前后排增强主线：`full_sliced/rear_adaptive + 可选 ROI SR/去噪去模糊 + pose-backbone fusion`。",
                "",
            ]
        )

    def build_outputs(self) -> Tuple[Dict[str, str], Dict[str, Any]]:
        self.resolve_roots()
        self.scan_inventory()
        self.scan_py_meta()
        entrypoints = self.build_entrypoints()
        pipeline = self.build_pipeline_steps()
        sr_rows = self.read_latest_sr_summary_rows()
        risks = self.collect_risks(sr_rows)
        stats = self.build_stats(entrypoints)

        all_rows = [
            {
                "relative_path": rec.relative_path,
                "root": rec.root_name,
                "dir": rec.dir_rel,
                "name": rec.name,
                "ext": rec.ext,
                "size_bytes": rec.size_bytes,
                "is_code": int(rec.is_code),
                "is_cache": int(rec.is_cache),
                "is_vendor_ultralytics": int(rec.is_vendor_ultralytics),
                "in_source_scope": int(rec.in_source_scope),
                "loc": rec.loc,
            }
            for rec in self.inventory
        ]
        source_rows = [row for row in all_rows if row["in_source_scope"] == 1]

        md_outputs: Dict[str, str] = {
            "00_全局总览.md": self.render_overview_md(stats, pipeline, risks, sr_rows),
            "01_运行流程总图.md": self.render_flow_md(pipeline),
            "02_快速定位.md": self.render_quick_locate_md(),
            "03_风险与异常.md": self.render_risks_md(risks),
            "README.md": self.build_readme(),
        }
        md_outputs.update(self.build_directory_docs())

        data_outputs = {
            "data/file_inventory_all.csv": all_rows,
            "data/source_scope_inventory.csv": source_rows,
            "data/entrypoints.json": entrypoints,
            "data/pipeline_steps.json": pipeline,
            "data/index_stats.json": stats,
            "data/risks.json": risks,
            "data/sr_ablation_latest.json": {"rows": sr_rows},
        }
        summary = {
            "stats": stats,
            "markdown_files": sorted(md_outputs.keys()),
            "data_files": sorted(data_outputs.keys()),
            "directory_doc_count": len([name for name in md_outputs if name.startswith("目录说明/")]),
            "sr_summary_rows": len(sr_rows),
            "risk_count": len(risks),
        }
        return md_outputs, {"tables_and_json": data_outputs, "summary": summary}

    def clean_generated_outputs(self) -> None:
        if self.dry_run:
            return
        if not self.out_dir.exists():
            return
        resolved_out = self.out_dir.resolve()
        if self.project_root not in [resolved_out, *resolved_out.parents]:
            raise RuntimeError(f"Refusing to clean outside project root: {resolved_out}")
        for name in ["00_全局总览.md", "01_运行流程总图.md", "02_快速定位.md", "03_风险与异常.md", "README.md"]:
            path = resolved_out / name
            if path.exists():
                path.unlink()
        for sub in ["目录说明", "data"]:
            subdir = resolved_out / sub
            if not subdir.exists():
                continue
            for path in subdir.glob("*"):
                if path.is_file() and path.suffix.lower() in {".md", ".json", ".csv"}:
                    path.unlink()

    def write_outputs(self, md_outputs: Dict[str, str], data_outputs: Dict[str, Any]) -> None:
        if self.dry_run:
            return
        self.clean_generated_outputs()
        for rel, content in md_outputs.items():
            target = self.out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        self.write_csv(self.out_dir / "data/file_inventory_all.csv", data_outputs["tables_and_json"]["data/file_inventory_all.csv"])
        self.write_csv(self.out_dir / "data/source_scope_inventory.csv", data_outputs["tables_and_json"]["data/source_scope_inventory.csv"])
        for key in [
            "data/entrypoints.json",
            "data/pipeline_steps.json",
            "data/index_stats.json",
            "data/risks.json",
            "data/sr_ablation_latest.json",
        ]:
            target = self.out_dir / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(data_outputs["tables_and_json"][key], ensure_ascii=False, indent=2), encoding="utf-8")

    def write_csv(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: List[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate project index docs under 索引/.")
    parser.add_argument("--out_dir", type=str, default="索引")
    parser.add_argument("--roots", nargs="+", default=["codex_reports", "scripts"])
    parser.add_argument("--scope", choices=["self_first"], default="self_first")
    parser.add_argument("--loc_mode", choices=["full"], default="full")
    parser.add_argument("--dry_run", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    builder = IndexBuilder(
        project_root=project_root,
        out_dir=out_dir,
        roots=list(args.roots),
        scope=str(args.scope),
        loc_mode=str(args.loc_mode),
        dry_run=bool(int(args.dry_run)),
    )
    md_outputs, data_outputs = builder.build_outputs()
    builder.write_outputs(md_outputs, data_outputs)
    print(json.dumps(data_outputs["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
