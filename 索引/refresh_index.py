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
from typing import Any, Dict, Iterable, List, Sequence, Tuple


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
    modified_at: str
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


def markdown_table(rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(esc_cell(x) for x in rows[0]) + " |"
    sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(esc_cell(x) for x in row) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def slug_for_dir(dir_rel: str) -> str:
    return dir_rel.replace("/", "__")


class IndexBuilder:
    def __init__(
        self,
        project_root: Path,
        out_dir: Path,
        roots: List[str],
        scope: str,
        loc_mode: str,
        dry_run: bool,
    ) -> None:
        self.project_root = project_root
        self.out_dir = out_dir
        self.roots = roots
        self.scope = scope
        self.loc_mode = loc_mode
        self.dry_run = dry_run
        self.root_paths: List[Tuple[str, Path]] = []
        self.inventory: List[FileRecord] = []
        self.py_meta: Dict[str, PyMeta] = {}

    def resolve_roots(self) -> None:
        self.root_paths = []
        for name in self.roots:
            path = self.project_root / name
            if path.exists():
                self.root_paths.append((name, path))

    def is_cache(self, rel: str, ext: str) -> bool:
        rel_lower = rel.lower()
        return (
            "__pycache__" in rel_lower
            or "/.git/" in rel_lower
            or rel_lower.endswith(".pyc")
            or "/node_modules/" in rel_lower
            or "/.mypy_cache/" in rel_lower
            or "/.pytest_cache/" in rel_lower
            or "/dist/" in rel_lower
            or "/build/" in rel_lower
            or rel_lower.endswith(".log")
            or rel_lower.endswith(".tmp")
        )

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
                try:
                    rel = posix_rel(path, self.project_root)
                except Exception:
                    continue
                ext = path.suffix.lower()
                is_code = ext in CODE_EXTENSIONS
                is_cache = self.is_cache(rel, ext)
                is_vendor = self.is_vendor_ultralytics(rel)
                in_scope = self.in_source_scope(is_cache, is_vendor) and (not self.is_generated_run_artifact(rel))
                try:
                    stat = path.stat()
                except Exception:
                    continue
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
                        size_bytes=stat.st_size,
                        modified_at=datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                        is_code=is_code,
                        is_cache=is_cache,
                        is_vendor_ultralytics=is_vendor,
                        in_source_scope=in_scope,
                        loc=loc,
                    )
                )
        self.inventory = sorted(records, key=lambda r: r.relative_path)

    def scan_py_meta(self) -> None:
        result: Dict[str, PyMeta] = {}
        for rec in self.inventory:
            if rec.ext != ".py":
                continue
            path = self.project_root / rec.relative_path
            text = read_text(path)
            try:
                tree = ast.parse(text)
                syntax_ok = True
                syntax_error = ""
                syntax_error_line = 0
            except SyntaxError as exc:
                tree = None
                syntax_ok = False
                syntax_error = str(exc)
                syntax_error_line = int(exc.lineno or 0)
            has_main = False
            main_line = 0
            argparse_description = ""
            if tree is not None:
                for node in ast.walk(tree):
                    if isinstance(node, ast.If):
                        test = node.test
                        if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__":
                            has_main = True
                            main_line = getattr(node, "lineno", 0)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                        if node.func.attr == "ArgumentParser":
                            for kw in node.keywords:
                                if kw.arg == "description" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                    argparse_description = kw.value.value
            result[rec.relative_path] = PyMeta(
                has_main=has_main,
                main_line=main_line,
                argparse_description=argparse_description,
                syntax_ok=syntax_ok,
                syntax_error=syntax_error,
                syntax_error_line=syntax_error_line,
            )
        self.py_meta = result

    def build_entrypoints(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for rel, meta in sorted(self.py_meta.items()):
            if meta.has_main:
                rows.append(
                    {
                        "file": rel,
                        "main_line": meta.main_line,
                        "description": meta.argparse_description,
                    }
                )
        return rows

    def parse_orchestrator_stages(self) -> Dict[str, Any]:
        path = self.project_root / "codex_reports/smart_classroom_yolo_feasibility/orchestrator.py"
        if not path.exists():
            return {"entry": path.as_posix(), "stage_order": []}
        text = read_text(path)
        match = re.search(r"STAGE_ORDER\s*=\s*\[(.*?)\]", text, re.S)
        stages = re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)) if match else []
        return {"entry": "codex_reports/smart_classroom_yolo_feasibility/orchestrator.py", "stage_order": stages}

    def parse_pipeline_09_steps(self) -> Dict[str, Any]:
        rel = "scripts/main/09_run_pipeline.py"
        path = self.project_root / rel
        steps: List[str] = []
        if path.exists():
            text = read_text(path)
            steps = re.findall(r"maybe_run\(\s*['\"]([^'\"]+)['\"]", text)
        return {"entry": rel, "steps": steps}

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

    def read_json_obj(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(read_text(path))
        except Exception:
            return default

    def read_multi_case_data(self) -> Dict[str, Any]:
        return self.read_json_obj(self.project_root / "output/frontend_bundle/multi_case_data.json", {})

    def collect_output_summary(self, sr_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        top_children: Dict[str, int] = defaultdict(int)
        root_files: List[str] = []
        for rec in self.inventory:
            if rec.root_name != "output":
                continue
            parts = Path(rec.relative_path).parts
            if len(parts) <= 2:
                root_files.append(parts[-1])
            else:
                top_children[parts[1]] += 1

        bundles: List[Dict[str, Any]] = []
        bundle_root = self.project_root / "output/frontend_bundle"
        if bundle_root.exists():
            for manifest_path in sorted(bundle_root.glob("*/frontend_data_manifest.json")):
                manifest = self.read_json_obj(manifest_path, {})
                metrics = self.read_json_obj(manifest_path.parent / "metrics_summary.json", {})
                counts = manifest.get("counts", {}) if isinstance(manifest, dict) else {}
                bundles.append(
                    {
                        "case_id": manifest.get("case_id", manifest_path.parent.name),
                        "students": manifest.get("tracked_students", counts.get("tracked_students", 0)),
                        "timeline_segments": counts.get("timeline_student_rows", 0),
                        "verified_events": counts.get("verified_events", 0),
                        "contract_status": manifest.get("contract_status", "unknown"),
                        "gt_status": metrics.get("gt_status", "unknown") if isinstance(metrics, dict) else "unknown",
                    }
                )

        gt_reports: List[Dict[str, Any]] = []
        rear_gt_dir = self.project_root / "output/codex_reports/rear_gt"
        if rear_gt_dir.exists():
            for report_path in sorted(rear_gt_dir.glob("*.report.json")):
                report = self.read_json_obj(report_path, {})
                gt_reports.append(
                    {
                        "name": report_path.stem.replace(".report", ""),
                        "status": report.get("status", "unknown"),
                        "video": Path(str(report.get("video", ""))).name,
                        "frames": report.get("frames", 0),
                        "fps": report.get("fps", 0),
                        "image_dir": report.get("image_dir", ""),
                    }
                )

        asr_dirs = []
        output_root = self.project_root / "output"
        if output_root.exists():
            for child in sorted(output_root.iterdir()):
                if child.is_dir() and child.name.startswith("asr_test_"):
                    asr_dirs.append(child.name)

        multi_case = self.read_multi_case_data()
        return {
            "top_children": dict(sorted(top_children.items())),
            "root_files": sorted(root_files),
            "bundle_cases": bundles,
            "rear_gt_reports": gt_reports,
            "asr_dirs": asr_dirs,
            "multi_case_cases": multi_case.get("cases", []) if isinstance(multi_case, dict) else [],
            "sr_rows_with_formal_gt": len([row for row in sr_rows if str(row.get("gt_status")) == "ok"]),
        }

    def collect_server_frontend_state(self) -> Dict[str, Any]:
        app_path = self.project_root / "server/app.py"
        app_text = read_text(app_path) if app_path.exists() else ""
        index_v2_path = self.project_root / "web_viz/templates/index_v2.html"
        index_v2_text = read_text(index_v2_path) if index_v2_path.exists() else ""
        templates_dir = self.project_root / "web_viz/templates"
        templates = sorted(path.name for path in templates_dir.glob("*.html")) if templates_dir.exists() else []
        return {
            "bundle_api": "/api/bundle/list" in app_text,
            "bundle_manifest_api": "/api/bundle/{case_id}/manifest" in app_text,
            "bundle_verified_api": "/api/bundle/{case_id}/verified" in app_text,
            "bff_case_data_api": "/api/v1/visualization/case_data" in app_text,
            "paper_bundle_page": "paper_demo.html" in app_text,
            "index_v2_exists": index_v2_path.exists(),
            "index_v2_uses_bff": "/api/v1/visualization/case_data" in index_v2_text,
            "templates": templates,
        }

    def collect_yolo_paper_state(self) -> Dict[str, Any]:
        paper_dir = self.project_root / "yolo\u8bba\u6587"
        background_docs: List[str] = []
        planning_docs: List[str] = []
        research_docs: List[str] = []
        if paper_dir.exists():
            for path in sorted(paper_dir.glob("*.md")):
                name = path.name
                if re.match(r"0[1-4]_", name) or name == "README.md":
                    background_docs.append(name)
                elif "深度研究报告" in name:
                    research_docs.append(name)
                else:
                    planning_docs.append(name)
        return {
            "background_docs": background_docs,
            "research_docs": research_docs,
            "planning_docs": planning_docs,
        }

    def build_pipeline_steps(self) -> Dict[str, Any]:
        return {
            "orchestrator": self.parse_orchestrator_stages(),
            "pipeline_09": self.parse_pipeline_09_steps(),
            "hybrid_pose_backbone": {
                "summary": "track_backend=hybrid: pose_tracks_smooth.jsonl is the identity backbone; behavior detections attach semantics.",
            },
            "rear_row_enhancement": {
                "summary": "ROI slicing + optional ROI SR/deartifacting + seat priors + ablation evaluation.",
                "ablation_script": "scripts/experiments/16_run_rear_row_sr_ablation.py",
                "paper_summary_script": "scripts/experiments/17_build_sr_ablation_paper_summary.py",
                "gt_template_script": "scripts/experiments/18_build_rear_row_gt_template.py",
                "metrics_eval_script": "scripts/experiments/19_eval_rear_row_metrics.py",
            },
        }

    def collect_risks(self, sr_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        risks: List[Dict[str, Any]] = []
        for rel, meta in sorted(self.py_meta.items()):
            if not meta.syntax_ok:
                risks.append({"level": "high", "type": "python_syntax_error", "file": rel, "line": meta.syntax_error_line, "message": meta.syntax_error})

        legacy_wrapper = "scripts/main/09b_run_pipeline.py"
        if (self.project_root / legacy_wrapper).exists():
            risks.append(
                {
                    "level": "medium",
                    "type": "legacy_wrapper",
                    "file": legacy_wrapper,
                    "line": 1,
                    "message": "Prefer scripts/main/09_run_pipeline.py as the main entrypoint.",
                }
            )

        missing_gt_cases = sorted({str(row.get("case", "")) for row in sr_rows if str(row.get("gt_status")) == "missing"})
        if missing_gt_cases:
            risks.append(
                {
                    "level": "medium",
                    "type": "missing_rear_row_gt",
                    "file": "output/codex_reports/sr_ablation_paper_summary.json",
                    "line": 1,
                    "message": f"{', '.join(missing_gt_cases)} still lack rear-row formal GT; keep them proxy-only.",
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
                                "file": "scripts/experiments/16_run_rear_row_sr_ablation.py",
                                "line": 1,
                                "message": f"{case_id} A8 improves coverage but increases track gaps.",
                            }
                        )
                except Exception:
                    pass

        for metrics in (self.project_root / "output/codex_reports").glob("front_*_sr_ablation/sr_ablation_metrics.json"):
            payload = self.read_json_obj(metrics, {})
            for row in payload.get("variants", []):
                if isinstance(row, dict) and row.get("variant_status") == "unavailable":
                    risks.append(
                        {
                            "level": "low",
                            "type": "external_backend_unavailable",
                            "file": metrics.as_posix(),
                            "line": 1,
                            "message": f"{row.get('variant')} unavailable: {row.get('sr_reason') or row.get('reason') or 'missing external dependency'}",
                        }
                    )

        multi_case = self.read_multi_case_data()
        asr_data = multi_case.get("asr_data", {}) if isinstance(multi_case, dict) else {}
        instruction_cases = 0
        total_instruction_events = 0
        if isinstance(asr_data, dict):
            for payload in asr_data.values():
                count = len(payload.get("instruction_events", [])) if isinstance(payload, dict) else 0
                if count > 0:
                    instruction_cases += 1
                    total_instruction_events += count
        if instruction_cases < 3:
            risks.append(
                {
                    "level": "medium",
                    "type": "audio_sample_shortage",
                    "file": "output/frontend_bundle/multi_case_data.json",
                    "line": 1,
                    "message": f"Only {instruction_cases} ASR cases with teacher instruction events were found ({total_instruction_events} events).",
                }
            )

        llm_fusion = multi_case.get("llm_fusion", {}) if isinstance(multi_case, dict) else {}
        if isinstance(llm_fusion, dict) and 0 < int(llm_fusion.get("total", 0) or 0) < 100:
            risks.append(
                {
                    "level": "medium",
                    "type": "llm_semantic_only_qualitative",
                    "file": "output/frontend_bundle/multi_case_data.json",
                    "line": 1,
                    "message": f"LLM semantic fusion currently has only {llm_fusion.get('total', 0)} pairs.",
                }
            )

        split_script = self.project_root / "scripts/experiments/25_split_test_set.py"
        if split_script.exists() and 'pure "by video" split is infeasible' in read_text(split_script):
            risks.append(
                {
                    "level": "medium",
                    "type": "test_split_not_video_grouped",
                    "file": "scripts/experiments/25_split_test_set.py",
                    "line": 1,
                    "message": "Current test split is not grouped by video.",
                }
            )

        if not any("ocr" in rec.relative_path.lower() for rec in self.inventory):
            risks.append(
                {
                    "level": "low",
                    "type": "ocr_not_implemented",
                    "file": "yolo\u8bba\u6587/\u6df1\u5ea6\u7814\u7a76\u62a5\u544a_2026-05-03.md",
                    "line": 161,
                    "message": "OCR / sensitive text recognition is not implemented in current code or outputs.",
                }
            )
        return risks

    def dependency_hints(self, dir_rel: str) -> Tuple[List[str], List[str]]:
        mapping: Dict[str, Tuple[List[str], List[str]]] = {
            "scripts": (
                ["课堂视频", "YOLO11x-pose / 8-class behavior detector", "fusion contract checker"],
                ["output/* pose/behavior/student_tracks/fusion/timeline/SR artifacts"],
            ),
            "server": (["output/frontend_bundle/*", "web_viz/templates/*"], ["/api/bundle/*", "/api/v1/visualization/case_data"]),
            "web_viz": (["server/app.py", "output/frontend_bundle/*"], ["paper_demo.html", "index_v2.html"]),
        }
        return mapping.get(dir_rel, (["自动扫描未命中固定规则"], ["自动扫描未命中固定规则"]))

    def responsibility_hints(self, dir_rel: str, files: List[FileRecord]) -> List[str]:
        explicit: Dict[str, List[str]] = {
            "scripts": [
                "项目主入口与实验脚本集合，核心入口是 scripts/main/09_run_pipeline.py。",
                "覆盖 pose、ROI-SR/切片、行为检测、ASR、融合、验证、timeline、前端 bundle 和论文脚本。",
            ],
            "server": [
                "FastAPI 服务入口，当前已落地 bundle API、paper demo 路由和 BFF 聚合接口。",
            ],
            "web_viz": [
                "前端模板目录，包含 paper_demo.html、index.html、index_v2.html。",
            ],
        }
        if dir_rel in explicit:
            return explicit[dir_rel]
        descs: List[str] = []
        for rec in files:
            meta = self.py_meta.get(rec.relative_path)
            if rec.ext == ".py" and meta and meta.argparse_description:
                descs.append(meta.argparse_description)
        unique = []
        for desc in descs:
            if desc not in unique:
                unique.append(desc)
        if unique:
            return [f"入口脚本职责：{desc}" for desc in unique[:4]]
        return ["该目录包含与主流程或项目工具相关的实现文件。"]

    def build_special_docs(self, output_summary: Dict[str, Any], server_state: Dict[str, Any], paper_state: Dict[str, Any]) -> Dict[str, str]:
        docs: Dict[str, str] = {}

        output_rows = [["child", "files"]]
        for name, count in output_summary.get("top_children", {}).items():
            output_rows.append([name, count])
        bundle_rows = [["bundle case", "students", "timeline", "verified", "contract", "gt_status"]]
        for row in output_summary.get("bundle_cases", []):
            bundle_rows.append([row["case_id"], row["students"], row["timeline_segments"], row["verified_events"], row["contract_status"], row["gt_status"]])
        gt_rows = [["gt report", "status", "video", "frames", "fps", "image_dir"]]
        for row in output_summary.get("rear_gt_reports", []):
            gt_rows.append([row["name"], row["status"], row["video"], row["frames"], row["fps"], row["image_dir"]])
        server_api_rows = [
            ["interface", "status"],
            ["/api/bundle/list", "ok" if server_state.get("bundle_api") else "missing"],
            ["/api/bundle/{case_id}/manifest", "ok" if server_state.get("bundle_manifest_api") else "missing"],
            ["/api/bundle/{case_id}/verified", "ok" if server_state.get("bundle_verified_api") else "missing"],
            ["/api/v1/visualization/case_data", "ok" if server_state.get("bff_case_data_api") else "missing"],
            ["/paper/bundle/{case_id}", "ok" if server_state.get("paper_bundle_page") else "missing"],
        ]
        template_rows = [["template", "role"]]
        for name in server_state.get("templates", []):
            role = "BFF template" if name == "index_v2.html" else ("paper demo" if name == "paper_demo.html" else "legacy/full page")
            template_rows.append([name, role])

        docs["\u76ee\u5f55\u8bf4\u660e/output.md"] = "\n".join(
            [
                "# 目录说明：`output`",
                "",
                "- 角色：运行产物、实验结果、GT、frontend bundle、ASR 测试和临时验证输出目录。",
                f"- frontend bundle case 数：`{len(output_summary.get('bundle_cases', []))}`",
                f"- formal GT 进入汇总表的行数：`{output_summary.get('sr_rows_with_formal_gt', 0)}`",
                "",
                "## 顶层子目录文件数",
                markdown_table(output_rows),
                "",
                "## 顶层散落文件",
                *([f"- `{name}`" for name in output_summary.get("root_files", [])] or ["- 无"]),
                "",
                "## Frontend Bundle 摘要",
                markdown_table(bundle_rows),
                "",
                "## Rear GT 报告",
                markdown_table(gt_rows),
                "",
                "## ASR 测试目录",
                *([f"- `{name}`" for name in output_summary.get("asr_dirs", [])] or ["- 无"]),
                "",
            ]
        )

        docs["\u76ee\u5f55\u8bf4\u660e/server.md"] = "\n".join(
            [
                "# 目录说明：`server`",
                "",
                "- 角色：FastAPI 服务入口，负责 bundle API、paper demo 路由和 BFF 聚合接口。",
                "",
                "## 当前接口",
                markdown_table(server_api_rows),
                "",
                "## 当前口径",
                "- `server/app.py` 已有 `/api/bundle/*`。",
                "- `server/app.py` 已有 `/api/v1/visualization/case_data`。",
                "- `paper_demo.html` 是现有单 case 展示页，`index_v2.html` 是已接 BFF 的新模板。",
                "",
            ]
        )

        docs["\u76ee\u5f55\u8bf4\u660e/web_viz.md"] = "\n".join(
            [
                "# 目录说明：`web_viz`",
                "",
                f"- `index_v2.html` 存在：`{server_state.get('index_v2_exists', False)}`",
                f"- `index_v2.html` 直接引用 BFF：`{server_state.get('index_v2_uses_bff', False)}`",
                "",
                "## 当前模板",
                markdown_table(template_rows),
                "",
                "## 当前口径",
                "- `paper_demo.html` 对应论文/单 case 演示页。",
                "- `index_v2.html` 已接 `/api/v1/visualization/case_data`，但是否作为默认首页仍应单独判断。",
                "",
            ]
        )

        docs["\u76ee\u5f55\u8bf4\u660e/yolo\u8bba\u6587.md"] = "\n".join(
            [
                "# 目录说明：`yolo论文`",
                "",
                "- 角色：项目背景、论文语境、研究定位、D3/BFF 改造方案和深度研究报告目录。",
                "- 用法：背景结论可引用；如果与仓库现状冲突，以代码、接口和产物为准。",
                "",
                "## 背景文档",
                *([f"- `{name}`" for name in paper_state.get("background_docs", [])] or ["- 无"]),
                "",
                "## 研究基线",
                *([f"- `{name}`" for name in paper_state.get("research_docs", [])] or ["- 无"]),
                "",
                "## 方案 / 待落地文档",
                *([f"- `{name}`" for name in paper_state.get("planning_docs", [])] or ["- 无"]),
                "",
                "## 当前口径",
                "- `01-04` 与深度研究报告用于解释背景、方法边界、论文语境。",
                "- D3/BFF/第三方方案默认归入“规划/待落地”，除非在代码或产物中找到直接证据。",
                "",
            ]
        )
        return docs

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
            rows: List[List[Any]] = [["file", "LOC", "entry", "description"]]
            for rec in files:
                meta = self.py_meta.get(rec.relative_path)
                rows.append([rec.name, rec.loc, "yes" if meta and meta.has_main else "no", meta.argparse_description if meta else ""])
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
                lines.append("- 无 `__main__` 入口，主要作为模块或被其他脚本调用。")
            lines.extend(["", "## 上游依赖"])
            for item in upstream:
                lines.append(f"- {item}")
            lines.extend(["", "## 下游产物"])
            for item in downstream:
                lines.append(f"- {item}")
            lines.extend(["", "## 文件清单", markdown_table(rows), ""])
            docs[f"\u76ee\u5f55\u8bf4\u660e/{slug_for_dir(dir_rel)}.md"] = "\n".join(lines)
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

    def render_overview_md(
        self,
        stats: Dict[str, Any],
        pipeline: Dict[str, Any],
        risks: List[Dict[str, Any]],
        sr_rows: List[Dict[str, Any]],
        output_summary: Dict[str, Any],
        server_state: Dict[str, Any],
    ) -> str:
        lines: List[str] = [
            "# 00 全局总览",
            "",
            f"- 生成时间：`{stats['generated_at']}`",
            f"- 扫描根目录：`{', '.join(stats['roots'])}`",
            f"- 全部文件数：`{stats['total_files']}`",
            f"- Source-scope 文件数：`{stats['source_scope_files']}`",
            "",
            "## 代码统计",
            f"- 代码文件数：`{stats['code_files_full']}`",
            f"- 代码总行数（LOC）：`{stats['code_loc_full']}`",
            f"- 自研范围代码：`{stats['code_files_source_scope']}` files / `{stats['code_loc_source_scope']}` LOC",
            f"- 第三方 `scripts/pipeline/ultralytics`：`{stats['vendor_code_files']}` files / `{stats['vendor_code_loc']}` LOC",
            f"- `__main__` 入口脚本数：`{stats['entrypoint_count']}`",
            "",
            "## 项目定位",
            "- 智慧课堂音视频上下文感知行为分析框架。",
            "- 已落地主线：YOLOv11 pose + 8 类行为检测 + hybrid 绑定 + rear-row sliced inference + Pipeline Contract。",
            "- 音频/LLM 当前更适合作为上下文增强与系统分析证据，不宜写成充分 benchmark 主结论。",
            "",
            "## 根目录文件分布",
        ]
        for root_name, count in stats["root_file_counts"].items():
            lines.append(f"- `{root_name}`: `{count}` files")
        lines.extend(["", "## Source-scope LOC Top"])
        for item in stats["top_source_dirs_by_loc"]:
            lines.append(f"- `{item['dir']}`: `{item['loc']}` LOC")
        lines.extend(
            [
                "",
                "## 主线入口",
                "- `scripts/main/09_run_pipeline.py`：正式端到端主流程。",
                "- `codex_reports/smart_classroom_yolo_feasibility/orchestrator.py`：profile 驱动的统一编排入口。",
                "- `scripts/experiments/16_run_rear_row_sr_ablation.py`：后排 ROI-SR / 切片 / 座位先验消融实验入口。",
                "- `scripts/experiments/18_build_rear_row_gt_template.py`：rear-row GT 模板生成。",
                "- `scripts/experiments/19_eval_rear_row_metrics.py`：formal 指标计算。",
                "",
                "## Output / 前后端现状",
                f"- `output/frontend_bundle` 当前 bundle：`{len(output_summary.get('bundle_cases', []))}` 个；`front_046_A8` 已进入 formal + bundle 链路。",
                f"- `server/app.py` bundle API：`{server_state.get('bundle_api', False)}`；BFF endpoint：`{server_state.get('bff_case_data_api', False)}`。",
                f"- `web_viz/templates/index_v2.html` 存在：`{server_state.get('index_v2_exists', False)}`；直接接 BFF：`{server_state.get('index_v2_uses_bff', False)}`。",
                "",
                "## 状态分层",
                "- 已落地现状：主流程、SR 消融、front_046 formal GT、frontend bundle、bundle API、BFF endpoint、paper_demo.html。",
                "- 已有代码/接口，但样本或实验未完全闭环：index_v2.html、音频上下文增强、LLM 语义融合、front_001/front_002 的 formal GT。",
                "- 规划/待落地：OCR/敏感文本、更强 D3 交互、更大音频 gold、未闭环的底层 YOLO 增强模块。",
                "",
                "## 后排增强最新结论",
                "- 当前工程方案是 ROI 切片 + 可选 ROI SR/去伪影 + pose identity backbone，而不是直接写成 DLSS 类方案。",
                "- `front_046` 已有 formal GT，可同时引用 proxy 与 formal 指标；`front_001/front_002` 仍以 proxy 为主。",
                "- `A8_adaptive_sliced_artifact_deblur_opencv` 提升覆盖，但 track gap 也可能上升，需要继续用 GT / ReID / seat prior 约束。",
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
        for risk in risks[:12]:
            lines.append(f"- [{risk['level']}] `{risk['file']}` line `{risk['line']}` / `{risk['type']}`: {risk['message']}")
        lines.append("")
        return "\n".join(lines)

    def render_flow_md(self, pipeline: Dict[str, Any]) -> str:
        stages = pipeline.get("orchestrator", {}).get("stage_order", [])
        maybe_steps = pipeline.get("pipeline_09", {}).get("steps", [])
        lines = ["# 01 运行流程总图", "", "## Orchestrator stages"]
        for stage in stages:
            lines.append(f"- `{stage}`")
        lines.extend(["", "## 09_run_pipeline maybe_run steps"])
        for step in maybe_steps:
            lines.append(f"- `{step}`")
        lines.append("")
        return "\n".join(lines)

    def render_quick_locate_md(self) -> str:
        rows = [
            ["任务", "脚本", "关键产物", "入口命令（PowerShell）"],
            ["刷新索引", "索引/refresh_index.py", "索引/*.md / 索引/data/*", "& F:/miniconda/envs/pytorch_env/python.exe ./索引/refresh_index.py --dry_run 0"],
            ["直接跑正式主流程", "scripts/main/09_run_pipeline.py", "actions.fusion_v2.jsonl / verified_events.jsonl / timeline_chart.png", "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/main/09_run_pipeline.py --video data/智慧课堂学生行为数据集/正方视角/002.mp4 --out_dir output/demo --track_backend hybrid --pose_conf 0.20"],
            ["后排 SR/切片消融", "scripts/experiments/16_run_rear_row_sr_ablation.py", "sr_ablation_metrics.csv/json / sr_ablation_contact_sheet.jpg", "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/experiments/16_run_rear_row_sr_ablation.py --video data/智慧课堂学生行为数据集/正方视角/046.mp4 --out_root output/codex_reports/front_046_sr_ablation --variants A8_adaptive_sliced_artifact_deblur_opencv --force 1"],
            ["生成后排 GT 标注模板", "scripts/experiments/18_build_rear_row_gt_template.py", "rear_gt_template.jsonl / gt_frames/*.jpg", "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/experiments/18_build_rear_row_gt_template.py --video data/智慧课堂学生行为数据集/正方视角/046.mp4 --out_jsonl output/codex_reports/rear_gt/front_046_rear_gt.jsonl --start_sec 0 --duration_sec 42 --every_sec 1"],
            ["计算后排正式论文指标", "scripts/experiments/19_eval_rear_row_metrics.py", "rear_row_metrics.json / rear_row_metrics.csv", "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/experiments/19_eval_rear_row_metrics.py --case_dir output/codex_reports/front_046_sr_ablation/A8_adaptive_sliced_artifact_deblur_opencv --gt_jsonl output/codex_reports/rear_gt/front_046_rear_gt.jsonl --video data/智慧课堂学生行为数据集/正方视角/046.mp4 --roi auto_rear"],
            ["汇总论文消融表", "scripts/experiments/17_build_sr_ablation_paper_summary.py", "sr_ablation_paper_summary.md/csv/json", "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/experiments/17_build_sr_ablation_paper_summary.py"],
            ["打包前端 bundle", "scripts/frontend/20_build_frontend_data_bundle.py", "frontend_data_manifest.json / timeline_students.json / metrics_summary.json", "& F:/miniconda/envs/pytorch_env/python.exe ./scripts/frontend/20_build_frontend_data_bundle.py --case_dir output/codex_reports/front_046_sr_ablation/A8_adaptive_sliced_artifact_deblur_opencv --out_dir output/frontend_bundle/front_046_A8 --ablation_summary output/codex_reports/sr_ablation_paper_summary.json"],
            ["查看 BFF 与 bundle API", "server/app.py", "/api/bundle/* /api/v1/visualization/case_data", "& F:/miniconda/envs/pytorch_env/python.exe -m uvicorn server.app:app --reload"],
            ["查看 BFF 驱动前端模板", "web_viz/templates/index_v2.html", "BFF dashboard template / D3 view composition", "模板中已引用 /api/v1/visualization/case_data 与 /api/bundle/list"],
        ]
        return "\n".join(["# 02 快速定位", "", "## 任务 -> 脚本 -> 产物 -> 命令", "", markdown_table(rows), ""])

    def render_risks_md(self, risks: List[Dict[str, Any]]) -> str:
        lines = ["# 03 风险与异常", "", "## 已检测异常"]
        for risk in risks:
            lines.append(f"- [{risk['level']}] `{risk['file']}` line `{risk['line']}` / `{risk['type']}`: {risk['message']}")
        lines.extend(
            [
                "",
                "## 已处理事项",
                "- `scripts/main/09b_run_pipeline.py` 只应视为 legacy wrapper，不应再当主入口。",
                "- hybrid 链路已按 pose identity backbone 口径写入索引。",
                "- 前端 bundle、bundle API 与 `/api/v1/visualization/case_data` 已作为现状写入索引。",
                "",
                "## 当前需要继续确认",
                "- `front_001/front_002` 仍缺 rear-row formal GT。",
                "- `A8` 覆盖提升与 identity stability 之间的权衡仍需继续验证。",
                "- 音频与 LLM 当前更适合上下文增强口径，不宜超前写成 benchmark 主结论。",
                "",
            ]
        )
        return "\n".join(lines)

    def build_readme(self) -> str:
        return "\n".join(
            [
                "# 索引文档体系",
                "",
                "本目录由 `refresh_index.py` 自动生成或刷新，用于让后续协作者快速理解项目结构、主流程、脚本定位、前后端现状和风险边界。",
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
                "- `data/output_inventory.csv`",
                "- `data/output_summary.json`",
                "- `data/entrypoints.json`",
                "- `data/pipeline_steps.json`",
                "- `data/index_stats.json`",
                "- `data/risks.json`",
                "- `data/server_frontend_state.json`",
                "- `data/yolo_paper_docs.json`",
                "",
                "## 口径说明",
                "- `roots` 默认扫描 `codex_reports scripts output server web_viz yolo论文`。",
                "- `scope=self_first`：目录说明跳过缓存目录与 `scripts/pipeline/ultralytics`。",
                "- `yolo论文/` 中的背景/研究报告可用于解释语境，但方案文档默认按“规划/待落地”处理。",
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
        output_summary = self.collect_output_summary(sr_rows)
        server_state = self.collect_server_frontend_state()
        paper_state = self.collect_yolo_paper_state()

        all_rows = []
        for rec in self.inventory:
            case_match = re.search(r"(front_\d+(?:_[A-Za-z0-9]+)?)", rec.relative_path)
            variant_match = re.search(r"/(A\d+_[^/]+)/", rec.relative_path)
            all_rows.append(
                {
                    "relative_path": rec.relative_path,
                    "root": rec.root_name,
                    "dir": rec.dir_rel,
                    "name": rec.name,
                    "ext": rec.ext,
                    "size_bytes": rec.size_bytes,
                    "modified_at": rec.modified_at,
                    "is_code": int(rec.is_code),
                    "is_cache": int(rec.is_cache),
                    "is_vendor_ultralytics": int(rec.is_vendor_ultralytics),
                    "in_source_scope": int(rec.in_source_scope),
                    "loc": rec.loc,
                    "case_hint": case_match.group(1) if case_match else "",
                    "variant_hint": variant_match.group(1) if variant_match else "",
                    "is_output_key_artifact": int(
                        rec.relative_path.startswith("output/frontend_bundle/")
                        or rec.relative_path.endswith("sr_ablation_paper_summary.json")
                        or "rear_gt" in rec.relative_path
                        or rec.relative_path.endswith("pipeline_contract_v2_report.json")
                        or rec.relative_path.endswith("verified_events.jsonl")
                    ),
                }
            )
        source_rows = [row for row in all_rows if row["in_source_scope"] == 1]
        output_rows = [row for row in all_rows if row["root"] == "output"]

        md_outputs: Dict[str, str] = {
            "00_\u5168\u5c40\u603b\u89c8.md": self.render_overview_md(stats, pipeline, risks, sr_rows, output_summary, server_state),
            "01_\u8fd0\u884c\u6d41\u7a0b\u603b\u56fe.md": self.render_flow_md(pipeline),
            "02_\u5feb\u901f\u5b9a\u4f4d.md": self.render_quick_locate_md(),
            "03_\u98ce\u9669\u4e0e\u5f02\u5e38.md": self.render_risks_md(risks),
            "README.md": self.build_readme(),
        }
        md_outputs.update(self.build_directory_docs())
        md_outputs.update(self.build_special_docs(output_summary, server_state, paper_state))

        data_outputs = {
            "data/file_inventory_all.csv": all_rows,
            "data/source_scope_inventory.csv": source_rows,
            "data/output_inventory.csv": output_rows,
            "data/entrypoints.json": entrypoints,
            "data/pipeline_steps.json": pipeline,
            "data/index_stats.json": stats,
            "data/risks.json": risks,
            "data/sr_ablation_latest.json": {"rows": sr_rows},
            "data/output_summary.json": output_summary,
            "data/server_frontend_state.json": server_state,
            "data/yolo_paper_docs.json": paper_state,
        }
        summary = {
            "stats": stats,
            "markdown_files": sorted(md_outputs.keys()),
            "data_files": sorted(data_outputs.keys()),
            "directory_doc_count": len([name for name in md_outputs if name.startswith("\u76ee\u5f55\u8bf4\u660e/")]),
            "sr_summary_rows": len(sr_rows),
            "risk_count": len(risks),
        }
        return md_outputs, {"tables_and_json": data_outputs, "summary": summary}

    def clean_generated_outputs(self) -> None:
        if self.dry_run or not self.out_dir.exists():
            return
        for name in [
            "00_\u5168\u5c40\u603b\u89c8.md",
            "01_\u8fd0\u884c\u6d41\u7a0b\u603b\u56fe.md",
            "02_\u5feb\u901f\u5b9a\u4f4d.md",
            "03_\u98ce\u9669\u4e0e\u5f02\u5e38.md",
            "README.md",
        ]:
            path = self.out_dir / name
            if path.exists():
                path.unlink()
        for sub in ["\u76ee\u5f55\u8bf4\u660e", "data"]:
            subdir = self.out_dir / sub
            if not subdir.exists():
                continue
            for path in subdir.glob("*"):
                if path.is_file() and path.suffix.lower() in {".md", ".json", ".csv"}:
                    path.unlink()

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

    def write_outputs(self, md_outputs: Dict[str, str], data_outputs: Dict[str, Any]) -> None:
        if self.dry_run:
            return
        self.clean_generated_outputs()
        for rel, content in md_outputs.items():
            target = self.out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        for rel, payload in data_outputs["tables_and_json"].items():
            target = self.out_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.suffix.lower() == ".csv":
                self.write_csv(target, payload)
            else:
                target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate project index docs under \\u7d22\\u5f15/.")
    parser.add_argument("--out_dir", type=str, default="\u7d22\u5f15")
    parser.add_argument("--roots", nargs="+", default=["codex_reports", "scripts", "output", "server", "web_viz", "yolo\u8bba\u6587"])
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
