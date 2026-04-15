#!/usr/bin/env python3
"""Build a paper-oriented static demo bundle for GitHub Pages."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _read_jsonl(path: Path, limit: int = 40) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            if len(rows) >= limit:
                break
    return rows


def _write_json(path: Path, obj: Dict[str, Any] | List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _create_case_frame_svg(case_title: str, subtitle: str, accent: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="675" viewBox="0 0 1200 675">
  <defs>
    <linearGradient id="bg" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#0f172a"/>
      <stop offset="100%" stop-color="#1f2937"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="675" fill="url(#bg)"/>
  <circle cx="980" cy="110" r="160" fill="{accent}" opacity="0.25"/>
  <rect x="64" y="80" width="1072" height="515" rx="28" fill="#0b1220" stroke="#334155" stroke-width="2"/>
  <rect x="96" y="120" width="580" height="320" rx="16" fill="#111827" stroke="#475569"/>
  <rect x="700" y="120" width="404" height="132" rx="16" fill="#111827" stroke="#475569"/>
  <rect x="700" y="272" width="404" height="168" rx="16" fill="#111827" stroke="#475569"/>
  <rect x="96" y="460" width="1008" height="100" rx="16" fill="#111827" stroke="#475569"/>
  <text x="112" y="170" fill="#f8fafc" font-size="34" font-family="'Segoe UI',sans-serif">{case_title}</text>
  <text x="112" y="212" fill="#cbd5e1" font-size="22" font-family="'Segoe UI',sans-serif">{subtitle}</text>
  <text x="720" y="165" fill="#93c5fd" font-size="18" font-family="'Segoe UI',sans-serif">Visual stream</text>
  <text x="720" y="315" fill="#fca5a5" font-size="18" font-family="'Segoe UI',sans-serif">Semantic stream</text>
  <text x="112" y="520" fill="#a7f3d0" font-size="18" font-family="'Segoe UI',sans-serif">Privacy-safe synthetic frame for paper demo recording</text>
</svg>
"""


def _create_pipeline_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="220" viewBox="0 0 1320 220">
  <rect width="1320" height="220" fill="#f8fafc"/>
  <style>
    .box { fill: #ffffff; stroke: #334155; stroke-width: 2; rx: 14; }
    .txt { font: 600 16px 'Segoe UI',sans-serif; fill: #1e293b; }
    .sub { font: 400 13px 'Segoe UI',sans-serif; fill: #475569; }
    .arrow { stroke: #0f766e; stroke-width: 2.5; marker-end: url(#m); }
  </style>
  <defs>
    <marker id="m" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L6,3 z" fill="#0f766e"/>
    </marker>
  </defs>
  <rect class="box" x="20" y="38" width="220" height="134"/>
  <rect class="box" x="280" y="38" width="220" height="134"/>
  <rect class="box" x="540" y="38" width="220" height="134"/>
  <rect class="box" x="800" y="38" width="220" height="134"/>
  <rect class="box" x="1060" y="38" width="240" height="134"/>
  <line class="arrow" x1="240" y1="105" x2="280" y2="105"/>
  <line class="arrow" x1="500" y1="105" x2="540" y2="105"/>
  <line class="arrow" x1="760" y1="105" x2="800" y2="105"/>
  <line class="arrow" x1="1020" y1="105" x2="1060" y2="105"/>
  <text class="txt" x="40" y="84">1. Pose / Action</text>
  <text class="sub" x="40" y="112">YOLOv11 + track smoothing</text>
  <text class="sub" x="40" y="136">UQ + stability estimation</text>
  <text class="txt" x="300" y="84">2. ASR / Query</text>
  <text class="sub" x="300" y="112">Text events + timestamps</text>
  <text class="sub" x="300" y="136">Noise-aware preprocessing</text>
  <text class="txt" x="560" y="84">3. Alignment</text>
  <text class="sub" x="560" y="112">Fixed / adaptive window</text>
  <text class="sub" x="560" y="136">Cross-modal candidates</text>
  <text class="txt" x="820" y="84">4. Dual Verify</text>
  <text class="sub" x="820" y="112">Visual + semantic fusion</text>
  <text class="sub" x="820" y="136">UQ reliability gating</text>
  <text class="txt" x="1080" y="84">5. Calibrated Output</text>
  <text class="sub" x="1080" y="112">verified_events + reliability</text>
  <text class="sub" x="1080" y="136">paper-ready metrics/artifacts</text>
</svg>
"""


def _create_reliability_svg(title: str, better: bool) -> str:
    before = "M120,220 C210,120 280,300 380,200 C480,110 580,250 680,170 C760,130 850,200 940,150"
    after = "M120,220 C220,205 320,190 420,180 C520,165 620,145 720,130 C810,118 900,108 960,100"
    path = after if better else before
    tone = "#0f766e" if better else "#b45309"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="980" height="340" viewBox="0 0 980 340">
  <rect width="980" height="340" fill="#ffffff"/>
  <line x1="100" y1="280" x2="940" y2="280" stroke="#94a3b8"/>
  <line x1="100" y1="280" x2="100" y2="40" stroke="#94a3b8"/>
  <text x="110" y="42" fill="#1e293b" font-size="22" font-family="'Segoe UI',sans-serif">{title}</text>
  <text x="890" y="304" fill="#475569" font-size="13" font-family="'Segoe UI',sans-serif">confidence</text>
  <text x="24" y="56" fill="#475569" font-size="13" font-family="'Segoe UI',sans-serif">accuracy</text>
  <path d="{path}" fill="none" stroke="{tone}" stroke-width="4"/>
  <line x1="100" y1="280" x2="940" y2="60" stroke="#cbd5e1" stroke-dasharray="6 6"/>
</svg>
"""


def _default_timeline(case_id: str) -> List[Dict[str, Any]]:
    if case_id == "case_visual_fix":
        return [
            {"start": 11.8, "end": 13.3, "stream": "visual", "label": "phone", "score": 0.71},
            {"start": 12.0, "end": 13.4, "stream": "semantic", "label": "note taking", "score": 0.82},
            {"start": 12.1, "end": 13.2, "stream": "alignment", "label": "aligned_top1", "score": 0.88},
            {"start": 12.2, "end": 13.2, "stream": "verification", "label": "match(note)", "score": 0.83},
        ]
    if case_id == "case_text_fix":
        return [
            {"start": 33.1, "end": 34.6, "stream": "visual", "label": "raise_hand", "score": 0.79},
            {"start": 33.0, "end": 35.0, "stream": "semantic", "label": "ASR noisy: phone", "score": 0.44},
            {"start": 33.2, "end": 34.7, "stream": "alignment", "label": "aligned_top1", "score": 0.84},
            {"start": 33.3, "end": 34.5, "stream": "verification", "label": "match(raise_hand)", "score": 0.81},
        ]
    return [
        {"start": 56.0, "end": 57.3, "stream": "visual", "label": "read", "score": 0.59},
        {"start": 56.1, "end": 57.5, "stream": "semantic", "label": "query: playing phone", "score": 0.52},
        {"start": 56.1, "end": 57.4, "stream": "alignment", "label": "aligned_top1", "score": 0.62},
        {"start": 56.2, "end": 57.3, "stream": "verification", "label": "uncertain", "score": 0.49},
    ]


def _default_verified_events(case_id: str) -> List[Dict[str, Any]]:
    if case_id == "case_visual_fix":
        return [
            {
                "query_id": "q_visual_fix_001",
                "query_time": 12.5,
                "visual_label": "phone",
                "semantic_label": "note",
                "final_label": "note",
                "decision": "match",
                "p_match": 0.82,
                "reliability_score": 0.86,
                "reason": "semantic supports writing context and object evidence excludes phone",
            }
        ]
    if case_id == "case_text_fix":
        return [
            {
                "query_id": "q_text_fix_001",
                "query_time": 33.8,
                "visual_label": "raise_hand",
                "semantic_label": "phone",
                "final_label": "raise_hand",
                "decision": "match",
                "p_match": 0.76,
                "reliability_score": 0.79,
                "reason": "ASR phrase has low confidence and visual temporal consistency is high",
            }
        ]
    return [
        {
            "query_id": "q_conflict_001",
            "query_time": 56.8,
            "visual_label": "read",
            "semantic_label": "phone",
            "final_label": "uncertain",
            "decision": "uncertain",
            "p_match": 0.52,
            "reliability_score": 0.41,
            "reason": "cross-modal conflict with high UQ and low overlap",
        }
    ]


def _sanitize_paths(obj: Any, project_root: Path) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_paths(v, project_root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_paths(v, project_root) for v in obj]
    if isinstance(obj, str):
        norm = obj.replace("\\", "/")
        if ":/" in norm or norm.startswith("/"):
            p = Path(norm)
            try:
                rel = p.resolve().relative_to(project_root.resolve())
                return f"./{rel.as_posix()}"
            except Exception:
                parts = [x for x in p.parts if x not in {"/", "\\"}]
                if "output" in parts:
                    i = parts.index("output")
                    return "./" + "/".join(parts[i:])
                if "contracts" in parts:
                    i = parts.index("contracts")
                    return "./" + "/".join(parts[i:])
                return f"./{p.name}" if p.name else obj
        return obj
    return obj


def _compose_metrics(output_root: Path, project_root: Path) -> Dict[str, Any]:
    exp_a = _read_json(output_root / "paper_experiments" / "exp_a_uq_align" / "metrics.json")
    exp_b = _read_json(output_root / "paper_experiments" / "exp_b_reliability_calibration" / "metrics.json")
    baseline = _read_json(output_root / "paper_experiments" / "baseline" / "metrics.json")
    exp_a = _sanitize_paths(exp_a, project_root)
    exp_b = _sanitize_paths(exp_b, project_root)
    baseline = _sanitize_paths(baseline, project_root)

    return {
        "generated_at": _now_iso(),
        "note": "Static demo metrics for visualization. Not final paper results.",
        "baseline": baseline or {"f1": 0.67, "precision": 0.70, "recall": 0.65},
        "alignment": exp_a or {"fixed_window_alignment": {"alignment_recall_at_1": 0.58}, "adaptive_uq_alignment": {"alignment_recall_at_1": 0.64}},
        "reliability": exp_b or {"no_uq_gate": {"ECE": 0.16, "Brier": 0.21}, "calibrated_uq_gate": {"ECE": 0.09, "Brier": 0.16}},
    }


def build(args: argparse.Namespace) -> Dict[str, Any]:
    docs_root = Path(args.docs_root).resolve()
    project_root = docs_root.parent.resolve()
    output_root = Path(args.output_root).resolve()
    data_root = docs_root / "data"
    assets_cases = docs_root / "assets" / "cases"
    assets_charts = docs_root / "assets" / "charts"
    assets_video = docs_root / "assets" / "videos"

    if args.clean:
        shutil.rmtree(data_root, ignore_errors=True)
        shutil.rmtree(assets_cases, ignore_errors=True)
        shutil.rmtree(assets_charts, ignore_errors=True)

    data_root.mkdir(parents=True, exist_ok=True)
    assets_cases.mkdir(parents=True, exist_ok=True)
    assets_charts.mkdir(parents=True, exist_ok=True)
    assets_video.mkdir(parents=True, exist_ok=True)

    case_specs = [
        {
            "id": "case_visual_fix",
            "title": "Case A: Visual false alarm corrected by semantic evidence",
            "scenario": "视觉误判但语义纠正",
            "accent": "#22d3ee",
            "analysis": "Pose-only branch tends to confuse note and phone; semantic context pushes the final decision to note.",
        },
        {
            "id": "case_text_fix",
            "title": "Case B: Semantic noise corrected by visual consistency",
            "scenario": "语义噪声但视觉纠正",
            "accent": "#f59e0b",
            "analysis": "ASR text is noisy while motion/pose continuity is stable, so verifier trusts visual evidence.",
        },
        {
            "id": "case_conflict_uncertain",
            "title": "Case C: Cross-modal conflict resolved as uncertain",
            "scenario": "视觉与语义冲突，系统输出 uncertain",
            "accent": "#ef4444",
            "analysis": "Both streams disagree and uncertainty is high; system returns uncertain instead of forcing a hard label.",
        },
    ]

    demo_cases: List[Dict[str, Any]] = []
    timeline_map: Dict[str, List[Dict[str, Any]]] = {}
    verified_map: Dict[str, List[Dict[str, Any]]] = {}

    for spec in case_specs:
        case_id = spec["id"]
        frame_name = f"{case_id}.svg"
        frame_rel = f"./assets/cases/{frame_name}"
        _write_text(
            assets_cases / frame_name,
            _create_case_frame_svg(spec["title"], spec["scenario"], spec["accent"]),
        )

        timeline_rows = _default_timeline(case_id)
        verified_rows = _default_verified_events(case_id)

        timeline_map[case_id] = timeline_rows
        verified_map[case_id] = verified_rows

        video_rel = None
        if args.include_real_media:
            for candidate in sorted(assets_video.glob("*.mp4")):
                video_rel = f"./assets/videos/{candidate.name}"
                break

        demo_cases.append(
            {
                "case_id": case_id,
                "title": spec["title"],
                "scenario": spec["scenario"],
                "privacy": "synthetic_or_desensitized",
                "media": {"frame": frame_rel, "video": video_rel},
                "timeline_ref": case_id,
                "verified_events_ref": case_id,
                "analysis": spec["analysis"],
                "tags": ["dual_verification", "reliability", "paper_demo"],
            }
        )

    metrics = _compose_metrics(output_root, project_root)

    # Prefer experiment reliability curves if present, otherwise write placeholders.
    copied_before = _copy_if_exists(
        output_root / "paper_experiments" / "exp_b_reliability_calibration" / "reliability_before.svg",
        assets_charts / "reliability_before.svg",
    )
    copied_after = _copy_if_exists(
        output_root / "paper_experiments" / "exp_b_reliability_calibration" / "reliability_after.svg",
        assets_charts / "reliability_after.svg",
    )
    if not copied_before:
        _write_text(assets_charts / "reliability_before.svg", _create_reliability_svg("Reliability (Before Calibration)", better=False))
    if not copied_after:
        _write_text(assets_charts / "reliability_after.svg", _create_reliability_svg("Reliability (After Calibration)", better=True))

    _write_text(assets_charts / "pipeline_flow.svg", _create_pipeline_svg())

    bundle = {
        "generated_at": _now_iso(),
        "project": {
            "name": "YOLOv11 Vision-Semantic Dual Verification Demo",
            "purpose": "Paper supplementary static visualization for GitHub Pages",
            "disclaimer": "All metrics and cases are for engineering smoke demo; not final paper conclusions.",
        },
        "cases": demo_cases,
    }

    _write_json(data_root / "demo_cases.json", bundle)
    _write_json(data_root / "metrics.json", metrics)
    _write_json(data_root / "timeline.json", {"generated_at": _now_iso(), "cases": timeline_map})
    _write_json(data_root / "verified_events.json", {"generated_at": _now_iso(), "cases": verified_map})

    # Compatibility files used by older viewers.
    _write_json(
        data_root / "manifest.json",
        {
            "generated_at": _now_iso(),
            "cases": [
                {
                    "video_id": c["case_id"],
                    "view": "paper_demo",
                    "case_id": c["case_id"],
                    "paths": {
                        "timeline": "./data/timeline.json",
                        "verified_events": "./data/verified_events.json",
                        "frame": c["media"]["frame"],
                        "video_overlay": c["media"]["video"],
                    },
                }
                for c in demo_cases
            ],
        },
    )
    _write_json(
        data_root / "list_cases.json",
        [{"video_id": c["case_id"], "case_id": c["case_id"], "view": "paper_demo"} for c in demo_cases],
    )

    return {
        "docs_root": str(docs_root),
        "cases": len(demo_cases),
        "files": [
            str(data_root / "demo_cases.json"),
            str(data_root / "metrics.json"),
            str(data_root / "timeline.json"),
            str(data_root / "verified_events.json"),
            str(assets_charts / "pipeline_flow.svg"),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static paper demo assets for docs/ (GitHub Pages).")
    parser.add_argument("--docs_root", default="docs", help="Target docs directory.")
    parser.add_argument("--output_root", default="output", help="Output directory used as optional metric source.")
    parser.add_argument("--clean", action="store_true", help="Clean docs/data and paper demo assets before writing.")
    parser.add_argument(
        "--include_real_media",
        action="store_true",
        help="If set, links the first docs/assets/videos/*.mp4 into demo cases. Default keeps privacy-safe frame-only demo.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(args)
    print(f"[DONE] docs_root={result['docs_root']}")
    print(f"[DONE] cases={result['cases']}")
    for path in result["files"]:
        print(f"[FILE] {path}")


if __name__ == "__main__":
    main()
