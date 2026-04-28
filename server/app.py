import os
import sys
import json
import csv
import re
from pathlib import Path
from functools import lru_cache
import inspect
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.templating import Jinja2Templates

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Avoid loky probing warnings/errors in restricted Windows environments.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

# 灏濊瘯瀵煎叆 Levenshtein锛屽鏋滄病鏈夊氨鐢ㄧ畝鍖栫増
try:
    import Levenshtein  # type: ignore
except Exception:
    Levenshtein = None


# =========================================================
# 1) 璺緞锛氫紭鍏堜娇鐢?paths.py锛堜笌浣?scripts/ 涓竴鑷达級
# =========================================================
def _try_import_paths():
    try:
        import paths  # type: ignore
        return paths
    except Exception:
        # 鍏佽浠?server 鐩綍瀵煎叆澶辫触鏃讹紝鎵嬪姩鍔犺矾寰?
        server_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(server_dir)
        try:
            import paths  # type: ignore
            return paths
        except Exception:
            return None


paths_mod = _try_import_paths()

# 娌℃湁 paths.py 灏辨寜椤圭洰缁撴瀯鎺ㄦ柇
BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(getattr(paths_mod, "PROJECT_ROOT", BASE_DIR)).resolve()
OUTPUT_DIR = Path(getattr(paths_mod, "OUTPUT_DIR", PROJECT_ROOT / "output")).resolve()
DATA_DIR = Path(getattr(paths_mod, "DATA_DIR", PROJECT_ROOT / "data")).resolve()
VIDEO_DIR = (DATA_DIR / "videos").resolve()
DOCS_DIR = (PROJECT_ROOT / "docs").resolve()
PAPER_TABLE_DIR = (DOCS_DIR / "assets" / "tables" / "paper_d3_selected").resolve()
PAPER_CHART_DIR = (DOCS_DIR / "assets" / "charts" / "paper_d3_selected").resolve()
PAPER_VIDEO_DIR = (DOCS_DIR / "assets" / "videos").resolve()
ASSETS_DIR = (DOCS_DIR / "assets").resolve()

# 浣犵殑鍓嶇妯℃澘涓€鑸湪 web_viz/templates锛涘鏋滄病鏈夊氨鍥為€€鍒伴」鐩牴鐩綍
TEMPLATE_DIR = (PROJECT_ROOT / "web_viz" / "templates")
if not TEMPLATE_DIR.exists():
    TEMPLATE_DIR = PROJECT_ROOT

# 闈欐€佽祫婧愮洰褰曪細web_viz/static 鎴栭」鐩牴
STATIC_DIR = (PROJECT_ROOT / "web_viz" / "static")
if not STATIC_DIR.exists():
    STATIC_DIR = PROJECT_ROOT

app = FastAPI(title="Classroom Multi-Modal Viz Server")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 2) 闈欐€佹枃浠舵寕杞斤紙鍏煎鏃ц矾寰勶細/output /data锛?
# =========================================================
# 鍏煎浣?.txt Flask 鐗堬細/output/<path:filename> -> output鐩綍 :contentReference[oaicite:2]{index=2}
if OUTPUT_DIR.exists():
    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
    # 涔熺粰涓€涓埆鍚嶏細/outputs
    app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# 鍏煎浣?.txt Flask 鐗堬細/data/<path:filename> -> data鐩綍 :contentReference[oaicite:3]{index=3}
if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# 瑙嗛鍒悕锛堟洿鐩磋锛?
if VIDEO_DIR.exists():
    app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")

# 闈欐€佽祫婧?
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if DOCS_DIR.exists():
    app.mount("/docs", StaticFiles(directory=str(DOCS_DIR)), name="docs")

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def _find_case_dir(video_id: str) -> Optional[Path]:
    """Find case directory by exact folder name, supporting nested view/case layout."""
    if not OUTPUT_DIR.exists():
        return None
    candidates: List[Path] = []
    direct = OUTPUT_DIR / video_id
    if direct.exists() and direct.is_dir() and _looks_like_case_dir(direct):
        candidates.append(direct)
    for candidate in OUTPUT_DIR.rglob(video_id):
        if candidate.is_dir() and _looks_like_case_dir(candidate):
            candidates.append(candidate)
    if not candidates:
        return None
    uniq = list(dict.fromkeys(candidates))
    return sorted(uniq, key=lambda p: (-_case_dir_quality(p), len(str(p))))[0]


def _find_case_dir_for_analysis(video_id: str) -> Optional[Path]:
    direct = OUTPUT_DIR / video_id
    candidates: List[Path] = []
    if direct.exists() and direct.is_dir() and _looks_like_case_dir(direct):
        candidates.append(direct)

    if OUTPUT_DIR.exists():
        for candidate in OUTPUT_DIR.rglob(video_id):
            if candidate.is_dir() and _looks_like_case_dir(candidate):
                candidates.append(candidate)

    if not candidates:
        return None

    def _rank(p: Path) -> Tuple[int, int]:
        has_bundle = int((p / "analysis" / "bundle.json").exists())
        # prefer bundle=1 first, then shorter path
        return (-has_bundle, len(str(p)))

    return sorted(candidates, key=_rank)[0]


def _to_output_url(path: Path) -> str:
    rel = path.resolve().relative_to(OUTPUT_DIR)
    return f"/output/{rel.as_posix()}"


def _to_data_url(path: Path) -> Optional[str]:
    try:
        rel = path.resolve().relative_to(DATA_DIR.resolve())
    except Exception:
        return None
    return f"/data/{rel.as_posix()}"


def _to_assets_url(path: Path) -> Optional[str]:
    try:
        rel = path.resolve().relative_to(ASSETS_DIR.resolve())
    except Exception:
        return None
    return f"/assets/{rel.as_posix()}"


def _path_to_public_url(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    if not path.exists():
        return None
    for fn in (_to_output_url, _to_assets_url, _to_docs_url, _to_data_url):
        try:
            u = fn(path)  # type: ignore[misc]
        except Exception:
            u = None
        if u:
            return u
    return None


_VIEW_TO_DATA_SUBDIR: Dict[str, str] = {
    "front": "正方视角",
    "rear": "后方视角",
    "teacher": "教师视角",
    "top1": "斜上方视角1",
    "top2": "斜上方视角2",
}


@lru_cache(maxsize=2)
def _guess_dataset_root() -> Optional[Path]:
    if not DATA_DIR.exists():
        return None
    wanted = set(_VIEW_TO_DATA_SUBDIR.values())
    for p in DATA_DIR.iterdir():
        if not p.is_dir():
            continue
        try:
            child_names = {c.name for c in p.iterdir() if c.is_dir()}
        except Exception:
            continue
        if wanted.intersection(child_names):
            return p
    return None


def _resolve_dataset_view_dir(view_name: str) -> Optional[Path]:
    root = _guess_dataset_root()
    if root is None:
        return None
    key = str(view_name).strip().lower()
    mapped = _VIEW_TO_DATA_SUBDIR.get(key)
    if not mapped:
        return None
    p = root / mapped
    return p if p.exists() and p.is_dir() else None


def _candidate_case_stems(case_id: str) -> List[str]:
    raw = str(case_id).strip()
    out: List[str] = []

    def _push(x: str) -> None:
        x = str(x).strip()
        if x and x not in out:
            out.append(x)

    _push(raw)
    digits = re.sub(r"\D+", "", raw)
    if digits:
        _push(digits)
        try:
            _push(str(int(digits)))
        except Exception:
            pass
        _push(digits.zfill(3))
        _push(digits.zfill(4))
        _push(digits.zfill(6))
    return out


def _extract_case_tokens(video_id: str, case_dir: Optional[Path] = None) -> Tuple[str, str]:
    view_name = ""
    if case_dir is not None and case_dir.parent != OUTPUT_DIR:
        view_name = case_dir.parent.name
    case_id = str(video_id).split("__")[-1] if "__" in str(video_id) else str(video_id)
    return str(view_name), str(case_id)


def _find_docs_video_path(video_id: str, variant: str) -> Optional[Path]:
    if not PAPER_VIDEO_DIR.exists():
        return None
    vid = str(video_id).strip()
    if not vid:
        return None

    candidates: List[str] = []
    if variant == "overlay":
        candidates = [f"{vid}.overlay.mp4"]
    elif variant == "original":
        candidates = [f"{vid}.original.mp4", f"{vid}.mp4"]
    else:
        candidates = [f"{vid}.{variant}.mp4", f"{vid}.mp4"]

    for name in candidates:
        p = PAPER_VIDEO_DIR / name
        if p.exists():
            return p
    return None


def _find_output_original_video_path(case_dir: Path, video_id: str, case_id: str) -> Optional[Path]:
    preferred = [
        case_dir / f"{video_id}.mp4",
        case_dir / f"{video_id}.original.mp4",
        case_dir / f"{case_id}.mp4",
        case_dir / "original.mp4",
        case_dir / "source.mp4",
        case_dir / "video.mp4",
    ]
    for p in preferred:
        if p.exists():
            return p

    for p in sorted(case_dir.glob("*.mp4")):
        low = p.name.lower()
        if low.endswith("_overlay.mp4") or low.endswith(".overlay.mp4"):
            continue
        if low in {"pose_demo_out.mp4", "objects_demo_out.mp4"}:
            continue
        return p
    return None


def _find_dataset_original_video_path(view_name: str, case_id: str) -> Optional[Path]:
    view_dir = _resolve_dataset_view_dir(view_name)
    if view_dir is None:
        return None
    for stem in _candidate_case_stems(case_id):
        p = view_dir / f"{stem}.mp4"
        if p.exists():
            return p
    return None


def _resolve_overlay_video_path(video_id: str, case_dir: Path) -> Optional[Path]:
    overlay_candidates = sorted(case_dir.glob("*_overlay.mp4"))
    if overlay_candidates:
        return overlay_candidates[0]
    return _find_docs_video_path(video_id, "overlay")


def _resolve_original_video_path(video_id: str, case_dir: Path) -> Optional[Path]:
    view_name, case_id = _extract_case_tokens(video_id, case_dir)
    p = _find_output_original_video_path(case_dir, video_id, case_id)
    if p is None:
        p = _find_docs_video_path(video_id, "original")
    if p is None:
        p = _find_dataset_original_video_path(view_name, case_id)
    if p is None:
        p = _resolve_overlay_video_path(video_id, case_dir)
    return p if p and p.exists() else None


def _is_nonempty_file(path: Path, min_bytes: int = 1) -> bool:
    try:
        return path.exists() and path.is_file() and int(path.stat().st_size) >= int(min_bytes)
    except Exception:
        return False


def _case_dir_quality(case_dir: Path) -> int:
    score = 0
    if _is_nonempty_file(case_dir / "timeline_chart.json", min_bytes=8):
        score += 120
    if _is_nonempty_file(case_dir / "timeline_viz.json", min_bytes=8):
        score += 90
    if _is_nonempty_file(case_dir / "timeline_data.json", min_bytes=8):
        score += 70
    if _is_nonempty_file(case_dir / "verified_events.jsonl", min_bytes=8):
        score += 60
    if _is_nonempty_file(case_dir / "pipeline_manifest.json", min_bytes=8):
        score += 45
    if _is_nonempty_file(case_dir / "pose_tracks_smooth.jsonl", min_bytes=16) or _is_nonempty_file(case_dir / "pose_tracks_smooth_uq.jsonl", min_bytes=16):
        score += 55
    if _is_nonempty_file(case_dir / "actions_fused.jsonl", min_bytes=8) or _is_nonempty_file(case_dir / "actions.jsonl", min_bytes=8):
        score += 35
    if _is_nonempty_file(case_dir / "transcript.jsonl", min_bytes=8):
        score += 25
    if _is_nonempty_file(case_dir / "student_projection.json", min_bytes=8) or _is_nonempty_file(case_dir / "static_projection.json", min_bytes=8):
        score += 15

    # prefer richer artifacts when multiple branches exist for the same case id
    for fn, cap, weight in [
        ("pose_tracks_smooth.jsonl", 4_000_000, 30),
        ("pose_tracks_smooth_uq.jsonl", 4_000_000, 20),
        ("actions.jsonl", 200_000, 15),
        ("actions.raw.jsonl", 200_000, 10),
        ("actions.fusion_v2.jsonl", 200_000, 18),
        ("actions_fused.jsonl", 200_000, 20),
        ("timeline_viz.json", 50_000, 12),
        ("verified_events.jsonl", 80_000, 10),
    ]:
        p = case_dir / fn
        if not _is_nonempty_file(p):
            continue
        try:
            score += int(min(cap, int(p.stat().st_size)) / cap * weight)
        except Exception:
            pass

    low = case_dir.as_posix().lower()
    if "asr_ablation" in low:
        score -= 120
    if "/_tmp_" in low or "/cache/" in low:
        score -= 80
    if "/_batch/" in low:
        score -= 30
    return score


def _looks_like_case_dir(case_dir: Path) -> bool:
    """Heuristic: a runnable case folder should contain at least one core pipeline artifact."""
    markers = [
        "timeline_chart.json",
        "pipeline_manifest.json",
        "timeline_viz.json",
        "per_person_sequences.json",
        "verified_events.jsonl",
        "event_queries.jsonl",
        "pose_tracks_smooth.jsonl",
        "pose_tracks_smooth_uq.jsonl",
        "actions.jsonl",
        "actions.raw.jsonl",
        "actions.fusion_v2.jsonl",
        "actions_fused.jsonl",
        "student_projection.json",
    ]
    return any(_is_nonempty_file(case_dir / m, min_bytes=2) for m in markers)


def _load_case_analysis_bundle(case_dir: Path) -> Optional[Dict[str, Any]]:
    p = case_dir / "analysis" / "bundle.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _coerce_scalar(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        t = v.strip()
        if t == "":
            return ""
        lo = t.lower()
        if lo == "true":
            return True
        if lo == "false":
            return False
        if re.fullmatch(r"[+-]?\d+", t):
            try:
                return int(t)
            except Exception:
                return t
        if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)", t):
            try:
                return float(t)
            except Exception:
                return t
    return v


def _coerce_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        clean = {str(k): _coerce_scalar(v) for k, v in row.items()}
        out.append(clean)
    return out


def _read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return default


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    encodings = ["utf-8-sig", "utf-8"]
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows = [dict(r) for r in reader]
            break
        except Exception:
            rows = []
            continue
    return _coerce_rows(rows)


def _to_docs_url(path: Path) -> Optional[str]:
    try:
        rel = path.resolve().relative_to(DOCS_DIR.resolve())
    except Exception:
        return None
    return f"/docs/{rel.as_posix()}"


def _load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
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
    return rows


def _timeline_from_verified_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for row in rows:
        win = row.get("window", {})
        if isinstance(win, dict):
            st = float(win.get("start", row.get("window_start", row.get("query_time", 0.0))))
            ed = float(win.get("end", row.get("window_end", row.get("query_time", st + 0.3))))
        else:
            st = float(row.get("window_start", row.get("query_time", 0.0)))
            ed = float(row.get("window_end", row.get("query_time", st + 0.3)))
        if ed <= st:
            ed = st + 0.3
        label = str(row.get("match_label", row.get("label", "mismatch")))
        if label == "match":
            action_id = 0
        elif label == "uncertain":
            action_id = 1
        else:
            action_id = 2
        items.append(
            {
                "type": "verified_event",
                "track_id": int(row.get("track_id", -1)),
                "start": st,
                "end": ed,
                "action_id": action_id,
                "action_label": label,
                "query_id": row.get("query_id", row.get("event_id")),
                "event_type": row.get("event_type"),
                "query_text": row.get("query_text"),
                "reliability": float(row.get("reliability_score", row.get("reliability", 0.0))),
                "match_score": float(row.get("p_match", row.get("match_score", 0.0))),
                "row": int(row.get("track_id", -1)),
            }
        )
    return {"items": items}


_ACTION_NAME_TO_ID: Dict[str, int] = {
    "listen": 0,
    "distract": 1,
    "phone": 2,
    "doze": 3,
    "chat": 4,
    "note": 5,
    "raise": 6,
    "stand": 7,
    "read": 8,
}


def _parse_action_id(action_name: Any, fallback: Any = None) -> int:
    name = str(action_name or "").strip().lower()
    if name in _ACTION_NAME_TO_ID:
        return _ACTION_NAME_TO_ID[name]
    try:
        return int(fallback)
    except Exception:
        return -1


def _timeline_from_actions_rows(rows: List[Dict[str, Any]], fps: float = 25.0) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    safe_fps = max(1e-6, float(fps))
    for row in rows:
        try:
            tid = int(row.get("track_id", -1))
        except Exception:
            tid = -1
        if tid < 0:
            continue

        action_name = str(row.get("action", row.get("action_label", "listen"))).strip().lower()
        action_id = _parse_action_id(action_name, row.get("action_id", -1))

        st = _safe_float(row.get("start", row.get("start_time", row.get("window_start", 0.0))), 0.0)
        ed = _safe_float(row.get("end", row.get("end_time", row.get("window_end", st))), st)

        if "start_frame" in row:
            st = _safe_float(row.get("start_frame"), 0.0) / safe_fps
        if "end_frame" in row:
            ed = _safe_float(row.get("end_frame"), st * safe_fps) / safe_fps

        if ed <= st:
            ed = st + 0.2

        items.append(
            {
                "type": "person",
                "track_id": tid,
                "start": float(st),
                "end": float(ed),
                "action_id": int(action_id),
                "action": action_name,
                "duration": float(ed - st),
                "row": tid,
            }
        )
    return {"items": items}


def _timeline_items_count(payload: Any) -> int:
    if isinstance(payload, dict):
        items = payload.get("items", [])
        if isinstance(items, list):
            return len(items)
        return 0
    if isinstance(payload, list):
        return len(payload)
    return 0


# =========================================================
# 3) 灏忓伐鍏凤細绠€鏄?Levenshtein锛堟棤涓夋柟搴撴椂鍏滃簳锛?
# =========================================================
def simple_levenshtein(seq1: str, seq2: str) -> float:
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=np.float32)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return float(matrix[size_x - 1, size_y - 1])


def _supports_param(cls, param: str) -> bool:
    try:
        return param in inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def _make_mds(dissimilarity: str, random_state: int = 42):
    kwargs = {"n_components": 2, "dissimilarity": dissimilarity, "random_state": random_state}
    if _supports_param(MDS, "n_jobs"):
        kwargs["n_jobs"] = 1
    return MDS(**kwargs)


def _fit_tsne(data: np.ndarray, metric: str, random_state: int = 42) -> np.ndarray:
    n_samples = data.shape[0]
    if n_samples <= 3:
        perp = float(max(1, n_samples - 1))
    else:
        perp = float(min(30, max(5, n_samples // 2)))
    perp = min(perp, float(n_samples) - 1e-3)
    kwargs = {
        "n_components": 2,
        "perplexity": perp,
        "metric": metric,
        "random_state": random_state,
    }
    if _supports_param(TSNE, "n_jobs"):
        kwargs["n_jobs"] = 1
    return TSNE(**kwargs).fit_transform(data)


# =========================================================
# 4) 缂撳瓨璇诲彇锛歵imeline/stats/transcript/tracks
# =========================================================
@lru_cache(maxsize=32)
def load_timeline_data_cached(video_id: str) -> Optional[Dict[str, Any]]:
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        return None

    candidates: List[Tuple[int, int, Dict[str, Any]]] = []

    def _push_timeline(obj: Any, priority: int) -> None:
        if isinstance(obj, dict):
            payload = obj
        elif isinstance(obj, list):
            payload = {"items": obj}
        else:
            return
        n = _timeline_items_count(payload)
        if n > 0:
            candidates.append((n, priority, payload))

    p1 = case_dir / "timeline_chart.json"
    if p1.exists():
        try:
            data = json.loads(p1.read_text(encoding="utf-8"))
            _push_timeline(data, priority=20)
        except Exception:
            pass

    p_viz = case_dir / "timeline_viz.json"
    if p_viz.exists():
        try:
            data = json.loads(p_viz.read_text(encoding="utf-8"))
            _push_timeline(data, priority=40)
        except Exception:
            pass

    p2 = case_dir / "timeline_data.json"
    if p2.exists():
        try:
            data = json.loads(p2.read_text(encoding="utf-8"))
            _push_timeline(data, priority=30)
        except Exception:
            pass

    fps = 25.0
    for meta_file in sorted(case_dir.glob("*.meta.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            fps = _safe_float(meta.get("fps"), 25.0)
            break
        except Exception:
            continue

    p_actions_fused = case_dir / "actions_fused.jsonl"
    if p_actions_fused.exists():
        rows = _load_jsonl_rows(p_actions_fused)
        if rows:
            _push_timeline(_timeline_from_actions_rows(rows, fps=fps), priority=60)

    p_actions = case_dir / "actions.jsonl"
    if p_actions.exists():
        rows = _load_jsonl_rows(p_actions)
        if rows:
            _push_timeline(_timeline_from_actions_rows(rows, fps=fps), priority=55)

    p3 = case_dir / "verified_events.jsonl"
    if p3.exists():
        rows = _load_jsonl_rows(p3)
        if rows:
            _push_timeline(_timeline_from_verified_rows(rows), priority=10)

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]

    return None


@lru_cache(maxsize=32)
def load_stats_data_cached(video_id: str) -> Dict[str, Any]:
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        return {"class_pie_chart": []}
    p = case_dir / "timeline_chart_stats.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"class_pie_chart": []}


@lru_cache(maxsize=32)
def load_transcript_data_cached(video_id: str) -> List[Dict[str, Any]]:
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        return []
    p = case_dir / "transcript.jsonl"
    data: List[Dict[str, Any]] = []
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue
    return data


@lru_cache(maxsize=32)
def load_verified_events_cached(video_id: str) -> List[Dict[str, Any]]:
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        return []
    p = case_dir / "verified_events.jsonl"
    return _load_jsonl_rows(p)


@lru_cache(maxsize=16)
def load_tracks_data_cached(video_id: str) -> Dict[str, Any]:
    """
    杩斿洖缁撴瀯:
    {
      "12": [{"id": 3, "box": [x1,y1,x2,y2]}, ...],
      "13": ...
    }
    """
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        return {}
    p = case_dir / "pose_tracks_smooth.jsonl"
    if not p.exists():
        p = case_dir / "pose_tracks_smooth_uq.jsonl"
    if not p.exists():
        return {}

    tracks: Dict[str, Any] = {}
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                frame = d.get("frame")
                persons = d.get("persons", [])
                tracks[str(frame)] = [{"id": pp.get("track_id"), "box": pp.get("bbox")} for pp in persons]
    except Exception:
        return {}

    return tracks


# =========================================================
# 5) 鏍稿績锛歅rojection锛堟敮鎸?euclidean / levenshtein + pca/mds/tsne锛?
# =========================================================
@lru_cache(maxsize=64)
def compute_projection_cached(video_id: str, method: str, metric: str) -> Dict[str, Any]:
    raw = load_timeline_data_cached(video_id)
    if metric != "spatial" and not raw:
        return {"method": method, "metric": metric, "points": []}

    items = raw.get("items", raw if isinstance(raw, list) else []) if raw else []
    if not isinstance(items, list):
        items = []

    # --- 鏋勫缓姣忎釜浜虹殑鏁版嵁 ---
    person_data: Dict[int, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue

        # 鍙鐞?person 琛岋紱group 琛屼細鐢?track_id = -1锛堜綘鐢熸垚 json 鏃跺氨鏄繖涔堝共鐨勶級 :contentReference[oaicite:5]{index=5}
        if item.get("type", "person") != "person":
            continue

        tid = int(item.get("track_id", -999))
        if tid < 0:
            continue

        if tid not in person_data:
            person_data[tid] = {"actions": [], "spatial": []}

        st = float(item.get("start", 0.0))
        ed = float(item.get("end", st))
        code = int(item.get("action_id", -1))
        row = int(item.get("row", 0))

        person_data[tid]["actions"].append({"code": code, "dur": max(0.0, ed - st), "start": st})
        person_data[tid]["spatial"].append(row)

    track_ids = sorted(person_data.keys())
    n_samples = len(track_ids)
    coords = None

    # -------- A) Spatial map from tracked bboxes --------
    if metric == "spatial":
        tracks = load_tracks_data_cached(video_id)
        if not tracks:
            return {"method": "spatial", "metric": "spatial", "points": []}

        agg: Dict[int, Dict[str, float]] = {}
        max_x = 1.0
        max_y = 1.0
        for persons in tracks.values():
            if not isinstance(persons, list):
                continue
            for pp in persons:
                try:
                    tid = int(pp.get("id", -1))
                except Exception:
                    tid = -1
                if tid < 0:
                    continue
                box = pp.get("box")
                if not isinstance(box, (list, tuple)) or len(box) < 4:
                    continue
                try:
                    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                except Exception:
                    continue
                cx = max(0.0, (x1 + x2) * 0.5)
                cy = max(0.0, (y1 + y2) * 0.5)
                max_x = max(max_x, x1, x2, cx)
                max_y = max(max_y, y1, y2, cy)
                s = agg.setdefault(tid, {"sx": 0.0, "sy": 0.0, "n": 0.0})
                s["sx"] += cx
                s["sy"] += cy
                s["n"] += 1.0

        if not agg:
            return {"method": "spatial", "metric": "spatial", "points": []}

        frame_w = max(1.0, max_x)
        frame_h = max(1.0, max_y)
        points = []
        for tid in sorted(agg.keys()):
            s = agg[tid]
            n = max(1.0, s["n"])
            x = min(1.0, max(0.0, (s["sx"] / n) / frame_w))
            y = min(1.0, max(0.0, (s["sy"] / n) / frame_h))
            points.append({"track_id": int(tid), "x": float(x), "y": float(y), "samples": int(n)})
        return {"method": "spatial", "metric": "spatial", "points": points, "frame_w": frame_w, "frame_h": frame_h}

    if n_samples < 2:
        return {"method": method, "metric": metric, "points": []}

    # -------- B) Vector-based: metric = euclidean --------
    if metric == "euclidean":
        # 9 actions + 1 spatial + 1 activity
        features = np.zeros((n_samples, 11), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            acts = sorted(person_data[tid]["actions"], key=lambda x: x["start"])
            total_dur = float(sum(a["dur"] for a in acts))

            counts = np.zeros(9, dtype=np.float32)
            for a in acts:
                c = int(a["code"])
                if 0 <= c < 9:
                    counts[c] += float(a["dur"])

            if total_dur > 1e-6:
                features[i, :9] = counts / total_dur

            ys = person_data[tid]["spatial"]
            features[i, 9] = float(sum(ys) / max(1, len(ys)))
            features[i, 10] = float(len(acts) / total_dur) if total_dur > 1e-6 else 0.0

        X = StandardScaler().fit_transform(features)

        if method == "pca":
            coords = PCA(n_components=2).fit_transform(X)
        elif method == "mds":
            coords = _make_mds("euclidean", random_state=42).fit_transform(X)
        elif method == "tsne":
            try:
                coords = _fit_tsne(X, metric="euclidean", random_state=42)
            except Exception:
                coords = PCA(n_components=2).fit_transform(X)
        else:
            # 涓嶈璇嗗氨鍥為€€
            coords = PCA(n_components=2).fit_transform(X)

    # -------- C) Sequence-based: metric = levenshtein --------
    elif metric == "levenshtein":
        sequences: List[str] = []
        for tid in track_ids:
            acts = sorted(person_data[tid]["actions"], key=lambda x: x["start"])
            # 绠€鍗曟嫾鎺?action_id 瀛楃涓?
            seq_str = "".join([str(int(a["code"])) for a in acts])
            sequences.append(seq_str)

        dist_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if Levenshtein is not None:
                    d = float(Levenshtein.distance(sequences[i], sequences[j]))
                else:
                    d = simple_levenshtein(sequences[i], sequences[j])

                max_len = max(len(sequences[i]), len(sequences[j]), 1)
                dist_matrix[i, j] = dist_matrix[j, i] = d / float(max_len)

        # PCA 涓嶈兘澶勭悊璺濈鐭╅樀锛屽己鍒跺洖钀藉埌 MDS
        if method == "pca":
            method = "mds"

        if method == "mds":
            coords = _make_mds("precomputed", random_state=42).fit_transform(dist_matrix)
        elif method == "tsne":
            try:
                coords = _fit_tsne(dist_matrix, metric="precomputed", random_state=42)
            except Exception:
                coords = _make_mds("precomputed", random_state=42).fit_transform(dist_matrix)
        else:
            coords = _make_mds("precomputed", random_state=42).fit_transform(dist_matrix)

    else:
        # 鏈煡 metric
        return {"method": method, "metric": metric, "points": []}

    # 褰掍竴鍖栧埌 0-1
    coords = MinMaxScaler().fit_transform(coords)

    points = []
    for i, tid in enumerate(track_ids):
        points.append({"track_id": int(tid), "x": float(coords[i, 0]), "y": float(coords[i, 1])})

    return {"method": method, "metric": metric, "points": points}


def _resolve_existing_path(path_text: str) -> Optional[Path]:
    if not path_text:
        return None
    p = Path(path_text)
    if p.is_absolute():
        return p if p.exists() else None
    probe = (PROJECT_ROOT / path_text).resolve()
    if probe.exists():
        return probe
    return None


def _load_docs_videos() -> List[Dict[str, Any]]:
    if not PAPER_VIDEO_DIR.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for p in sorted(PAPER_VIDEO_DIR.glob("*.mp4")):
        name = p.name
        variant = "default"
        case_id = name[:-4]
        if name.endswith(".overlay.mp4"):
            variant = "overlay"
            case_id = name[: -len(".overlay.mp4")]
        elif name.endswith(".original.mp4"):
            variant = "original"
            case_id = name[: -len(".original.mp4")]
        rows.append(
            {
                "name": name,
                "case_id": case_id,
                "variant": variant,
                "url": _to_docs_url(p),
            }
        )
    return rows


def _build_run_summary(run_metric_ci: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for row in run_metric_ci:
        run_name = str(row.get("run_name", ""))
        metric = str(row.get("metric", ""))
        if not run_name:
            continue
        summary.setdefault(run_name, {"run_name": run_name})
        if metric == "verified_p_match_mean":
            summary[run_name]["verified_mean"] = _safe_float(row.get("mean"), 0.0)
            summary[run_name]["verified_ci_low"] = _safe_float(row.get("ci95_low"), 0.0)
            summary[run_name]["verified_ci_high"] = _safe_float(row.get("ci95_high"), 0.0)
        elif metric == "align_avg_candidates":
            summary[run_name]["align_mean"] = _safe_float(row.get("mean"), 0.0)
        elif metric == "elapsed_sec":
            summary[run_name]["elapsed_mean"] = _safe_float(row.get("mean"), 0.0)
    out = list(summary.values())
    out.sort(key=lambda x: x.get("verified_mean", -1.0), reverse=True)
    return out


@lru_cache(maxsize=4)
def load_paper_v2_bundle_cached() -> Dict[str, Any]:
    showcase_path = PAPER_TABLE_DIR / "showcase_data.json"
    fallback_ci = PAPER_TABLE_DIR / "tbl01_run_metric_ci_enhanced.csv"
    fallback_pairs = PAPER_TABLE_DIR / "tbl02_case_pairs_mainline_behavior.csv"
    fallback_all_cases = PAPER_TABLE_DIR / "tbl03_all_case_metrics_long.csv"
    fallback_selection = PAPER_TABLE_DIR / "chart_selection_matrix.csv"
    fallback_family = PAPER_TABLE_DIR / "d3og_family_counts.csv"
    figure_manifest = PAPER_TABLE_DIR / "tbl05_figure_manifest.csv"

    payload = _read_json_file(showcase_path, {})
    if not isinstance(payload, dict):
        payload = {}

    run_metric_ci = payload.get("run_metric_ci", [])
    case_pairs = payload.get("case_pairs", [])
    noise_curve = payload.get("noise_curve", [])
    reliability_bins = payload.get("reliability_bins", [])
    chart_selection = payload.get("chart_selection_matrix", [])
    family_counts = payload.get("d3og_family_counts", [])

    if not isinstance(run_metric_ci, list) or not run_metric_ci:
        run_metric_ci = _read_csv_rows(fallback_ci)
    else:
        run_metric_ci = _coerce_rows(run_metric_ci)

    if not isinstance(case_pairs, list) or not case_pairs:
        case_pairs = _read_csv_rows(fallback_pairs)
    else:
        case_pairs = _coerce_rows(case_pairs)

    if not isinstance(noise_curve, list):
        noise_curve = []
    else:
        noise_curve = _coerce_rows(noise_curve)

    if not isinstance(reliability_bins, list):
        reliability_bins = []
    else:
        reliability_bins = _coerce_rows(reliability_bins)

    if not isinstance(chart_selection, list) or not chart_selection:
        chart_selection = _read_csv_rows(fallback_selection)
    else:
        chart_selection = _coerce_rows(chart_selection)

    if not isinstance(family_counts, list) or not family_counts:
        family_counts = _read_csv_rows(fallback_family)
    else:
        family_counts = _coerce_rows(family_counts)

    all_case_metrics = _read_csv_rows(fallback_all_cases)

    for r in case_pairs:
        r["d_verified_p_match_mean"] = _safe_float(r.get("d_verified_p_match_mean"), 0.0)
        r["d_align_avg_candidates"] = _safe_float(r.get("d_align_avg_candidates"), 0.0)
        r["d_elapsed_sec"] = _safe_float(r.get("d_elapsed_sec"), 0.0)
        r["duration_sec"] = _safe_float(r.get("duration_sec"), 0.0)
        if "abs_d_verified_p_match_mean" not in r:
            r["abs_d_verified_p_match_mean"] = abs(_safe_float(r.get("d_verified_p_match_mean"), 0.0))
        else:
            r["abs_d_verified_p_match_mean"] = _safe_float(r.get("abs_d_verified_p_match_mean"), 0.0)

    for r in all_case_metrics:
        r["duration_sec"] = _safe_float(r.get("duration_sec"), 0.0)
        r["elapsed_sec"] = _safe_float(r.get("elapsed_sec"), 0.0)
        r["verified_p_match_mean"] = _safe_float(r.get("verified_p_match_mean"), 0.0)
        r["align_avg_candidates"] = _safe_float(r.get("align_avg_candidates"), 0.0)

    case_pairs_sorted = sorted(case_pairs, key=lambda x: _safe_float(x.get("abs_d_verified_p_match_mean"), 0.0), reverse=True)
    top_cases = case_pairs_sorted[:80]
    case_pair_index = {str(r.get("case_id", "")): r for r in case_pairs if str(r.get("case_id", ""))}

    by_case_runs: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_case_metrics:
        key = str(r.get("case_id", ""))
        if not key:
            continue
        by_case_runs.setdefault(key, []).append(r)
    for key in by_case_runs:
        by_case_runs[key] = sorted(by_case_runs[key], key=lambda x: str(x.get("run_name", "")))

    selection_lookup = {str(r.get("chart_id", "")): r for r in chart_selection}
    figures: List[Dict[str, Any]] = []
    for row in _read_csv_rows(figure_manifest):
        fig_id = str(row.get("figure_id", ""))
        path_text = str(row.get("path", ""))
        fig_path = _resolve_existing_path(path_text)
        fig_url = _to_docs_url(fig_path) if fig_path else None
        sel = selection_lookup.get(fig_id, {})
        figures.append(
            {
                "figure_id": fig_id,
                "url": fig_url,
                "title": sel.get("analysis_task", fig_id),
                "d3_title": sel.get("d3_title"),
                "d3_url": sel.get("d3_url"),
            }
        )

    videos = _load_docs_videos()
    run_summary = _build_run_summary(run_metric_ci)

    return {
        "meta": {
            "generated_at": str(payload.get("meta", {}).get("generated_at", "")),
            "table_dir": str(PAPER_TABLE_DIR),
            "chart_dir": str(PAPER_CHART_DIR),
            "has_showcase_json": showcase_path.exists(),
            "has_case_pairs_csv": fallback_pairs.exists(),
            "has_run_metric_ci_csv": fallback_ci.exists(),
            "has_all_case_metrics_csv": fallback_all_cases.exists(),
        },
        "run_metric_ci": run_metric_ci,
        "run_summary": run_summary,
        "case_pairs": case_pairs,
        "top_cases": top_cases,
        "all_case_metrics": all_case_metrics,
        "noise_curve": noise_curve,
        "reliability_bins": reliability_bins,
        "chart_selection": chart_selection,
        "d3og_family_counts": family_counts,
        "figures": figures,
        "videos": videos,
        "case_pair_index": case_pair_index,
        "by_case_runs": by_case_runs,
    }


# =========================================================
# 6) 璺敱锛欻TML + API
# =========================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    涓婚〉锛氶粯璁ゆ覆鏌?templates/index.html
    濡傛灉浣犳殏鏃舵病鍋氭ā鏉匡紝涔熶細鍥為€€鍒?PROJECT_ROOT/index.html
    """
    # 浼樺厛 templates
    index_tpl = Path(TEMPLATE_DIR) / "index.html"
    if index_tpl.exists():
        return templates.TemplateResponse("index.html", {"request": request})

    # 鍥為€€鍒版牴鐩綍 index.html
    index_file = PROJECT_ROOT / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))

    raise HTTPException(404, "index.html not found in templates/ or project root")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    candidates = [
        DOCS_DIR / "favicon.ico",
        PROJECT_ROOT / "favicon.ico",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return FileResponse(str(p))
    return Response(status_code=204)


@app.get("/paper/v2", response_class=HTMLResponse)
async def paper_v2_dashboard():
    page = DOCS_DIR / "paper_v2_dashboard.html"
    if page.exists():
        return FileResponse(str(page))
    raise HTTPException(404, "paper_v2_dashboard.html not found under docs/")


@app.get("/api/paper/v2/overview")
def get_paper_v2_overview():
    bundle = load_paper_v2_bundle_cached().copy()
    bundle.pop("case_pair_index", None)
    bundle.pop("by_case_runs", None)
    return bundle


@app.get("/api/paper/v2/case/{case_id}")
def get_paper_v2_case(case_id: str):
    bundle = load_paper_v2_bundle_cached()
    key = str(case_id).strip()
    case_row = bundle.get("case_pair_index", {}).get(key)
    if case_row is None:
        raise HTTPException(404, f"case not found in paper_v2 data: {case_id}")
    runs = bundle.get("by_case_runs", {}).get(key, [])
    media = None
    try:
        media = get_media(key)
    except Exception:
        media = None
    return {
        "case_id": key,
        "case_pair": case_row,
        "runs": runs,
        "media": media,
    }


@app.post("/api/paper/v2/refresh")
def refresh_paper_v2_cache():
    load_paper_v2_bundle_cached.cache_clear()
    return {
        "ok": True,
        "message": "paper_v2 cache cleared",
    }


@app.get("/api/list_cases")
def list_cases():
    """
    鑷姩鎵弿 output 鐩綍锛屽憡璇夊墠绔湁鍝簺 demo 鍙敤锛堝吋瀹逛綘 txt 鐨?Flask 鎺ュ彛锛?:contentReference[oaicite:6]{index=6}
    """
    if not OUTPUT_DIR.exists():
        return []
    # 閫掑綊鎵弿鈥滅湡姝ｇ殑 case 鐩綍鈥濓紝閬垮厤鎶娾€滄暀甯堣瑙?鏂滀笂鏂硅瑙?鈥濊繖绉嶈瑙掔埗鐩綍璇綋鎴?case銆?
    case_dirs = [d for d in OUTPUT_DIR.rglob("*") if d.is_dir() and _looks_like_case_dir(d)]
    cases: List[Dict[str, Any]] = []
    for case_dir in case_dirs:
        rel_parts = case_dir.relative_to(OUTPUT_DIR).parts
        view_name = ""
        if len(rel_parts) >= 2:
            # 鏂扮粨鏋勯€氬父鏄?.../<view>/<video_id>
            view_name = rel_parts[-2]
        video_id = case_dir.name
        original_path = _resolve_original_video_path(video_id, case_dir)
        cases.append({
            "view": view_name,
            "video_id": video_id,
            "case_id": video_id.split("__")[-1],
            "path": str(case_dir),
            "has_original_video": bool(original_path is not None),
        })

    # 鍘婚噸锛堝悓鍚?video_id 鍙兘鍑虹幇鍦ㄤ笉鍚岀紦瀛樼洰褰曪級
    uniq: Dict[str, Dict[str, Any]] = {}
    for c in cases:
        key = c["video_id"]
        if key not in uniq:
            uniq[key] = c
            continue
        old = uniq[key]
        old_q = _case_dir_quality(Path(old["path"]))
        new_q = _case_dir_quality(Path(c["path"]))
        if new_q > old_q:
            uniq[key] = c
        elif new_q == old_q and len(str(c["path"])) < len(str(old["path"])):
            uniq[key] = c

    return sorted(
        list(uniq.values()),
        key=lambda x: (str(x.get("view", "")) == "", str(x.get("view", "")), str(x.get("video_id", ""))),
    )


@app.get("/api/analysis/list_cases")
def list_analysis_cases():
    cases = list_cases()
    out: List[Dict[str, Any]] = []
    for c in cases:
        video_id = str(c.get("video_id", ""))
        case_dir = _find_case_dir_for_analysis(video_id)
        if case_dir is None:
            continue
        bundle = _load_case_analysis_bundle(case_dir)
        bundle_path = case_dir / "analysis" / "bundle.json"
        proj_path = case_dir / "analysis" / "projection.json"
        out.append(
            {
                "video_id": video_id,
                "view": str(c.get("view", "")),
                "case_id": str(c.get("case_id", "")),
                "has_analysis_bundle": bundle is not None,
                "analysis_bundle": _to_output_url(bundle_path) if bundle_path.exists() else None,
                "projection": _to_output_url(proj_path) if proj_path.exists() else None,
                "num_slices": len(bundle.get("slices", [])) if bundle else 0,
            }
        )
    return out


@app.get("/api/analysis/case/{case_id}")
def get_analysis_case(case_id: str):
    case_dir = _find_case_dir_for_analysis(case_id)
    if case_dir is None:
        raise HTTPException(404, f"Case not found: {case_id}")
    bundle = _load_case_analysis_bundle(case_dir)
    if bundle is None:
        raise HTTPException(404, f"Analysis bundle not found: {case_id}")
    return bundle


@app.get("/api/analysis/slices/{case_id}")
def get_analysis_slices(case_id: str):
    case_dir = _find_case_dir_for_analysis(case_id)
    if case_dir is None:
        raise HTTPException(404, f"Case not found: {case_id}")
    bundle = _load_case_analysis_bundle(case_dir)
    if bundle is None:
        raise HTTPException(404, f"Analysis bundle not found: {case_id}")
    return {
        "video_id": case_dir.name,
        "view": case_dir.parent.name if case_dir.parent != OUTPUT_DIR else "",
        "meta": bundle.get("meta", {}),
        "video_meta": bundle.get("video_meta", {}),
        "slices": bundle.get("slices", []),
    }


@app.get("/api/media/{video_id}")
def get_media(video_id: str):
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        raise HTTPException(404, f"Case not found: {video_id}")

    view_name, case_id = _extract_case_tokens(video_id, case_dir)
    overlay_path = _resolve_overlay_video_path(video_id, case_dir)
    overlay_url = _path_to_public_url(overlay_path)

    original_path = _resolve_original_video_path(video_id, case_dir)
    original_url = f"/api/media/{video_id}/original" if original_path and original_path.exists() else None

    payload = {
        "video_id": case_dir.name,
        "view": view_name,
        "case_id": case_id,
        "original": original_url,
        "overlay": overlay_url,
        "pose_demo": _to_output_url(case_dir / "pose_demo_out.mp4") if (case_dir / "pose_demo_out.mp4").exists() else None,
        "objects_demo": _to_output_url(case_dir / "objects_demo_out.mp4") if (case_dir / "objects_demo_out.mp4").exists() else None,
        "timeline_png": _to_output_url(case_dir / "timeline_chart.png") if (case_dir / "timeline_chart.png").exists() else None,
        "projection_json": _to_output_url(case_dir / "student_projection.json") if (case_dir / "student_projection.json").exists() else None,
        "verified_events": _to_output_url(case_dir / "verified_events.jsonl") if (case_dir / "verified_events.jsonl").exists() else None,
        "event_queries": _to_output_url(case_dir / "event_queries.jsonl") if (case_dir / "event_queries.jsonl").exists() else None,
        "actions": _to_output_url(case_dir / "actions_fused.jsonl") if (case_dir / "actions_fused.jsonl").exists() else (
            _to_output_url(case_dir / "actions.jsonl") if (case_dir / "actions.jsonl").exists() else None
        ),
    }
    return payload


@app.get("/api/media/{video_id}/original")
def get_media_original(video_id: str):
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        raise HTTPException(404, f"Case not found: {video_id}")
    p = _resolve_original_video_path(video_id, case_dir)
    if p is None or (not p.exists()):
        raise HTTPException(404, f"Original video not found: {video_id}")
    return FileResponse(str(p))


@app.get("/api/case/{video_id}/manifest")
def get_case_manifest(video_id: str):
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        raise HTTPException(404, f"Case not found: {video_id}")

    expected = [
        "pipeline_manifest.json",
        "pose_keypoints_v2.jsonl", "pose_tracks_smooth.jsonl", "pose_tracks_smooth_uq.jsonl",
        "objects.jsonl", "objects.semantic.jsonl", "behavior_det.jsonl", "behavior_det.semantic.jsonl",
        "actions.raw.jsonl", "actions.jsonl", "actions_fused.jsonl", "actions.fusion_v2.jsonl",
        "actions.behavior.jsonl", "actions.behavior.semantic.jsonl", "actions.behavior_aug.jsonl",
        "transcript.jsonl", "asr_quality_report.json", "event_queries.jsonl",
        "event_queries.visual_fallback.jsonl", "event_queries.fusion_v2.jsonl",
        "align_multimodal.json", "verified_events.jsonl", "per_person_sequences.json",
        "timeline_chart.json", "timeline_chart.png", "timeline_students.csv",
        "student_projection.json", "group_events.jsonl", "embeddings.pkl",
        "fusion_contract_report.json", "pipeline_contract_v2_report.json",
        "verifier.pt", "verifier_samples.raw.jsonl", "verifier_samples_train.jsonl",
        "verifier_report.raw.json", "verifier_eval_report.json",
        "verifier_calibration_report.json", "verifier_reliability_diagram.svg",
        "pose_demo_out.mp4", "objects_demo_out.mp4",
    ]
    files = []
    for name in expected:
        p = case_dir / name
        files.append({
            "name": name,
            "exists": p.exists(),
            "size": int(p.stat().st_size) if p.exists() else 0,
            "url": _to_output_url(p) if p.exists() else None,
        })

    schema_version = ""
    pm = case_dir / "pipeline_manifest.json"
    if pm.exists():
        try:
            obj = json.loads(pm.read_text(encoding="utf-8"))
            schema_version = str(obj.get("schema_version", ""))
        except Exception:
            schema_version = ""

    return {
        "video_id": case_dir.name,
        "view": case_dir.parent.name if case_dir.parent != OUTPUT_DIR else "",
        "path": str(case_dir),
        "schema_version": schema_version,
        "files": files,
        "ready_for_frontend": any(
            (f["name"] == "timeline_chart.json" and f["exists"])
            or (f["name"] == "verified_events.jsonl" and f["exists"])
            for f in files
        ),
    }


@app.get("/api/config")
def get_config():
    """
    鏆撮湶閮ㄥ垎閰嶇疆锛堝吋瀹逛綘 txt 鐨?Flask 鎺ュ彛锛?:contentReference[oaicite:7]{index=7}
    """
    return {
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "data_dir": str(DATA_DIR),
        "video_dir": str(VIDEO_DIR),
        "template_dir": str(TEMPLATE_DIR),
    }


@app.get("/api/timeline/{video_id}")
def get_timeline(video_id: str):
    data = load_timeline_data_cached(video_id)
    if not data:
        return {"items": []}
    # 缁熶竴杈撳嚭 {"items": [...]}
    if isinstance(data, list):
        return {"items": data}
    return {"items": data.get("items", [])}


@app.get("/api/stats/{video_id}")
def get_stats(video_id: str):
    return load_stats_data_cached(video_id)


@app.get("/api/transcript/{video_id}")
def get_transcript(video_id: str):
    raw = load_transcript_data_cached(video_id)
    verified_rows = load_verified_events_cached(video_id)

    def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    out = []
    for l in raw:
        st = _safe_float(l.get("start"), 0.0)
        ed = _safe_float(l.get("end"), st)
        if ed < st:
            st, ed = ed, st

        linked = []
        for v in verified_rows:
            q_t = _safe_float(v.get("query_time"), 0.0)
            w_st = _safe_float(v.get("window_start", q_t), q_t)
            w_ed = _safe_float(v.get("window_end", q_t), q_t)
            if w_ed < w_st:
                w_st, w_ed = w_ed, w_st
            if _overlap(st, ed, w_st, w_ed) > 0 or (st <= q_t <= ed):
                linked.append(v)

        linked.sort(key=lambda x: _safe_float(x.get("reliability"), 0.0), reverse=True)
        top = linked[0] if linked else None
        label = str(top.get("label")) if top else ""

        out.append({
            "start": st,
            "end": ed,
            "text": l.get("text", l.get("sentence", "")),
            "verified": bool(label == "match" or l.get("verified", False)),
            "reliability": _safe_float(top.get("reliability"), 0.0) if top else 0.0,
            "verification_label": label if label else None,
            "event_id": top.get("event_id") if top else None,
            "event_type": top.get("event_type") if top else None,
        })
    return out


@app.get("/api/tracks/{video_id}")
def get_tracks(video_id: str):
    return load_tracks_data_cached(video_id)


@app.get("/api/projection/{video_id}")
def get_projection(video_id: str, method: str = "pca", metric: str = "euclidean"):
    method = method.lower().strip()
    metric = metric.lower().strip()
    if method not in {"pca", "mds", "tsne"}:
        raise HTTPException(400, "method must be one of: pca, mds, tsne")
    if metric not in {"euclidean", "levenshtein", "spatial"}:
        raise HTTPException(400, "metric must be one of: euclidean, levenshtein, spatial")
    return compute_projection_cached(video_id, method, metric)


# =========================================================
# 7) 鍚姩
# =========================================================
if __name__ == "__main__":
    import uvicorn

    print("Server starting...")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"OUTPUT_DIR  : {OUTPUT_DIR}")
    print(f"DATA_DIR    : {DATA_DIR}")
    print("Open http://localhost:8000")

    uvicorn.run(app, host="0.0.0.0", port=8000)

