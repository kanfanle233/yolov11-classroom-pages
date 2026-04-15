import os
import sys
import json
from pathlib import Path
from functools import lru_cache
import inspect
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

# 浣犵殑鍓嶇妯℃澘涓€鑸湪 web_viz/templates锛涘鏋滄病鏈夊氨鍥為€€鍒伴」鐩牴鐩綍
TEMPLATE_DIR = (PROJECT_ROOT / "web_viz" / "templates")
if not TEMPLATE_DIR.exists():
    TEMPLATE_DIR = PROJECT_ROOT

# 闈欐€佽祫婧愮洰褰曪細web_viz/static 鎴栭」鐩牴
STATIC_DIR = (PROJECT_ROOT / "web_viz" / "static")
if not STATIC_DIR.exists():
    STATIC_DIR = PROJECT_ROOT
DOCS_DIR = (PROJECT_ROOT / "docs").resolve()

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

# Paper pages demo (GitHub Pages-equivalent local serving)
if DOCS_DIR.exists():
    app.mount("/docs", StaticFiles(directory=str(DOCS_DIR), html=True), name="docs")

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def _find_case_dir(video_id: str) -> Optional[Path]:
    """Find case directory by exact folder name, supporting nested view/case layout."""
    direct = OUTPUT_DIR / video_id
    if direct.exists() and direct.is_dir() and _looks_like_case_dir(direct):
        return direct
    if not OUTPUT_DIR.exists():
        return None
    for candidate in OUTPUT_DIR.rglob(video_id):
        if candidate.is_dir() and _looks_like_case_dir(candidate):
            return candidate
    return None


def _to_output_url(path: Path) -> str:
    rel = path.resolve().relative_to(OUTPUT_DIR)
    return f"/output/{rel.as_posix()}"


def _looks_like_case_dir(case_dir: Path) -> bool:
    """Heuristic: a runnable case folder should contain at least one core pipeline artifact."""
    markers = [
        "timeline_chart.json",
        "per_person_sequences.json",
        "verified_events.jsonl",
        "event_queries.jsonl",
        "pose_tracks_smooth.jsonl",
        "pose_tracks_smooth_uq.jsonl",
        "actions.jsonl",
        "actions_fused.jsonl",
        "student_projection.json",
    ]
    return any((case_dir / m).exists() for m in markers)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


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
        kwargs["n_jobs"] = -1
    return MDS(**kwargs)


def _fit_tsne(data: np.ndarray, metric: str, random_state: int = 42) -> np.ndarray:
    n_samples = data.shape[0]
    perp = min(30, max(5, n_samples - 1))
    kwargs = {
        "n_components": 2,
        "perplexity": perp,
        "metric": metric,
        "random_state": random_state,
    }
    if _supports_param(TSNE, "n_jobs"):
        kwargs["n_jobs"] = -1
    return TSNE(**kwargs).fit_transform(data)


# =========================================================
# 4) 缂撳瓨璇诲彇锛歵imeline/stats/transcript/tracks
# =========================================================
@lru_cache(maxsize=32)
def load_timeline_data_cached(video_id: str) -> Optional[Dict[str, Any]]:
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        return None
    # 鍏堟壘 timeline_chart.json锛堜綘鐜板湪 step10 鐢熸垚鐨勬槸 *.json锛岀粨鏋?{"items": [...]}锛?:contentReference[oaicite:4]{index=4}
    p1 = case_dir / "timeline_chart.json"
    if p1.exists():
        return json.loads(p1.read_text(encoding="utf-8"))

    # 鍥為€€ timeline_data.json
    p2 = case_dir / "timeline_data.json"
    if p2.exists():
        return json.loads(p2.read_text(encoding="utf-8"))

    p3 = case_dir / "verified_events.jsonl"
    if p3.exists():
        rows = _load_jsonl_rows(p3)
        if rows:
            return _timeline_from_verified_rows(rows)

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
    if not raw:
        return {"method": method, "metric": metric, "points": []}

    items = raw.get("items", raw if isinstance(raw, list) else [])
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
    if n_samples < 2:
        return {"method": method, "metric": metric, "points": []}

    coords = None

    # -------- A) Vector-based: metric = euclidean --------
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
            coords = _fit_tsne(X, metric="euclidean", random_state=42)
        else:
            # 涓嶈璇嗗氨鍥為€€
            coords = PCA(n_components=2).fit_transform(X)

    # -------- B) Sequence-based: metric = levenshtein --------
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
            coords = _fit_tsne(dist_matrix, metric="precomputed", random_state=42)
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


@app.get("/api/list_cases")
def list_cases():
    """
    鑷姩鎵弿 output 鐩綍锛屽憡璇夊墠绔湁鍝簺 demo 鍙敤锛堝吋瀹逛綘 txt 鐨?Flask 鎺ュ彛锛?:contentReference[oaicite:6]{index=6}
    """
    if not OUTPUT_DIR.exists():
        return []
    # 閫掑綊鎵弿鈥滅湡姝ｇ殑 case 鐩綍鈥濓紝閬垮厤鎶娾€滄暀甯堣瑙?鏂滀笂鏂硅瑙?鈥濊繖绉嶈瑙掔埗鐩綍璇綋鎴?case銆?
    case_dirs = [d for d in OUTPUT_DIR.rglob("*") if d.is_dir() and _looks_like_case_dir(d)]
    cases: List[Dict[str, str]] = []
    for case_dir in case_dirs:
        rel_parts = case_dir.relative_to(OUTPUT_DIR).parts
        view_name = ""
        if len(rel_parts) >= 2:
            # 鏂扮粨鏋勯€氬父鏄?.../<view>/<video_id>
            view_name = rel_parts[-2]
        video_id = case_dir.name
        cases.append({
            "view": view_name,
            "video_id": video_id,
            "case_id": video_id.split("__")[-1],
            "path": str(case_dir),
        })

    # 鍘婚噸锛堝悓鍚?video_id 鍙兘鍑虹幇鍦ㄤ笉鍚岀紦瀛樼洰褰曪級
    uniq: Dict[str, Dict[str, str]] = {}
    for c in cases:
        key = c["video_id"]
        if key not in uniq or len(c["path"]) < len(uniq[key]["path"]):
            uniq[key] = c

    return sorted(list(uniq.values()), key=lambda x: (x["view"], x["video_id"]))


@app.get("/api/media/{video_id}")
def get_media(video_id: str):
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        raise HTTPException(404, f"Case not found: {video_id}")

    overlay_candidates = sorted(case_dir.glob("*_overlay.mp4"))
    overlay = overlay_candidates[0] if overlay_candidates else None

    payload = {
        "video_id": case_dir.name,
        "view": case_dir.parent.name if case_dir.parent != OUTPUT_DIR else "",
        "overlay": _to_output_url(overlay) if overlay and overlay.exists() else None,
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


@app.get("/api/case/{video_id}/manifest")
def get_case_manifest(video_id: str):
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        raise HTTPException(404, f"Case not found: {video_id}")

    expected = [
        "actions.jsonl", "actions_fused.jsonl", "transcript.jsonl", "pose_tracks_smooth.jsonl",
        "pose_tracks_smooth_uq.jsonl", "event_queries.jsonl", "align_multimodal.json", "verified_events.jsonl",
        "pose_demo_out.mp4", "objects_demo_out.mp4", "student_projection.json", "timeline_chart.json",
        "timeline_chart.png", "group_events.jsonl", "per_person_sequences.json", "embeddings.pkl",
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

    return {
        "video_id": case_dir.name,
        "view": case_dir.parent.name if case_dir.parent != OUTPUT_DIR else "",
        "path": str(case_dir),
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
        raise HTTPException(404, "Timeline data not found")
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
    if metric not in {"euclidean", "levenshtein"}:
        raise HTTPException(400, "metric must be one of: euclidean, levenshtein")
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

