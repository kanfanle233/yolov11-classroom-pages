import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules.peer_context import (
    apply_peer_correction,
    build_spatial_neighbor_index,
    extract_peer_features,
)

DEFAULT_VIDEO_FPS = None
DEFAULT_VISUAL_TIME_MODE = "start"
SCHEMA_VERSION = "1.2.0"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    if not path.exists():
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    data.append(obj)
            except Exception:
                continue
    return data


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None


def normalize_visual_actions(
    raw_actions: List[Dict[str, Any]],
    fps: Optional[float] = None,
    time_mode: str = "start",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if time_mode not in ("start", "end", "mid"):
        time_mode = "start"

    for a in raw_actions:
        if not isinstance(a, dict):
            continue
        track_id = a.get("track_id", a.get("id", a.get("student_id", a.get("tid"))))
        action = a.get("action", a.get("label", a.get("class_name", a.get("class"))))
        conf = a.get("confidence", a.get("conf", a.get("score", 1.0)))
        action_code = a.get("action_code")

        track_id_i = _to_int(track_id)
        if track_id_i is None or action is None:
            continue

        conf_f = _to_float(conf)
        if conf_f is None:
            conf_f = 1.0

        st = _to_float(a.get("start_time"))
        et = _to_float(a.get("end_time"))
        if st is None and et is None:
            sf = _to_float(a.get("start_frame"))
            ef = _to_float(a.get("end_frame"))
            if fps and (sf is not None or ef is not None):
                if sf is not None:
                    st = sf / float(fps)
                if ef is not None:
                    et = ef / float(fps)

        t: Optional[float] = None
        if st is not None and et is not None:
            if time_mode == "start":
                t = st
            elif time_mode == "end":
                t = et
            else:
                t = (st + et) / 2.0
        elif st is not None:
            t = st
        elif et is not None:
            t = et

        dur = _to_float(a.get("duration"))
        if dur is None and st is not None and et is not None:
            dur = max(0.0, et - st)

        item: Dict[str, Any] = {
            "track_id": track_id_i,
            "action": str(action).lower().strip(),
            "confidence": float(conf_f),
            "action_code": int(action_code) if action_code is not None else None,
            "time": float(t) if t is not None else None,
            "start_time": float(st) if st is not None else None,
            "end_time": float(et) if et is not None else None,
            "duration": float(dur) if dur is not None else None,
        }

        for k in ("side", "start_frame", "end_frame", "objects_found"):
            if k in a:
                item[k] = a[k]
        out.append(item)

    out.sort(key=lambda x: (x["track_id"], x["time"] if x["time"] is not None else 1e18))
    return out


def normalize_transcripts(raw_transcripts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in raw_transcripts:
        if not isinstance(r, dict):
            continue
        text = r.get("text")
        st = _to_float(r.get("start", r.get("time_start", r.get("ts_start"))))
        et = _to_float(r.get("end", r.get("time_end", r.get("ts_end"))))
        if text is None or st is None or et is None:
            continue
        if et < st:
            st, et = et, st
        out.append({"start": float(st), "end": float(et), "text": str(text)})
    out.sort(key=lambda x: x["start"])
    return out


def _validate_visual_actions(visual_actions: List[Dict[str, Any]]) -> Tuple[int, int]:
    valid = 0
    invalid = 0
    for a in visual_actions:
        ok = True
        if _to_int(a.get("track_id")) is None:
            ok = False
        if not a.get("action"):
            ok = False
        if ok:
            valid += 1
        else:
            invalid += 1
    return valid, invalid


def apply_peer_aware_stage(
    visual_actions: List[Dict[str, Any]],
    pose_tracks: List[Dict[str, Any]],
    peer_radius: float = 0.15,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Apply peer-aware correction and return (corrected_actions, peer_context_map)."""
    if not visual_actions or not pose_tracks:
        return visual_actions, {}

    neighbor_index = build_spatial_neighbor_index(pose_tracks, radius=peer_radius)
    by_tid: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for a in visual_actions:
        tid = _to_int(a.get("track_id"))
        if tid is not None:
            by_tid[tid].append(a)

    corrected_all: List[Dict[str, Any]] = []
    peer_ctx_map: Dict[int, Dict[str, Any]] = {}

    for tid, seq in by_tid.items():
        frame_to_neighbors = neighbor_index.get(tid, {})
        neighbors = set()
        for _, ids in frame_to_neighbors.items():
            neighbors.update(ids)

        peer_feat = extract_peer_features(target_id=tid, neighbors=neighbors, actions=visual_actions)
        corrected = apply_peer_correction(seq, peer_feat, threshold=0.55)

        corrected_all.extend(corrected["actions"])
        peer_ctx_map[tid] = {
            "neighbor_count": int(peer_feat.get("neighbor_count", 0)),
            "dominant_peer_action": peer_feat.get("dominant_peer_action", "none"),
            "peer_agreement_score": float(peer_feat.get("peer_agreement_score", 0.0)),
            "correction_applied": bool(corrected.get("changed", False)),
        }

    corrected_all.sort(key=lambda x: (x.get("track_id", 1e18), x.get("time", 1e18)))
    return corrected_all, peer_ctx_map


def build_per_person_sequences(
    visual_actions: List[Dict[str, Any]],
    transcripts: List[Dict[str, Any]],
    duplicate_speech_per_person: bool = True,
    peer_context_map: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    per_map: Dict[int, Dict[str, Any]] = {}
    ids = sorted({a["track_id"] for a in visual_actions if isinstance(a, dict) and "track_id" in a})
    for tid in ids:
        per_map[tid] = {
            "track_id": int(tid),
            "person_id": int(tid),
            "visual_sequence": [],
            "peer_context": (peer_context_map or {}).get(
                tid,
                {
                    "neighbor_count": 0,
                    "dominant_peer_action": "none",
                    "peer_agreement_score": 0.0,
                    "correction_applied": False,
                },
            ),
        }
        if duplicate_speech_per_person:
            per_map[tid]["speech_sequence"] = transcripts

    for a in visual_actions:
        tid = _to_int(a.get("track_id"))
        if tid is None:
            continue
        if tid not in per_map:
            per_map[tid] = {
                "track_id": int(tid),
                "person_id": int(tid),
                "visual_sequence": [],
                "peer_context": (peer_context_map or {}).get(
                    tid,
                    {
                        "neighbor_count": 0,
                        "dominant_peer_action": "none",
                        "peer_agreement_score": 0.0,
                        "correction_applied": False,
                    },
                ),
            }
            if duplicate_speech_per_person:
                per_map[tid]["speech_sequence"] = transcripts
        per_map[tid]["visual_sequence"].append(a)

    people: List[Dict[str, Any]] = []
    for tid in sorted(per_map.keys()):
        p = per_map[tid]
        vs = p.get("visual_sequence", [])
        if isinstance(vs, list):
            vs.sort(key=lambda x: x.get("time", 1e18) if x.get("time") is not None else 1e18)
        people.append(p)
    return people


def main():
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Step07: Dual Verification Merge + Peer-Aware correction")
    parser.add_argument("--actions", type=str, required=True, help="actions.jsonl path")
    parser.add_argument("--transcript", type=str, required=True, help="transcript.jsonl path")
    parser.add_argument("--out", type=str, required=True, help="per_person_sequences.json output path")
    parser.add_argument("--fps", type=float, default=DEFAULT_VIDEO_FPS, help="fallback fps if actions only have frames")
    parser.add_argument("--time_mode", type=str, default=DEFAULT_VISUAL_TIME_MODE, choices=["start", "end", "mid"])
    parser.add_argument("--duplicate_speech", type=int, default=1, help="1=copy transcript into each person")
    parser.add_argument("--pose_tracks", type=str, default="", help="pose_tracks_smooth.jsonl for peer-aware")
    parser.add_argument("--enable_peer_aware", type=int, default=0, help="1=enable peer-aware correction")
    parser.add_argument("--peer_radius", type=float, default=0.15, help="normalized neighbor radius")
    args = parser.parse_args()

    action_path = Path(args.actions)
    transcript_path = Path(args.transcript)
    out_path = Path(args.out)
    pose_tracks_path = Path(args.pose_tracks) if args.pose_tracks else None

    if not action_path.is_absolute():
        action_path = (base_dir / action_path).resolve()
    if not transcript_path.is_absolute():
        transcript_path = (base_dir / transcript_path).resolve()
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()
    if pose_tracks_path and not pose_tracks_path.is_absolute():
        pose_tracks_path = (base_dir / pose_tracks_path).resolve()

    if not action_path.exists():
        raise FileNotFoundError(f"actions.jsonl not found: {action_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"transcript.jsonl not found: {transcript_path}")

    raw_actions = load_jsonl(action_path)
    raw_transcripts = load_jsonl(transcript_path)
    visual_actions = normalize_visual_actions(raw_actions, fps=args.fps, time_mode=args.time_mode)
    transcripts = normalize_transcripts(raw_transcripts)

    peer_ctx_map: Dict[int, Dict[str, Any]] = {}
    if int(args.enable_peer_aware) == 1 and pose_tracks_path and pose_tracks_path.exists():
        pose_tracks = load_jsonl(pose_tracks_path)
        visual_actions, peer_ctx_map = apply_peer_aware_stage(
            visual_actions=visual_actions,
            pose_tracks=pose_tracks,
            peer_radius=float(args.peer_radius),
        )

    valid_cnt, invalid_cnt = _validate_visual_actions(visual_actions)
    duplicate = bool(int(args.duplicate_speech) == 1)
    people = build_per_person_sequences(
        visual_actions, transcripts, duplicate_speech_per_person=duplicate, peer_context_map=peer_ctx_map
    )

    result: Dict[str, Any] = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "visual_time_mode": args.time_mode,
            "fps": args.fps,
            "total_people": len(people),
            "total_visual_actions": len(visual_actions),
            "total_visual_actions_valid": valid_cnt,
            "total_visual_actions_invalid": invalid_cnt,
            "total_speech_segments": len(transcripts),
            "duplicate_speech_per_person": duplicate,
            "enable_peer_aware": int(args.enable_peer_aware),
            "peer_radius": float(args.peer_radius),
            "inputs": {
                "actions": str(action_path),
                "transcript": str(transcript_path),
                "pose_tracks": str(pose_tracks_path) if pose_tracks_path else "",
            },
        },
        "speech_sequence": transcripts,
        "people": people,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("[Done] Step07 output:", out_path)
    print("meta:", result["meta"])
    print(f"[CHECK] people type={type(result['people']).__name__}, len={len(result['people'])}")


if __name__ == "__main__":
    main()

