import argparse
import json
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
            except Exception:
                continue
            if isinstance(obj, dict):
                data.append(obj)
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
        action_code = a.get("action_code", a.get("code"))

        track_id_i = _to_int(track_id)
        if track_id_i is None or action is None:
            continue

        conf_f = _to_float(conf)
        if conf_f is None:
            conf_f = 1.0

        st = _to_float(a.get("start_time"))
        et = _to_float(a.get("end_time"))
        sf = _to_int(a.get("start_frame"))
        ef = _to_int(a.get("end_frame"))

        if st is None and et is None and fps and (sf is not None or ef is not None):
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
            "time": float(t) if t is not None else None,
            "start_time": float(st) if st is not None else None,
            "end_time": float(et) if et is not None else None,
            "duration": float(dur) if dur is not None else None,
        }

        if action_code is not None:
            try:
                item["action_code"] = int(action_code)
            except Exception:
                pass
        if sf is not None:
            item["start_frame"] = sf
        if ef is not None:
            item["end_frame"] = ef
        if "frame" in a:
            item["frame"] = a.get("frame")
        if "side" in a:
            item["side"] = a["side"]
        if "objects_found" in a:
            item["objects_found"] = a["objects_found"]

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


def build_per_person_sequences(
    visual_actions: List[Dict[str, Any]],
    transcripts: List[Dict[str, Any]],
    duplicate_speech_per_person: bool = True,
) -> List[Dict[str, Any]]:
    per_map: Dict[int, Dict[str, Any]] = {}
    ids = sorted({a["track_id"] for a in visual_actions if isinstance(a, dict) and "track_id" in a})
    for tid in ids:
        per_map[tid] = {"track_id": int(tid), "person_id": int(tid), "visual_sequence": []}
        if duplicate_speech_per_person:
            per_map[tid]["speech_sequence"] = transcripts

    for a in visual_actions:
        tid = _to_int(a.get("track_id"))
        if tid is None:
            continue
        if tid not in per_map:
            per_map[tid] = {"track_id": int(tid), "person_id": int(tid), "visual_sequence": []}
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


def _frame_of_action(action: Dict[str, Any], fps: Optional[float]) -> int:
    for key in ("start_frame", "frame", "end_frame"):
        val = _to_int(action.get(key))
        if val is not None and val >= 0:
            return val
    t = _to_float(action.get("time", action.get("start_time")))
    if t is not None:
        return int(max(0, round(t * float(fps or 25.0))))
    return 0


def apply_peer_aware_to_people(
    people: List[Dict[str, Any]],
    visual_actions: List[Dict[str, Any]],
    neighbor_index: Dict[int, Dict[int, List[int]]],
    fps: Optional[float],
) -> int:
    action_to_code = {
        "listen": 0,
        "listening": 0,
        "distract": 1,
        "phone": 2,
        "doze": 3,
        "chat": 4,
        "note": 5,
        "raise_hand": 6,
        "stand": 7,
        "read": 8,
    }

    action_by_tid: Dict[int, List[Dict[str, Any]]] = {}
    for a in visual_actions:
        tid = _to_int(a.get("track_id"))
        if tid is None:
            continue
        action_by_tid.setdefault(tid, []).append(a)
    for acts in action_by_tid.values():
        acts.sort(key=lambda x: _frame_of_action(x, fps))

    total_corrections = 0
    for p in people:
        tid = int(p.get("track_id", -1))
        seq = p.get("visual_sequence", [])
        if not isinstance(seq, list):
            continue

        peer_scores: List[float] = []
        no_neighbor_count = 0
        corrected_count = 0
        dominant_counter: Dict[str, int] = {}

        for item in seq:
            frame_idx = _frame_of_action(item, fps)
            peer_feat = extract_peer_features(
                target_id=tid,
                frame_idx=frame_idx,
                neighbors_dict=neighbor_index,
                target_actions=action_by_tid.get(tid, []),
                neighbor_actions=action_by_tid,
            )
            old_conf = float(item.get("confidence", 1.0))
            pred_label = str(item.get("action", "unknown"))
            conf_map = {pred_label: max(1e-6, old_conf)}
            dominant_label = str(peer_feat.get("dominant_peer_action", "unknown"))
            if dominant_label not in {"", "unknown"} and dominant_label != pred_label:
                conf_map[dominant_label] = max(1e-6, 1.0 - old_conf)
            corrected_map, changed = apply_peer_correction(conf_map, peer_feat)

            final_label = pred_label
            if corrected_map:
                final_label = max(corrected_map.items(), key=lambda kv: kv[1])[0]
            if final_label != pred_label:
                item["action_raw"] = pred_label
                item["action"] = final_label
                if final_label in action_to_code:
                    item["action_code"] = action_to_code[final_label]
            new_conf = float(corrected_map.get(final_label, old_conf))

            item["confidence_raw"] = old_conf
            item["confidence"] = new_conf
            item["peer_context"] = {
                "neighbor_count": int(peer_feat.get("neighbor_count", 0)),
                "dominant_peer_action": str(peer_feat.get("dominant_peer_action", "unknown")),
                "peer_agreement_score": float(peer_feat.get("peer_agreement_score", 0.0)),
                "correction_applied": bool(changed),
            }

            if changed:
                corrected_count += 1
                total_corrections += 1
            if item["peer_context"]["neighbor_count"] <= 0:
                no_neighbor_count += 1
            peer_scores.append(float(item["peer_context"]["peer_agreement_score"]))
            dom = item["peer_context"]["dominant_peer_action"]
            dominant_counter[dom] = dominant_counter.get(dom, 0) + 1

        dominant_peer_action = "unknown"
        if dominant_counter:
            dominant_peer_action = max(dominant_counter.items(), key=lambda kv: kv[1])[0]
        p["peer_context"] = {
            "neighbor_count": int(sum(int(x.get("peer_context", {}).get("neighbor_count", 0)) for x in seq)),
            "dominant_peer_action": dominant_peer_action,
            "peer_agreement_score": float(round(sum(peer_scores) / max(1, len(peer_scores)), 4)),
            "correction_applied": bool(corrected_count > 0),
            "isolation_ratio": float(round(no_neighbor_count / max(1, len(seq)), 4)),
            "corrected_segments": int(corrected_count),
        }

    return total_corrections


def main():
    base_dir = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Step07: Dual Verification Merge")
    parser.add_argument("--actions", type=str, required=True, help="actions.jsonl path")
    parser.add_argument("--transcript", type=str, required=True, help="transcript.jsonl path")
    parser.add_argument("--out", type=str, required=True, help="per_person_sequences.json output path")
    parser.add_argument("--fps", type=float, default=DEFAULT_VIDEO_FPS, help="fallback fps if actions only have frames")
    parser.add_argument("--time_mode", type=str, default=DEFAULT_VISUAL_TIME_MODE, choices=["start", "end", "mid"])
    parser.add_argument("--duplicate_speech", type=int, default=1, help="1=copy transcript into each person")
    parser.add_argument("--pose_tracks", type=str, default="", help="pose_tracks_smooth.jsonl for peer-aware context")
    parser.add_argument("--enable_peer_aware", type=int, default=0, help="1=enable peer-aware correction")
    parser.add_argument("--peer_radius", type=float, default=0.15, help="spatial neighbor search radius")
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
    if pose_tracks_path is not None and not pose_tracks_path.is_absolute():
        pose_tracks_path = (base_dir / pose_tracks_path).resolve()

    if not action_path.exists():
        raise FileNotFoundError(f"actions.jsonl not found: {action_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"transcript.jsonl not found: {transcript_path}")

    raw_actions = load_jsonl(action_path)
    raw_transcripts = load_jsonl(transcript_path)
    visual_actions = normalize_visual_actions(raw_actions, fps=args.fps, time_mode=args.time_mode)
    transcripts = normalize_transcripts(raw_transcripts)
    valid_cnt, invalid_cnt = _validate_visual_actions(visual_actions)

    duplicate = bool(int(args.duplicate_speech) == 1)
    people = build_per_person_sequences(visual_actions, transcripts, duplicate_speech_per_person=duplicate)

    peer_enabled = bool(int(args.enable_peer_aware) == 1)
    peer_corrections = 0
    if peer_enabled:
        if pose_tracks_path is None or not pose_tracks_path.exists():
            print("[WARN] --enable_peer_aware=1 but pose tracks file is missing. Skip peer-aware stage.")
        else:
            neighbor_index = build_spatial_neighbor_index(pose_tracks_path, peer_radius=args.peer_radius)
            peer_corrections = apply_peer_aware_to_people(
                people=people,
                visual_actions=visual_actions,
                neighbor_index=neighbor_index,
                fps=args.fps,
            )
            print(
                f"[INFO] Peer-aware correction enabled. "
                f"radius={args.peer_radius}, corrections={peer_corrections}"
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
            "peer_aware_enabled": peer_enabled,
            "peer_radius": args.peer_radius,
            "peer_corrections": peer_corrections,
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

    print(f"[DONE] Step07 output: {out_path}")
    print("meta:", result["meta"])
    print(f"[CHECK] people type = {type(result['people']).__name__}, len={len(result['people'])}")


if __name__ == "__main__":
    main()
