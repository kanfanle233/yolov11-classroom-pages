import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


ATTENTION_WEIGHTS = {
    0: 1.0,  # listen
    1: 0.0,  # distract
    2: 0.0,  # phone
    3: 0.0,  # doze
    4: 0.5,  # chat
    5: 1.0,  # note
    6: 1.0,  # raise_hand
    7: 0.5,  # stand
    8: 1.0,  # read
}

ACTION_TO_CODE = {
    "listen": 0,
    "listening": 0,
    "distract": 1,
    "phone": 2,
    "playing_phone": 2,
    "doze": 3,
    "sleeping": 3,
    "chat": 4,
    "chatting": 4,
    "note": 5,
    "writing": 5,
    "raise_hand": 6,
    "raising_hand": 6,
    "stand": 7,
    "standing": 7,
    "read": 8,
    "reading": 8,
}

MLLM_TO_ACTION_CODE = {
    "lecture": 0,  # listen
    "group_discuss": 4,  # chat
    "pair_chat": 4,  # chat
    "individual_work": 5,  # note
    "break": 1,  # distract
    "transition": 7,  # stand/move
}


def _action_code(item: Dict[str, Any]) -> int:
    if item.get("corrected_action_code") is not None:
        return int(item["corrected_action_code"])
    if item.get("action_code") is not None:
        return int(item["action_code"])
    return ACTION_TO_CODE.get(str(item.get("action", "")).lower().strip(), 0)


def _duration_frames(item: Dict[str, Any], fps: float) -> float:
    sf = item.get("start_frame")
    ef = item.get("end_frame")
    if sf is not None and ef is not None:
        return max(1.0, float(ef) - float(sf))
    st = item.get("start_time", item.get("time", 0.0))
    et = item.get("end_time", st)
    try:
        st = float(st)
        et = float(et)
    except Exception:
        return 1.0
    return max(1.0, (et - st) * fps)


def _from_mllm_items(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for seg in items:
        try:
            tid = int(seg.get("track_id"))
        except Exception:
            continue
        key = str(tid)
        if key not in grouped:
            grouped[key] = {
                "track_id": tid,
                "visual_sequence": [],
                "peer_context": {"neighbor_count": 0, "peer_agreement_score": 0.0},
            }
        label = str(seg.get("mllm_label", seg.get("source_action", "transition"))).strip().lower()
        code = MLLM_TO_ACTION_CODE.get(label, 0)
        grouped[key]["visual_sequence"].append(
            {
                "action": label,
                "action_code": code,
                "corrected_action_code": code,
                "confidence": float(seg.get("mllm_confidence", 0.5) or 0.5),
                "start_time": float(seg.get("start_time", 0.0) or 0.0),
                "end_time": float(seg.get("end_time", seg.get("start_time", 0.0)) or 0.0),
            }
        )
    return grouped


def calculate_features(person_data: Dict[str, Any], fps: float = 25.0) -> Dict[str, Any] | None:
    items = person_data.get("visual_sequence", []) or []
    if not items:
        return None

    total_frames = 0.0
    attention_score_sum = 0.0
    action_switch_count = 0
    last_action = None
    action_counts = {k: 0.0 for k in ATTENTION_WEIGHTS.keys()}

    for item in items:
        dur = _duration_frames(item, fps=fps)
        code = _action_code(item)
        total_frames += dur
        attention_score_sum += dur * ATTENTION_WEIGHTS.get(code, 0.5)
        if code != last_action:
            action_switch_count += 1
            last_action = code
        if code in action_counts:
            action_counts[code] += dur

    if total_frames <= 0:
        return None

    total_seconds = total_frames / max(fps, 1e-6)
    avg_attention = attention_score_sum / total_frames
    activity_freq = (action_switch_count / max(total_seconds, 1e-6)) * 60.0
    interaction_ratio = (action_counts[4] + action_counts[6] + action_counts[7]) / total_frames
    distract_ratio = (action_counts[1] + action_counts[2]) / total_frames

    peer_ctx = person_data.get("peer_context", {}) or {}
    peer_agreement = float(peer_ctx.get("peer_agreement_score", 0.0))
    social_isolation = 1.0 if int(peer_ctx.get("neighbor_count", 0)) <= 0 else 0.0

    return {
        "track_id": person_data.get("track_id"),
        "Avg Attention": round(float(avg_attention), 3),
        "Activity Lvl": round(float(activity_freq), 1),
        "Interaction": round(float(interaction_ratio), 3),
        "Distraction": round(float(distract_ratio), 3),
        "Peer Agreement": round(float(peer_agreement), 3),
        "Social Isolation": round(float(social_isolation), 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="per_person_sequences.json")
    parser.add_argument("--out", required=True, help="student_features.json")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    src_path = Path(args.src)
    if not src_path.is_absolute():
        src_path = (base_dir / src_path).resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fps = float((data.get("meta", {}) or {}).get("fps", 25.0) or 25.0)
    people = data.get("people", {})
    if isinstance(people, list):
        people_dict = {}
        for p in people:
            pid = p.get("track_id") or p.get("id")
            people_dict[str(pid)] = p
        people = people_dict
    elif not people and isinstance(data.get("items"), list):
        people = _from_mllm_items(data.get("items", []))

    features_list: List[Dict[str, Any]] = []
    for _, p_data in people.items():
        feat = calculate_features(p_data, fps=fps)
        if feat:
            features_list.append(feat)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(features_list, f, ensure_ascii=False, indent=2)

    print(f"[Done] Exported features for {len(features_list)} students -> {out_path}")


if __name__ == "__main__":
    main()
