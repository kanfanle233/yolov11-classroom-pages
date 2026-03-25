import argparse
import json
from pathlib import Path
from typing import Dict, List


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

ACTION_NAME_TO_CODE = {
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


def action_code_of(item: Dict) -> int:
    if "action_code" in item:
        try:
            return int(item["action_code"])
        except Exception:
            pass
    name = str(item.get("action", "")).strip().lower()
    return ACTION_NAME_TO_CODE.get(name, 0)


def duration_frames(item: Dict, fps: float) -> float:
    start_f = item.get("start_frame")
    end_f = item.get("end_frame")
    if start_f is not None and end_f is not None:
        return max(1.0, float(end_f) - float(start_f))
    st = float(item.get("start_time", item.get("time", 0.0)) or 0.0)
    ed = float(item.get("end_time", st + 1.0 / fps) or (st + 1.0 / fps))
    return max(1.0, (ed - st) * fps)


def calculate_features(person_data: Dict, fps: float = 25.0):
    items = person_data.get("visual_sequence", [])
    if not items:
        return None

    total_frames = 0.0
    attention_score_sum = 0.0
    action_switch_count = 0
    last_action = -1
    action_counts = {k: 0.0 for k in ATTENTION_WEIGHTS.keys()}

    peer_agreements: List[float] = []
    isolated_frames = 0.0

    for item in items:
        duration = duration_frames(item, fps=fps)
        code = action_code_of(item)

        total_frames += duration
        attention_score_sum += duration * ATTENTION_WEIGHTS.get(code, 0.5)

        if code != last_action:
            action_switch_count += 1
            last_action = code
        if code in action_counts:
            action_counts[code] += duration

        pc = item.get("peer_context", {})
        if isinstance(pc, dict):
            agreement = pc.get("peer_agreement_score")
            try:
                if agreement is not None:
                    peer_agreements.append(float(agreement))
            except Exception:
                pass
            try:
                if int(pc.get("neighbor_count", 0)) <= 0:
                    isolated_frames += duration
            except Exception:
                isolated_frames += duration

    if total_frames <= 0:
        return None

    avg_attention = attention_score_sum / total_frames
    total_seconds = total_frames / max(1e-9, fps)
    activity_freq = (action_switch_count / max(1e-9, total_seconds)) * 60.0
    interaction_ratio = (action_counts[4] + action_counts[6] + action_counts[7]) / total_frames
    distract_ratio = (action_counts[1] + action_counts[2]) / total_frames
    peer_agreement = sum(peer_agreements) / max(1, len(peer_agreements))
    social_isolation = isolated_frames / total_frames

    return {
        "track_id": person_data.get("track_id"),
        "Avg Attention": round(avg_attention, 3),
        "Activity Lvl": round(activity_freq, 1),
        "Interaction": round(interaction_ratio, 3),
        "Distraction": round(distract_ratio, 3),
        "Peer Agreement": round(peer_agreement, 3),
        "Social Isolation": round(social_isolation, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="per_person_sequences.json or mllm_verified_sequences.json")
    parser.add_argument("--out", required=True, help="student_features.json")
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    src_path = Path(args.src)
    if not src_path.exists():
        print(f"Source file not found: {src_path}")
        return

    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    people = data.get("people", {})
    if isinstance(people, list):
        people_dict = {}
        for p in people:
            pid = p.get("track_id", p.get("id"))
            people_dict[str(pid)] = p
        people = people_dict

    features_list = []
    for _, p_data in people.items():
        feat = calculate_features(p_data, fps=float(args.fps))
        if feat:
            features_list.append(feat)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(features_list, f, ensure_ascii=False, indent=2)

    print(f"[Done] Exported features for {len(features_list)} students to {args.out}")


if __name__ == "__main__":
    main()

