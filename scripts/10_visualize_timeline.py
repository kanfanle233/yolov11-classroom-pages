import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


COLOR_MAP = {
    0: "#2ecc71",
    1: "#34495e",
    2: "#e74c3c",
    3: "#3498db",
    4: "#e67e22",
    5: "#9b59b6",
    6: "#f1c40f",
    7: "#1abc9c",
    8: "#d35400",
}
LABELS = {
    0: "Listen",
    1: "Distract",
    2: "Phone",
    3: "Doze",
    4: "Chat",
    5: "Note",
    6: "Raise",
    7: "Stand",
    8: "Read",
}
GROUP_COLOR_MAP = {
    "lecture": "#A8E6CF",
    "discussion": "#AEDFF7",
    "break": "#FFD3B6",
    "individual_work": "#E2DBBE",
    "group_discuss": "#AEDFF7",
    "pair_chat": "#FFC8A2",
    "transition": "#DDDDDD",
}
ACTION_STR_TO_ID = {
    "listening": 0,
    "listen": 0,
    "distract": 1,
    "playing_phone": 2,
    "phone": 2,
    "cell phone": 2,
    "sleeping": 3,
    "doze": 3,
    "chatting": 4,
    "chat": 4,
    "writing": 5,
    "reading_writing": 5,
    "note": 5,
    "raising_hand": 6,
    "raise_hand": 6,
    "raise": 6,
    "standing": 7,
    "stand": 7,
    "reading": 8,
    "read": 8,
}


def _safe_int(x: Any, default: int = 10**18) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _people_to_list(people: Any) -> List[Dict[str, Any]]:
    if isinstance(people, list):
        return [p for p in people if isinstance(p, dict)]
    if isinstance(people, dict):
        out = []
        for _, p in people.items():
            if isinstance(p, dict):
                out.append(p)
        return out
    return []


def _extract_visual_seq(person: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("visual_sequence", "visual_actions", "actions"):
        val = person.get(key)
        if isinstance(val, list):
            return val
    return []


def _action_code(act: Dict[str, Any]) -> int:
    if "action_code" in act:
        try:
            return int(act["action_code"])
        except Exception:
            pass
    if "action" in act:
        return ACTION_STR_TO_ID.get(str(act["action"]).lower().strip(), 0)
    return 0


def _time_range(act: Dict[str, Any], fps: Optional[float], default_dur: float = 0.2) -> Optional[Tuple[float, float]]:
    st = act.get("start_time", act.get("start"))
    ed = act.get("end_time", act.get("end"))
    if st is not None or ed is not None:
        try:
            stf = float(st) if st is not None else None
            edf = float(ed) if ed is not None else None
        except Exception:
            stf, edf = None, None
        if stf is not None and edf is not None and edf > stf:
            return stf, edf
        if stf is not None:
            return stf, stf + default_dur
        if edf is not None:
            return max(0.0, edf - default_dur), edf

    if fps and fps > 0:
        sf = act.get("start_frame", act.get("frame"))
        ef = act.get("end_frame", act.get("frame"))
        try:
            sf = float(sf) if sf is not None else None
            ef = float(ef) if ef is not None else None
        except Exception:
            sf, ef = None, None
        if sf is not None and ef is not None:
            if ef < sf:
                sf, ef = ef, sf
            return sf / fps, ef / fps

    t = act.get("t", act.get("time"))
    if t is not None:
        try:
            t = float(t)
            return t, t + default_dur
        except Exception:
            return None
    return None


def merge_segments(
    segs: List[Tuple[float, float, int, bool, Optional[str]]], gap_tol: float = 0.12, min_dur: float = 0.1
) -> List[Tuple[float, float, int, bool, Optional[str]]]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda x: x[0])
    merged = [list(segs[0])]
    for st, ed, code, corrected, mlabel in segs[1:]:
        last = merged[-1]
        if code == last[2] and st <= last[1] + gap_tol and mlabel == last[4]:
            last[1] = max(last[1], ed)
            last[3] = bool(last[3] or corrected)
        else:
            merged.append([st, ed, code, corrected, mlabel])
    out = []
    for s in merged:
        if s[1] - s[0] >= min_dur:
            out.append((float(s[0]), float(s[1]), int(s[2]), bool(s[3]), s[4]))
    return out


def load_group_events(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def draw_timeline(
    data: Dict[str, Any],
    out_path: str,
    group_events: List[Dict[str, Any]],
    fps: Optional[float],
    gap_tol: float,
    min_dur: float,
    top_n: int = 50,
) -> None:
    people = _people_to_list(data.get("people", {}))
    if not people:
        raise ValueError("No people found in source JSON.")

    if fps is None:
        try:
            fps = float(data.get("meta", {}).get("fps"))
        except Exception:
            fps = None

    # Build per-person segments
    per_segments: Dict[int, List[Tuple[float, float, int, bool, Optional[str]]]] = {}
    per_total: Dict[int, float] = {}
    max_t = 0.0

    for p in people:
        tid = int(p.get("track_id", p.get("person_id", 0)))
        seq = _extract_visual_seq(p)
        segs = []
        for act in seq:
            tr = _time_range(act, fps=fps)
            if not tr:
                continue
            st, ed = tr
            if ed <= st:
                continue
            code = _action_code(act)
            corrected = bool(act.get("peer_context", {}).get("correction_applied", False))
            mlabel = act.get("mllm_label")
            segs.append((st, ed, code, corrected, str(mlabel) if mlabel else None))
            max_t = max(max_t, ed)
        segs = merge_segments(segs, gap_tol=gap_tol, min_dur=min_dur)
        if segs:
            per_segments[tid] = segs
            per_total[tid] = sum(ed - st for st, ed, _, _, _ in segs)

    ranked = sorted(per_total.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    sorted_ids = [tid for tid, _ in ranked]
    row_of_id = {tid: i for i, tid in enumerate(sorted_ids)}
    has_group = len(group_events) > 0
    has_mllm = any(seg[4] for tid in sorted_ids for seg in per_segments.get(tid, []))

    extra_rows = (1 if has_group else 0) + (1 if has_mllm else 0)
    fig_h = max(4, min(0.35 * len(sorted_ids) + 2 + extra_rows, 30))
    fig, ax = plt.subplots(figsize=(18, fig_h))
    if max_t <= 0:
        max_t = 10.0

    # person rows
    for row, tid in enumerate(sorted_ids):
        for st, ed, code, corrected, mlabel in per_segments.get(tid, []):
            ax.barh(row, ed - st, left=st, height=0.72, color=COLOR_MAP.get(code, "#333333"), linewidth=0, zorder=2)
            if corrected:
                ax.barh(
                    row,
                    ed - st,
                    left=st,
                    height=0.72,
                    facecolor="none",
                    edgecolor="#ff3366",
                    linewidth=1.2,
                    zorder=3,
                )

    # group row
    current_top = len(sorted_ids)
    if has_group:
        gy = current_top
        for ev in group_events:
            gst = float(ev.get("start_time", 0.0))
            ged = float(ev.get("end_time", 0.0))
            label = str(ev.get("group_event", "unknown"))
            if ged <= gst:
                continue
            ax.barh(gy, ged - gst, left=gst, height=0.8, color=GROUP_COLOR_MAP.get(label, "#EEEEEE"), alpha=0.85, zorder=1)
            if ged - gst > 1.5:
                ax.text(gst + (ged - gst) * 0.5, gy, label, ha="center", va="center", fontsize=7, color="#444")

            # interaction arcs
            pairs = ev.get("interaction_pairs", [])
            xmid = gst + (ged - gst) * 0.5
            for pair in pairs:
                try:
                    a = int(pair.get("id_a"))
                    b = int(pair.get("id_b"))
                except Exception:
                    continue
                if a not in row_of_id or b not in row_of_id:
                    continue
                y1 = row_of_id[a]
                y2 = row_of_id[b]
                ax.annotate(
                    "",
                    xy=(xmid, y1),
                    xytext=(xmid, y2),
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "#666666",
                        "lw": 1.0,
                        "alpha": 0.7,
                        "connectionstyle": "arc3,rad=0.25",
                    },
                )
        current_top += 1

    # mllm row
    if has_mllm:
        my = current_top
        mlabel_colors: Dict[str, str] = {}
        palette = ["#f39c12", "#16a085", "#8e44ad", "#2c3e50", "#c0392b", "#7f8c8d"]
        pi = 0
        for tid in sorted_ids:
            for st, ed, _, _, mlabel in per_segments.get(tid, []):
                if not mlabel:
                    continue
                if mlabel not in mlabel_colors:
                    mlabel_colors[mlabel] = palette[pi % len(palette)]
                    pi += 1
                c = mlabel_colors[mlabel]
                ax.barh(my, ed - st, left=st, height=0.7, color=c, alpha=0.45, edgecolor="none")
                if ed - st > 1.0:
                    ax.text(st + (ed - st) * 0.5, my, mlabel, ha="center", va="center", fontsize=6, color="#111")

    # axes and legend
    yticks = list(range(len(sorted_ids)))
    ylabels = [f"ID {tid}" for tid in sorted_ids]
    if has_group:
        yticks.append(len(sorted_ids))
        ylabels.append("Classroom State")
    if has_mllm:
        yticks.append(len(sorted_ids) + (1 if has_group else 0))
        ylabels.append("MLLM Layer")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title("Classroom Behavior Timeline")
    ax.set_xlim(0, max_t * 1.02)

    legend_items = [mpatches.Patch(color=COLOR_MAP[i], label=LABELS[i]) for i in range(9)]
    legend_items.append(mpatches.Patch(facecolor="none", edgecolor="#ff3366", label="Peer corrected"))
    for k, v in GROUP_COLOR_MAP.items():
        legend_items.append(mpatches.Patch(color=v, label=f"Group:{k}"))
    ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=8, fontsize=8, frameon=False)

    plt.tight_layout()
    _ensure_dir(out_path)
    plt.savefig(out_path, dpi=240)
    plt.close(fig)

    # Export timeline JSON
    items = []
    for row, tid in enumerate(sorted_ids):
        for st, ed, code, corrected, mlabel in per_segments.get(tid, []):
            items.append(
                {
                    "type": "person",
                    "track_id": int(tid),
                    "row": int(row),
                    "start": float(round(st, 2)),
                    "end": float(round(ed, 2)),
                    "action_id": int(code),
                    "action_label": LABELS.get(code, "Unknown"),
                    "peer_corrected": bool(corrected),
                    "mllm_label": mlabel,
                    "color": COLOR_MAP.get(code, "#333333"),
                }
            )
    for ev in group_events:
        items.append(
            {
                "type": "group",
                "start": float(ev.get("start_time", 0.0)),
                "end": float(ev.get("end_time", 0.0)),
                "label": str(ev.get("group_event", "unknown")),
                "interaction_pairs": ev.get("interaction_pairs", []),
            }
        )
    json_path = os.path.splitext(out_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Timeline PNG: {out_path}")
    print(f"[DONE] Timeline JSON: {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="per_person_sequences.json or mllm_verified_sequences.json")
    parser.add_argument("--out", required=True, help="output png path")
    parser.add_argument("--group_src", default="", help="group_events.jsonl")
    parser.add_argument("--mllm_src", default="", help="optional mllm source (same as --src is allowed)")
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--gap_tol", type=float, default=0.12)
    parser.add_argument("--min_dur", type=float, default=0.35)
    args = parser.parse_args()

    with open(args.src, "r", encoding="utf-8") as f:
        data = json.load(f)
    group_events = load_group_events(args.group_src)

    draw_timeline(
        data=data,
        out_path=args.out,
        group_events=group_events,
        fps=args.fps,
        gap_tol=float(args.gap_tol),
        min_dur=float(args.min_dur),
        top_n=int(args.top_n),
    )


if __name__ == "__main__":
    main()

