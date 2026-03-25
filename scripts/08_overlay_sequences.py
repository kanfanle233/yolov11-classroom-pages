import argparse
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if isinstance(d, dict):
                out.append(d)
    return out


def normalize_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for a in actions:
        label = str(a.get("action", a.get("label", ""))).strip()
        if not label:
            continue
        tid = a.get("track_id", a.get("id"))
        conf = a.get("conf", a.get("confidence"))
        st = a.get("start_time", a.get("start"))
        ed = a.get("end_time", a.get("end"))
        t = a.get("t", a.get("time"))
        try:
            conf = float(conf) if conf is not None else None
        except Exception:
            conf = None
        try:
            st = float(st) if st is not None else None
        except Exception:
            st = None
        try:
            ed = float(ed) if ed is not None else None
        except Exception:
            ed = None
        try:
            t = float(t) if t is not None else None
        except Exception:
            t = None
        out.append({"id": tid, "label": label, "conf": conf, "start": st, "end": ed, "t": t})
    out.sort(key=lambda x: x["t"] if x["t"] is not None else (x["start"] if x["start"] is not None else 1e18))
    return out


def normalize_transcript(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for x in items:
        st = x.get("start")
        ed = x.get("end")
        txt = str(x.get("text", "")).strip()
        if not txt:
            continue
        try:
            st = float(st)
            ed = float(ed)
        except Exception:
            continue
        if ed < st:
            st, ed = ed, st
        out.append({"start": st, "end": ed, "text": txt})
    out.sort(key=lambda x: x["start"])
    return out


def load_pose_tracks_map(path: Path) -> Dict[int, Dict[int, Tuple[int, int, int, int]]]:
    frame_map: Dict[int, Dict[int, Tuple[int, int, int, int]]] = {}
    if not path.exists():
        return frame_map
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            frame = int(d.get("frame", 0))
            m = {}
            for p in d.get("persons", []):
                try:
                    tid = int(p.get("track_id", p.get("id")))
                    b = p.get("bbox", [0, 0, 0, 0])
                    m[tid] = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                except Exception:
                    continue
            frame_map[frame] = m
    return frame_map


def load_group_events(path: Path) -> List[Dict[str, Any]]:
    return load_jsonl(path) if path.exists() else []


def load_mllm_segments(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            return []
    people = data.get("people", [])
    if isinstance(people, dict):
        people = list(people.values())
    out = []
    for p in people:
        tid = p.get("track_id", p.get("person_id", -1))
        for item in p.get("visual_sequence", []):
            label = item.get("mllm_label")
            if not label:
                continue
            st = item.get("start_time")
            ed = item.get("end_time")
            if st is None or ed is None:
                continue
            out.append(
                {
                    "track_id": int(tid),
                    "start": float(st),
                    "end": float(ed),
                    "mllm_label": str(label),
                    "mllm_confidence": float(item.get("mllm_confidence", 0.0)),
                }
            )
    return out


def active_transcript(transcript: List[Dict[str, Any]], t: float) -> Optional[str]:
    for seg in transcript:
        if seg["start"] <= t <= seg["end"]:
            return seg["text"]
    return None


def active_actions(actions: List[Dict[str, Any]], t: float, window: float = 0.25) -> List[Dict[str, Any]]:
    out = []
    for a in actions:
        if a["t"] is not None:
            if abs(a["t"] - t) <= window:
                out.append(a)
        else:
            st = a["start"]
            ed = a["end"]
            if st is not None and ed is not None and st <= t <= ed:
                out.append(a)
    out.sort(key=lambda x: -(x["conf"] if x["conf"] is not None else 0.0))
    dedup = {}
    for x in out:
        k = (x["id"], x["label"])
        if k not in dedup:
            dedup[k] = x
    return list(dedup.values())[:6]


def active_group_event(events: List[Dict[str, Any]], t: float) -> Optional[Dict[str, Any]]:
    for e in events:
        st = float(e.get("start_time", -1))
        ed = float(e.get("end_time", -1))
        if st <= t <= ed:
            return e
    return None


def active_mllm_labels(segments: List[Dict[str, Any]], t: float) -> List[Dict[str, Any]]:
    out = []
    for s in segments:
        if s["start"] <= t <= s["end"]:
            out.append(s)
    return out


def bbox_center(b: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = b
    return (x1 + x2) // 2, (y1 + y2) // 2


def clamp_text(s: str, max_len: int = 64) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_len else (s[: max_len - 3] + "...")


def main():
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=str(base_dir / "data/videos/demo3.mp4"))
    parser.add_argument("--actions", type=str, default=str(base_dir / "output/demo3/actions.jsonl"))
    parser.add_argument("--transcript", type=str, default=str(base_dir / "output/demo3/transcript.jsonl"))
    parser.add_argument("--group_events", type=str, default="", help="group_events.jsonl")
    parser.add_argument("--mllm_seq", type=str, default="", help="mllm_verified_sequences.json")
    parser.add_argument("--pose_tracks", type=str, default="", help="pose_tracks_smooth.jsonl")
    parser.add_argument("--out_dir", type=str, default=str(base_dir / "output/demo3"))
    parser.add_argument("--name", type=str, default="demo3")
    parser.add_argument("--mux_audio", type=int, default=1)
    parser.add_argument("--out_video", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_video) if args.out_video else out_dir / f"{args.name}_overlay.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / f"{args.name}_overlay_noaudio.mp4"

    actions = normalize_actions(load_jsonl(Path(args.actions)))
    transcript = normalize_transcript(load_jsonl(Path(args.transcript)))
    group_events = load_group_events(Path(args.group_events)) if args.group_events else []
    mllm_segments = load_mllm_segments(Path(args.mllm_seq)) if args.mllm_seq else []
    pose_map = load_pose_tracks_map(Path(args.pose_tracks)) if args.pose_tracks else {}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"[INFO] actions={len(actions)} transcript={len(transcript)} group_events={len(group_events)} mllm={len(mllm_segments)}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps

        cv2.putText(frame, f"t={t:.2f}s frame={frame_idx}", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        acts = active_actions(actions, t=t, window=0.25)
        y = 60
        cv2.rectangle(frame, (10, 45), (min(680, w - 10), 45 + 28 * max(2, len(acts) + 1)), (0, 0, 0), -1)
        cv2.putText(frame, "Actions", (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for a in acts:
            line = f"ID:{a.get('id')} {a.get('label')}"
            cv2.putText(frame, clamp_text(line, 52), (18, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            y += 28

        tr = active_transcript(transcript, t=t)
        if tr:
            cv2.rectangle(frame, (0, h - 88), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, clamp_text("Speech: " + tr, 90), (16, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        event = active_group_event(group_events, t=t)
        if event:
            txt = f"Group: {event.get('group_event', 'unknown')}"
            cv2.putText(frame, txt, (w - 330, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 220, 255), 2)
            pairs = event.get("interaction_pairs", [])
            frame_boxes = pose_map.get(frame_idx, {})
            for pair in pairs:
                try:
                    a = int(pair.get("id_a"))
                    b = int(pair.get("id_b"))
                    if a not in frame_boxes or b not in frame_boxes:
                        continue
                    ca = bbox_center(frame_boxes[a])
                    cb = bbox_center(frame_boxes[b])
                    cv2.line(frame, ca, cb, (255, 200, 0), 2)
                    mid = ((ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2)
                    cv2.putText(frame, f"{pair.get('type', 'pair')}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
                except Exception:
                    continue

        mllm_items = active_mllm_labels(mllm_segments, t=t)
        frame_boxes = pose_map.get(frame_idx, {})
        for m in mllm_items:
            tid = int(m["track_id"])
            label = str(m["mllm_label"])
            if tid in frame_boxes:
                x1, y1, x2, y2 = frame_boxes[tid]
                cv2.putText(
                    frame,
                    f"MLLM:{label}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 165, 255),
                    2,
                )
            else:
                cv2.putText(frame, f"ID{tid}:{label}", (w - 330, 62 + 22 * (tid % 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 250 == 0:
            print(f"[INFO] frame={frame_idx}")

    cap.release()
    writer.release()

    if int(args.mux_audio) == 1:
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(tmp_path),
                    "-i",
                    args.video,
                    "-c:v",
                    "libx264",
                    "-crf",
                    "23",
                    "-preset",
                    "fast",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-shortest",
                    str(out_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if tmp_path.exists():
                os.remove(tmp_path)
            print(f"[DONE] overlay video: {out_path}")
        except Exception as e:
            print(f"[WARN] ffmpeg mux failed: {e}")
            if out_path.exists():
                out_path.unlink()
            tmp_path.rename(out_path)
    else:
        if out_path.exists():
            out_path.unlink()
        tmp_path.rename(out_path)
        print(f"[DONE] overlay video: {out_path}")


if __name__ == "__main__":
    main()

