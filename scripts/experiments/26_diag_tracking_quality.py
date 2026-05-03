"""Diagnose tracking quality for paper comparison table.

Reads a single pose_tracks_smooth.jsonl and outputs a JSON report
with standard MOT metrics + classroom-specific diagnostics.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose tracking fragmentation and quality.")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", default="")
    parser.add_argument("--label", default="", help="Short label for this run (e.g. TK-0)")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    tracks: Dict[int, List[int]] = defaultdict(list)
    track_bboxes: Dict[int, List[List[float]]] = defaultdict(list)
    frame_set = set()

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            frame = int(row.get("frame", -1))
            frame_set.add(frame)
            for p in row.get("persons", []):
                tid = p.get("track_id")
                bbox = p.get("bbox", [])
                if tid is None:
                    continue
                tracks[int(tid)].append(frame)
                if len(bbox) == 4:
                    cx = (bbox[0] + bbox[2]) / 2.0
                    track_bboxes[int(tid)].append([cx, bbox[3] - bbox[1]])

    frames_total = len(frame_set)
    track_count = len(tracks)
    total_detections = sum(len(v) for v in tracks.values())

    # Gap analysis
    gaps_total = 0
    gaps_ge_1s = 0
    gaps_ge_3s = 0
    max_gap = 0
    track_gap_details: List[Dict[str, Any]] = []

    for tid in sorted(tracks.keys()):
        fs = sorted(tracks[tid])
        if len(fs) < 2:
            continue
        for a, b in zip(fs, fs[1:]):
            g = b - a
            if g > 1:
                gaps_total += 1
                if g >= 30:
                    gaps_ge_1s += 1
                if g >= 90:
                    gaps_ge_3s += 1
                max_gap = max(max_gap, g)
        # Record tracks with significant gaps
        max_g = 0
        for a, b in zip(fs, fs[1:]):
            max_g = max(max_g, b - a)
        if max_g >= 10:
            track_gap_details.append({
                "track_id": tid,
                "frames": len(fs),
                "first_frame": fs[0],
                "last_frame": fs[-1],
                "max_gap_frames": max_g,
                "max_gap_sec": round(max_g / 30.0, 1),
            })

    track_gap_details.sort(key=lambda x: -x["max_gap_frames"])

    # CX stability
    cx_stdevs = []
    for tid, bboxes in track_bboxes.items():
        if len(bboxes) < 5:
            continue
        cxs = [b[0] for b in bboxes]
        mean_cx = sum(cxs) / len(cxs)
        stdev = (sum((x - mean_cx) ** 2 for x in cxs) / len(cxs)) ** 0.5
        cx_stdevs.append(stdev)
    cx_stdev_mean = sum(cx_stdevs) / max(1, len(cx_stdevs))

    # Average track length
    avg_track_length = total_detections / max(1, track_count)

    # Short tracks (< 1 second)
    short_tracks = sum(1 for tid, fs in tracks.items() if len(fs) < 30)

    report = {
        "label": args.label,
        "input": str(in_path),
        "frames_total": frames_total,
        "track_count": track_count,
        "total_detections": total_detections,
        "avg_track_length": round(avg_track_length, 1),
        "short_tracks_lt_1s": short_tracks,
        "gaps_total": gaps_total,
        "gaps_ge_1s": gaps_ge_1s,
        "gaps_ge_3s": gaps_ge_3s,
        "max_gap_frames": max_gap,
        "max_gap_sec": round(max_gap / 30.0, 1),
        "cx_stdev_mean_px": round(cx_stdev_mean, 2),
        "top_gaps": track_gap_details[:10],
    }

    if args.out_path:
        out = Path(args.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[DONE] Report: {out}")

    # Print summary
    print(f"{args.label or 'Report'}: {track_count} tracks, {total_detections} dets, "
          f"avg_len={avg_track_length:.0f}fr, gaps={gaps_total}, "
          f">=1s={gaps_ge_1s}, >=3s={gaps_ge_3s}, max={max_gap}fr, "
          f"cx_std={cx_stdev_mean:.1f}px, short={short_tracks}")


if __name__ == "__main__":
    main()
