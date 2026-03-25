import argparse
import json
from pathlib import Path
from collections import defaultdict

def load_index(p: Path):
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {"meta": {}, "videos": data}
    return {"meta": data.get("meta", {}), "videos": data.get("videos", [])}

def sort_key(v):
    stem = str(v.get("stem", ""))
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--views", type=str, default="front,rear,teacher,top1,top2")
    args = ap.parse_args()

    in_path = Path(args.in_path).resolve()
    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_index(in_path)
    videos = data["videos"]
    allow_views = {x.strip() for x in args.views.split(",") if x.strip()}

    by_view = defaultdict(list)
    for v in videos:
        vc = (v.get("view_code") or v.get("view") or "").strip()
        if vc in allow_views:
            by_view[vc].append(v)

    picked = []
    counts = {}
    for vc in sorted(allow_views):
        arr = sorted(by_view.get(vc, []), key=sort_key)
        sel = arr[: args.k]
        picked.extend(sel)
        counts[vc] = len(sel)

    out = {
        "meta": {
            **data.get("meta", {}),
            "subset": f"{args.k}_per_view",
            "source_index": str(in_path),
            "selected_view_counts": counts,
            "total_videos": len(picked),
        },
        "videos": picked,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[OK] wrote:", out_path)
    print("[OK] counts:", counts)
    print("[OK] total :", len(picked))

if __name__ == "__main__":
    main()
