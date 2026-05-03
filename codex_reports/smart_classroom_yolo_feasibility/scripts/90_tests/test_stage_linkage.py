from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check downstream linkage requirements from manifest.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--upstream_stage", type=str, default="infer_full_pipeline")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage = data.get("stage_results", {}).get(args.upstream_stage)
    if not isinstance(stage, dict):
        raise RuntimeError(f"Missing upstream stage result: {args.upstream_stage}")

    meta = stage.get("meta")
    if not isinstance(meta, dict):
        raise RuntimeError(f"Missing {args.upstream_stage}.meta")

    if not meta.get("output_dir"):
        raise RuntimeError(f"Missing {args.upstream_stage}.meta.output_dir")

    rel = meta.get("required_relpaths")
    if not isinstance(rel, list) or not rel:
        raise RuntimeError(f"Missing {args.upstream_stage}.meta.required_relpaths")

    print(
        json.dumps(
            {
                "status": "ok",
                "upstream_stage": args.upstream_stage,
                "output_dir": meta.get("output_dir"),
                "required_relpaths_count": len(rel),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
