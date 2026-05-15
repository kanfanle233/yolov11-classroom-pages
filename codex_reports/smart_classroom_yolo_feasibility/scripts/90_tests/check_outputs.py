from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _resolve_path(base: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def _load_required_from_json(path: Path) -> List[str]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError("required_from_json must be a JSON list.")
    return [str(x) for x in obj]


def check_outputs(output_dir: Path, required: List[str]) -> Dict[str, Any]:
    present: List[str] = []
    missing: List[str] = []
    empty: List[str] = []

    for rel in required:
        target = _resolve_path(output_dir, rel)
        if not target.exists():
            missing.append(str(target))
            continue
        if target.is_file() and target.stat().st_size <= 0:
            empty.append(str(target))
            continue
        present.append(str(target))

    return {
        "output_dir": str(output_dir),
        "required_count": len(required),
        "present_count": len(present),
        "missing_count": len(missing),
        "empty_count": len(empty),
        "present": present,
        "missing": missing,
        "empty": empty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check required output artifacts.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--required", action="append", default=[])
    parser.add_argument("--required_from_json", type=str, default="")
    parser.add_argument("--report", type=str, default="")
    parser.add_argument("--strict", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    required = list(args.required or [])
    if args.required_from_json:
        required.extend(_load_required_from_json(Path(args.required_from_json).resolve()))

    if not required:
        raise ValueError("No required outputs specified.")

    result = check_outputs(output_dir, required)
    result["status"] = "ok" if result["missing_count"] == 0 and result["empty_count"] == 0 else "failed"

    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if int(args.strict) == 1 and result["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
