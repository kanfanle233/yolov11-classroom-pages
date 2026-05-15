from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_RUN_OUTPUTS = [
    "output/codex_reports/run_full_001/full_integration_001",
    "output/codex_reports/run_full_wisdom8_001/full_integration_001",
    "output/codex_reports/run_full_e150_001/full_integration_001",
]


def _resolve_repo_root(anchor: Path) -> Path:
    for candidate in [anchor] + list(anchor.parents):
        if (candidate / "data").exists() and (candidate / "scripts").exists():
            return candidate.resolve()
    raise RuntimeError(f"Cannot resolve repo root from: {anchor}")


def _resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _build_command(
    *,
    py: str,
    script: Path,
    output_dir: Path,
    taxonomy: Path,
    display_language: str,
    mode: str,
    strict: int,
) -> List[str]:
    return [
        py,
        str(script),
        "--output_dir",
        str(output_dir),
        "--taxonomy",
        str(taxonomy),
        "--display_language",
        str(display_language),
        "--mode",
        str(mode),
        "--strict",
        str(int(strict)),
        "--emit_only",
        "0",
        "--dry_run",
        "0",
        "--print_json",
        "0",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill semantic bridge outputs for historical runs.")
    parser.add_argument("--run_output", action="append", default=[])
    parser.add_argument("--taxonomy", default="codex_reports/smart_classroom_yolo_feasibility/profiles/action_semantics_8class.yaml")
    parser.add_argument("--display_language", default="bilingual")
    parser.add_argument("--mode", default="compatible_enrich")
    parser.add_argument("--strict", type=int, default=1)
    parser.add_argument("--py", default=sys.executable)
    parser.add_argument("--emit_only", type=int, default=1)
    parser.add_argument("--dry_run", type=int, default=0)
    parser.add_argument("--print_json", type=int, default=1)
    args = parser.parse_args()

    repo_root = _resolve_repo_root(Path(__file__).resolve())
    bridge_script = _resolve_path(
        repo_root,
        "codex_reports/smart_classroom_yolo_feasibility/scripts/40_semantics/semantic_bridge.py",
    )
    taxonomy = _resolve_path(repo_root, args.taxonomy)
    runs = list(args.run_output or []) or list(DEFAULT_RUN_OUTPUTS)
    output_dirs = [_resolve_path(repo_root, run) for run in runs]

    commands = [
        _build_command(
            py=str(_resolve_path(repo_root, args.py)),
            script=bridge_script,
            output_dir=out_dir,
            taxonomy=taxonomy,
            display_language=args.display_language,
            mode=args.mode,
            strict=args.strict,
        )
        for out_dir in output_dirs
    ]

    payload = {
        "stage": "semantic_backfill",
        "bridge_script": str(bridge_script),
        "taxonomy": str(taxonomy),
        "run_outputs": [str(p) for p in output_dirs],
        "commands": commands,
        "emit_only": int(args.emit_only),
        "dry_run": int(args.dry_run),
    }
    if int(args.print_json) == 1:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for cmd in commands:
            print(" ".join(cmd))

    if int(args.emit_only) == 1 or int(args.dry_run) == 1:
        return

    for cmd in commands:
        ret = subprocess.run(cmd, check=False)
        if ret.returncode != 0:
            raise RuntimeError(f"Backfill failed, exit={ret.returncode}, command={cmd}")


if __name__ == "__main__":
    main()
