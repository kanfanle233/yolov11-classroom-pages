from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_manifest(run_id: str, profile_name: str, global_env: Dict[str, Any]) -> Dict[str, Any]:
    ts = now_iso()
    return {
        "run_id": run_id,
        "profile": profile_name,
        "global_env": global_env,
        "stage_results": {},
        "artifacts": {},
        "errors": [],
        "created_at": ts,
        "updated_at": ts,
    }


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Manifest root must be dict.")
    data.setdefault("stage_results", {})
    data.setdefault("artifacts", {})
    data.setdefault("errors", [])
    return data


def save_manifest(manifest_path: Path, manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = now_iso()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def mark_stage_start(
    manifest: Dict[str, Any],
    stage_name: str,
    command_ps1: str,
    inputs: List[str],
    outputs: List[str],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    record = manifest.setdefault("stage_results", {}).get(stage_name, {})
    record.update(
        {
            "inputs": inputs,
            "outputs": outputs,
            "command": command_ps1,
            "status": "running",
            "started_at": now_iso(),
            "finished_at": None,
        }
    )
    if meta is not None:
        record["meta"] = meta
    manifest["stage_results"][stage_name] = record


def mark_stage_end(
    manifest: Dict[str, Any],
    stage_name: str,
    status: str,
    *,
    error: str = "",
    log_file: str = "",
) -> None:
    record = manifest.setdefault("stage_results", {}).setdefault(stage_name, {})
    record["status"] = status
    record["finished_at"] = now_iso()
    if error:
        record["error"] = error
    if log_file:
        record["log_file"] = log_file


def append_error(manifest: Dict[str, Any], message: str) -> None:
    errors = manifest.setdefault("errors", [])
    errors.append({"time": now_iso(), "message": message})
