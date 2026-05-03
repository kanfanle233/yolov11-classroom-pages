from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


PROFILE_REQUIRED_TOP_KEYS = (
    "env",
    "video",
    "models",
    "pipeline_args",
    "stages_enabled",
    "checks",
)


def _load_as_json(text: str) -> Dict[str, Any]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Profile root must be a JSON object/dict.")
    return data


def _load_as_yaml(text: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Profile is not valid JSON-formatted YAML and PyYAML is unavailable."
        ) from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Profile root must be a mapping/dict.")
    return data


def load_profile(profile_path: Path) -> Dict[str, Any]:
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_path}")

    text = profile_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Profile is empty: {profile_path}")

    try:
        profile = _load_as_json(text)
    except json.JSONDecodeError:
        profile = _load_as_yaml(text)

    missing = [k for k in PROFILE_REQUIRED_TOP_KEYS if k not in profile]
    if missing:
        raise ValueError(f"Profile missing keys: {missing}")

    if not isinstance(profile.get("env"), dict):
        raise ValueError("Profile key `env` must be a dict.")
    if not isinstance(profile.get("models"), dict):
        raise ValueError("Profile key `models` must be a dict.")
    if not isinstance(profile.get("pipeline_args"), dict):
        raise ValueError("Profile key `pipeline_args` must be a dict.")
    if not isinstance(profile.get("stages_enabled"), dict):
        raise ValueError("Profile key `stages_enabled` must be a dict.")
    if not isinstance(profile.get("checks"), dict):
        raise ValueError("Profile key `checks` must be a dict.")
    if "semantic" in profile and (not isinstance(profile.get("semantic"), dict)):
        raise ValueError("Profile key `semantic` must be a dict when present.")
    if not isinstance(profile.get("video"), str) or not profile.get("video").strip():
        raise ValueError("Profile key `video` must be a non-empty string.")

    profile.setdefault("checks", {})
    checks = profile["checks"]
    checks.setdefault("required_outputs", [])
    if not isinstance(checks.get("required_outputs"), list):
        raise ValueError("Profile checks.required_outputs must be a list.")
    if "required_outputs_base" in checks and (not isinstance(checks.get("required_outputs_base"), list)):
        raise ValueError("Profile checks.required_outputs_base must be a list when present.")

    return profile
