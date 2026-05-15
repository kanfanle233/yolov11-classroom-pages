from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent
COMMON_DIR = BASE_DIR / "scripts" / "00_common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from command_executor import run_command, to_powershell_command  # type: ignore
from config_loader import load_profile  # type: ignore
from log_utils import stage_log_path, write_commands_ps1  # type: ignore
from manifest_store import (  # type: ignore
    append_error,
    load_manifest,
    mark_stage_end,
    mark_stage_start,
    new_manifest,
    save_manifest,
)
from pathing import ensure_dir, resolve_path, resolve_repo_root  # type: ignore


STAGE_ORDER = [
    "train_wisdom_s",
    "train_wisdom_m",
    "scb_clean",
    "scb_train",
    "infer_full_pipeline",
    "semantic_bridge",
    "validate_outputs",
]


@dataclass
class StageSpec:
    stage: str
    argv: List[str]
    env: Dict[str, str]
    inputs: List[str]
    outputs: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)


def _now_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _str_dict(raw: Dict[str, Any]) -> Dict[str, str]:
    return {str(k): str(v) for k, v in raw.items()}


def _profile_path(plan_name: str) -> Path:
    return (BASE_DIR / "profiles" / f"{plan_name}.yaml").resolve()


def _resolve_python(repo_root: Path, profile: Dict[str, Any]) -> str:
    env = profile.get("env", {})
    py = str(env.get("python", sys.executable))
    return str(resolve_path(repo_root, py))


def _resolve_env(repo_root: Path, profile: Dict[str, Any]) -> Dict[str, str]:
    env = _str_dict(profile.get("env", {}))
    yolo_dir = env.get("yolo_config_dir", "")
    if yolo_dir:
        env["YOLO_CONFIG_DIR"] = str(resolve_path(repo_root, yolo_dir))
    else:
        env["YOLO_CONFIG_DIR"] = str((repo_root / ".ultralytics").resolve())
    env.pop("yolo_config_dir", None)
    return env


def _build_train_stage(
    *,
    repo_root: Path,
    python_exe: str,
    env: Dict[str, str],
    profile: Dict[str, Any],
    run_id: str,
    variant: str,
) -> StageSpec:
    wrapper = BASE_DIR / "scripts" / "10_train" / "train_wisdom.py"
    models = profile.get("models", {})
    model_key = "train_s" if variant == "s" else "train_m"
    model_ref = str(models.get(model_key, "yolo11s.pt" if variant == "s" else "yolo11m.pt"))
    data_ref = str(profile.get("train_data", "data/processed/classroom_yolo/dataset.yaml"))
    project_ref = str(profile.get("train_project", "runs/detect"))
    run_name = f"wisdom8_yolo11{variant}_detect_{run_id}"
    batch = 8 if variant == "s" else 6

    argv = [
        python_exe,
        str(wrapper),
        "--variant",
        variant,
        "--data",
        data_ref,
        "--model",
        model_ref,
        "--epochs",
        str(int(profile.get("train_epochs", 80))),
        "--imgsz",
        str(int(profile.get("train_imgsz", 832))),
        "--batch",
        str(int(profile.get("train_batch_s", batch) if variant == "s" else profile.get("train_batch_m", batch))),
        "--device",
        str(profile.get("train_device", "0")),
        "--workers",
        str(int(profile.get("train_workers", 4))),
        "--project",
        project_ref,
        "--name",
        run_name,
        "--py",
        python_exe,
        "--emit_only",
        "0",
        "--dry_run",
        "0",
        "--print_json",
        "0",
    ]

    project_path = resolve_path(repo_root, project_ref)
    outputs = [
        str((project_path / run_name / "weights" / "best.pt").resolve()),
        str((project_path / run_name / "results.csv").resolve()),
        str((project_path / run_name / "confusion_matrix.png").resolve()),
    ]
    return StageSpec(
        stage=f"train_wisdom_{variant}",
        argv=argv,
        env=env,
        inputs=[
            str(resolve_path(repo_root, data_ref)),
            str(resolve_path(repo_root, model_ref)),
        ],
        outputs=outputs,
    )


def _build_scb_stage(
    *,
    repo_root: Path,
    python_exe: str,
    env: Dict[str, str],
    task: str,
) -> StageSpec:
    wrapper = BASE_DIR / "scripts" / "20_scb" / "scb_tasks.py"
    argv = [
        python_exe,
        str(wrapper),
        "--task",
        task,
        "--py",
        python_exe,
        "--emit_only",
        "0",
        "--dry_run",
        "0",
        "--print_json",
        "0",
    ]
    if task == "clean":
        outputs = [str((repo_root / "data" / "processed" / "scb_yolo_clean").resolve())]
    else:
        outputs = [str((repo_root / "runs" / "detect" / "scb_hrw_yolo11s_detect_v1" / "weights" / "best.pt").resolve())]

    return StageSpec(
        stage=f"scb_{task}",
        argv=argv,
        env=env,
        inputs=[str((repo_root / "data" / "SCB-Dataset3 yolo dataset").resolve())],
        outputs=outputs,
    )


def _build_infer_stage(
    *,
    repo_root: Path,
    python_exe: str,
    env: Dict[str, str],
    profile: Dict[str, Any],
    run_id: str,
) -> StageSpec:
    wrapper = BASE_DIR / "scripts" / "30_infer" / "run_full_integration.py"
    models = profile.get("models", {})
    pipeline_args = profile.get("pipeline_args", {})
    video_ref = str(profile["video"])

    output_dir = (repo_root / "output" / "codex_reports" / run_id / str(profile.get("infer_case_name", "full_integration_001"))).resolve()
    checks = profile.get("checks", {})
    required_rel = [str(x) for x in checks.get("required_outputs_base", checks.get("required_outputs", []))]
    outputs = [str((output_dir / rel).resolve()) for rel in required_rel]

    reserved_keys = {
        "asr_backend",
        "from_step",
        "export_compat",
        "enable_behavior_det",
    }
    extra = {k: v for k, v in pipeline_args.items() if k not in reserved_keys}

    argv = [
        python_exe,
        str(wrapper),
        "--video",
        video_ref,
        "--out_dir",
        str(output_dir),
        "--asr_backend",
        str(pipeline_args.get("asr_backend", "whisper")),
        "--pose_model",
        str(models.get("pose_model", "auto")),
        "--det_model",
        str(models.get("det_model", "yolo11x.pt")),
        "--enable_behavior_det",
        str(int(pipeline_args.get("enable_behavior_det", 1))),
        "--behavior_det_model",
        str(models.get("behavior_det", "runs/detect/case_yolo_train/weights/best.pt")),
        "--from_step",
        str(int(pipeline_args.get("from_step", 2))),
        "--export_compat",
        str(int(pipeline_args.get("export_compat", 1))),
        "--extra_args_json",
        json.dumps(extra, ensure_ascii=False),
        "--py",
        python_exe,
        "--emit_only",
        "0",
        "--dry_run",
        "0",
        "--print_json",
        "0",
    ]

    return StageSpec(
        stage="infer_full_pipeline",
        argv=argv,
        env=env,
        inputs=[
            str(resolve_path(repo_root, video_ref)),
            str(resolve_path(repo_root, str(models.get("behavior_det", "runs/detect/case_yolo_train/weights/best.pt")))),
        ],
        outputs=outputs,
        meta={
            "output_dir": str(output_dir),
            "required_relpaths": required_rel,
        },
    )


def _build_semantic_stage(
    *,
    repo_root: Path,
    python_exe: str,
    env: Dict[str, str],
    profile: Dict[str, Any],
    manifest: Dict[str, Any],
) -> StageSpec:
    infer_record = manifest.get("stage_results", {}).get("infer_full_pipeline")
    if not isinstance(infer_record, dict):
        raise RuntimeError("Missing upstream stage result: infer_full_pipeline")
    infer_meta = infer_record.get("meta")
    if not isinstance(infer_meta, dict):
        raise RuntimeError("Missing infer_full_pipeline.meta in manifest")
    output_dir = infer_meta.get("output_dir")
    if not output_dir:
        raise RuntimeError("Missing infer_full_pipeline.meta.output_dir in manifest")

    semantic_cfg = profile.get("semantic", {})
    if not isinstance(semantic_cfg, dict):
        semantic_cfg = {}
    taxonomy_ref = str(
        semantic_cfg.get(
            "map_path",
            "codex_reports/smart_classroom_yolo_feasibility/profiles/action_semantics_8class.yaml",
        )
    )
    mode = str(semantic_cfg.get("mode", "compatible_enrich"))
    display_language = str(semantic_cfg.get("display_language", "bilingual"))
    strict = int(semantic_cfg.get("strict", 1))

    checks = profile.get("checks", {})
    final_required_rel = [str(x) for x in checks.get("required_outputs", [])]
    if not final_required_rel:
        final_required_rel = [
            "pose_keypoints_v2.jsonl",
            "pose_tracks_smooth.jsonl",
            "transcript.jsonl",
            "event_queries.jsonl",
            "align_multimodal.json",
            "verified_events.jsonl",
            "pipeline_manifest.json",
            "behavior_det.semantic.jsonl",
            "actions.behavior.semantic.jsonl",
            "actions.behavior_aug.semantic.jsonl",
            "align_multimodal.semantic.json",
            "verified_events.semantic.jsonl",
            "semantics_manifest.json",
        ]

    wrapper = BASE_DIR / "scripts" / "40_semantics" / "semantic_bridge.py"
    argv = [
        python_exe,
        str(wrapper),
        "--output_dir",
        str(output_dir),
        "--taxonomy",
        taxonomy_ref,
        "--display_language",
        display_language,
        "--mode",
        mode,
        "--strict",
        str(strict),
        "--emit_only",
        "0",
        "--dry_run",
        "0",
        "--print_json",
        "0",
    ]
    inputs = [
        str((Path(output_dir) / "behavior_det.jsonl").resolve()),
        str((Path(output_dir) / "actions.behavior.jsonl").resolve()),
        str((Path(output_dir) / "actions.behavior_aug.jsonl").resolve()),
        str((Path(output_dir) / "align_multimodal.json").resolve()),
        str((Path(output_dir) / "verified_events.jsonl").resolve()),
        str(resolve_path(repo_root, taxonomy_ref)),
    ]
    outputs = [str((Path(output_dir) / rel).resolve()) for rel in final_required_rel]
    return StageSpec(
        stage="semantic_bridge",
        argv=argv,
        env=env,
        inputs=inputs,
        outputs=outputs,
        meta={
            "output_dir": str(output_dir),
            "required_relpaths": final_required_rel,
            "taxonomy_path": str(resolve_path(repo_root, taxonomy_ref)),
            "mode": mode,
            "display_language": display_language,
        },
    )


def _build_validate_stage(
    *,
    repo_root: Path,
    python_exe: str,
    env: Dict[str, str],
    run_dir: Path,
    manifest: Dict[str, Any],
) -> StageSpec:
    stage_results = manifest.get("stage_results", {})
    source_stage = "semantic_bridge"
    source_record = stage_results.get(source_stage)
    if not isinstance(source_record, dict):
        source_stage = "infer_full_pipeline"
        source_record = stage_results.get(source_stage)
    if not isinstance(source_record, dict):
        raise RuntimeError("Missing upstream stage result: infer_full_pipeline/semantic_bridge")

    meta = source_record.get("meta")
    if not isinstance(meta, dict):
        raise RuntimeError(f"Missing {source_stage}.meta in manifest")
    output_dir = meta.get("output_dir")
    required_relpaths = meta.get("required_relpaths")

    if not output_dir:
        raise RuntimeError(f"Missing {source_stage}.meta.output_dir in manifest")
    if not isinstance(required_relpaths, list) or not required_relpaths:
        raise RuntimeError(f"Missing {source_stage}.meta.required_relpaths in manifest")

    wrapper = BASE_DIR / "scripts" / "90_tests" / "check_outputs.py"
    report_path = (run_dir / "logs" / "validate_outputs.report.json").resolve()
    argv = [
        python_exe,
        str(wrapper),
        "--output_dir",
        str(output_dir),
        "--report",
        str(report_path),
        "--strict",
        "1",
    ]
    for rel in required_relpaths:
        argv.extend(["--required", str(rel)])

    return StageSpec(
        stage="validate_outputs",
        argv=argv,
        env=env,
        inputs=[str((Path(output_dir) / str(rel)).resolve()) for rel in required_relpaths],
        outputs=[str(report_path)],
    )


def _build_spec(
    *,
    stage_name: str,
    repo_root: Path,
    python_exe: str,
    env: Dict[str, str],
    profile: Dict[str, Any],
    run_id: str,
    run_dir: Path,
    manifest: Dict[str, Any],
) -> StageSpec:
    if stage_name == "train_wisdom_s":
        return _build_train_stage(repo_root=repo_root, python_exe=python_exe, env=env, profile=profile, run_id=run_id, variant="s")
    if stage_name == "train_wisdom_m":
        return _build_train_stage(repo_root=repo_root, python_exe=python_exe, env=env, profile=profile, run_id=run_id, variant="m")
    if stage_name == "scb_clean":
        return _build_scb_stage(repo_root=repo_root, python_exe=python_exe, env=env, task="clean")
    if stage_name == "scb_train":
        return _build_scb_stage(repo_root=repo_root, python_exe=python_exe, env=env, task="train")
    if stage_name == "infer_full_pipeline":
        return _build_infer_stage(repo_root=repo_root, python_exe=python_exe, env=env, profile=profile, run_id=run_id)
    if stage_name == "semantic_bridge":
        return _build_semantic_stage(repo_root=repo_root, python_exe=python_exe, env=env, profile=profile, manifest=manifest)
    if stage_name == "validate_outputs":
        return _build_validate_stage(repo_root=repo_root, python_exe=python_exe, env=env, run_dir=run_dir, manifest=manifest)
    raise ValueError(f"Unsupported stage: {stage_name}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified codex_reports orchestrator.")
    parser.add_argument("--plan", type=str, required=True, help="profile name under profiles/*.yaml")
    parser.add_argument("--stage", type=str, default="all", choices=["all"] + STAGE_ORDER)
    parser.add_argument("--dry_run", type=int, default=0, choices=[0, 1])
    parser.add_argument("--emit_only", type=int, default=0, choices=[0, 1], help="emit commands but do not execute")
    parser.add_argument("--run_id", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = resolve_repo_root(BASE_DIR)

    profile_name = args.plan.strip()
    profile_path = _profile_path(profile_name)
    profile = load_profile(profile_path)

    python_exe = _resolve_python(repo_root, profile)
    env = _resolve_env(repo_root, profile)

    run_id = args.run_id.strip() if args.run_id.strip() else f"{profile_name}_{_now_compact()}"
    run_dir = ensure_dir((BASE_DIR / "runs" / run_id).resolve())
    logs_dir = ensure_dir((run_dir / "logs").resolve())
    manifest_path = (run_dir / "manifest.json").resolve()

    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        manifest["profile"] = profile_name
        manifest["global_env"] = env
    else:
        manifest = new_manifest(run_id=run_id, profile_name=profile_name, global_env=env)
    save_manifest(manifest_path, manifest)

    command_blocks: List[Dict[str, str]] = []
    explicit_stage = args.stage != "all"
    run_failed = False

    for stage_name in STAGE_ORDER:
        if explicit_stage and args.stage != stage_name:
            continue

        enabled_by_profile = _enabled(profile.get("stages_enabled", {}).get(stage_name, False))
        if (not explicit_stage) and (not enabled_by_profile):
            # Keep disabled stages visible in commands.ps1 for manual replay.
            try:
                spec = _build_spec(
                    stage_name=stage_name,
                    repo_root=repo_root,
                    python_exe=python_exe,
                    env=env,
                    profile=profile,
                    run_id=run_id,
                    run_dir=run_dir,
                    manifest=manifest,
                )
            except Exception as exc:
                err = f"{stage_name}: failed to build disabled stage spec: {exc}"
                append_error(manifest, err)
                command_blocks.append(
                    {
                        "stage": stage_name,
                        "status": "failed_build",
                        "command": f"# failed to build stage spec\n# error: {exc}",
                    }
                )
                mark_stage_start(manifest, stage_name, "<build_spec_failed>", [], [], meta={})
                mark_stage_end(manifest, stage_name, "failed", error=err)
                save_manifest(manifest_path, manifest)
                run_failed = True
                break
            ps_cmd = to_powershell_command(spec.argv, spec.env)
            command_blocks.append({"stage": stage_name, "status": "disabled", "command": ps_cmd})
            mark_stage_start(
                manifest,
                stage_name,
                ps_cmd,
                spec.inputs,
                spec.outputs,
                meta=spec.meta,
            )
            mark_stage_end(manifest, stage_name, "disabled")
            save_manifest(manifest_path, manifest)
            continue

        try:
            spec = _build_spec(
                stage_name=stage_name,
                repo_root=repo_root,
                python_exe=python_exe,
                env=env,
                profile=profile,
                run_id=run_id,
                run_dir=run_dir,
                manifest=manifest,
            )
        except Exception as exc:
            err = f"{stage_name}: failed to build stage spec: {exc}"
            append_error(manifest, err)
            command_blocks.append(
                {
                    "stage": stage_name,
                    "status": "failed_build",
                    "command": f"# failed to build stage spec\n# error: {exc}",
                }
            )
            mark_stage_start(manifest, stage_name, "<build_spec_failed>", [], [], meta={})
            mark_stage_end(manifest, stage_name, "failed", error=err)
            save_manifest(manifest_path, manifest)
            run_failed = True
            break

        ps_cmd = to_powershell_command(spec.argv, spec.env)
        command_blocks.append({"stage": stage_name, "status": "planned", "command": ps_cmd})
        mark_stage_start(
            manifest,
            stage_name,
            ps_cmd,
            spec.inputs,
            spec.outputs,
            meta=spec.meta,
        )
        save_manifest(manifest_path, manifest)

        if int(args.emit_only) == 1:
            mark_stage_end(manifest, stage_name, "emit_only")
            manifest.setdefault("artifacts", {})[stage_name] = spec.outputs
            save_manifest(manifest_path, manifest)
            continue

        if int(args.dry_run) == 1:
            mark_stage_end(manifest, stage_name, "dry_run")
            manifest.setdefault("artifacts", {})[stage_name] = spec.outputs
            save_manifest(manifest_path, manifest)
            continue

        log_path = stage_log_path(logs_dir, stage_name)
        rc, log_file = run_command(spec.argv, cwd=repo_root, log_file=log_path, env=spec.env)
        if rc != 0:
            err = f"{stage_name}: exit={rc}, log={log_file}"
            append_error(manifest, err)
            mark_stage_end(manifest, stage_name, "failed", error=err, log_file=log_file)
            save_manifest(manifest_path, manifest)
            run_failed = True
            break

        mark_stage_end(manifest, stage_name, "succeeded", log_file=log_file)
        manifest.setdefault("artifacts", {})[stage_name] = spec.outputs
        save_manifest(manifest_path, manifest)

    commands_ps1_path = (run_dir / "commands.ps1").resolve()
    write_commands_ps1(commands_ps1_path, command_blocks)

    if run_failed:
        raise SystemExit(1)

    print(f"[DONE] run_id={run_id}")
    print(f"[DONE] manifest: {manifest_path}")
    print(f"[DONE] commands: {commands_ps1_path}")


if __name__ == "__main__":
    main()
