import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from object_evidence_mapping import load_object_evidence_config, normalize_object_name


DEFAULT_CONFIG_PATH = Path("contracts/object_evidence_mapping.json")


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _window_scores(object_rows: List[dict], query_t: float, window: float) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    for row in object_rows:
        t = row.get("t")
        if t is None:
            frame = row.get("frame")
            if isinstance(frame, int):
                t = float(frame) / 25.0
        if t is None or abs(float(t) - query_t) > window:
            continue

        for obj in row.get("objects", []):
            name = str(obj.get("name", "")).strip().lower()
            conf = float(obj.get("conf", 0.0) or 0.0)
            if name:
                scores[name] = max(scores[name], conf)
    return scores


def _obj_evidence_to_actions(obj_scores: Dict[str, float], object_action_priors: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    mapped: Dict[str, float] = defaultdict(float)
    for object_name, score in obj_scores.items():
        priors = object_action_priors.get(object_name, {})
        if not priors:
            continue
        for action_name, weight in priors.items():
            mapped[action_name] = max(mapped[action_name], float(score) * float(weight))
    return mapped


def fuse(
    actions: List[dict],
    objects: List[dict],
    alpha: float,
    beta: float,
    window: float,
    object_action_priors: Dict[str, Dict[str, float]],
    object_name_aliases: Dict[str, str],
) -> Tuple[List[dict], int]:
    fused_rows: List[dict] = []
    changed = 0

    for a in actions:
        action = str(a.get("action", "unknown"))
        conf = float(a.get("conf", 0.0) or 0.0)
        t = a.get("t")
        if t is None:
            t = a.get("start_time", 0.0)
        t = float(t or 0.0)

        obj_scores = _window_scores(objects, query_t=t, window=window)
        normalized_scores: Dict[str, float] = defaultdict(float)
        for key, value in obj_scores.items():
            normalized_scores[normalize_object_name(key, object_name_aliases)] = max(
                normalized_scores[normalize_object_name(key, object_name_aliases)],
                float(value),
            )
        obj_action_scores = _obj_evidence_to_actions(normalized_scores, object_action_priors)

        # default keep original
        fused_action = action
        fused_score = conf
        for candidate, obj_score in obj_action_scores.items():
            base = conf if candidate == action else conf * 0.6
            candidate_score = max(conf if candidate == action else 0.0, alpha * base + beta * obj_score)
            if candidate_score > fused_score:
                fused_score = candidate_score
                fused_action = candidate

        row = dict(a)
        row["raw_action"] = action
        row["raw_conf"] = round(conf, 4)
        row["fused_action"] = fused_action
        row["fused_conf"] = round(float(fused_score), 4)
        row["fusion_object_evidence"] = {k: round(float(v), 4) for k, v in normalized_scores.items()}
        row["fusion_weights"] = {"alpha": alpha, "beta": beta, "window": window}

        # keep downstream compatibility by writing fused values to canonical fields
        row["action"] = fused_action
        row["conf"] = round(float(fused_score), 4)

        if fused_action != action:
            changed += 1

        fused_rows.append(row)

    return fused_rows, changed


def main():
    parser = argparse.ArgumentParser(description="Fuse SlowFast actions with object detection evidence")
    parser.add_argument("--actions", type=str, required=True, help="input actions.jsonl")
    parser.add_argument("--objects", type=str, required=True, help="input objects.jsonl")
    parser.add_argument("--out", type=str, required=True, help="output actions_fused.jsonl")
    parser.add_argument("--window", type=float, default=0.8, help="time window in seconds")
    parser.add_argument("--alpha", type=float, default=0.75, help="SlowFast confidence weight")
    parser.add_argument("--beta", type=float, default=0.25, help="Object evidence weight")
    parser.add_argument(
        "--mapping_config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="centralized object-action mapping json",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]

    actions_path = Path(args.actions)
    if not actions_path.is_absolute():
        actions_path = (base_dir / actions_path).resolve()

    objects_path = Path(args.objects)
    if not objects_path.is_absolute():
        objects_path = (base_dir / objects_path).resolve()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()
    config_path = Path(args.mapping_config)
    if not config_path.is_absolute():
        config_path = (base_dir / config_path).resolve()

    if not objects_path.exists():
        raise FileNotFoundError(
            "objects.jsonl not found: "
            f"{objects_path}. "
            "Please run scripts/02b_export_objects_jsonl.py or use sample objects file: "
            "paper_experiments/samples/objects_object_fusion.sample.jsonl"
        )

    actions = _read_jsonl(actions_path)
    objects = _read_jsonl(objects_path)
    object_name_aliases, object_action_priors = load_object_evidence_config(config_path)

    if not actions:
        _write_jsonl(out_path, [])
        print(f"[WARN] No actions found, wrote empty fused file: {out_path}")
        return

    alpha = max(0.0, min(1.0, float(args.alpha)))
    beta = max(0.0, float(args.beta))
    if alpha + beta <= 1e-6:
        alpha, beta = 1.0, 0.0
    else:
        s = alpha + beta
        alpha, beta = alpha / s, beta / s

    fused_rows, changed = fuse(
        actions,
        objects,
        alpha=alpha,
        beta=beta,
        window=max(0.05, float(args.window)),
        object_action_priors=object_action_priors,
        object_name_aliases=object_name_aliases,
    )
    _write_jsonl(out_path, fused_rows)

    print(f"[INFO] actions={len(actions)}, objects={len(objects)}")
    print(f"[INFO] fusion alpha={alpha:.3f}, beta={beta:.3f}, window={args.window:.2f}s")
    print(f"[INFO] mapping_config={config_path}")
    print(f"[DONE] wrote fused actions: {out_path} (changed={changed})")


if __name__ == "__main__":
    main()
