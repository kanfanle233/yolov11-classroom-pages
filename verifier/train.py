import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from verifier.dataset import (
    build_training_samples,
    save_training_samples,
    select_samples_for_setting,
)
from verifier.eval import compute_binary_metrics
from verifier.model import (
    VerifierMLP,
    VerifierRuntimeConfig,
    build_feature_vector,
    build_feature_vector_from_scores,
)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _normalize_label_binary(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "match", "positive", "pos", "true"}:
            return 1
    return 0


def _split_samples_legacy(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train, val = [], []
    for s in samples:
        qid = str(s.get("query_id", ""))
        h = sum(ord(ch) for ch in qid) % 10
        if h < 8:
            train.append(s)
        else:
            val.append(s)
    if not val and train:
        val = train[-max(1, len(train) // 5) :]
        train = train[: len(train) - len(val)]
    return train, val


def _is_fixed_sample_schema(row: Dict[str, Any]) -> bool:
    keys = {
        "query_id",
        "candidate_id",
        "overlap",
        "action_confidence",
        "text_score",
        "uq_score",
        "stability_score",
        "label",
        "negative_type",
    }
    return keys.issubset(set(row.keys()))


def _tensorize(samples: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    feats: List[List[float]] = []
    labels: List[int] = []
    for s in samples:
        if _is_fixed_sample_schema(s):
            feats.append(
                build_feature_vector_from_scores(
                    overlap=_safe_float(s.get("overlap", 0.0), 0.0),
                    action_confidence=_safe_float(s.get("action_confidence", 0.0), 0.0),
                    text_score=_safe_float(s.get("text_score", 0.0), 0.0),
                    uq_score=_safe_float(s.get("uq_score", 0.5), 0.5),
                    stability_score=_safe_float(s.get("stability_score", -1.0), -1.0),
                )
            )
            labels.append(_normalize_label_binary(s.get("label", 0)))
            continue
        feats.append(
            build_feature_vector(
                event_type=str(s.get("event_type", "unknown")),
                query_text=str(s.get("query_text", "")),
                action_label=str(s.get("action_label", "")),
                overlap=float(s.get("overlap", 0.0)),
                action_confidence=float(s.get("action_confidence", 0.0)),
                uq_score=float(s.get("uq_score", 0.5)),
            )
        )
        labels.append(int(s.get("target", 0)))
    x = torch.tensor(feats, dtype=torch.float32) if feats else torch.zeros((0, 4), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0,), dtype=torch.float32)
    return x, y


def _fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.numel() == 0 or labels.numel() == 0:
        return 1.0
    best_t = 1.0
    best_brier = 10.0
    for t in [0.50, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60]:
        probs = torch.sigmoid(logits / t).detach().cpu().numpy().tolist()
        ys = labels.detach().cpu().numpy().tolist()
        metric = compute_binary_metrics(scores=probs, labels=ys, threshold=0.5, num_bins=10)
        brier = float(metric["Brier"])
        if brier < best_brier:
            best_brier = brier
            best_t = float(t)
    return best_t


def _runtime_thresholds(probs: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    if probs.numel() == 0 or labels.numel() == 0:
        return 0.60, 0.40
    pos_mask = labels >= 0.5
    neg_mask = ~pos_mask
    match_threshold = 0.60
    uncertain_threshold = 0.40
    if bool(pos_mask.any()) and bool(neg_mask.any()):
        pos_probs = probs[pos_mask]
        neg_probs = probs[neg_mask]
        match_threshold = _clamp(float(torch.quantile(pos_probs, 0.40).item()), 0.50, 0.80)
        uncertain_threshold = _clamp(float(torch.quantile(neg_probs, 0.80).item()), 0.20, match_threshold - 0.05)
    return match_threshold, uncertain_threshold


def _prepare_sample_splits(
    *,
    samples_path: Path,
    setting: str,
    train_split: str,
    eval_split: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Counter]:
    all_samples = _load_jsonl(samples_path)
    if not all_samples:
        raise RuntimeError(f"no samples loaded: {samples_path}")
    sample_types = Counter(str(s.get("negative_type", s.get("sample_type", "unknown"))) for s in all_samples)

    if _is_fixed_sample_schema(all_samples[0]):
        train_samples = select_samples_for_setting(all_samples, setting=setting, train_split=train_split)
        eval_samples = [dict(s) for s in all_samples if str(s.get("split", "eval")) == eval_split]
    else:
        # Legacy path: ignore setting-specific filtering.
        train_samples, eval_samples = _split_samples_legacy(all_samples)

    if not eval_samples:
        eval_samples = [dict(s) for s in all_samples if s not in train_samples]
    if not eval_samples:
        eval_samples = [dict(s) for s in train_samples]
    return train_samples, eval_samples, sample_types


def main() -> None:
    parser = argparse.ArgumentParser(description="Train verifier (binary match/mismatch) and export checkpoint.")
    parser.add_argument("--event_queries", default="", type=str)
    parser.add_argument("--aligned", default="", type=str)
    parser.add_argument("--actions", default="", type=str)
    parser.add_argument("--samples", default="", type=str, help="prebuilt training samples jsonl")
    parser.add_argument(
        "--setting",
        default="positive_plus_both",
        type=str,
        choices=[
            "positive_only",
            "positive_plus_temporal_shift",
            "positive_plus_semantic_mismatch",
            "positive_plus_both",
        ],
        help="training setting used when --samples is fixed-schema sample set",
    )
    parser.add_argument("--train_split", default="train", type=str)
    parser.add_argument("--eval_split", default="eval", type=str)
    parser.add_argument("--out_model", required=True, type=str, help="verifier.pt")
    parser.add_argument("--out_report", required=True, type=str, help="verifier_report.json")
    parser.add_argument("--out_samples", default="", type=str, help="optional saved sample jsonl")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    args = parser.parse_args()

    out_model = Path(args.out_model).resolve()
    out_report = Path(args.out_report).resolve()
    out_samples = Path(args.out_samples).resolve() if args.out_samples else None

    samples: List[Dict[str, Any]] = []
    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []
    sample_type_counter: Counter = Counter()

    if args.samples:
        samples_path = Path(args.samples).resolve()
        train_samples, val_samples, sample_type_counter = _prepare_sample_splits(
            samples_path=samples_path,
            setting=str(args.setting),
            train_split=str(args.train_split),
            eval_split=str(args.eval_split),
        )
        samples = _load_jsonl(samples_path)
    else:
        if not args.event_queries or not args.aligned or not args.actions:
            raise ValueError("Either --samples or all of --event_queries/--aligned/--actions is required")
        event_queries = Path(args.event_queries).resolve()
        aligned = Path(args.aligned).resolve()
        actions = Path(args.actions).resolve()
        samples = build_training_samples(
            event_queries_path=event_queries,
            aligned_path=aligned,
            actions_path=actions,
        )
        if not samples:
            raise RuntimeError("no training samples generated for verifier")
        train_samples, val_samples = _split_samples_legacy(samples)
        sample_type_counter = Counter(str(s.get("sample_type", "unknown")) for s in samples)

    if out_samples:
        out_samples.parent.mkdir(parents=True, exist_ok=True)
        save_training_samples(out_samples, samples)

    if not train_samples:
        raise RuntimeError("empty training split after filtering")
    if not val_samples:
        raise RuntimeError("empty eval split")

    x_train, y_train = _tensorize(train_samples)
    x_val, y_val = _tensorize(val_samples)
    if x_train.numel() == 0 or y_train.numel() == 0:
        raise RuntimeError("no valid tensorized training samples")

    model = VerifierMLP(in_dim=x_train.shape[1], hidden_dim=int(args.hidden_dim))
    pos_count = float(y_train.sum().item())
    neg_count = float(len(y_train) - pos_count)
    if pos_count > 0 and neg_count > 0:
        pos_weight_value = max(1.0, neg_count / pos_count)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        pos_weight_value = 1.0
        criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))

    model.train()
    for _ in range(int(args.epochs)):
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_logits = model(x_train)
        val_logits = model(x_val)
    temperature = _fit_temperature(val_logits, y_val)

    train_probs = torch.sigmoid(train_logits / temperature).detach().cpu().numpy().tolist()
    val_probs = torch.sigmoid(val_logits / temperature).detach().cpu().numpy().tolist()
    train_labels = y_train.detach().cpu().numpy().tolist()
    val_labels = y_val.detach().cpu().numpy().tolist()
    threshold = _clamp(float(args.score_threshold), 0.1, 0.9)

    train_metrics = compute_binary_metrics(scores=train_probs, labels=train_labels, threshold=threshold, num_bins=10)
    val_metrics = compute_binary_metrics(scores=val_probs, labels=val_labels, threshold=threshold, num_bins=10)

    val_probs_tensor = torch.tensor(val_probs, dtype=torch.float32)
    match_threshold, uncertain_threshold = _runtime_thresholds(val_probs_tensor, y_val)

    high_uq_train = [
        s
        for s in train_samples
        if _safe_float(s.get("uq_score", 1.0 - _safe_float(s.get("stability_score", 0.5), 0.5)), 0.5) >= 0.6
    ]
    high_uq_neg_rate = 0.0
    if high_uq_train:
        high_uq_neg_rate = sum(1 for s in high_uq_train if _normalize_label_binary(s.get("label", s.get("target", 0))) == 0) / len(
            high_uq_train
        )

    runtime_cfg = VerifierRuntimeConfig(
        match_threshold=float(match_threshold),
        uncertain_threshold=float(uncertain_threshold),
        uq_gate=min(0.85, max(0.45, 0.50 + 0.35 * high_uq_neg_rate)),
        temperature=float(temperature),
    )

    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "trainable_verifier",
            "version": "2.1",
            "in_dim": int(x_train.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "state_dict": model.state_dict(),
            "runtime_config": runtime_cfg.to_dict(),
            "training_setting": str(args.setting),
        },
        out_model,
    )

    report = {
        "kind": "verifier_train_report",
        "setting": str(args.setting),
        "num_samples_total": len(samples),
        "num_train": len(train_samples),
        "num_eval": len(val_samples),
        "sample_types": dict(sample_type_counter),
        "train_metrics": train_metrics,
        "eval_metrics": val_metrics,
        "class_balance": {
            "train_positive": int(pos_count),
            "train_negative": int(neg_count),
            "pos_weight": float(pos_weight_value),
        },
        "runtime_config": runtime_cfg.to_dict(),
        "paths": {
            "samples": str(Path(args.samples).resolve()) if args.samples else "",
            "model": str(out_model),
            "saved_samples": str(out_samples) if out_samples else "",
        },
    }

    out_report.parent.mkdir(parents=True, exist_ok=True)
    with out_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[DONE] saved verifier model: {out_model}")
    print(f"[DONE] saved training report: {out_report}")
    print(
        "[INFO] "
        f"setting={args.setting} train={len(train_samples)} eval={len(val_samples)} "
        f"F1={val_metrics['F1']:.4f} AUROC={val_metrics['AUROC']:.4f} "
        f"ECE={val_metrics['ECE']:.4f} Brier={val_metrics['Brier']:.4f}"
    )


if __name__ == "__main__":
    main()
