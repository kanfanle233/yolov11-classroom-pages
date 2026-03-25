"""Step14: MLLM semantic verification with optional CCA + DQD."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from models.cca_module import CascadedCoAttention, DynamicQueryDriver
from models.mllm_inference import format_classroom_prompt, load_mllm


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
    return out


def _extract_vec(item: Any) -> np.ndarray:
    if isinstance(item, dict) and "embedding" in item:
        return np.asarray(item["embedding"], dtype=np.float32)
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return np.asarray(item[1], dtype=np.float32)
    return np.asarray(item, dtype=np.float32)


def _parse_emb(emb: Dict[Any, Any]) -> Dict[int, List[np.ndarray]]:
    out: Dict[int, List[np.ndarray]] = {}
    for k, vals in emb.items():
        tid = int(k)
        arr = []
        for v in vals:
            vec = _extract_vec(v).reshape(-1)
            if vec.size > 0:
                arr.append(vec)
        out[tid] = arr
    return out


def _segment_time(seg: Dict[str, Any]) -> tuple[float, float]:
    st = float(seg.get("start_time", seg.get("time", 0.0)) or 0.0)
    ed = float(seg.get("end_time", st + 1.0) or (st + 1.0))
    if ed < st:
        st, ed = ed, st
    return st, ed


def _action_label(seg: Dict[str, Any]) -> str:
    return str(seg.get("action", seg.get("label", "unknown")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_person", required=True, help="per_person_sequences.json")
    parser.add_argument("--emb", required=True, help="embeddings.pkl")
    parser.add_argument("--transcript", required=True, help="transcript.jsonl")
    parser.add_argument("--video", required=True, help="source video path")
    parser.add_argument("--out", required=True, help="mllm_verified_sequences.json")
    parser.add_argument("--mllm_model", default="heuristic")
    parser.add_argument("--quantize", default="none")
    parser.add_argument("--keyframe_interval", type=int, default=30)
    parser.add_argument("--enable_cca", type=int, default=1)
    parser.add_argument("--enable_dqd", type=int, default=1)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    per_person_path = Path(args.per_person)
    emb_path = Path(args.emb)
    transcript_path = Path(args.transcript)
    out_path = Path(args.out)
    if not per_person_path.is_absolute():
        per_person_path = (base_dir / per_person_path).resolve()
    if not emb_path.is_absolute():
        emb_path = (base_dir / emb_path).resolve()
    if not transcript_path.is_absolute():
        transcript_path = (base_dir / transcript_path).resolve()
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()

    with open(per_person_path, "r", encoding="utf-8") as f:
        per_person = json.load(f)
    with open(emb_path, "rb") as f:
        emb_raw = pickle.load(f)
    transcript = load_jsonl(transcript_path)

    emb_by_track = _parse_emb(emb_raw if isinstance(emb_raw, dict) else {})
    people = per_person.get("people", [])
    if isinstance(people, dict):
        people = list(people.values())

    dim = 256
    cca = CascadedCoAttention(dim=dim) if int(args.enable_cca) == 1 else None
    dqd = DynamicQueryDriver(dim=dim) if int(args.enable_dqd) == 1 else None
    mllm = load_mllm(args.mllm_model, device="cpu", quantize=args.quantize)

    results = []
    for p in people:
        tid = int(p.get("track_id", p.get("person_id", -1)))
        seq = p.get("visual_sequence", [])
        track_vecs = emb_by_track.get(tid, [])
        if track_vecs:
            tv = np.mean(np.stack(track_vecs, axis=0), axis=0).astype(np.float32)
        else:
            tv = np.zeros((2304,), dtype=np.float32)

        # Project to a stable working dim for CCA/DQD.
        if tv.size < dim:
            tv = np.pad(tv, (0, dim - tv.size))
        else:
            tv = tv[:dim]

        for seg in seq:
            st, ed = _segment_time(seg)
            local_asr = [x for x in transcript if float(x.get("start", 0.0)) <= ed and float(x.get("end", 0.0)) >= st]

            f_temporal = torch.tensor(tv, dtype=torch.float32).view(1, 1, dim)
            f_visual = torch.tensor(tv[::-1].copy(), dtype=torch.float32).view(1, 1, dim)
            if cca is not None:
                fused = cca(f_temporal, f_visual)
            else:
                fused = torch.cat([f_temporal, f_visual], dim=1)
            if dqd is not None:
                q = fused[:, :1, :]
                visual = dqd(q, fused[:, 1:, :])
            else:
                visual = fused[:, 1:, :]

            prompt = format_classroom_prompt(
                actions=[seg],
                transcript=local_asr,
                context={"peer_context": p.get("peer_context", {})},
            )
            raw = mllm.infer(image=None, prompt=prompt)

            label = _action_label(seg)
            conf = float(seg.get("confidence", seg.get("conf", 0.5)) or 0.5)
            reasoning = str(raw)
            # Best-effort parse for JSON-like output.
            if isinstance(raw, str) and raw.strip().startswith("{") and raw.strip().endswith("}"):
                try:
                    obj = json.loads(raw)
                    label = str(obj.get("label", label))
                    conf = float(obj.get("confidence", conf))
                    reasoning = str(obj.get("reasoning", reasoning))
                except Exception:
                    pass

            results.append(
                {
                    "track_id": tid,
                    "start_time": st,
                    "end_time": ed,
                    "source_action": _action_label(seg),
                    "mllm_label": label,
                    "mllm_confidence": float(max(0.0, min(1.0, conf))),
                    "attention_region": "global_frame",
                    "reasoning_text": reasoning[:500],
                    "asr_count": len(local_asr),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "mllm_model": args.mllm_model,
                    "quantize": args.quantize,
                    "enable_cca": int(args.enable_cca),
                    "enable_dqd": int(args.enable_dqd),
                    "keyframe_interval": int(args.keyframe_interval),
                    "count": len(results),
                },
                "items": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Done] Step14 output: {out_path} (items={len(results)})")


if __name__ == "__main__":
    main()

