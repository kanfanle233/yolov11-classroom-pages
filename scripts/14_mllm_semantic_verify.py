import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from models.cca_module import CascadedCoAttention, DynamicQueryDriver
from models.mllm_inference import format_classroom_prompt, load_mllm


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if isinstance(d, dict):
                out.append(d)
    return out


def load_embeddings(path: Path) -> Dict[int, List[Tuple[Tuple[int, int], np.ndarray]]]:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        raw = pickle.load(f)

    norm: Dict[int, List[Tuple[Tuple[int, int], np.ndarray]]] = {}
    for tid, seq in (raw or {}).items():
        try:
            tid_i = int(tid)
        except Exception:
            continue
        out_seq: List[Tuple[Tuple[int, int], np.ndarray]] = []
        for item in seq:
            # New format: ((start_frame, end_frame), vec)
            if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], (tuple, list)):
                fr = item[0]
                vec = np.asarray(item[1], dtype=np.float32)
                if len(fr) >= 2:
                    out_seq.append(((int(fr[0]), int(fr[1])), vec))
                    continue
            # Old format: vec only
            vec = np.asarray(item, dtype=np.float32)
            out_seq.append(((0, 0), vec))
        norm[tid_i] = out_seq
    return norm


def asr_keyword_query(transcript: List[Dict[str, Any]], t0: float, t1: float, dim: int) -> torch.Tensor:
    keywords = []
    for seg in transcript:
        st = float(seg.get("start", -1))
        ed = float(seg.get("end", -1))
        if ed < t0 or st > t1:
            continue
        text = str(seg.get("text", "")).lower()
        for k in ("讨论", "提问", "回答", "互动", "question", "discuss", "answer", "chat"):
            if k in text:
                keywords.append(k)
    # A lightweight deterministic query embedding.
    q_len = max(1, len(keywords))
    q = torch.zeros((1, q_len, dim), dtype=torch.float32)
    for i, kw in enumerate(keywords[:q_len]):
        seed = sum(ord(c) for c in kw) % dim
        q[0, i, seed] = 1.0
    if not keywords:
        q[0, 0, 0] = 1.0
    return q


def nearest_embedding(
    emb_seq: List[Tuple[Tuple[int, int], np.ndarray]],
    sf: int,
    ef: int,
) -> np.ndarray:
    if not emb_seq:
        return np.zeros((256,), dtype=np.float32)
    best = None
    best_dist = 10**18
    mid = (sf + ef) * 0.5
    for (s, e), vec in emb_seq:
        c = (s + e) * 0.5
        d = abs(c - mid)
        if d < best_dist:
            best = vec
            best_dist = d
    v = np.asarray(best, dtype=np.float32).reshape(-1)
    if v.size >= 256:
        return v[:256]
    pad = np.zeros((256,), dtype=np.float32)
    pad[: v.size] = v
    return pad


def frame_range_of_action(item: Dict[str, Any], fps: float) -> Tuple[int, int, float, float]:
    sf = item.get("start_frame")
    ef = item.get("end_frame")
    st = item.get("start_time")
    ed = item.get("end_time")
    if sf is None and st is not None:
        sf = int(float(st) * fps)
    if ef is None and ed is not None:
        ef = int(float(ed) * fps)
    if sf is None:
        sf = int(float(item.get("frame", 0)))
    if ef is None:
        ef = sf
    sf = int(sf)
    ef = int(ef)
    if ef < sf:
        sf, ef = ef, sf
    st = float(st if st is not None else sf / fps)
    ed = float(ed if ed is not None else ef / fps)
    return sf, ef, st, ed


def run_semantic_verify(
    per_person: Dict[str, Any],
    embeddings: Dict[int, List[Tuple[Tuple[int, int], np.ndarray]]],
    transcript: List[Dict[str, Any]],
    keyframes_dir: str,
    mllm_model: str,
    quantize: str,
    enable_cca: bool,
    enable_dqd: bool,
    fps: float,
) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cca = CascadedCoAttention(dim=256).to(device)
    dqd = DynamicQueryDriver(dim=256).to(device)
    mllm = load_mllm(model_name=mllm_model, device=device, quantize=quantize)

    people = per_person.get("people", [])
    if isinstance(people, dict):
        people = list(people.values())

    for p in people:
        tid = int(p.get("track_id", p.get("person_id", -1)))
        seq = p.get("visual_sequence", [])
        if not isinstance(seq, list):
            continue

        emb_seq = embeddings.get(tid, [])
        for item in seq:
            sf, ef, st, ed = frame_range_of_action(item, fps=fps)
            t_vec = nearest_embedding(emb_seq, sf, ef)
            f_temporal = torch.from_numpy(t_vec).to(device).view(1, 1, -1)

            # Placeholder visual semantics vector for current stage.
            f_visual = torch.from_numpy(t_vec).to(device).view(1, 1, -1)
            fused = torch.cat([f_temporal, f_visual], dim=1)

            if enable_cca:
                with torch.no_grad():
                    fused = cca(f_temporal, f_visual)
            if enable_dqd:
                q = asr_keyword_query(transcript, st, ed, dim=fused.shape[-1]).to(device)
                with torch.no_grad():
                    fused = dqd(q, fused)

            context = {"peer_context": item.get("peer_context", p.get("peer_context", {}))}
            prompt = format_classroom_prompt(actions=[item], transcript=transcript, context=context)
            raw = mllm.infer(image=None, prompt=prompt)

            label = item.get("action", "individual_work")
            conf = float(item.get("confidence", 0.5))
            reasoning = "fallback_from_visual_action"
            try:
                j = json.loads(raw)
                if isinstance(j, dict):
                    label = str(j.get("label", label))
                    conf = float(j.get("confidence", conf))
                    reasoning = str(j.get("reasoning", reasoning))
            except Exception:
                pass

            item["mllm_label"] = label
            item["mllm_confidence"] = float(round(conf, 4))
            item["attention_region"] = {
                "start_frame": sf,
                "end_frame": ef,
                "start_time": st,
                "end_time": ed,
            }
            item["reasoning_text"] = reasoning

    out = dict(per_person)
    out["people"] = people
    out.setdefault("meta", {})
    out["meta"]["mllm_verified"] = True
    out["meta"]["mllm_model"] = mllm_model
    out["meta"]["enable_cca"] = bool(enable_cca)
    out["meta"]["enable_dqd"] = bool(enable_dqd)
    out["meta"]["quantize"] = quantize
    out["meta"]["keyframes_dir"] = keyframes_dir
    return out


def main():
    parser = argparse.ArgumentParser(description="Step14: MLLM semantic verify")
    parser.add_argument("--per_person", required=True, help="per_person_sequences.json")
    parser.add_argument("--emb", required=True, help="embeddings.pkl")
    parser.add_argument("--transcript", required=True, help="transcript.jsonl")
    parser.add_argument("--keyframes", type=str, default="", help="keyframes directory from step05")
    parser.add_argument("--out", required=True, help="mllm_verified_sequences.json")
    parser.add_argument("--mllm_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--quantize", type=str, default="", help="4bit/8bit/empty")
    parser.add_argument("--enable_cca", type=int, default=1)
    parser.add_argument("--enable_dqd", type=int, default=1)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    per_person_path = Path(args.per_person)
    emb_path = Path(args.emb)
    transcript_path = Path(args.transcript)
    out_path = Path(args.out)

    with open(per_person_path, "r", encoding="utf-8") as f:
        per_person = json.load(f)
    transcript = load_jsonl(transcript_path)
    embeddings = load_embeddings(emb_path)

    verified = run_semantic_verify(
        per_person=per_person,
        embeddings=embeddings,
        transcript=transcript,
        keyframes_dir=args.keyframes,
        mllm_model=args.mllm_model,
        quantize=args.quantize,
        enable_cca=bool(int(args.enable_cca)),
        enable_dqd=bool(int(args.enable_dqd)),
        fps=float(args.fps),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(verified, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Step14 output: {out_path}")


if __name__ == "__main__":
    main()
