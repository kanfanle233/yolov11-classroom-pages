import argparse
import json
import os
import site
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _video_duration(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.5
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0:
        return 0.5
    return max(0.5, float(n) / float(fps))


def _write_placeholder(path: Path, video_path: Path, reason: str) -> None:
    row = {
        "start": 0.0,
        "end": round(_video_duration(video_path), 3),
        "text": f"[ASR_EMPTY:{reason}]",
        "source": "asr_placeholder",
        "is_placeholder": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_wav(video_path: Path, wav_path: Path, ffmpeg: str, sr: int) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _configure_windows_cuda_dll_paths() -> list[str]:
    if os.name != "nt":
        return []
    added: list[str] = []
    roots: list[Path] = []
    try:
        roots.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            roots.append(Path(user_site))
    except Exception:
        pass
    roots.append(Path(sys.prefix) / "Lib" / "site-packages")

    seen: set[str] = set()
    for root in roots:
        nvidia_root = root / "nvidia"
        if not nvidia_root.exists():
            continue
        for bin_dir in nvidia_root.glob("*/bin"):
            if not bin_dir.is_dir():
                continue
            key = str(bin_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                os.add_dll_directory(key)
            except Exception:
                continue
            os.environ["PATH"] = key + os.pathsep + os.environ.get("PATH", "")
            added.append(key)
    return added


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Whisper ASR to transcript.jsonl with non-empty guarantee.")
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--lang", type=str, default="zh")
    parser.add_argument("--model", type=str, default="small")
    parser.add_argument("--device", type=str, default="cpu", help="cpu/cuda/auto")
    parser.add_argument("--compute_type", type=str, default="int8", help="int8/float16/int8_float16")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--vad_filter", type=int, default=0, help="1=enable VAD filter")
    parser.add_argument("--condition_on_previous_text", type=int, default=0, help="1=enable context carry-over")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    if not video_path.is_absolute():
        video_path = (base_dir / video_path).resolve()
    if not out_dir.is_absolute():
        out_dir = (base_dir / out_dir).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "asr_audio_16k.wav"
    transcript_path = out_dir / "transcript.jsonl"
    added_dll_paths = _configure_windows_cuda_dll_paths()

    try:
        from faster_whisper import WhisperModel
    except Exception:
        _write_placeholder(transcript_path, video_path, "faster_whisper_not_installed")
        print(f"[WARN] faster-whisper missing, wrote placeholder: {transcript_path}")
        return

    try:
        _extract_wav(video_path, wav_path, ffmpeg=args.ffmpeg, sr=int(args.sr))
    except Exception:
        _write_placeholder(transcript_path, video_path, "ffmpeg_extract_failed")
        print(f"[WARN] ffmpeg failed, wrote placeholder: {transcript_path}")
        return

    try:
        device = str(args.device).strip().lower()
        if device == "auto":
            device = "cuda"
        model = WhisperModel(args.model, device=device, compute_type=str(args.compute_type))
        segments, _ = model.transcribe(
            str(wav_path),
            language=args.lang,
            vad_filter=bool(int(args.vad_filter)),
            beam_size=max(1, int(args.beam_size)),
            condition_on_previous_text=bool(int(args.condition_on_previous_text)),
        )
    except Exception:
        _write_placeholder(transcript_path, video_path, "whisper_inference_failed")
        print(f"[WARN] whisper inference failed, wrote placeholder: {transcript_path}")
        return

    try:
        count = 0
        with transcript_path.open("w", encoding="utf-8") as f:
            for seg in segments:
                text = str(seg.text).strip()
                if not text:
                    continue
                start = _safe_float(seg.start, 0.0)
                end = _safe_float(seg.end, start + 0.2)
                if end <= start:
                    end = start + 0.2
                row = {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": text,
                    "source": f"whisper:{args.model}",
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    except Exception:
        _write_placeholder(transcript_path, video_path, "whisper_decode_failed")
        print(f"[WARN] whisper decode failed, wrote placeholder: {transcript_path}")
        return

    if count == 0:
        _write_placeholder(transcript_path, video_path, "empty_whisper_result")

    print(f"[DONE] transcript: {transcript_path}")
    print(f"[INFO] segments: {count}")
    print(
        "[INFO] whisper config:",
        f"model={args.model}",
        f"device={device}",
        f"compute_type={args.compute_type}",
        f"beam_size={args.beam_size}",
        f"vad_filter={int(args.vad_filter)}",
    )
    if added_dll_paths:
        print(f"[INFO] added_dll_paths={len(added_dll_paths)}")


if __name__ == "__main__":
    main()
