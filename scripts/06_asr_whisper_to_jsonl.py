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


def _extract_wav(video_path: Path, wav_path: Path, ffmpeg: str, sr: int, audio_filter: str = "") -> None:
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
    ]
    if str(audio_filter).strip():
        cmd.extend(["-af", str(audio_filter).strip()])
    cmd.extend([
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _optional_float(raw: str) -> float | None:
    text = str(raw).strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def _segment_metrics(seg: Any) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in ("avg_logprob", "no_speech_prob", "compression_ratio"):
        value = getattr(seg, key, None)
        try:
            out[key] = None if value is None else float(value)
        except Exception:
            out[key] = None
    return out


def _rejection_reason(text: str, metrics: dict[str, float | None], args: argparse.Namespace) -> str:
    min_chars = max(0, int(args.min_text_chars))
    if min_chars and len(text.strip()) < min_chars:
        return "too_short"

    min_avg_logprob = _optional_float(args.min_avg_logprob)
    if min_avg_logprob is not None:
        avg_logprob = metrics.get("avg_logprob")
        if avg_logprob is not None and avg_logprob < min_avg_logprob:
            return "low_avg_logprob"

    max_no_speech_prob = _optional_float(args.max_no_speech_prob)
    if max_no_speech_prob is not None:
        no_speech_prob = metrics.get("no_speech_prob")
        if no_speech_prob is not None and no_speech_prob > max_no_speech_prob:
            return "high_no_speech_prob"

    max_compression_ratio = _optional_float(args.max_compression_ratio)
    if max_compression_ratio is not None:
        compression_ratio = metrics.get("compression_ratio")
        if compression_ratio is not None and compression_ratio > max_compression_ratio:
            return "high_compression_ratio"

    return ""


def _write_quality_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


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
    parser.add_argument("--audio_filter", type=str, default="", help="optional ffmpeg -af filter chain")
    parser.add_argument("--min_avg_logprob", type=str, default="", help="reject segments below this avg_logprob")
    parser.add_argument("--max_no_speech_prob", type=str, default="", help="reject segments above this no_speech_prob")
    parser.add_argument("--max_compression_ratio", type=str, default="", help="reject segments above this compression_ratio")
    parser.add_argument("--min_text_chars", type=int, default=1)
    parser.add_argument("--quality_report", type=str, default="")
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
    quality_report_path = Path(args.quality_report) if str(args.quality_report).strip() else out_dir / "asr_quality_report.json"
    if not quality_report_path.is_absolute():
        quality_report_path = (out_dir / quality_report_path).resolve() if quality_report_path.parent == Path(".") else (base_dir / quality_report_path).resolve()
    added_dll_paths = _configure_windows_cuda_dll_paths()

    try:
        from faster_whisper import WhisperModel
    except Exception:
        _write_placeholder(transcript_path, video_path, "faster_whisper_not_installed")
        print(f"[WARN] faster-whisper missing, wrote placeholder: {transcript_path}")
        return

    try:
        _extract_wav(video_path, wav_path, ffmpeg=args.ffmpeg, sr=int(args.sr), audio_filter=str(args.audio_filter))
    except Exception:
        _write_placeholder(transcript_path, video_path, "ffmpeg_extract_failed")
        print(f"[WARN] ffmpeg failed, wrote placeholder: {transcript_path}")
        return

    info_summary: dict[str, Any] = {}
    try:
        device = str(args.device).strip().lower()
        if device == "auto":
            device = "cuda"
        model = WhisperModel(args.model, device=device, compute_type=str(args.compute_type))
        segments, info = model.transcribe(
            str(wav_path),
            language=args.lang,
            vad_filter=bool(int(args.vad_filter)),
            beam_size=max(1, int(args.beam_size)),
            condition_on_previous_text=bool(int(args.condition_on_previous_text)),
        )
        info_summary = {
            "language": getattr(info, "language", None),
            "language_probability": _safe_float(getattr(info, "language_probability", None), 0.0),
            "duration": _safe_float(getattr(info, "duration", None), 0.0),
        }
    except Exception:
        _write_placeholder(transcript_path, video_path, "whisper_inference_failed")
        print(f"[WARN] whisper inference failed, wrote placeholder: {transcript_path}")
        return

    try:
        count = 0
        raw_count = 0
        rejected: list[dict[str, Any]] = []
        with transcript_path.open("w", encoding="utf-8") as f:
            for seg in segments:
                text = str(seg.text).strip()
                if not text:
                    continue
                raw_count += 1
                metrics = _segment_metrics(seg)
                reason = _rejection_reason(text, metrics, args)
                if reason:
                    rejected.append(
                        {
                            "start": round(_safe_float(seg.start, 0.0), 3),
                            "end": round(_safe_float(seg.end, 0.0), 3),
                            "text": text,
                            "reason": reason,
                            **metrics,
                        }
                    )
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
                    "asr_quality_status": "accepted",
                    **metrics,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    except Exception:
        _write_placeholder(transcript_path, video_path, "whisper_decode_failed")
        print(f"[WARN] whisper decode failed, wrote placeholder: {transcript_path}")
        return

    if count == 0:
        reason = "low_confidence_whisper_result" if raw_count > 0 else "empty_whisper_result"
        _write_placeholder(transcript_path, video_path, reason)

    quality_report = {
        "video": str(video_path),
        "wav": str(wav_path),
        "transcript": str(transcript_path),
        "model": str(args.model),
        "device": device,
        "compute_type": str(args.compute_type),
        "beam_size": int(args.beam_size),
        "vad_filter": int(args.vad_filter),
        "condition_on_previous_text": int(args.condition_on_previous_text),
        "audio_filter": str(args.audio_filter),
        "thresholds": {
            "min_avg_logprob": _optional_float(args.min_avg_logprob),
            "max_no_speech_prob": _optional_float(args.max_no_speech_prob),
            "max_compression_ratio": _optional_float(args.max_compression_ratio),
            "min_text_chars": int(args.min_text_chars),
        },
        "info": info_summary,
        "segments_raw": raw_count,
        "segments_accepted": count,
        "segments_rejected": len(rejected),
        "rejected": rejected[:20],
        "status": "ok" if count > 0 else "placeholder",
    }
    _write_quality_report(quality_report_path, quality_report)

    print(f"[DONE] transcript: {transcript_path}")
    print(f"[INFO] segments: {count}")
    print(f"[INFO] raw_segments={raw_count} rejected={len(rejected)} quality_report={quality_report_path}")
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
