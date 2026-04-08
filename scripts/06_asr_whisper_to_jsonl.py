import argparse
import json
import subprocess
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


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Whisper ASR to transcript.jsonl with non-empty guarantee.")
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--lang", type=str, default="zh")
    parser.add_argument("--model", type=str, default="small")
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
        model = WhisperModel(args.model, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(
            str(wav_path),
            language=args.lang,
            vad_filter=False,
            beam_size=5,
            condition_on_previous_text=False,
        )
    except Exception:
        _write_placeholder(transcript_path, video_path, "whisper_inference_failed")
        print(f"[WARN] whisper inference failed, wrote placeholder: {transcript_path}")
        return

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
                "source": "whisper",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    if count == 0:
        _write_placeholder(transcript_path, video_path, "empty_whisper_result")

    print(f"[DONE] transcript: {transcript_path}")
    print(f"[INFO] segments: {count}")


if __name__ == "__main__":
    main()
