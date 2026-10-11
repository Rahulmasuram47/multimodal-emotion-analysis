"""
Utility script to dump ASR results per audio file in a directory.

Usage:
    python scripts/dump_asr.py --dir sample_audio --out asr_dump.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.speech_to_text import SpeechToText


def dump_asr(audio_dir: str, out_file: str) -> None:
    stt = SpeechToText()
    results = []

    files = sorted([
        f for f in os.listdir(audio_dir)
        if f.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg"))
    ])

    for fname in files:
        fpath = os.path.join(audio_dir, fname)
        try:
            res = stt.transcribe(fpath)
            raw_segs = res.get("segments", []) or []
            segs = [
                {
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "text": s.get("text"),
                }
                for s in raw_segs
            ]
            results.append({
                "file": fname,
                "language": res.get("language"),
                "language_probability": res.get("language_probability"),
                "segments": segs,
            })
        except Exception as e:
            results.append({
                "file": fname,
                "error": str(e),
                "language": None,
                "language_probability": None,
                "segments": [],
            })

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Dumped ASR results for {len(results)} files to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Dump ASR transcription per clip to JSON.")
    parser.add_argument("--dir", default="sample_audio", help="Directory containing audio files.")
    parser.add_argument("--out", required=True, help="Output JSON file path.")
    args = parser.parse_args()

    dump_asr(args.dir, args.out)


if __name__ == "__main__":
    main()
