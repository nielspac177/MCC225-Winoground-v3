#!/usr/bin/env python3
"""Sintetiza la narración (narration.py) con Kokoro (TTS neural, español).

Debe ejecutarse con el venv que tiene kokoro_onnx (el del toolkit lnm_animations):
    KOKORO_HOME/.venv/bin/python synth_kokoro.py --voice em_santa --out build/audio

Carga el modelo UNA vez y escribe build/audio/n00.wav ... n29.wav.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from narration import NARRATION  # noqa: E402

KOKORO_HOME = Path(os.environ.get(
    "KOKORO_HOME", "/Volumes/Niels_mac/circuitpyper-inference/lnm_animations"))
os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY",
                      "/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="em_santa")
    ap.add_argument("--lang", default="es")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--out", default=str(HERE / "build" / "audio"))
    args = ap.parse_args()

    import soundfile as sf
    from kokoro_onnx import Kokoro

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = KOKORO_HOME / "kokoro_models" / "kokoro-v1.0.onnx"
    voices = KOKORO_HOME / "kokoro_models" / "voices-v1.0.bin"
    k = Kokoro(str(model), str(voices))

    for i, text in enumerate(NARRATION):
        s, sr = k.create(text, voice=args.voice, speed=args.speed, lang=args.lang)
        sf.write(str(out / f"n{i:02d}.wav"), s, sr)
        print(f"  kokoro {i+1:02d}/{len(NARRATION)}  ({len(s)/sr:.1f}s)")
    print(f"OK -> {out}  (voz {args.voice})")


if __name__ == "__main__":
    main()
