#!/usr/bin/env python3
"""Genera un video narrado (español) de las slides de la defensa.

Cadena: PDF de slides --pdftoppm--> PNG por página
        narración (narration.py) --say (Paulina es_MX)--> aiff --ffmpeg--> wav
        cada PNG + su wav --ffmpeg--> segmento mp4 (1920x1080)
        segmentos --ffmpeg concat--> video final.

Uso:  python video/make_video.py            # voz por defecto Paulina (es_MX)
      VOICE=Monica python video/make_video.py
Requisitos: macOS `say`, ffmpeg, ffprobe, pdftoppm (todos presentes en este equipo).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PDF = ROOT / "slides" / "latex" / "defensa_winoground.pdf"
BUILD = HERE / "build"
FRAMES = BUILD / "frames"
AUDIO = BUILD / "audio"
SEGS = BUILD / "segments"
FINAL = HERE / "defensa_winoground_video.mp4"
ENGINE = os.environ.get("TTS", "kokoro")             # kokoro (neural) | say (macOS)
VOICE = os.environ.get("VOICE", "em_santa" if os.environ.get("TTS", "kokoro") == "kokoro" else "Paulina")
RATE = os.environ.get("RATE", "165")                 # solo para `say`
KOKORO_HOME = Path(os.environ.get(
    "KOKORO_HOME", "/Volumes/Niels_mac/circuitpyper-inference/lnm_animations"))
W, H = 1920, 1080

sys.path.insert(0, str(HERE))
from narration import NARRATION  # noqa: E402


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def ffprobe_dur(path: Path) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nokey=1:noprint_wrappers=1", str(path)]).stdout
    return float(out.strip())


def main():
    for d in (FRAMES, AUDIO, SEGS):
        d.mkdir(parents=True, exist_ok=True)
    if not PDF.exists():
        sys.exit(f"No existe el PDF de slides: {PDF}")

    # 1) PDF -> PNG por página
    print("[1/4] PDF -> PNG ...")
    run(["pdftoppm", "-png", "-r", "200", str(PDF), str(FRAMES / "slide")])
    pages = sorted(FRAMES.glob("slide-*.png"))
    print(f"      {len(pages)} páginas")
    n = min(len(pages), len(NARRATION))
    if len(pages) != len(NARRATION):
        print(f"      AVISO: {len(pages)} páginas vs {len(NARRATION)} narraciones; uso {n}.")

    # 2) narración -> wav
    print(f"[2/4] TTS ({ENGINE}, voz {VOICE}) ...")
    if ENGINE == "kokoro":
        # genera todos los wav cargando el modelo una sola vez (en el venv de Kokoro)
        kpy = KOKORO_HOME / ".venv" / "bin" / "python"
        run([str(kpy), str(HERE / "synth_kokoro.py"),
             "--voice", VOICE, "--out", str(AUDIO)])
    else:
        for i in range(n):
            aiff = AUDIO / f"n{i:02d}.aiff"
            wav = AUDIO / f"n{i:02d}.wav"
            run(["say", "-v", VOICE, "-r", RATE, "-o", str(aiff), NARRATION[i]])
            run(["ffmpeg", "-y", "-i", str(aiff), "-ar", "44100", "-ac", "2", str(wav)])

    # 3) segmentos (imagen + audio)
    print("[3/4] segmentos ...")
    seg_list = SEGS / "segments.txt"
    with open(seg_list, "w") as lf:
        for i in range(n):
            page = pages[i]
            wav = AUDIO / f"n{i:02d}.wav"
            seg = SEGS / f"seg{i:02d}.mp4"
            dur = ffprobe_dur(wav)
            # segmento: imagen fija (escalada/pad a 1920x1080) + audio, +0.4s de cola
            vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                  f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:white,setsar=1,format=yuv420p")
            run(["ffmpeg", "-y", "-loop", "1", "-i", str(page), "-i", str(wav),
                 "-t", f"{dur + 0.4:.2f}", "-vf", vf, "-r", "25",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                 "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", str(seg)])
            lf.write(f"file '{seg.as_posix()}'\n")
            print(f"      slide {i+1:02d}/{n}  ({dur:.1f}s)")

    # 4) concatenar
    print("[4/4] concatenando ...")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(seg_list),
         "-c", "copy", str(FINAL)])
    total = ffprobe_dur(FINAL)
    mins = int(total // 60)
    print(f"\nLISTO -> {FINAL}")
    print(f"Duración: {mins} min {int(total % 60):02d} s · voz: {VOICE}")


if __name__ == "__main__":
    main()
