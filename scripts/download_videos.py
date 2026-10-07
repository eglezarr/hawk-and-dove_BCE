"""Descarga los vídeos de las ruedas de prensa en data/raw/<fecha>.mp4.

Uso:
    python scripts/download_videos.py               # rueda de referencia (config.yaml)
    python scripts/download_videos.py 2026-09-10    # fechas concretas
    python scripts/download_videos.py --all         # todas las de scripts/videos.yaml

Las URLs están en scripts/videos.yaml. Se descarga a 720p como máximo (suficiente
para la cara y mucho más ligero) con vídeo y audio en un solo mp4, que es lo que
reproduce la app. No hace falta instalar ffmpeg: se usa el que trae imageio-ffmpeg.
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RAW_DIR, load_config  # noqa: E402

VIDEOS_FILE = Path(__file__).with_name("videos.yaml")
MAX_HEIGHT = 720


def load_urls() -> dict[str, str]:
    with open(VIDEOS_FILE, encoding="utf-8") as f:
        return {str(k): v for k, v in (yaml.safe_load(f) or {}).items()}


def download(event_date: str, url: str, force: bool = False) -> Path:
    import imageio_ffmpeg
    import yt_dlp

    out = RAW_DIR / f"{event_date}.mp4"
    if out.exists() and not force:
        print(f"[{event_date}] ya descargado: {out}")
        return out
    out.unlink(missing_ok=True)  # con --force, yt-dlp no sobrescribe si el fichero ya existe
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    opts = {
        # H.264 (avc1): lo decodifican OpenCV y todos los navegadores, en cualquier sistema
        "format": f"bv*[height<={MAX_HEIGHT}][vcodec^=avc1]+ba[ext=m4a]/b[height<={MAX_HEIGHT}]",
        "merge_output_format": "mp4",
        "outtmpl": str(RAW_DIR / f"{event_date}.%(ext)s"),
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "noplaylist": True,
    }
    if shutil.which("node"):  # YouTube necesita un intérprete de JavaScript para algunos formatos
        opts["js_runtimes"] = {"node": {}}

    print(f"[{event_date}] descargando {url}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    print(f"[{event_date}] guardado en {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dates", nargs="*", help="fechas AAAA-MM-DD (por defecto, la rueda de referencia)")
    parser.add_argument("--all", action="store_true", help="descarga todas las de videos.yaml")
    parser.add_argument("--force", action="store_true", help="vuelve a descargar aunque ya exista")
    args = parser.parse_args()

    urls = load_urls()
    dates = list(urls) if args.all else (args.dates or [load_config()["reference_event"]])
    missing = [d for d in dates if d not in urls]
    if missing:
        sys.exit(f"Sin URL en {VIDEOS_FILE.name} para: {', '.join(missing)}")
    for d in dates:
        download(d, urls[d], force=args.force)


if __name__ == "__main__":
    main()
