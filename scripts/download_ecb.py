"""Descarga vídeos y transcripciones oficiales de las ruedas de prensa del BCE.

Uso:
    python scripts/download_ecb.py 2026-09-10              # una rueda
    python scripts/download_ecb.py 2026-09-10 2026-07-23   # varias

Los vídeos se guardan en data/raw/<fecha>.mp4 (fuera de git).
Las transcripciones oficiales se guardan en data/raw/<fecha>_transcript.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Añadir raíz del proyecto al path para importar src.config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RAW_DIR


# ---------------------------------------------------------------------------
# URLs del BCE
# ---------------------------------------------------------------------------
# Las ruedas de prensa del BCE se publican en:
# - Vídeo: YouTube (canal del BCE) o el webcast del BCE
# - Transcripción oficial: https://www.ecb.europa.eu/press/pressconf/

# Mapa de fechas a URLs de YouTube (rellenar manualmente o con yt-dlp search)
# El BCE no tiene una API pública de vídeos; lo más fiable es yt-dlp.
RUEDAS = {
    "2026-09-10": {
        "youtube": None,  # TODO: rellenar con la URL real de YouTube
        "transcript_url": "https://www.ecb.europa.eu/press/pressconf/2026/html/ecb.is260910~PLACEHOLDER.en.html",
    },
    # Añadir más fechas según se necesite
}


def descargar_video(fecha: str, output_dir: Path | None = None) -> Path:
    """Descarga el vídeo de una rueda de prensa del BCE desde YouTube.

    Requiere: pip install yt-dlp
    """
    import subprocess

    if output_dir is None:
        output_dir = RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{fecha}.mp4"
    if output_path.exists():
        print(f"Ya existe: {output_path}")
        return output_path

    rueda = RUEDAS.get(fecha)
    if rueda is None or rueda.get("youtube") is None:
        # Intentar buscar en YouTube automáticamente
        query = f"ECB press conference {fecha}"
        print(f"Buscando en YouTube: {query}")
        cmd = [
            "yt-dlp",
            f"ytsearch1:{query}",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            "-o", str(output_path),
            "--merge-output-format", "mp4",
        ]
    else:
        url = rueda["youtube"]
        cmd = [
            "yt-dlp",
            url,
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            "-o", str(output_path),
            "--merge-output-format", "mp4",
        ]

    print(f"Descargando vídeo: {fecha}")
    subprocess.run(cmd, check=True)
    print(f"Guardado: {output_path}")

    return output_path


def descargar_transcripcion(fecha: str, output_dir: Path | None = None) -> Path:
    """Descarga la transcripción oficial del BCE.

    Se usa como referencia (ground truth) para el benchmark de ASR.
    """
    import requests
    from bs4 import BeautifulSoup

    if output_dir is None:
        output_dir = RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{fecha}_transcript.txt"
    if output_path.exists():
        print(f"Ya existe: {output_path}")
        return output_path

    rueda = RUEDAS.get(fecha)
    if rueda is None or rueda.get("transcript_url") is None:
        raise ValueError(f"No hay URL de transcripción para la fecha {fecha}")

    url = rueda["transcript_url"]
    print(f"Descargando transcripción: {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    # Extraer el texto limpio del HTML
    soup = BeautifulSoup(resp.text, "html.parser")
    # El contenido de la transcripción suele estar en <div class="section">
    content = soup.find("div", class_="section")
    if content is None:
        content = soup.find("main") or soup.find("body")

    texto = content.get_text(separator="\n", strip=True) if content else resp.text
    output_path.write_text(texto, encoding="utf-8")
    print(f"Guardado: {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python scripts/download_ecb.py <fecha> [<fecha> ...]")
        print("Ejemplo: python scripts/download_ecb.py 2026-09-10")
        sys.exit(1)

    fechas = sys.argv[1:]
    for fecha in fechas:
        try:
            descargar_video(fecha)
        except Exception as e:
            print(f"Error descargando vídeo de {fecha}: {e}")

        try:
            descargar_transcripcion(fecha)
        except Exception as e:
            print(f"Error descargando transcripción de {fecha}: {e}")
