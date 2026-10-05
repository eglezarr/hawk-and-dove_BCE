"""Lectura de los resultados para la app: solo lee data/, nunca ejecuta modelos.

Cada función devuelve None si el fichero aún no existe, para que la app pueda
mostrar lo disponible mientras los bloques terminan sus entregas.
"""
import json
from pathlib import Path

import pandas as pd

from src.config import EVENTS_DIR, HISTORY_DIR, RAW_DIR, ROOT

EVENT_FILES = {
    "transcript.csv": "B1",
    "voice.csv": "B1",
    "briefing.mp3": "B1",
    "stance.csv": "B2",
    "signals.csv": "B2",
    "summary.json": "B2",
    "face.csv": "B3",
    "projections.csv": "B3",
}


def list_events() -> list[str]:
    """Fechas de las ruedas con resultados, de la más reciente a la más antigua."""
    if not EVENTS_DIR.exists():
        return []
    return sorted((d.name for d in EVENTS_DIR.iterdir() if d.is_dir()), reverse=True)


def missing_files(event_date: str) -> dict[str, str]:
    """{fichero: bloque} de los ficheros que aún faltan en la carpeta del evento."""
    folder = EVENTS_DIR / event_date
    return {f: b for f, b in EVENT_FILES.items() if not (folder / f).exists()}


def _csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def load_csv(event_date: str, name: str) -> pd.DataFrame | None:
    return _csv(EVENTS_DIR / event_date / name)


def load_summary(event_date: str) -> dict | None:
    path = EVENTS_DIR / event_date / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def briefing_audio(event_date: str) -> Path | None:
    path = EVENTS_DIR / event_date / "briefing.mp3"
    return path if path.exists() else None


def video_path(event_date: str) -> Path | None:
    """Vídeo de la rueda (descargado con scripts/, fuera de git)."""
    path = RAW_DIR / f"{event_date}.mp4"
    return path if path.exists() else None


def load_history() -> pd.DataFrame | None:
    return _csv(HISTORY_DIR / "stance_by_conference.csv")


def load_benchmarks() -> dict[str, pd.DataFrame]:
    """Tablas de resultados de los notebooks de benchmark (benchmarks/results/*.csv)."""
    folder = ROOT / "benchmarks" / "results"
    return {p.stem: pd.read_csv(p) for p in sorted(folder.glob("*.csv"))}


def mmss(seconds: float) -> str:
    """1421.0 → '23:41'."""
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"
