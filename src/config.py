"""Rutas del proyecto y carga de config.yaml, compartidas por los tres bloques.

Las rutas se calculan a partir de la ubicación de este fichero, así que funcionan
igual desde notebooks, scripts o la app, sin depender de la carpeta desde la que
se ejecuten.
"""
from pathlib import Path
from dotenv import load_dotenv

import yaml

# Raíz del repositorio: la carpeta que contiene src/
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # carga las claves de .env como variables de entorno (p. ej., HF_TOKEN)

DATA_DIR = ROOT / "data"
EVENTS_DIR = DATA_DIR / "events"    # una subcarpeta por rueda de prensa (AAAA-MM-DD)
HISTORY_DIR = DATA_DIR / "history"  # transcripciones oficiales e índices del histórico
RAW_DIR = DATA_DIR / "raw"          # vídeos y audios descargados (fuera de git)


def load_config() -> dict:
    """Lee config.yaml: evento de referencia, ruedas a procesar y modelo elegido por etapa."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def event_dir(event_date: str) -> Path:
    """Carpeta de resultados de una rueda de prensa, p. ej. event_dir("2026-09-10")."""
    return EVENTS_DIR / event_date
