"""Transcripción automática de audio (ASR) · Bloque 1.

Funciones:
    transcribir(audio_path, modelo_id) -> DataFrame con start, end, text
    voz_a_texto(audio)                 -> str  (función del chat en vivo)
    cargar_modelo_asr(modelo_id)       -> pipeline cargado en memoria
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def transcribir(audio_path: Union[str, Path], modelo_id: str | None = None) -> pd.DataFrame:
    """Transcribe un fichero de audio y devuelve un DataFrame con timestamps.

    Parámetros
    ----------
    audio_path : ruta al fichero de audio (.wav, .mp3, .m4a).
    modelo_id  : identificador del modelo ASR de Hugging Face.
                 Si es None, usa el que indique config.yaml.

    Devuelve
    --------
    DataFrame con columnas: start, end, text.
    Cada fila es un segmento (frase o fragmento corto) con sus tiempos.
    """
    from src.audio.models.asr_backends import crear_backend_asr
    from src.config import load_config

    if modelo_id is None:
        modelo_id = load_config()["models"].get("asr")

    backend = crear_backend_asr(modelo_id)
    segmentos = backend.transcribir(audio_path)  # list[dict] con start, end, text

    return pd.DataFrame(segmentos, columns=["start", "end", "text"])


def voz_a_texto(audio: Union[bytes, str, Path]) -> str:
    """Transcribe audio del usuario a texto (función del chat en vivo).

    Parámetros
    ----------
    audio : bytes con el audio grabado, o ruta a un fichero temporal.

    Devuelve
    --------
    Texto transcrito como una sola cadena.
    """
    import tempfile

    # Si es bytes, guardar en un fichero temporal
    if isinstance(audio, bytes):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            audio_path = f.name
    else:
        audio_path = str(audio)

    df = transcribir(audio_path)
    return " ".join(df["text"].tolist())


def cargar_modelo_asr(modelo_id: str | None = None):
    """Precarga el modelo ASR en memoria y lo devuelve.

    Se usa desde preparar() para que la primera llamada a voz_a_texto
    no tenga latencia de carga.
    """
    from src.audio.models.asr_backends import crear_backend_asr
    from src.config import load_config

    if modelo_id is None:
        modelo_id = load_config()["models"].get("asr")

    return crear_backend_asr(modelo_id)
