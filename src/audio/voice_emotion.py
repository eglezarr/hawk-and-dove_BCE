"""Análisis de emoción en la voz · Bloque 1.

Extrae arousal, valence y dominance del audio en ventanas temporales.

Funciones:
    analizar_emocion(audio_path, modelo_id) -> DataFrame con start, end, arousal, valence, dominance
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


# Tamaño de la ventana de análisis en segundos
VENTANA_SEGUNDOS = 2.0
# Solapamiento entre ventanas (0.5 = 50%)
SOLAPAMIENTO = 0.5


def analizar_emocion(
    audio_path: Union[str, Path],
    modelo_id: str | None = None,
    ventana: float = VENTANA_SEGUNDOS,
    solapamiento: float = SOLAPAMIENTO,
) -> pd.DataFrame:
    """Analiza el tono de voz del audio en ventanas temporales.

    Parámetros
    ----------
    audio_path   : ruta al fichero de audio.
    modelo_id    : identificador del modelo de emoción.
                   Si es None, usa el que indique config.yaml.
    ventana      : duración de cada ventana de análisis en segundos.
    solapamiento : fracción de solapamiento entre ventanas consecutivas.

    Devuelve
    --------
    DataFrame con columnas: start, end, arousal, valence, dominance.
    Cada fila corresponde a una ventana temporal.
    dominance puede estar vacío (NaN) si el modelo no la proporciona.
    """
    from src.audio.models.emotion_backends import crear_backend_emocion
    from src.config import load_config

    if modelo_id is None:
        modelo_id = load_config()["models"].get("voice_emotion")

    backend = crear_backend_emocion(modelo_id)
    resultados = backend.analizar(audio_path, ventana, solapamiento)

    return pd.DataFrame(
        resultados,
        columns=["start", "end", "arousal", "valence", "dominance"],
    )


def agregar_a_frases(
    voice: pd.DataFrame,
    transcript: pd.DataFrame,
) -> pd.DataFrame:
    """Agrega las emociones de voz a nivel de frase por solapamiento temporal.

    Para cada frase de la transcripción, calcula la media ponderada
    de arousal y valence de las ventanas de voz que solapan con ella.

    Parámetros
    ----------
    voice      : DataFrame de analizar_emocion() con start, end, arousal, valence.
    transcript : DataFrame con start, end (las frases de la transcripción).

    Devuelve
    --------
    DataFrame con las mismas filas que transcript y columnas:
    voice_arousal, voice_valence (medias ponderadas por solapamiento).
    """
    voice_arousal = []
    voice_valence = []

    for _, frase in transcript.iterrows():
        f_start, f_end = frase["start"], frase["end"]

        # Calcular solapamiento de cada ventana de voz con la frase
        overlap_start = voice["start"].clip(lower=f_start)
        overlap_end = voice["end"].clip(upper=f_end)
        overlap = (overlap_end - overlap_start).clip(lower=0)

        total_overlap = overlap.sum()
        if total_overlap == 0:
            voice_arousal.append(None)
            voice_valence.append(None)
            continue

        pesos = overlap / total_overlap
        voice_arousal.append((voice["arousal"] * pesos).sum())
        voice_valence.append((voice["valence"] * pesos).sum())

    return pd.DataFrame({
        "voice_arousal": voice_arousal,
        "voice_valence": voice_valence,
    }, index=transcript.index)
