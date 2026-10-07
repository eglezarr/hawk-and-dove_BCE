"""Tiempo de cada frase oficial en el vídeo · Bloque 2, fase 5.

El bloque 1 transcribe el audio con marcas de tiempo (transcript.csv); la transcripción
oficial del BCE no las tiene. Las dos se emparejan palabra a palabra y cada frase oficial
toma el tiempo de su primera y de su última palabra.
"""
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


def tokens(texto: str) -> list[str]:
    """Palabras normalizadas: minúsculas y sin puntuación; "%" y "per cent" se escriben igual."""
    t = str(texto).lower().replace("’", "'")
    t = re.sub(r"%", " percent ", t)
    t = re.sub(r"\bper cent\b", "percent", t)
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", t)


def palabras_con_tiempo(transcripcion: pd.DataFrame) -> pd.DataFrame:
    """Una fila por palabra del audio, con su inicio y su fin en segundos.

    Admite transcript.csv por palabras o por segmentos: la duración de cada fila se reparte
    entre sus palabras en proporción a su longitud (con una palabra por fila, no cambia nada).
    La columna de texto puede llamarse "text" o "word".
    """
    columna = "text" if "text" in transcripcion else "word"
    filas = []
    for fila in transcripcion.sort_values("start").itertuples():
        palabras = tokens(getattr(fila, columna))
        if not palabras:
            continue
        longitudes = np.array([len(p) + 1 for p in palabras], dtype=float)
        bordes = fila.start + (fila.end - fila.start) * np.concatenate([[0.0], np.cumsum(longitudes)]) / longitudes.sum()
        filas += list(zip(palabras, bordes[:-1], bordes[1:]))
    return pd.DataFrame(filas, columns=["word", "start", "end"])


def alinear(frases: pd.DataFrame, transcripcion: pd.DataFrame) -> pd.DataFrame:
    """Inicio y fin en el vídeo de cada frase de la transcripción oficial.

    Args:
        frases: transcripción oficial completa de la rueda (sentence_id, text), con las
            preguntas de los periodistas incluidas: el audio también las contiene.
        transcripcion: transcript.csv del bloque 1 (start, end, text).

    Returns:
        sentence_id, start, end y cobertura (proporción de palabras de la frase que se han
        emparejado con el audio). Las palabras sin pareja toman un tiempo interpolado entre
        las emparejadas vecinas, de modo que toda frase recibe un tiempo; la cobertura indica
        cuánto fiarse de él.
    """
    oficiales, ids = [], []
    for frase in frases.sort_values("sentence_id").itertuples():
        palabras = tokens(frase.text)
        oficiales += palabras
        ids += [frase.sentence_id] * len(palabras)
    audio = palabras_con_tiempo(transcripcion)

    # Bloques de palabras idénticas en el mismo orden en las dos transcripciones
    inicio = np.full(len(oficiales), np.nan)
    fin = np.full(len(oficiales), np.nan)
    inicio_audio, fin_audio = audio["start"].to_numpy(), audio["end"].to_numpy()
    for i, j, n in SequenceMatcher(None, oficiales, audio["word"].tolist(), autojunk=False).get_matching_blocks():
        inicio[i:i + n], fin[i:i + n] = inicio_audio[j:j + n], fin_audio[j:j + n]

    emparejada = ~np.isnan(inicio)
    if not emparejada.any():
        raise ValueError("Ninguna palabra de la transcripción oficial aparece en la del audio")
    posicion = np.arange(len(oficiales))
    palabras = pd.DataFrame({
        "sentence_id": ids,
        "start": np.interp(posicion, posicion[emparejada], inicio[emparejada]),
        "end": np.interp(posicion, posicion[emparejada], fin[emparejada]),
        "emparejada": emparejada,
    })
    return (palabras.groupby("sentence_id")
                    .agg(start=("start", "min"), end=("end", "max"), cobertura=("emparejada", "mean"))
                    .reset_index())


def calidad(tiempos: pd.DataFrame) -> dict:
    """Resumen de la alineación: proporción de frases bien emparejadas y orden temporal."""
    return {"frases": len(tiempos),
            "cobertura_media": float(tiempos["cobertura"].mean()),
            "frases_cobertura_50": float((tiempos["cobertura"] >= 0.5).mean()),
            "orden_correcto": bool(tiempos.sort_values("sentence_id")["start"].is_monotonic_increasing)}
