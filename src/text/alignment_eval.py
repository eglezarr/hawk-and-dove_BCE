"""Validación de la alineación con transcripciones de audio simuladas · Bloque 2, fase 5.

Se simula lo que entregaría un reconocedor de voz a partir de la transcripción oficial: cada
palabra recibe un tiempo, se introducen errores de reconocimiento (palabras omitidas, mal
reconocidas y muletillas) y el texto se agrupa en segmentos como los de Whisper. Como los
tiempos reales se conocen, se puede medir el error de la alineación.
"""
import random

import pandas as pd

from src.text.alignment import alinear

SUSTITUTAS = ["the", "a", "this", "that", "and", "it", "we", "is", "in", "of"]


def transcripcion_simulada(corpus_evento: pd.DataFrame, error: float = 0.08, ritmo: float = 2.4,
                           semilla: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transcripción del audio simulada (transcript.csv por segmentos) y tiempos reales de cada frase.

    error: proporción de palabras mal reconocidas, repartida a partes iguales entre omitidas y
    sustituidas por otra palabra; además, un 2 % de muletillas ("uh"). ritmo: palabras por segundo.
    """
    rng = random.Random(semilla)
    t, palabras, reales, rol_anterior = 35.0, [], [], None
    for frase in corpus_evento.sort_values("sentence_id").itertuples():
        if rol_anterior is not None and frase.role != rol_anterior:
            t += rng.uniform(1.0, 3.0)   # cambio de turno entre pregunta y respuesta
        rol_anterior, inicio = frase.role, t
        for palabra in str(frase.text).split():
            duracion = max(0.12, rng.gauss(1 / ritmo, 0.08))
            azar = rng.random()
            if azar >= error / 2:   # si no se omite, se reconoce bien o se sustituye
                palabras.append((rng.choice(SUSTITUTAS) if azar < error else palabra, t, t + duracion))
            t += duracion
            if rng.random() < 0.02:
                palabras.append(("uh", t, t + 0.3))
                t += 0.3
        reales.append((frase.sentence_id, inicio, t))
        t += rng.uniform(0.2, 0.6)   # pausa entre frases

    segmentos, i = [], 0
    while i < len(palabras):
        bloque = palabras[i:i + rng.randint(8, 20)]
        segmentos.append((bloque[0][1], bloque[-1][2], " ".join(p[0] for p in bloque)))
        i += len(bloque)
    return (pd.DataFrame(segmentos, columns=["start", "end", "text"]),
            pd.DataFrame(reales, columns=["sentence_id", "start_real", "end_real"]))


def errores(unidas: pd.DataFrame) -> dict:
    """Error absoluto del inicio de cada frase frente a su tiempo real (columnas start y start_real), en segundos."""
    error = (unidas["start"] - unidas["start_real"]).abs()
    return {"mediana_s": round(float(error.median()), 2), "p95_s": round(float(error.quantile(0.95)), 2),
            "max_s": round(float(error.max()), 2), "menos_de_1s": round(float((error < 1).mean()), 3)}


def validar(corpus_evento: pd.DataFrame, niveles: list[float], semillas: int = 5) -> pd.DataFrame:
    """Error de la alineación para cada nivel de error de reconocimiento (varias simulaciones por nivel)."""
    filas = []
    for nivel in niveles:
        simulaciones = (transcripcion_simulada(corpus_evento, error=nivel, semilla=s) for s in range(semillas))
        unidas = pd.concat([alinear(corpus_evento, transcripcion).merge(reales, on="sentence_id")
                            for transcripcion, reales in simulaciones], ignore_index=True)
        filas.append({"error_palabras": nivel, **errores(unidas),
                      "cobertura_media": round(float(unidas["cobertura"].mean()), 3)})
    return pd.DataFrame(filas)
