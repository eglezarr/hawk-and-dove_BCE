"""Validación de la alineación con transcripciones de audio simuladas · Bloque 2, fase 5.

Se simula lo que entregaría un reconocedor de voz a partir de la transcripción oficial: cada
palabra recibe un tiempo, se introducen errores de reconocimiento (palabras omitidas, mal
reconocidas y muletillas) y el texto se agrupa en segmentos como los de Whisper. Como los
tiempos reales se conocen, se puede medir el error de la alineación.
"""
import random

import numpy as np
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


# ---------------------------------------------------------------------------
# Comprobaciones con datos reales
# ---------------------------------------------------------------------------
def ritmo_anomalo(tiempos: pd.DataFrame, frases: pd.DataFrame, minimo: int = 8,
                  rango: tuple[float, float] = (0.2, 1.2)) -> int:
    """Frases de al menos `minimo` palabras con un ritmo imposible (segundos por palabra fuera de `rango`).

    El habla normal ronda 0,4-0,5 s por palabra; menos de 0,2 (cinco palabras por segundo) o más
    de 1,2 indican un inicio o un fin mal situados.
    """
    unidas = tiempos.merge(frases[["sentence_id", "text"]], on="sentence_id")
    palabras = unidas["text"].str.split().str.len()
    ritmo = (unidas["end"] - unidas["start"]) / palabras
    return int(((palabras >= minimo) & ((ritmo < rango[0]) | (ritmo > rango[1]))).sum())


def periodista_en_pantalla(tiempos: pd.DataFrame, frases: pd.DataFrame, cara: pd.DataFrame) -> pd.Series:
    """Proporción media de primeros planos de otra persona durante las frases de cada parte de la rueda.

    Comprobación independiente de la alineación con face.csv (bloque 3): durante las preguntas,
    la realización enfoca a los periodistas (person = "otro"); durante la declaración y las
    respuestas, a la presidenta. Con tiempos correctos, la proporción es alta en las preguntas
    y casi nula en el resto. Solo cuentan los segundos con una cara en primer plano.
    """
    con_cara = cara[cara["face_detected"].astype(bool)]
    otro = set(con_cara.loc[con_cara["person"] == "otro", "start"].astype(int))
    validos = set(con_cara["start"].astype(int))
    unidas = tiempos.merge(frases[["sentence_id", "role"]], on="sentence_id")

    def proporcion(inicio: float, fin: float) -> float:
        segundos = set(range(int(np.floor(inicio)), int(np.ceil(fin)))) & validos
        return len(segundos & otro) / len(segundos) if segundos else np.nan

    unidas["periodista"] = [proporcion(a, b) for a, b in zip(unidas["start"], unidas["end"])]
    return unidas.groupby("role")["periodista"].mean()
