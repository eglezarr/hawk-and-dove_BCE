"""Postura de cada frase a partir de las probabilidades del clasificador · Bloque 2.

Lógica común a todos los modelos (benchmark, histórico y rueda de referencia):
etiqueta en tres clases, puntuación continua y relevancia.
"""
import pandas as pd

ETIQUETAS = ["hawkish", "neutral", "dovish"]


def etiquetar(probs: pd.DataFrame) -> pd.DataFrame:
    """Añade a las probabilidades la etiqueta, la puntuación y la relevancia de cada frase.

    - label: clase más probable entre hawkish, neutral y dovish; "irrelevant" se suma a
      neutral, porque en la app esas frases se muestran como neutrales.
    - score: p_hawkish − p_dovish, en [−1, 1]; positivo = hawkish.
    - relevant: False si "irrelevant" es la clase más probable de las cuatro. En los
      modelos sin esa clase (p_irrelevant vacío), todas las frases cuentan como relevantes.
    """
    p_irrelevant = probs["p_irrelevant"].fillna(0.0)
    tres = pd.DataFrame({"hawkish": probs["p_hawkish"],
                         "neutral": probs["p_neutral"] + p_irrelevant,
                         "dovish": probs["p_dovish"]})
    cuatro = probs[["p_hawkish", "p_neutral", "p_dovish", "p_irrelevant"]].fillna(-1.0)

    salida = probs.copy()
    salida["label"] = tres.idxmax(axis=1)
    salida["score"] = probs["p_hawkish"] - probs["p_dovish"]
    salida["relevant"] = cuatro.idxmax(axis=1) != "p_irrelevant"
    return salida
