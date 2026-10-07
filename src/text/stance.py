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


def puntuacion_rueda(frases: pd.DataFrame) -> dict:
    """Puntuación de una rueda de prensa a partir de la postura de sus frases.

    - Cada frase pesa su probabilidad de ser relevante (1 − p_irrelevant): un saludo pesa
      casi 0 y una frase con postura clara, casi 1. Así, los errores de la detección de
      relevancia se atenúan en lugar de convertirse en exclusiones. En los modelos sin la
      clase "irrelevant", todas las frases pesan 1.
    - score_statement y score_qa: media ponderada de la declaración y de las respuestas.
    - score: 50 % declaración y 50 % respuestas, para que la puntuación no dependa de cuánto
      dure el turno de preguntas y sea comparable entre ruedas.
    """
    peso = 1.0 - frases["p_irrelevant"].fillna(0.0)

    def media_ponderada(mascara: pd.Series) -> float:
        w = peso[mascara]
        return float((w * frases.loc[mascara, "score"]).sum() / w.sum()) if w.sum() > 0 else float("nan")

    score_statement = media_ponderada(frases["section"] == "statement")
    score_qa = media_ponderada(frases["section"] == "qa")
    partes = [s for s in (score_statement, score_qa) if s == s]   # descarta NaN si falta una parte
    return {
        "score": sum(partes) / len(partes) if partes else float("nan"),
        "score_statement": score_statement,
        "score_qa": score_qa,
        "n_sentences": len(frases),
    }
