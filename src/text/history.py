"""Índice histórico de postura por rueda de prensa · Bloque 2, fase 3."""
import re

import pandas as pd
from scipy.stats import spearmanr

from src.text.stance import puntuacion_rueda

# Fórmula con la que la declaración anuncia la decisión de tipos
# ("The Governing Council today decided to raise the three key ECB interest rates...")
PATRON_DECISION = re.compile(
    r"decided to (raise|increase|lower|reduce|cut|keep|leave|maintain)\s+the\s+(?:three\s+)?"
    r"(?:key ECB interest rates|deposit facility rate)", re.IGNORECASE)
VERBO_A_DECISION = {"raise": "subida", "increase": "subida", "lower": "bajada", "reduce": "bajada",
                    "cut": "bajada", "keep": "mantenimiento", "leave": "mantenimiento", "maintain": "mantenimiento"}

# Entre estas fechas los tipos oficiales no cambiaron (facilidad de depósito en −0,50 %). Si en
# ese periodo la declaración no usa la fórmula habitual, la decisión fue de mantenimiento.
PERIODO_SIN_CAMBIOS = ("2019-12-01", "2022-06-30")

CODIGO_DECISION = {"bajada": -1, "mantenimiento": 0, "subida": 1}


def decisiones_de_tipos(corpus: pd.DataFrame) -> pd.DataFrame:
    """Decisión de tipos de cada rueda (subida, bajada o mantenimiento), extraída de su declaración."""
    filas = []
    for fecha, frases in corpus[corpus["role"] == "statement"].groupby("date"):
        coincidencia = PATRON_DECISION.search(" ".join(frases.sort_values("sentence_id")["text"]))
        if coincidencia:
            decision, origen = VERBO_A_DECISION[coincidencia.group(1).lower()], "declaración"
        elif PERIODO_SIN_CAMBIOS[0] <= fecha <= PERIODO_SIN_CAMBIOS[1]:
            decision, origen = "mantenimiento", "periodo sin cambios"
        else:
            raise ValueError(f"No se encuentra la decisión de tipos en la declaración del {fecha}")
        filas.append({"date": fecha, "decision": decision, "origen": origen})
    return pd.DataFrame(filas)


def indice_historico(frases: pd.DataFrame, listado: pd.DataFrame, decisiones: pd.DataFrame) -> pd.DataFrame:
    """Una fila por rueda: puntuaciones, decisión de tipos, percentil en el histórico y enlace oficial."""
    indice = pd.DataFrame([{"date": fecha, **puntuacion_rueda(grupo)} for fecha, grupo in frases.groupby("date")])
    indice = (indice.merge(decisiones[["date", "decision"]], on="date", how="left")
                    .merge(listado[["date", "url"]], on="date", how="left"))
    # Percentil: porcentaje de ruedas del histórico con una puntuación igual o inferior
    indice["percentile"] = (indice["score"].rank(method="max", pct=True) * 100).round().astype(int)
    columnas = ["date", "score", "score_statement", "score_qa", "n_sentences", "decision", "percentile", "url"]
    return indice[columnas].sort_values("date").reset_index(drop=True)


def medias_por_decision(indice: pd.DataFrame) -> pd.DataFrame:
    """Puntuación media de las respuestas según la decisión del día y la de la reunión siguiente."""
    datos = indice.sort_values("date").assign(decision_siguiente=lambda d: d["decision"].shift(-1))
    del_dia = datos.groupby("decision")["score_qa"].agg(["mean", "count"])
    siguiente = datos.dropna(subset=["decision_siguiente"]).groupby("decision_siguiente")["score_qa"].agg(["mean", "count"])
    tabla = pd.concat({"misma reunión": del_dia, "reunión siguiente": siguiente}, axis=1)
    return tabla.reindex(["bajada", "mantenimiento", "subida"])


def correlaciones_con_decisiones(indice: pd.DataFrame) -> pd.DataFrame:
    """Correlación de Spearman entre cada puntuación y la decisión codificada (−1, 0, +1).

    La declaración frente a la decisión del día sirve de referencia: es en parte mecánica,
    porque la propia declaración anuncia la decisión.
    """
    datos = indice.sort_values("date").assign(decision_siguiente=lambda d: d["decision"].shift(-1))
    casos = [("Declaración", "score_statement", "decision", "misma reunión (referencia, en parte mecánica)"),
             ("Respuestas", "score_qa", "decision", "misma reunión"),
             ("Respuestas", "score_qa", "decision_siguiente", "reunión siguiente"),
             ("Global", "score", "decision_siguiente", "reunión siguiente")]
    filas = []
    for parte, puntuacion, decision, frente_a in casos:
        validos = datos.dropna(subset=[decision])
        rho, p = spearmanr(validos[puntuacion], validos[decision].map(CODIGO_DECISION))
        filas.append({"puntuacion": parte, "decision": frente_a, "rho_spearman": rho, "p_valor": p,
                      "n": len(validos)})
    return pd.DataFrame(filas)
