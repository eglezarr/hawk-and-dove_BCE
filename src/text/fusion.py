"""Postura, tono de voz y expresión facial de cada frase (signals.csv) · Bloque 2, fase 5.

Une lo que se dice (postura del texto, bloque 2) con cómo se dice (voz, bloque 1) y cómo se
ve (cara, bloque 3) en la ventana de tiempo de cada frase del panel.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import EVENTS_DIR

UMBRAL_DESVIACION = 1.5   # desviaciones típicas para que la voz o la cara cuenten como momento clave
DURACION_MINIMA = 3.0     # segundos: una frase más corta tiene 1-2 fotogramas y su media es poco fiable


# ---------------------------------------------------------------------------
# Señales por frase
# ---------------------------------------------------------------------------
def media_por_ventana(ventanas: pd.DataFrame, senal: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Media de cada columna de la señal en la ventana [start, end) de cada frase.

    Cada fila de la señal pesa los segundos que solapa con la ventana. Si ninguna fila con
    valor solapa, el resultado queda vacío.
    """
    solape = np.clip(np.minimum(ventanas["end"].to_numpy()[:, None], senal["end"].to_numpy()[None, :])
                     - np.maximum(ventanas["start"].to_numpy()[:, None], senal["start"].to_numpy()[None, :]),
                     0, None)
    salida = {}
    for columna in columnas:
        valores = senal[columna].to_numpy(dtype=float)
        con_valor = ~np.isnan(valores)
        pesos = solape[:, con_valor]
        total = pesos.sum(axis=1)
        salida[columna] = np.where(total > 0, pesos @ valores[con_valor] / np.where(total > 0, total, 1), np.nan)
    return pd.DataFrame(salida, index=ventanas.index)


def filas_de_la_presidenta(cara: pd.DataFrame) -> pd.DataFrame:
    """Fotogramas con la cara de la presidenta en primer plano (durante las preguntas se enfoca a los periodistas)."""
    return cara[cara["face_detected"].astype(bool) & (cara["person"] == "presidenta")]


def desviacion(valores: pd.Series, grupos: pd.Series) -> pd.Series:
    """Desviación respecto a la media de cada hablante en la rueda, en desviaciones típicas.

    En una banquera central el nivel dice poco (sale «neutral» casi siempre); lo informativo
    es el cambio respecto a su forma habitual de hablar y gesticular.
    """
    agrupado = valores.groupby(grupos)
    return (valores - agrupado.transform("mean")) / agrupado.transform("std")


def senales_por_frase(postura: pd.DataFrame, cara: pd.DataFrame | None, voz: pd.DataFrame | None) -> pd.DataFrame:
    """Tabla de signals.csv: stance.csv con tiempos más voz y cara en la ventana de cada frase.

    Sin voz o sin cara, sus columnas quedan vacías: la tabla se puede generar con lo que haya.
    """
    senales = postura.copy()
    con_tiempo = senales["start"].notna() & senales["end"].notna()
    for columna in ["voice_arousal", "voice_valence", "face_valence", "face_arousal"]:
        senales[columna] = np.nan

    if voz is not None and con_tiempo.any():
        columnas = [c for c in ["arousal", "valence"] if c in voz]
        medias = media_por_ventana(senales[con_tiempo], voz, columnas)
        for c in columnas:
            senales.loc[con_tiempo, f"voice_{c}"] = medias[c]
    if cara is not None and con_tiempo.any():
        medias = media_por_ventana(senales[con_tiempo], filas_de_la_presidenta(cara), ["valence", "arousal"])
        senales.loc[con_tiempo, "face_valence"] = medias["valence"]
        senales.loc[con_tiempo, "face_arousal"] = medias["arousal"]

    senales["voice_arousal_z"] = desviacion(senales["voice_arousal"], senales["speaker"])
    senales["face_valence_z"] = desviacion(senales["face_valence"], senales["speaker"])
    senales["key_moment"] = False
    return senales


# ---------------------------------------------------------------------------
# Resumen del evento (summary.json)
# ---------------------------------------------------------------------------
def _frente_al_historico(valor: float, columna: str, fecha: str) -> float | None:
    """Desviación de la media de esta rueda frente a las demás ruedas procesadas (None con menos de 3)."""
    otras = []
    for ruta in EVENTS_DIR.glob("*/signals.csv"):
        if ruta.parent.name == fecha:
            continue
        datos = pd.read_csv(ruta)
        if columna in datos and datos[columna].notna().any():
            otras.append(datos.loc[datos["speaker"] == "presidenta", columna].mean())
    if len(otras) < 3 or np.std(otras, ddof=1) == 0 or np.isnan(valor):
        return None
    return round(float((valor - np.mean(otras)) / np.std(otras, ddof=1)), 2)


def resumen_voz(senales: pd.DataFrame, fecha: str) -> dict | None:
    """Bloque voice de summary.json: tono de voz medio de la presidenta y su posición frente a otras ruedas."""
    valores = senales.loc[senales["speaker"] == "presidenta", "voice_arousal"]
    if valores.notna().sum() == 0:
        return None
    media = float(valores.mean())
    return {"arousal_mean": round(media, 3), "arousal_z_vs_history": _frente_al_historico(media, "voice_arousal", fecha)}


def resumen_cara(senales: pd.DataFrame, cara: pd.DataFrame | None, fecha: str) -> dict | None:
    """Bloque face de summary.json: expresión media de la presidenta frente a sus otras ruedas.

    La etiqueta es relativa: en la rueda del 10/09/2026 ninguna emoción supera el 21 % de
    probabilidad media, así que la emoción "ganadora" no describe nada. Se compara la
    valencia media con la de las demás ruedas procesadas (necesita al menos 3).
    """
    if cara is None:
        return None
    filas = filas_de_la_presidenta(cara)
    if filas.empty:
        return None
    z = _frente_al_historico(senales.loc[senales["speaker"] == "presidenta", "face_valence"].mean(), "face_valence", fecha)
    etiqueta = None if z is None else ("more positive than usual" if z >= 1 else "more tense than usual" if z <= -1
                                       else "usual")
    return {"valence_mean": round(float(filas["valence"].mean()), 3), "arousal_mean": round(float(filas["arousal"].mean()), 3),
            "label": etiqueta, "frames": int(len(filas)), "valence_z_vs_history": z}


def momentos_de_entrega(senales: pd.DataFrame, excluidas: set[int]) -> list[dict]:
    """Hasta dos momentos clave por cómo se dice: la frase relevante de la presidenta con la voz
    más activada y la de expresión facial más alejada de su media, si superan el umbral.

    Solo compiten las frases de al menos DURACION_MINIMA segundos: la media de una frase muy
    corta sale de uno o dos fotogramas, es más variable y ganaría por azar.
    """
    candidatas = senales[senales["relevant"] & (senales["speaker"] == "presidenta")
                         & ~senales["sentence_id"].isin(excluidas)
                         & ((senales["end"] - senales["start"]) >= DURACION_MINIMA)]
    momentos = []
    for columna, absoluto, motivo in [
            ("voice_arousal_z", False, "Emphatic delivery: voice arousal {z:+.1f} SD from her average in this press conference"),
            ("face_valence_z", True, "Facial expression shift: valence {z:+.1f} SD from her average in this press conference")]:
        valores = candidatas[columna].abs() if absoluto else candidatas[columna]
        if valores.notna().sum() == 0 or valores.max() < UMBRAL_DESVIACION:
            continue
        fila = candidatas.loc[valores.idxmax()]
        momentos.append({"start": round(float(fila["start"]), 1), "end": round(float(fila["end"]), 1),
                         "sentence_id": int(fila["sentence_id"]),
                         "speaker": fila["speaker"], "text": fila["text"], "score": round(float(fila["score"]), 3),
                         "reason": motivo.format(z=fila[columna]), "signal": columna.split("_")[0]})
    return momentos


def actualizar_resumen(resumen: dict, senales: pd.DataFrame, cara: pd.DataFrame | None, fecha: str) -> dict:
    """Añade al summary.json los tiempos de citas y momentos clave, y los bloques voice y face."""
    tiempos = senales.set_index("sentence_id")[["start", "end"]]

    def con_tiempo(elemento: dict) -> dict:
        if elemento.get("sentence_id") in tiempos.index:
            inicio, fin = tiempos.loc[elemento["sentence_id"]]
            if pd.notna(inicio):
                elemento = {**elemento, "start": round(float(inicio), 1), "end": round(float(fin), 1)}
        return elemento

    momentos = [{**con_tiempo(m), "signal": m.get("signal", "stance")} for m in resumen.get("key_moments", [])
                if m.get("signal", "stance") == "stance"]
    momentos += momentos_de_entrega(senales, {m["sentence_id"] for m in momentos})
    momentos.sort(key=lambda m: (m["start"] is None, m["start"] or 0))

    return {**resumen,
            "voice": resumen_voz(senales, fecha),
            "face": resumen_cara(senales, cara, fecha),
            "key_moments": momentos,
            "citations": [con_tiempo(c) for c in resumen.get("citations", [])]}


# ---------------------------------------------------------------------------
# Ficheros del evento
# ---------------------------------------------------------------------------
def leer_opcional(carpeta: Path, nombre: str) -> pd.DataFrame | None:
    ruta = carpeta / nombre
    return pd.read_csv(ruta) if ruta.exists() else None


def guardar(carpeta: Path, postura: pd.DataFrame, senales: pd.DataFrame, resumen: dict) -> None:
    """Escribe stance.csv (con tiempos), signals.csv y summary.json, y marca los momentos clave."""
    claves = {m["sentence_id"] for m in resumen["key_moments"]}
    senales = senales.assign(key_moment=senales["sentence_id"].isin(claves))
    postura.to_csv(carpeta / "stance.csv", index=False)
    senales.to_csv(carpeta / "signals.csv", index=False)
    (carpeta / "summary.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
