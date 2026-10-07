"""Diarización de hablantes · Bloque 1.

Identifica quién habla en cada momento del audio y asigna la etiqueta
de hablante (presidenta / vicepresidente / periodista) a cada segmento
de la transcripción.

Funciones:
    diarizar(audio_path, modelo_id) -> DataFrame con start, end, speaker
    asignar_hablantes(transcripcion, diarizacion) -> DataFrame enriquecido
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def diarizar(self, audio_path: Union[str, Path]) -> list[dict]:
    import torchaudio
    
    # Cargar audio con torchaudio (evita torchcodec)
    waveform, sample_rate = torchaudio.load(str(audio_path))
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}
    
    diarization = self.pipeline(audio_input)

    turnos = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turnos.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker_id": speaker,
        })

    return turnos


def asignar_hablantes(
    transcripcion: pd.DataFrame,
    diarizacion: pd.DataFrame,
) -> pd.DataFrame:
    """Cruza la transcripción (ASR) con la diarización para asignar hablantes.

    Cada segmento de la transcripción hereda el hablante mayoritario
    en su ventana temporal (por solapamiento).

    Además:
    - Mapea los speaker_id genéricos a nombres: el hablante con más tiempo
      total en la primera parte (statement) se asigna como 'presidenta';
      el segundo como 'vicepresidente'; el resto como 'periodista'.
    - Asigna la columna 'section': todo antes de la primera intervención
      de un periodista es 'statement'; lo posterior es 'qa'.

    Parámetros
    ----------
    transcripcion : DataFrame de transcribir() con start, end, text.
    diarizacion   : DataFrame de diarizar() con start, end, speaker_id.

    Devuelve
    --------
    DataFrame con columnas: start, end, speaker, section, text.
    """
    # 1. Para cada segmento de la transcripción, encontrar el speaker_id
    #    que más solapa con su ventana [start, end].
    speakers = []
    for _, fila in transcripcion.iterrows():
        solapamiento = _calcular_solapamiento(fila, diarizacion)
        speakers.append(solapamiento)

    transcripcion = transcripcion.copy()
    transcripcion["speaker_id"] = speakers

    # 2. Mapear speaker_id a nombres reales
    mapa = _mapear_hablantes(transcripcion, diarizacion)
    transcripcion["speaker"] = transcripcion["speaker_id"].map(mapa)

    # 3. Asignar sección (statement / qa)
    transcripcion["section"] = _asignar_seccion(transcripcion)

    return transcripcion[["start", "end", "speaker", "section", "text"]]


def _calcular_solapamiento(fila: pd.Series, diarizacion: pd.DataFrame) -> str:
    """Devuelve el speaker_id con mayor solapamiento en la ventana de la fila."""
    seg_start, seg_end = fila["start"], fila["end"]

    # Solapamiento: max(0, min(end1, end2) - max(start1, start2))
    overlap_start = diarizacion["start"].clip(lower=seg_start)
    overlap_end = diarizacion["end"].clip(upper=seg_end)
    overlap = (overlap_end - overlap_start).clip(lower=0)

    if overlap.sum() == 0:
        return "UNKNOWN"

    # Speaker con más solapamiento
    idx_max = overlap.idxmax()
    return diarizacion.loc[idx_max, "speaker_id"]


def _mapear_hablantes(transcripcion: pd.DataFrame, diarizacion: pd.DataFrame) -> dict:
    """Asigna nombres a los speaker_id genéricos.

    Heurística: en la primera mitad del audio (statement), la presidenta
    habla mucho más que el vicepresidente. El speaker_id con más tiempo
    acumulado en esa zona se asigna como 'presidenta'; el segundo como
    'vicepresidente'; el resto como 'periodista'.
    """
    # Tiempo total por speaker en la primera mitad del audio
    mitad = diarizacion["end"].max() / 2
    primera_mitad = diarizacion[diarizacion["start"] < mitad].copy()
    primera_mitad["duracion"] = primera_mitad["end"].clip(upper=mitad) - primera_mitad["start"]

    tiempo_por_speaker = primera_mitad.groupby("speaker_id")["duracion"].sum().sort_values(ascending=False)

    speakers_ordenados = tiempo_por_speaker.index.tolist()

    mapa = {}
    if len(speakers_ordenados) >= 1:
        mapa[speakers_ordenados[0]] = "presidenta"
    if len(speakers_ordenados) >= 2:
        mapa[speakers_ordenados[1]] = "vicepresidente"
    for s in speakers_ordenados[2:]:
        mapa[s] = "periodista"

    mapa["UNKNOWN"] = "periodista"
    return mapa


def _asignar_seccion(transcripcion: pd.DataFrame) -> pd.Series:
    """Asigna 'statement' o 'qa' según la primera intervención de un periodista."""
    primera_pregunta = transcripcion[transcripcion["speaker"] == "periodista"]

    if primera_pregunta.empty:
        # Sin periodistas detectados: todo es statement
        return pd.Series("statement", index=transcripcion.index)

    corte = primera_pregunta.iloc[0]["start"]
    return transcripcion["start"].apply(lambda t: "statement" if t < corte else "qa")
