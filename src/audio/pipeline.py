"""Pipeline del Bloque 1: procesa una rueda de prensa completa.

Encadena: extracción de audio → ASR → diarización → emoción → TTS briefing.
Escribe transcript.csv, voice.csv y briefing.mp3 en data/events/<fecha>/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union


def extraer_audio(video_path: Union[str, Path], output_path: Union[str, Path] | None = None) -> Path:
    """Extrae la pista de audio de un vídeo usando ffmpeg.

    Parámetros
    ----------
    video_path  : ruta al vídeo (.mp4, .webm, etc.).
    output_path : ruta de salida (.wav). Si es None, se genera junto al vídeo.

    Devuelve
    --------
    Ruta al fichero de audio extraído.
    """
    import subprocess

    video_path = Path(video_path)
    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    output_path = Path(output_path)

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",                    # sin vídeo
            "-acodec", "pcm_s16le",   # WAV 16-bit
            "-ar", "16000",           # 16 kHz (estándar para ASR)
            "-ac", "1",               # mono
            str(output_path),
        ],
        capture_output=True,
        check=True,
    )

    return output_path


def procesar_evento(event_date: str, video_path: Union[str, Path]) -> dict:
    """Procesa una rueda de prensa completa y escribe los ficheros de B1.

    Pasos:
    1. Extrae el audio del vídeo.
    2. Transcribe con el modelo ASR ganador.
    3. Diariza para identificar hablantes.
    4. Cruza transcripción con diarización → transcript.csv.
    5. Analiza la emoción de voz → voice.csv.
    6. Genera la píldora de audio → briefing.mp3 (requiere summary.json de B2).

    Parámetros
    ----------
    event_date : fecha del evento (AAAA-MM-DD).
    video_path : ruta al vídeo de la rueda de prensa.

    Devuelve
    --------
    dict con las rutas de los ficheros generados:
    {"transcript": Path, "voice": Path, "briefing": Path | None}.
    """
    from src.audio.diarization import asignar_hablantes, diarizar
    from src.audio.transcription import transcribir
    from src.audio.tts import generar_briefing
    from src.audio.voice_emotion import analizar_emocion
    from src.config import event_dir

    carpeta = event_dir(event_date)
    carpeta.mkdir(parents=True, exist_ok=True)

    # 1. Extraer audio
    audio_path = carpeta / "audio.wav"
    extraer_audio(video_path, audio_path)
    print(f"Audio extraído: {audio_path}")

    # 2. Transcribir
    transcripcion = transcribir(audio_path)
    print(f"Transcripción: {len(transcripcion)} segmentos")

    # 3. Diarizar
    diarizacion = diarizar(audio_path)
    print(f"Diarización: {diarizacion['speaker_id'].nunique()} hablantes detectados")

    # 4. Cruzar → transcript.csv
    transcript = asignar_hablantes(transcripcion, diarizacion)
    transcript_path = carpeta / "transcript.csv"
    transcript.to_csv(transcript_path, index=False)
    print(f"Guardado: {transcript_path}")

    # 5. Emoción de voz → voice.csv
    voice = analizar_emocion(audio_path)
    voice_path = carpeta / "voice.csv"
    voice.to_csv(voice_path, index=False)
    print(f"Guardado: {voice_path}")

    # 6. Briefing (solo si existe summary.json de B2)
    briefing_path = None
    summary_path = carpeta / "summary.json"
    if summary_path.exists():
        try:
            briefing_path = generar_briefing(event_date)
            print(f"Guardado: {briefing_path}")
        except Exception as e:
            print(f"No se pudo generar briefing: {e}")

    return {
        "transcript": transcript_path,
        "voice": voice_path,
        "briefing": briefing_path,
    }


# ---------------------------------------------------------------------------
# Integración con signals.csv (cruce B1 + B2 + B3)
# ---------------------------------------------------------------------------
def generar_signals(event_date: str) -> Path:
    """Genera signals.csv cruzando stance.csv (B2) con voice.csv (B1) y face.csv (B3).

    Para cada frase de stance.csv, agrega las señales de voz y cara de la
    ventana temporal correspondiente (media ponderada por solapamiento).
    Añade ``key_moment`` desde summary.json.

    Parámetros
    ----------
    event_date : fecha del evento (AAAA-MM-DD).

    Devuelve
    --------
    Ruta al fichero signals.csv generado.
    """
    import numpy as np
    import pandas as pd

    from src.config import event_dir

    carpeta = event_dir(event_date)

    # --- 1. Cargar stance.csv (B2) -------------------------------------------
    stance_path = carpeta / "stance.csv"
    if not stance_path.exists():
        raise FileNotFoundError(
            f"No se encuentra stance.csv en {carpeta}. "
            "Ejecuta primero el bloque 2 para generar las etiquetas de stance."
        )
    stance = pd.read_csv(stance_path)

    # --- 2. Agregar emoción de voz (B1) a nivel de frase --------------------
    voice_path = carpeta / "voice.csv"
    if voice_path.exists():
        voice = pd.read_csv(voice_path)
        stance = _agregar_señal(stance, voice, "arousal", "voice_arousal")
        stance = _agregar_señal(stance, voice, "valence", "voice_valence")
    else:
        stance["voice_arousal"] = np.nan
        stance["voice_valence"] = np.nan

    # --- 3. Agregar expresión facial (B3) a nivel de frase -------------------
    face_path = carpeta / "face.csv"
    if face_path.exists():
        face = pd.read_csv(face_path)
        stance = _agregar_señal(stance, face, "valence", "face_valence")
        stance = _agregar_señal(stance, face, "arousal", "face_arousal")
    else:
        stance["face_valence"] = np.nan
        stance["face_arousal"] = np.nan

    # --- 4. Marcar key_moments desde summary.json ----------------------------
    stance["key_moment"] = False
    summary_path = carpeta / "summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        key_ids = {km["sentence_id"] for km in summary.get("key_moments", [])}
        if "sentence_id" in stance.columns and key_ids:
            stance.loc[stance["sentence_id"].isin(key_ids), "key_moment"] = True

    # --- 5. Seleccionar columnas del contrato y escribir ---------------------
    columnas_signals = [
        "start", "end", "speaker", "section", "text",
        "label", "score",
        "voice_arousal", "voice_valence",
        "face_valence", "face_arousal",
        "key_moment",
    ]
    # Solo incluir columnas que existen en el DataFrame
    columnas_presentes = [c for c in columnas_signals if c in stance.columns]
    signals = stance[columnas_presentes]

    signals_path = carpeta / "signals.csv"
    signals.to_csv(signals_path, index=False)
    print(f"Guardado: {signals_path}")

    return signals_path


def _agregar_señal(
    frases: "pd.DataFrame",
    ventanas: "pd.DataFrame",
    col_origen: str,
    col_destino: str,
) -> "pd.DataFrame":
    """Agrega una señal temporal (ventanas) a nivel de frase por solapamiento.

    Para cada frase, calcula la media ponderada de ``col_origen`` en las
    ventanas que solapan con [start, end] de la frase.  El peso de cada
    ventana es la duración del solapamiento.

    Parámetros
    ----------
    frases     : DataFrame con columnas ``start`` y ``end`` (segundos).
    ventanas   : DataFrame con columnas ``start``, ``end`` y ``col_origen``.
    col_origen : nombre de la columna en ``ventanas`` que se agrega.
    col_destino: nombre de la columna que se crea en ``frases``.

    Devuelve
    --------
    DataFrame ``frases`` con la nueva columna ``col_destino``.
    """
    import numpy as np

    valores = []
    for _, frase in frases.iterrows():
        f_start = frase.get("start")
        f_end = frase.get("end")

        # Si la frase no tiene timestamps, no se puede agregar
        if f_start is None or f_end is None or np.isnan(f_start) or np.isnan(f_end):
            valores.append(np.nan)
            continue

        # Ventanas que solapan con la frase
        overlap_start = np.maximum(ventanas["start"].values, f_start)
        overlap_end = np.minimum(ventanas["end"].values, f_end)
        overlap_dur = np.maximum(overlap_end - overlap_start, 0.0)

        total_overlap = overlap_dur.sum()
        if total_overlap < 1e-6:
            valores.append(np.nan)
        else:
            # Media ponderada por duración del solapamiento
            val = (ventanas[col_origen].values * overlap_dur).sum() / total_overlap
            valores.append(round(val, 4))

    frases = frases.copy()
    frases[col_destino] = valores
    return frases
