"""Síntesis de voz (TTS) · Bloque 1.

Genera audio hablado a partir de texto.

Funciones:
    texto_a_voz(texto, modelo_id)       -> bytes (WAV)
    generar_briefing(event_date)        -> escribe briefing.mp3
    cargar_modelo_tts(modelo_id)        -> modelo cargado en memoria
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

SAMPLE_RATE = 24000  # frecuencia de muestreo por defecto de la mayoría de modelos TTS


def texto_a_voz(texto: str, modelo_id: str | None = None) -> bytes:
    """Genera audio hablado a partir de texto (función del chat en vivo).

    Parámetros
    ----------
    texto     : texto a convertir en voz.
    modelo_id : identificador del modelo TTS. Si es None, usa config.yaml.

    Devuelve
    --------
    bytes con el audio en formato WAV.
    """
    from src.audio.models.tts_backends import crear_backend_tts
    from src.config import load_config

    if modelo_id is None:
        modelo_id = load_config()["models"].get("tts")

    backend = crear_backend_tts(modelo_id)
    audio_array, sr = backend.sintetizar(texto)

    return _array_a_wav(audio_array, sr)


def generar_briefing(event_date: str, modelo_id: str | None = None) -> Path:
    """Genera briefing.mp3 a partir de summary.json de un evento.

    Lee el campo `briefing_tts` del fichero summary.json del evento
    y genera el audio correspondiente.

    Parámetros
    ----------
    event_date : fecha del evento (AAAA-MM-DD).
    modelo_id  : identificador del modelo TTS.

    Devuelve
    --------
    Ruta al fichero briefing.mp3 generado.
    """
    from src.config import event_dir

    carpeta = event_dir(event_date)
    summary_path = carpeta / "summary.json"

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    texto = summary.get("briefing_tts", summary.get("briefing", ""))
    if not texto:
        raise ValueError(f"No se encontró briefing_tts en {summary_path}")

    audio_wav = texto_a_voz(texto, modelo_id)
    output_path = carpeta / "briefing.mp3"
    _wav_a_mp3(audio_wav, output_path)

    return output_path


def cargar_modelo_tts(modelo_id: str | None = None):
    """Precarga el modelo TTS en memoria y lo devuelve."""
    from src.audio.models.tts_backends import crear_backend_tts
    from src.config import load_config

    if modelo_id is None:
        modelo_id = load_config()["models"].get("tts")

    return crear_backend_tts(modelo_id)


def _array_a_wav(audio_array, sample_rate: int) -> bytes:
    """Convierte un array numpy de audio a bytes WAV."""
    import io
    import struct
    import wave

    import numpy as np

    # Normalizar a int16
    if audio_array.dtype != np.int16:
        audio_array = (audio_array * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_array.tobytes())

    return buffer.getvalue()


def _wav_a_mp3(wav_bytes: bytes, output_path: Union[str, Path]) -> None:
    """Convierte WAV a MP3 usando ffmpeg."""
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_path, "-codec:a", "libmp3lame", "-qscale:a", "2", str(output_path)],
        capture_output=True,
        check=True,
    )

    Path(tmp_path).unlink(missing_ok=True)
