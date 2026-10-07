"""Backends de síntesis de voz (TTS) · Bloque 1.

Todos los candidatos exponen la misma interfaz:
    sintetizar(texto) -> tuple[np.ndarray, int]  (audio array, sample_rate)
    memoria_gb()      -> VRAM aproximada

Candidatos:
    - hexgrad/Kokoro-82M             (rápido, calidad profesional, MIT)
    - parler-tts/parler-tts-mini-v1  (controlable por prompt de texto)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.audio.models.asr_backends import dispositivo, liberar_memoria


# ---------------------------------------------------------------------------
# Interfaz base
# ---------------------------------------------------------------------------
class BackendTTS(ABC):
    """Interfaz que todos los backends de TTS deben implementar."""

    @abstractmethod
    def sintetizar(self, texto: str) -> tuple[np.ndarray, int]:
        """Genera audio hablado a partir de texto.

        Devuelve
        --------
        Tupla (audio_array, sample_rate):
        - audio_array: numpy array float32 en [-1, 1].
        - sample_rate: frecuencia de muestreo en Hz.
        """

    @abstractmethod
    def memoria_gb(self) -> float:
        """VRAM aproximada en GB."""


# ---------------------------------------------------------------------------
# Kokoro
# ---------------------------------------------------------------------------
class KokoroBackend(BackendTTS):
    """TTS con Kokoro (hexgrad/Kokoro-82M).

    Modelo ligero (82M parámetros), calidad profesional, licencia MIT.
    Requiere: pip install kokoro>=0.8
    """

    def __init__(self):
        try:
            from kokoro import KPipeline
        except ImportError:
            raise ImportError(
                "Kokoro no está instalado. Instálalo con: pip install kokoro>=0.8 soundfile"
            )

        # Pipeline en inglés americano (voz por defecto: af_heart)
        self.pipeline = KPipeline(lang_code="a")
        self.voice = "af_heart"  # voz femenina profesional
        self.sample_rate = 24000

    def sintetizar(self, texto: str) -> tuple[np.ndarray, int]:
        # Kokoro genera por fragmentos; los concatenamos
        fragmentos = []
        for _, _, audio in self.pipeline(texto, voice=self.voice):
            fragmentos.append(audio)

        if not fragmentos:
            return np.array([], dtype=np.float32), self.sample_rate

        audio_completo = np.concatenate(fragmentos)
        return audio_completo, self.sample_rate

    def memoria_gb(self) -> float:
        return 0.4


# ---------------------------------------------------------------------------
# Parler-TTS (placeholder)
# ---------------------------------------------------------------------------
class ParlerTTSBackend(BackendTTS):
    """TTS con Parler-TTS mini v1.

    Modelo controlable por descripción textual del estilo de voz.
    Requiere: pip install parler-tts
    """

    def __init__(self):
        # TODO: implementar para el benchmark comparativo
        raise NotImplementedError(
            "Backend Parler-TTS pendiente de implementación. "
            "Usa Kokoro como primera opción."
        )

    def sintetizar(self, texto: str) -> tuple[np.ndarray, int]:
        raise NotImplementedError

    def memoria_gb(self) -> float:
        return 1.0


# ---------------------------------------------------------------------------
# Fábrica
# ---------------------------------------------------------------------------
BACKENDS = {
    "hexgrad/Kokoro-82M": KokoroBackend,
    "kokoro": KokoroBackend,
    "parler-tts/parler-tts-mini-v1": ParlerTTSBackend,
}


def crear_backend_tts(modelo_id: str | None) -> BackendTTS:
    """Crea el backend adecuado según el identificador del modelo."""
    if modelo_id is None:
        modelo_id = "kokoro"

    cls = BACKENDS.get(modelo_id)
    if cls is None:
        raise ValueError(
            f"Modelo TTS no soportado: {modelo_id}. "
            f"Opciones: {list(BACKENDS.keys())}"
        )
    return cls()
