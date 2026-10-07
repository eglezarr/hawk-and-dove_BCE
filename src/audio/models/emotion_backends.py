"""Backends de análisis de emoción en la voz · Bloque 1.

Todos los candidatos exponen la misma interfaz:
    analizar(audio_path, ventana, solapamiento) -> list[dict] con {start, end, arousal, valence, dominance}
    memoria_gb()                                 -> VRAM aproximada

Candidatos:
    - audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim  (arousal, valence, dominance directos)
    - emotion2vec/emotion2vec_plus_large                      (emociones categóricas + embeddings)
"""
from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np

from src.audio.models.asr_backends import dispositivo, liberar_memoria


# ---------------------------------------------------------------------------
# Interfaz base
# ---------------------------------------------------------------------------
class BackendEmocion(ABC):
    """Interfaz que todos los backends de emoción deben implementar."""

    @abstractmethod
    def analizar(
        self,
        audio_path: Union[str, Path],
        ventana: float = 2.0,
        solapamiento: float = 0.5,
    ) -> list[dict]:
        """Analiza el tono de voz en ventanas temporales.

        Devuelve
        --------
        Lista de ventanas, cada una con:
        - start (float), end (float): intervalo en segundos.
        - arousal (float): activación [0, 1].
        - valence (float): valencia [0, 1].
        - dominance (float | None): dominancia [0, 1] o None si no disponible.
        """

    @abstractmethod
    def memoria_gb(self) -> float:
        """VRAM aproximada en GB."""


# ---------------------------------------------------------------------------
# wav2vec2 audEERING
# ---------------------------------------------------------------------------
class Wav2Vec2AudeeringBackend(BackendEmocion):
    """Emoción dimensional con wav2vec2 (audEERING).

    Modelo: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
    Produce directamente arousal, dominance, valence en [0, 1].
    """

    MODELO_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    SAMPLE_RATE = 16000

    def __init__(self):
        import torch
        import transformers

        self.device = dispositivo()
        self.processor = transformers.Wav2Vec2Processor.from_pretrained(self.MODELO_ID)
        self.model = transformers.Wav2Vec2ForSequenceClassification.from_pretrained(
            self.MODELO_ID
        ).to(self.device)
        self.model.eval()

    def analizar(
        self,
        audio_path: Union[str, Path],
        ventana: float = 2.0,
        solapamiento: float = 0.5,
    ) -> list[dict]:
        import torch
        import torchaudio

        waveform, sr = torchaudio.load(str(audio_path))

        # Convertir a mono y resamplear si es necesario
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)
        duracion_total = len(waveform) / self.SAMPLE_RATE

        # Ventanas deslizantes
        paso = ventana * (1 - solapamiento)
        resultados = []
        t = 0.0

        while t < duracion_total:
            t_end = min(t + ventana, duracion_total)
            inicio_muestra = int(t * self.SAMPLE_RATE)
            fin_muestra = int(t_end * self.SAMPLE_RATE)

            fragmento = waveform[inicio_muestra:fin_muestra]

            if len(fragmento) < self.SAMPLE_RATE * 0.5:  # fragmento muy corto
                break

            inputs = self.processor(
                fragmento.numpy(),
                sampling_rate=self.SAMPLE_RATE,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits.squeeze().cpu().numpy()

            # logits: [arousal, dominance, valence] en escala ~ [0, 1]
            resultados.append({
                "start": round(t, 2),
                "end": round(t_end, 2),
                "arousal": float(np.clip(logits[0], 0, 1)),
                "valence": float(np.clip(logits[2], 0, 1)),
                "dominance": float(np.clip(logits[1], 0, 1)),
            })

            t += paso

        return resultados

    def memoria_gb(self) -> float:
        return 1.3


# ---------------------------------------------------------------------------
# emotion2vec+ (placeholder)
# ---------------------------------------------------------------------------
class Emotion2VecBackend(BackendEmocion):
    """Emoción con emotion2vec+ large.

    Modelo: emotion2vec/emotion2vec_plus_large
    Produce emociones categóricas; se mapean a arousal/valence.
    """

    def __init__(self):
        # TODO: implementar para el benchmark comparativo
        raise NotImplementedError(
            "Backend emotion2vec+ pendiente de implementación. "
            "Usa wav2vec2 (audEERING) como primera opción."
        )

    def analizar(self, audio_path, ventana=2.0, solapamiento=0.5) -> list[dict]:
        raise NotImplementedError

    def memoria_gb(self) -> float:
        return 1.5


# ---------------------------------------------------------------------------
# Fábrica
# ---------------------------------------------------------------------------
BACKENDS = {
    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim": Wav2Vec2AudeeringBackend,
    "emotion2vec_plus_large": Emotion2VecBackend,
}

# Alias cortos para config.yaml
ALIASES = {
    "wav2vec2-audeering": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    "emotion2vec+": "emotion2vec_plus_large",
}


def crear_backend_emocion(modelo_id: str | None) -> BackendEmocion:
    """Crea el backend adecuado según el identificador del modelo."""
    if modelo_id is None:
        modelo_id = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

    modelo_id = ALIASES.get(modelo_id, modelo_id)

    cls = BACKENDS.get(modelo_id)
    if cls is None:
        raise ValueError(
            f"Modelo de emoción no soportado: {modelo_id}. "
            f"Opciones: {list(BACKENDS.keys())}"
        )
    return cls()
