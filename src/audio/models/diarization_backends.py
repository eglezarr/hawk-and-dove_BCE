"""Backends de diarización de hablantes · Bloque 1.

Todos los candidatos exponen la misma interfaz:
    diarizar(audio_path) -> list[dict] con {start, end, speaker_id}
    memoria_gb()         -> VRAM aproximada

Candidatos:
    - pyannote/speaker-diarization-3.1   (estado del arte, requiere HF_TOKEN)
    - NeMo MSDD                          (alternativa sin licencia restringida)

NOTA: pyannote requiere aceptar la licencia en Hugging Face y configurar HF_TOKEN.
      https://huggingface.co/pyannote/speaker-diarization-3.1
"""
from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from src.audio.models.asr_backends import dispositivo, liberar_memoria


# ---------------------------------------------------------------------------
# Interfaz base
# ---------------------------------------------------------------------------
class BackendDiarization(ABC):
    """Interfaz que todos los backends de diarización deben implementar."""

    @abstractmethod
    def diarizar(self, audio_path: Union[str, Path]) -> list[dict]:
        """Diariza un fichero de audio.

        Devuelve
        --------
        Lista de turnos de palabra, cada uno con:
        - start (float): segundo de inicio.
        - end (float): segundo de fin.
        - speaker_id (str): identificador genérico (SPEAKER_00, SPEAKER_01, …).
        """

    @abstractmethod
    def memoria_gb(self) -> float:
        """VRAM aproximada en GB que ocupa el modelo cargado."""


# ---------------------------------------------------------------------------
# pyannote
# ---------------------------------------------------------------------------
class PyannoteBackend(BackendDiarization):
    """Diarización con pyannote/speaker-diarization-3.1.

    Requiere:
    - pip install pyannote.audio
    - Aceptar la licencia en HF: https://huggingface.co/pyannote/speaker-diarization-3.1
    - HF_TOKEN configurado en .env
    """

    def __init__(self, modelo_id: str = "pyannote/speaker-diarization-3.1"):
        import os

        from pyannote.audio import Pipeline

        self.modelo_id = modelo_id
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN no configurado. pyannote requiere un token de Hugging Face "
                "con acceso al modelo. Configúralo en .env"
            )

        self.pipeline = Pipeline.from_pretrained(modelo_id, use_auth_token=hf_token)

        device = dispositivo()
        if device != "cpu":
            import torch
            self.pipeline.to(torch.device(device))

    def diarizar(self, audio_path: Union[str, Path]) -> list[dict]:
        diarization = self.pipeline(str(audio_path))

        turnos = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turnos.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker_id": speaker,
            })

        return turnos

    def memoria_gb(self) -> float:
        return 1.5


# ---------------------------------------------------------------------------
# NeMo (alternativa)
# ---------------------------------------------------------------------------
class NemoBackend(BackendDiarization):
    """Diarización con NVIDIA NeMo MSDD.

    Requiere: pip install nemo_toolkit[asr]
    Más pesado de instalar pero sin licencia restringida.
    """

    def __init__(self, modelo_id: str = "nemo_msdd"):
        # TODO: implementar cuando se compare con pyannote
        self.modelo_id = modelo_id
        raise NotImplementedError(
            "Backend NeMo pendiente de implementación. "
            "Usa pyannote como primera opción."
        )

    def diarizar(self, audio_path: Union[str, Path]) -> list[dict]:
        raise NotImplementedError

    def memoria_gb(self) -> float:
        return 2.0


# ---------------------------------------------------------------------------
# Fábrica
# ---------------------------------------------------------------------------
BACKENDS = {
    "pyannote/speaker-diarization-3.1": PyannoteBackend,
    "nemo_msdd": NemoBackend,
}


def crear_backend_diarization(modelo_id: str | None) -> BackendDiarization:
    """Crea el backend adecuado según el identificador del modelo."""
    if modelo_id is None:
        modelo_id = "pyannote/speaker-diarization-3.1"

    cls = BACKENDS.get(modelo_id)
    if cls is None:
        raise ValueError(
            f"Modelo de diarización no soportado: {modelo_id}. "
            f"Opciones: {list(BACKENDS.keys())}"
        )
    return cls(modelo_id)
