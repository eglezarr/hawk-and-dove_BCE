"""Backends de ASR (transcripción automática) · Bloque 1.

Todos los candidatos exponen la misma interfaz:
    transcribir(audio_path) -> list[dict] con {start, end, text}
    memoria_gb()            -> VRAM aproximada que ocupa el modelo

Candidatos:
    - openai/whisper-large-v3         (mejor calidad, más lento)
    - openai/whisper-large-v3-turbo   (casi igual, mucho más rápido)
    - distil-whisper/distil-large-v3  (más ligero, buena calidad)
"""
from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union


# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------
def dispositivo() -> str:
    """GPU disponible: CUDA > MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def liberar_memoria() -> None:
    """Libera la memoria de la GPU entre modelos."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Interfaz base
# ---------------------------------------------------------------------------
class BackendASR(ABC):
    """Interfaz que todos los backends de ASR deben implementar."""

    @abstractmethod
    def transcribir(self, audio_path: Union[str, Path]) -> list[dict]:
        """Transcribe un fichero de audio.

        Devuelve
        --------
        Lista de segmentos, cada uno con:
        - start (float): segundo de inicio.
        - end (float): segundo de fin.
        - text (str): texto transcrito.
        """

    @abstractmethod
    def memoria_gb(self) -> float:
        """VRAM aproximada en GB que ocupa el modelo cargado."""


# ---------------------------------------------------------------------------
# Whisper (transformers pipeline)
# ---------------------------------------------------------------------------
class WhisperBackend(BackendASR):
    """Whisper vía Hugging Face transformers (large-v3, turbo o distil)."""

    def __init__(self, modelo_id: str):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.modelo_id = modelo_id
        self.device = dispositivo()
        self.torch_dtype = torch.float16 if self.device != "cpu" else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            modelo_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)

        processor = AutoProcessor.from_pretrained(modelo_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            return_timestamps=True,
        )

    def transcribir(self, audio_path: Union[str, Path]) -> list[dict]:
        resultado = self.pipe(
            str(audio_path),
            return_timestamps=True,
            generate_kwargs={"language": "english", "task": "transcribe"},
        )

        segmentos = []
        for chunk in resultado.get("chunks", []):
            ts = chunk.get("timestamp", (None, None))
            segmentos.append({
                "start": ts[0] if ts[0] is not None else 0.0,
                "end": ts[1] if ts[1] is not None else ts[0] or 0.0,
                "text": chunk["text"].strip(),
            })

        return segmentos

    def memoria_gb(self) -> float:
        modelos_gb = {
            "openai/whisper-large-v3": 3.1,
            "openai/whisper-large-v3-turbo": 1.6,
            "distil-whisper/distil-large-v3": 1.5,
        }
        return modelos_gb.get(self.modelo_id, 3.0)


# ---------------------------------------------------------------------------
# Fábrica
# ---------------------------------------------------------------------------
BACKENDS = {
    "openai/whisper-large-v3": WhisperBackend,
    "openai/whisper-large-v3-turbo": WhisperBackend,
    "distil-whisper/distil-large-v3": WhisperBackend,
}


def crear_backend_asr(modelo_id: str | None) -> BackendASR:
    """Crea el backend adecuado según el identificador del modelo.

    Si modelo_id es None, usa whisper-large-v3-turbo como valor por defecto.
    """
    if modelo_id is None:
        modelo_id = "openai/whisper-large-v3-turbo"

    cls = BACKENDS.get(modelo_id)
    if cls is None:
        raise ValueError(
            f"Modelo ASR no soportado: {modelo_id}. "
            f"Opciones: {list(BACKENDS.keys())}"
        )
    return cls(modelo_id)
