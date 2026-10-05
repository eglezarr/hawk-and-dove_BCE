"""Interfaz común de los backends de visión.

Cada candidato del benchmark (HSEmotion, py-feat, SigLIP...) es un backend que
cumple esta interfaz, así que la lógica y los notebooks los usan sin distinguirlos.
La detección de la cara es común (src/vision/detect.py): los backends de cara
reciben ya el recorte.
"""
from typing import Protocol

import numpy as np


class FaceBackend(Protocol):
    """Analiza la expresión de un recorte de cara."""

    name: str

    def analyze(self, face: np.ndarray) -> dict:
        """Recibe el recorte RGB de una cara y devuelve un dict con:

        - valence, arousal: float en [-1, 1] (None si el modelo no los estima)
        - emotions: {emoción: probabilidad} con las claves de schema.EMOTIONS, suman 1
        - action_units: {columna: intensidad en [0, 1]} con las claves de
          schema.ACTION_UNITS (dict vacío si el modelo no los estima)
        """
        ...


class ChartBackend(Protocol):
    """Lee los datos de un gráfico de proyecciones."""

    name: str

    def read_chart(self, image: np.ndarray, variable: str) -> list[dict]:
        """Recibe la imagen de un gráfico y devuelve [{"year": int, "value": float}, ...]."""
        ...
