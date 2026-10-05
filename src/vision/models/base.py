"""Interfaz común de los backends de visión.

Cada candidato del benchmark (py-feat, HSEmotion, SigLIP...) es un backend que
cumple esta interfaz, así que la lógica y los notebooks los usan sin distinguirlos.
"""
from typing import Protocol

import numpy as np


class FaceBackend(Protocol):
    """Analiza la expresión facial de la cara principal de un fotograma."""

    name: str

    def analyze(self, image: np.ndarray) -> dict | None:
        """Recibe un fotograma RGB y devuelve None si no hay cara, o un dict con:

        - valence, arousal: float en [-1, 1] (None si el modelo no los estima)
        - emotions: {emoción: probabilidad} con las claves de schema.EMOTIONS, suman 1
        - action_units: {columna: intensidad en [0, 1]} con las claves de
          schema.ACTION_UNITS (dict vacío si el modelo no los estima)
        - box: (x, y, ancho, alto) de la cara, en píxeles
        """
        ...


class ChartBackend(Protocol):
    """Lee los datos de un gráfico de proyecciones."""

    name: str

    def read_chart(self, image: np.ndarray, variable: str) -> list[dict]:
        """Recibe la imagen de un gráfico y devuelve [{"year": int, "value": float}, ...]."""
        ...
