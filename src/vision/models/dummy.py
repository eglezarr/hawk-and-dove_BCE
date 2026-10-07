"""Backend de prueba: valores neutros fijos, sin modelo.

Sirve para probar el flujo completo (fotogramas → face.csv → app) en cualquier
equipo y sin descargar modelos de emoción.
"""
import numpy as np

from src.vision.schema import ACTION_UNITS, EMOTIONS


class DummyFaceBackend:
    name = "dummy"

    def analyze(self, face: np.ndarray) -> dict:
        emotions = {e: 0.0 for e in EMOTIONS}
        emotions["neutral"] = 1.0
        return {
            "valence": 0.0,
            "arousal": 0.0,
            "emotions": emotions,
            "action_units": {au: 0.0 for au in ACTION_UNITS},
        }
