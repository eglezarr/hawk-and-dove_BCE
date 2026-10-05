"""Detección e identificación de caras, común a todos los candidatos de expresión.

Todos los backends reciben el mismo recorte de cara, así que el benchmark compara
solo el reconocimiento de la expresión y no la detección. Usamos dos modelos
ligeros de OpenCV (opencv_zoo), que van en CPU y no añaden dependencias:

- YuNet (~230 KB): detecta las caras y sus puntos de referencia.
- SFace (~37 MB): saca un vector por cara para reconocer a la presidenta frente a
  otras personas (p. ej., el gobernador anfitrión que abre la rueda).

Los modelos se descargan la primera vez en data/raw/models/ (fuera de git).
"""
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import RAW_DIR

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
YUNET = ("face_detection_yunet_2023mar.onnx", ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx")
SFACE = ("face_recognition_sface_2021dec.onnx", ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx")

MIN_SCORE = 0.7         # confianza mínima de la detección
CROP_MARGIN = 0.25      # margen alrededor de la caja: los modelos de emoción esperan algo de contexto
SAME_PERSON = 0.363     # umbral de similitud coseno recomendado por OpenCV para SFace


def _model_path(name: str, url: str) -> Path:
    path = RAW_DIR / "models" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


@dataclass
class Face:
    box: tuple[int, int, int, int]  # x, y, ancho, alto en píxeles
    raw: np.ndarray                 # fila completa de YuNet (caja, 5 puntos y confianza), la usa SFace

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]


class FaceDetector:
    def __init__(self, min_score: float = MIN_SCORE):
        import cv2

        self._cv2 = cv2
        self._detector = cv2.FaceDetectorYN.create(str(_model_path(*YUNET)), "", (320, 320),
                                                   score_threshold=min_score)

    def detect(self, image: np.ndarray) -> list[Face]:
        """Caras de un fotograma RGB, de mayor a menor tamaño."""
        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))
        _, rows = self._detector.detect(self._cv2.cvtColor(image, self._cv2.COLOR_RGB2BGR))
        if rows is None:
            return []
        faces = [Face(tuple(int(v) for v in r[:4]), r) for r in rows]
        return sorted(faces, key=lambda f: f.area, reverse=True)

    @staticmethod
    def crop(image: np.ndarray, face: Face, margin: float = CROP_MARGIN) -> np.ndarray:
        """Recorte cuadrado de la cara con margen, ajustado a los bordes de la imagen."""
        x, y, bw, bh = face.box
        side = int(max(bw, bh) * (1 + 2 * margin))
        cx, cy = x + bw // 2, y + bh // 2
        h, w = image.shape[:2]
        x0, y0 = max(cx - side // 2, 0), max(cy - side // 2, 0)
        x1, y1 = min(x0 + side, w), min(y0 + side, h)
        return image[y0:y1, x0:x1]


class FaceIdentifier:
    """Reconoce a una persona comparando con caras de referencia suyas."""

    def __init__(self):
        import cv2

        self._cv2 = cv2
        self._model = cv2.FaceRecognizerSF.create(str(_model_path(*SFACE)), "")
        self._references: dict[str, list[np.ndarray]] = {}

    def embed(self, image: np.ndarray, face: Face) -> np.ndarray:
        bgr = self._cv2.cvtColor(image, self._cv2.COLOR_RGB2BGR)
        return self._model.feature(self._model.alignCrop(bgr, face.raw)).copy()

    def add_reference(self, person: str, image: np.ndarray, face: Face) -> None:
        self._references.setdefault(person, []).append(self.embed(image, face))

    def identify(self, image: np.ndarray, face: Face) -> tuple[str | None, float]:
        """(persona, similitud) de la referencia más parecida; persona None si ninguna supera el umbral."""
        emb = self.embed(image, face)
        best, best_sim = None, -1.0
        for person, refs in self._references.items():
            sim = max(self._model.match(emb, r, self._cv2.FaceRecognizerSF_FR_COSINE) for r in refs)
            if sim > best_sim:
                best, best_sim = person, sim
        return (best if best_sim >= SAME_PERSON else None), float(best_sim)
