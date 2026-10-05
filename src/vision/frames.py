"""Extracción de fotogramas del vídeo de una rueda de prensa.

No usamos un modelo de vídeo: el vídeo se convierte en fotogramas a intervalos
fijos (por defecto, uno por segundo) y cada fotograma se analiza como una imagen.
"""
from collections.abc import Iterator

import numpy as np


def iter_frames(video_path: str, fps: float = 1.0) -> Iterator[tuple[float, np.ndarray]]:
    """Recorre el vídeo y devuelve (segundo, fotograma RGB) cada 1/fps segundos.

    Salta directamente a cada instante en lugar de decodificar todo el vídeo, así
    que una rueda de una hora a 1 fps son unas 3.600 lecturas.
    """
    import cv2  # import diferido: solo hace falta al procesar vídeo, no en la app

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se puede abrir el vídeo: {video_path}")
    try:
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration = n_frames / video_fps
        t = 0.0
        while t < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                break
            yield t, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t += 1.0 / fps
    finally:
        cap.release()
