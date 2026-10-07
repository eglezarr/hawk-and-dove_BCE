"""Extracción de fotogramas del vídeo de una rueda de prensa.

No usamos un modelo de vídeo: el vídeo se convierte en fotogramas a intervalos
fijos (por defecto, uno por segundo) y cada fotograma se analiza como una imagen.
"""
from collections.abc import Iterator

import numpy as np


def iter_frames(video_path: str, fps: float = 1.0) -> Iterator[tuple[float, np.ndarray]]:
    """Recorre el vídeo y devuelve (segundo, fotograma RGB) cada 1/fps segundos.

    Lee el vídeo de forma secuencial: grab() avanza sin decodificar la imagen y solo
    se decodifica el fotograma que toca. Es mucho más rápido que saltar con
    CAP_PROP_POS_MSEC, que obliga a volver al fotograma clave anterior en cada salto.
    """
    import cv2  # import diferido: solo hace falta al procesar vídeo, no en la app

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se puede abrir el vídeo: {video_path}")
    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = 1.0 / fps
        next_t, i = 0.0, 0
        while cap.grab():
            t = i / video_fps
            if t + 1e-6 >= next_t:
                ok, frame = cap.retrieve()
                if ok:
                    yield round(next_t, 3), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                next_t += step
            i += 1
    finally:
        cap.release()
