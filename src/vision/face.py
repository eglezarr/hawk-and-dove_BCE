"""Expresión facial a lo largo de la rueda de prensa → face.csv · Bloque 3.

Flujo: fotogramas a 1 fps → backend de cara → una fila por segundo con valence,
arousal, probabilidades de emoción y Action Units. Los segundos sin una cara en
primer plano (plano general, periodistas) quedan con face_detected=False.
"""
from pathlib import Path

import pandas as pd

from src.vision.frames import iter_frames
from src.vision.models.base import FaceBackend
from src.vision.schema import ACTION_UNITS, EMOTIONS, FACE_COLUMNS, empty_face_row

# Una cara se considera primer plano si ocupa al menos esta fracción del fotograma
MIN_CLOSEUP_AREA = 0.02


def classify_shot(box: tuple[int, int, int, int] | None, frame_shape: tuple) -> str:
    """'closeup' si la cara es grande en el fotograma, 'wide' si es pequeña o no hay."""
    if box is None:
        return "wide"
    area = box[2] * box[3] / (frame_shape[0] * frame_shape[1])
    return "closeup" if area >= MIN_CLOSEUP_AREA else "wide"


def analyze_video(video_path: str | Path, backend: FaceBackend, fps: float = 1.0) -> pd.DataFrame:
    """Recorre el vídeo y devuelve el DataFrame de face.csv."""
    rows = []
    for t, frame in iter_frames(video_path, fps=fps):
        start, end = round(t, 3), round(t + 1.0 / fps, 3)
        result = backend.analyze(frame)
        shot = classify_shot(result["box"] if result else None, frame.shape)
        if result is None or shot != "closeup":
            rows.append(empty_face_row(start, end, shot_type=shot))
            continue
        row = {
            "start": start, "end": end, "face_detected": True,
            "person": None,  # identificación de la persona: pendiente (presidenta / vicepresidente)
            "shot_type": shot,
            "valence": result.get("valence"), "arousal": result.get("arousal"),
        }
        row.update({f"p_{e}": result["emotions"].get(e) for e in EMOTIONS})
        row.update({au: result.get("action_units", {}).get(au) for au in ACTION_UNITS})
        rows.append(row)
    return pd.DataFrame(rows, columns=FACE_COLUMNS)


def write_face_csv(df: pd.DataFrame, out_dir: str | Path) -> Path:
    """Escribe face.csv en la carpeta del evento."""
    out = Path(out_dir) / "face.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.3f")
    return out
