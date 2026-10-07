"""Expresión facial a lo largo de la rueda de prensa → face.csv · Bloque 3.

Flujo por segundo de vídeo:
    fotograma → detector común (YuNet) → ¿primer plano? → ¿quién es? (SFace)
    → recorte de la cara → backend de expresión → fila de face.csv

Los segundos sin una cara en primer plano (plano general del panel) quedan con
face_detected=False. Uso desde la raíz del repo:

    python -m src.vision.face 2026-09-10 --backend hsemotion
"""
import argparse
import time
from pathlib import Path

import cv2
import pandas as pd

from src.config import DATA_DIR, RAW_DIR, event_dir
from src.vision.detect import Face, FaceDetector, FaceIdentifier
from src.vision.frames import iter_frames
from src.vision.models.base import FaceBackend
from src.vision.schema import ACTION_UNITS, EMOTIONS, FACE_COLUMNS, empty_face_row

# Una cara es primer plano si ocupa al menos esta fracción del fotograma. Medido en
# la rueda del 10/09: la presidenta en el atril ocupa ~1,7-2,8 %; en el plano
# general del panel cada cara ocupa ~0,15-0,2 %.
MIN_CLOSEUP_AREA = 0.008

REFERENCES_DIR = DATA_DIR / "references"  # una subcarpeta por persona con fotos de su cara


def classify_shot(face: Face | None, frame_shape: tuple) -> str:
    """'closeup' si la cara principal es grande en el fotograma, 'wide' si es pequeña o no hay."""
    if face is None:
        return "wide"
    return "closeup" if face.area / (frame_shape[0] * frame_shape[1]) >= MIN_CLOSEUP_AREA else "wide"


def load_identifier(detector: FaceDetector) -> FaceIdentifier:
    """Carga las caras de referencia de data/references/<persona>/*.jpg."""
    identifier = FaceIdentifier()
    for person_dir in sorted(p for p in REFERENCES_DIR.iterdir() if p.is_dir()):
        for img_path in sorted(person_dir.glob("*.jpg")):
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            faces = detector.detect(img)
            if faces:
                identifier.add_reference(person_dir.name, img, faces[0])
    return identifier


def analyze_video(video_path: str | Path, backend: FaceBackend, fps: float = 1.0) -> pd.DataFrame:
    """Recorre el vídeo y devuelve el DataFrame de face.csv."""
    detector = FaceDetector()
    identifier = load_identifier(detector)
    rows = []
    for t, frame in iter_frames(video_path, fps=fps):
        start, end = round(t, 3), round(t + 1.0 / fps, 3)
        faces = detector.detect(frame)
        face = faces[0] if faces else None
        shot = classify_shot(face, frame.shape)
        if shot != "closeup":
            rows.append(empty_face_row(start, end, shot_type=shot))
            continue
        person, _ = identifier.identify(frame, face)
        result = backend.analyze(detector.crop(frame, face))
        row = {
            "start": start, "end": end, "face_detected": True,
            "person": person or "otro", "shot_type": shot,
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


def main() -> None:
    from src.vision.models import get_face_backend

    parser = argparse.ArgumentParser(description="Genera face.csv de una rueda de prensa")
    parser.add_argument("event_date", help="AAAA-MM-DD (vídeo en data/raw/<fecha>.mp4)")
    parser.add_argument("--backend", default="hsemotion")
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args()

    t0 = time.perf_counter()
    df = analyze_video(RAW_DIR / f"{args.event_date}.mp4", get_face_backend(args.backend), fps=args.fps)
    out = write_face_csv(df, event_dir(args.event_date))
    secs = time.perf_counter() - t0
    print(f"{out}: {len(df)} segundos, {df['face_detected'].sum()} con cara en primer plano, {secs:.0f} s de proceso")


if __name__ == "__main__":
    main()
