"""Formato de los ficheros que entrega el bloque 3 (face.csv y projections.csv).

Es la única fuente de verdad sobre columnas y rangos: la usan la lógica del bloque
para escribir y la app para validar lo que lee. El detalle está en docs/contrato_b3.md.
"""
import pandas as pd

# Emociones discretas que devuelven todos los backends de cara (orden fijo)
EMOTIONS = ["neutral", "happiness", "sadness", "surprise", "fear", "anger", "disgust", "contempt"]

# Action Units que guardamos: los más informativos en caras contenidas
ACTION_UNITS = {
    "au04_brow_lowerer": "Ceño fruncido (preocupación, concentración)",
    "au12_lip_corner_puller": "Comisuras hacia arriba (sonrisa)",
    "au24_lip_pressor": "Labios apretados (tensión, contención)",
}

FACE_COLUMNS = (
    ["start", "end", "face_detected", "person", "shot_type", "valence", "arousal"]
    + [f"p_{e}" for e in EMOTIONS]
    + list(ACTION_UNITS)
)

PROJECTION_COLUMNS = ["variable", "year", "value", "unit", "source"]

# Variables de las proyecciones macroeconómicas del staff
PROJECTION_VARIABLES = {
    "hicp": "Inflación (HICP)",
    "hicp_ex_energy_food": "Inflación subyacente (HICP sin energía ni alimentos)",
    "gdp": "Crecimiento del PIB real",
    "unemployment": "Tasa de paro",
}


def empty_face_row(start: float, end: float, shot_type: str = "other") -> dict:
    """Fila de face.csv para un segundo sin cara del panel en primer plano."""
    row = {c: None for c in FACE_COLUMNS}
    row.update(start=start, end=end, face_detected=False, shot_type=shot_type)
    return row


def validate_face(df: pd.DataFrame) -> list[str]:
    """Devuelve una lista de problemas de formato (vacía si face.csv es correcto)."""
    problems = [f"falta la columna {c}" for c in FACE_COLUMNS if c not in df.columns]
    if problems:
        return problems
    if (df["end"] < df["start"]).any():
        problems.append("hay filas con end < start")
    detected = df[df["face_detected"].astype(bool)]
    for col in ["valence", "arousal"]:
        if not detected[col].between(-1, 1).all():
            problems.append(f"{col} fuera de [-1, 1]")
    p = detected[[f"p_{e}" for e in EMOTIONS]].sum(axis=1)
    if not ((p - 1).abs() < 0.01).all():
        problems.append("las probabilidades de emoción no suman 1")
    return problems


def validate_projections(df: pd.DataFrame) -> list[str]:
    """Devuelve una lista de problemas de formato (vacía si projections.csv es correcto)."""
    problems = [f"falta la columna {c}" for c in PROJECTION_COLUMNS if c not in df.columns]
    if problems:
        return problems
    unknown = set(df["variable"]) - set(PROJECTION_VARIABLES)
    if unknown:
        problems.append(f"variables desconocidas: {sorted(unknown)}")
    return problems
