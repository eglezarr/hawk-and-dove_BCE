"""Proyecciones macroeconómicas del staff → projections.csv · Bloque 3.

Los gráficos no salen del vídeo (en la rueda casi todo es la presidenta hablando):
se leen del documento de proyecciones que el BCE publica el mismo día. Ese mismo
documento trae las tablas con las cifras exactas, que sirven de referencia en el
benchmark de lectura de gráficos.
"""
from pathlib import Path

import pandas as pd

from src.vision.schema import PROJECTION_COLUMNS


def to_projections_df(readings: dict[str, list[dict]], source: str) -> pd.DataFrame:
    """Convierte {variable: [{"year", "value"}, ...]} en el DataFrame de projections.csv."""
    rows = [
        {"variable": var, "year": int(r["year"]), "value": float(r["value"]), "unit": "pct", "source": source}
        for var, values in readings.items()
        for r in values
    ]
    return pd.DataFrame(rows, columns=PROJECTION_COLUMNS)


def write_projections_csv(df: pd.DataFrame, out_dir: str | Path) -> Path:
    """Escribe projections.csv en la carpeta del evento."""
    out = Path(out_dir) / "projections.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
