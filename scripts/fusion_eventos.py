"""Combina texto, voz y cara de las ruedas procesadas · Bloque 2, fase 5.

Uso, desde la raíz del repositorio:
    python scripts/fusion_eventos.py               # todas las ruedas de data/events/ con transcript.csv
    python scripts/fusion_eventos.py 2026-09-10    # ruedas concretas

Para cada rueda:
1. Alinea la transcripción oficial con transcript.csv (bloque 1): tiempo de cada frase.
2. Rellena start y end en stance.csv.
3. Calcula la voz (voice.csv, bloque 1) y la cara (face.csv, bloque 3) en la ventana de cada
   frase y escribe signals.csv. voice.csv y face.csv son opcionales: sin ellos, sus columnas
   quedan vacías.
4. Añade a summary.json los tiempos de las citas y de los momentos clave, los momentos clave
   de voz y cara y los bloques voice y face.
Se puede volver a ejecutar cuando llegue un fichero nuevo: el resultado no se duplica.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import EVENTS_DIR, HISTORY_DIR, load_config  # noqa: E402
from src.text import fusion  # noqa: E402
from src.text.alignment import alinear, calidad  # noqa: E402


def procesar(fecha: str, corpus: pd.DataFrame) -> None:
    carpeta = EVENTS_DIR / fecha
    transcripcion = fusion.leer_opcional(carpeta, "transcript.csv")
    if transcripcion is None:
        print(f"[{fecha}] falta transcript.csv (bloque 1): sin él no se puede situar cada frase en el vídeo")
        return

    tiempos = alinear(corpus[corpus["date"] == fecha], transcripcion)
    postura = (pd.read_csv(carpeta / "stance.csv").drop(columns=["start", "end"])
                 .merge(tiempos[["sentence_id", "start", "end"]], on="sentence_id", how="left"))
    postura = postura[["start", "end"] + [c for c in postura.columns if c not in ("start", "end")]]

    cara = fusion.leer_opcional(carpeta, "face.csv")
    voz = fusion.leer_opcional(carpeta, "voice.csv")
    senales = fusion.senales_por_frase(postura, cara, voz)
    resumen = json.loads((carpeta / "summary.json").read_text(encoding="utf-8"))
    resumen = fusion.actualizar_resumen(resumen, senales, cara, fecha)
    modelos = load_config()["models"]
    resumen["generated_with"] = {**resumen.get("generated_with", {}),
                                 **{clave: modelos[clave] for clave in ("asr", "voice_emotion", "face") if modelos.get(clave)}}
    fusion.guardar(carpeta, postura, senales, resumen)

    panel = calidad(tiempos[tiempos["sentence_id"].isin(postura["sentence_id"])])
    print(f"[{fecha}] frases del panel con al menos la mitad de sus palabras en el audio: {panel['frases_cobertura_50']:.0%} "
          f"| orden temporal correcto: {panel['orden_correcto']} "
          f"| con voz: {senales['voice_arousal'].notna().mean():.0%} | con cara: {senales['face_valence'].notna().mean():.0%} "
          f"| momentos clave: {len(resumen['key_moments'])}")


def main() -> None:
    corpus = pd.read_parquet(HISTORY_DIR / "transcripts.parquet")
    fechas = sys.argv[1:] or sorted(p.parent.name for p in EVENTS_DIR.glob("*/transcript.csv"))
    if not fechas:
        print("Ninguna rueda tiene todavía transcript.csv (bloque 1)")
    for fecha in fechas:
        procesar(fecha, corpus)


if __name__ == "__main__":
    main()
