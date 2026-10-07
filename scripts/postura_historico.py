"""Clasifica la postura de todas las frases del panel del histórico · Bloque 2, fase 3.

Aplica el modelo elegido en la fase 2 (models.stance de config.yaml) a las frases de la
presidenta y del vicepresidente de las 55 ruedas de prensa.

Uso, desde la raíz del repositorio (caffeinate evita que el Mac se duerma mientras corre):
    caffeinate -i python scripts/postura_historico.py

Cada rueda se guarda en data/raw/postura_historico/ en cuanto termina: si el proceso se
interrumpe, basta con volver a lanzarlo y continúa donde lo dejó. Al final une todas las
ruedas en data/history/stance_sentences.parquet.
"""
import re
import time

import pandas as pd

from src.config import HISTORY_DIR, RAW_DIR, load_config
from src.text.models.stance_backends import ClasificadorLLM
from src.text.stance import etiquetar

HABLANTES_PANEL = ["presidenta", "vicepresidente"]
COLUMNAS_FRASE = ["date", "section", "subsection", "qa_id", "role", "speaker", "paragraph_id",
                  "sentence_id", "text"]


def main() -> None:
    modelo_id = load_config()["models"]["stance"]
    if not modelo_id:
        raise SystemExit("config.yaml no indica el modelo de postura (models.stance)")

    corpus = pd.read_parquet(HISTORY_DIR / "transcripts.parquet")
    panel = corpus[corpus["speaker"].isin(HABLANTES_PANEL)]

    # Una carpeta por modelo, para no mezclar nunca resultados de modelos distintos
    partes_dir = RAW_DIR / "postura_historico" / re.sub(r"[^a-z0-9]+", "_", modelo_id.lower())
    partes_dir.mkdir(parents=True, exist_ok=True)
    pendientes = [f for f in sorted(panel["date"].unique()) if not (partes_dir / f"{f}.parquet").exists()]
    print(f"{panel['date'].nunique()} ruedas y {len(panel)} frases del panel | "
          f"pendientes: {len(pendientes)} ruedas | modelo: {modelo_id}", flush=True)

    if pendientes:
        clasificador = ClasificadorLLM(modelo_id)
        inicio = time.perf_counter()
        for i, fecha in enumerate(pendientes, start=1):
            frases = panel[panel["date"] == fecha][COLUMNAS_FRASE].reset_index(drop=True)
            t0 = time.perf_counter()
            probs = clasificador.predecir(frases["text"].tolist())
            pd.concat([frases, etiquetar(probs)], axis=1).to_parquet(partes_dir / f"{fecha}.parquet", index=False)
            # Estimación del tiempo restante a partir del ritmo medio hasta ahora
            restante = (time.perf_counter() - inicio) / i * (len(pendientes) - i)
            print(f"[{i}/{len(pendientes)}] {fecha}: {len(frases)} frases en {time.perf_counter() - t0:.0f} s | "
                  f"quedan unos {restante / 60:.0f} min", flush=True)

    todas = pd.concat([pd.read_parquet(r) for r in sorted(partes_dir.glob("*.parquet"))], ignore_index=True)
    todas["model"] = modelo_id
    salida = HISTORY_DIR / "stance_sentences.parquet"
    todas.to_parquet(salida, index=False)
    print(f"Guardado: {salida} ({len(todas)} frases de {todas['date'].nunique()} ruedas)")


if __name__ == "__main__":
    main()
