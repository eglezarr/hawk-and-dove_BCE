"""Preguntas sobre el histórico de ruedas de prensa (RAG) · Bloque 2, fase 4.

Contrato con la app (bloque 3): `responder(pregunta, k)` devuelve la respuesta y sus citas.
Flujo: la pregunta se compara con los fragmentos del histórico (declaración y respuestas del
panel), los k más parecidos se pasan al LLM y este redacta la respuesta citándolos.
"""
import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import EVENTS_DIR, HISTORY_DIR, load_config
from src.text.models import llm
from src.text.models.retrieval_backends import crear_recuperador
from src.text.models.stance_backends import mlx_disponible

INDICE_DIR = HISTORY_DIR / "rag_index"
MIN_PALABRAS = 8     # los fragmentos más cortos son fórmulas ("Thank you very much, Vice-President.")
MAX_PALABRAS = 150   # los párrafos más largos se dividen en bloques de frases consecutivas
MESES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
         "October", "November", "December"]
HABLANTES = {"presidenta": "the President", "vicepresidente": "the Vice-President", "otro": "a guest governor"}

PROMPT_SISTEMA = """You answer questions about European Central Bank (ECB) monetary policy press conferences for treasury and fixed-income professionals, using only the numbered excerpts provided.

Rules:
- Use only the information in the excerpts. If they do not answer the question, say so plainly and do not cite any excerpt.
- Support each statement with the number of the excerpt it comes from, in brackets: [1], or [2][3].
- Mention the date of the press conference when it matters.
- Report what the ECB said. Do not add forecasts, opinions or investment recommendations.
- Answer in the same language as the question, in no more than 150 words."""


def fecha_en_ingles(fecha: str) -> str:
    """'2026-09-10' -> '10 September 2026'."""
    anio, mes, dia = fecha.split("-")
    return f"{int(dia)} {MESES[int(mes) - 1]} {anio}"


# ---------------------------------------------------------------------------
# Fragmentos e índice
# ---------------------------------------------------------------------------
def _bloques(frases: pd.DataFrame) -> pd.Series:
    """Número de bloque de cada frase dentro de su párrafo.

    Un párrafo de más de MAX_PALABRAS palabras se reparte en ceil(palabras / MAX_PALABRAS)
    bloques de frases consecutivas de tamaño parecido: cada frase va al bloque en el que cae
    su punto medio. Así no quedan restos diminutos al final del párrafo.
    """
    claves = [frases["date"], frases["paragraph_id"]]
    palabras = frases["text"].str.split().str.len()
    total = palabras.groupby(claves).transform("sum")
    tamano = total / np.ceil(total / MAX_PALABRAS)
    punto_medio = palabras.groupby(claves).cumsum() - palabras / 2
    return (punto_medio // tamano).astype(int)


def construir_fragmentos(corpus: pd.DataFrame) -> pd.DataFrame:
    """Fragmentos del índice: párrafos de la declaración y de las intervenciones del panel.

    - Se excluyen las preguntas de los periodistas (no son palabras del BCE y son las consultas
      de la evaluación) y el moderador.
    - Los párrafos de más de MAX_PALABRAS palabras se dividen en bloques: así caben en el
      contexto de todos los modelos (el menor, el de MiniLM, admite unas 190 palabras) y las
      citas son más precisas. Los fragmentos de menos de MIN_PALABRAS palabras se descartan.
    - Cada fragmento lleva un encabezado con la fecha y quién habla. Se indexa junto al texto,
      porque ayuda en las preguntas con fecha, y se muestra al LLM para que pueda citarla.
    - sentence_id es la primera frase del fragmento: permite situar la cita en el vídeo en las
      ruedas cuyas frases tengan segundos.
    """
    panel = corpus[corpus["role"].isin(["statement", "answer", "remarks"])
                   & corpus["speaker"].isin(list(HABLANTES))].sort_values(["date", "sentence_id"]).copy()
    panel["block"] = _bloques(panel)
    fragmentos = (panel.groupby(["date", "paragraph_id", "block"], as_index=False)
                       .agg(section=("section", "first"), subsection=("subsection", "first"),
                            qa_id=("qa_id", "first"), role=("role", "first"), speaker=("speaker", "first"),
                            sentence_id=("sentence_id", "first"), text=("text", " ".join), url=("url", "first")))
    fragmentos = fragmentos[fragmentos["text"].str.split().str.len() >= MIN_PALABRAS].reset_index(drop=True)

    def encabezado(fila) -> str:
        inicio = f"ECB press conference, {fecha_en_ingles(fila.date)}"
        if fila.role == "statement":
            apartado = f" ({fila.subsection})" if isinstance(fila.subsection, str) else ""
            return f"{inicio}. Monetary policy statement{apartado}"
        tipo = "Answer" if fila.role == "answer" else "Remarks"
        return f"{inicio}. {tipo} by {HABLANTES[fila.speaker]}"

    fragmentos["header"] = [encabezado(f) for f in fragmentos.itertuples()]
    fragmentos["indexed_text"] = fragmentos["header"] + ": " + fragmentos["text"]
    identificador = (fragmentos["date"] + "_p" + fragmentos["paragraph_id"].astype(str).str.zfill(3)
                     + "_" + fragmentos["block"].astype(str))
    fragmentos.insert(0, "fragment_id", identificador)
    return fragmentos


def guardar_indice(fragmentos: pd.DataFrame, modelo_id: str, matriz: np.ndarray | None = None,
                   carpeta: Path = INDICE_DIR) -> None:
    """Guarda los fragmentos y sus embeddings, para que el chat funcione sin recalcular nada.

    matriz: embeddings de los fragmentos (None con BM25, que se reconstruye al cargar).
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    fragmentos.to_parquet(carpeta / "fragments.parquet", index=False)
    (carpeta / "embeddings.npy").unlink(missing_ok=True)   # no dejar embeddings de un modelo anterior
    if matriz is not None:
        np.save(carpeta / "embeddings.npy", matriz.astype(np.float16))   # float16: mitad de tamaño
    meta = {"model": modelo_id, "n_fragments": len(fragmentos), "created": date.today().isoformat()}
    (carpeta / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cargar_indice.cache_clear()   # la siguiente pregunta usa el índice nuevo


@lru_cache(maxsize=1)
def cargar_indice(carpeta: Path = INDICE_DIR):
    """Fragmentos y recuperador listo para buscar (se carga una vez por sesión)."""
    if not (carpeta / "meta.json").exists():
        raise FileNotFoundError(f"No existe el índice del chat en {carpeta}: "
                                "se crea al ejecutar benchmarks/b2_04_chat.ipynb")
    meta = json.loads((carpeta / "meta.json").read_text(encoding="utf-8"))
    fragmentos = pd.read_parquet(carpeta / "fragments.parquet")
    recuperador = crear_recuperador(meta["model"])
    if meta["model"] == "bm25":
        recuperador.indexar(fragmentos["indexed_text"].tolist())
    else:
        recuperador.indexar(matriz=np.load(carpeta / "embeddings.npy"))
    return fragmentos, recuperador


# ---------------------------------------------------------------------------
# Respuesta
# ---------------------------------------------------------------------------
def _modelo_llm() -> str:
    """LLM del chat: models.llm de config.yaml o, si no se ha indicado, el modelo de postura."""
    modelos = load_config()["models"]
    modelo = modelos.get("llm") or modelos.get("stance")
    if not modelo:
        raise RuntimeError("config.yaml no indica ningún LLM (models.llm)")
    return modelo


def _mensaje(pregunta: str, recuperados: pd.DataFrame) -> str:
    extractos = "\n\n".join(f"[{i}] {f.header}: {f.text}" for i, f in enumerate(recuperados.itertuples(), start=1))
    return f"Excerpts:\n\n{extractos}\n\nQuestion: {pregunta}"


@lru_cache(maxsize=None)
def _tiempos_de_frases(fecha: str) -> dict:
    """{sentence_id: (start, end)} de una rueda procesada con vídeo; vacío si no tiene tiempos."""
    ruta = EVENTS_DIR / fecha / "stance.csv"
    if not ruta.exists():
        return {}
    frases = pd.read_csv(ruta).dropna(subset=["start", "end"])
    return {int(f.sentence_id): (round(float(f.start), 1), round(float(f.end), 1)) for f in frases.itertuples()}


def cita(fragmento) -> dict:
    """Cita en el formato del contrato.

    start y end (segundos del vídeo) solo existen en las ruedas cuyas frases se han situado en
    el vídeo (fase 5); en el resto valen None y la cita enlaza a la transcripción oficial.
    """
    texto = fragmento.text if len(fragmento.text) <= 400 else fragmento.text[:400].rsplit(" ", 1)[0] + "…"
    inicio, fin = None, None
    frase = getattr(fragmento, "sentence_id", None)   # los índices anteriores a la fase 6 no lo tienen
    if frase is not None and pd.notna(frase):
        inicio, fin = _tiempos_de_frases(fragmento.date).get(int(frase), (None, None))
    return {"date": fragmento.date, "source": fragmento.header, "start": inicio, "end": fin,
            "snippet": texto, "url": fragmento.url}


def procesar_citas(respuesta: str, n_extractos: int) -> tuple[str, list[int]]:
    """Renumera las citas por orden de aparición: el marcador [n] del texto remite a citations[n-1].

    Admite listas ("[1, 3]") y elimina las referencias a extractos que no existen. Devuelve el
    texto renumerado y la posición (desde 0) de los extractos citados, en orden de aparición.
    """
    patron = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
    usados = []
    for grupo in patron.findall(respuesta):
        for n in re.findall(r"\d+", grupo):
            i = int(n) - 1
            if 0 <= i < n_extractos and i not in usados:
                usados.append(i)
    nuevo = {i: j + 1 for j, i in enumerate(usados)}

    def sustituir(coincidencia) -> str:
        numeros = [int(n) - 1 for n in re.findall(r"\d+", coincidencia.group(1))]
        return "".join(f"[{nuevo[i]}]" for i in numeros if i in nuevo)

    texto = patron.sub(sustituir, respuesta)
    # Espacios que quedan al eliminar citas no válidas
    texto = re.sub(r"[ \t]{2,}", " ", re.sub(r"[ \t]+([.,;:])", r"\1", texto))
    return texto, usados


def preparar() -> None:
    """Carga el índice y el LLM. La app lo llama al arrancar para que la primera pregunta no espere."""
    cargar_indice()
    if mlx_disponible():
        llm.cargar(_modelo_llm())


def responder(pregunta: str, k: int = 5) -> dict:
    """Responde a una pregunta sobre las ruedas de prensa del histórico.

    Args:
        pregunta: pregunta del usuario en texto (si llega por voz, la transcribe antes el bloque 1).
        k: número de fragmentos del histórico que se recuperan y se pasan al LLM.

    Returns:
        {"answer": str,
         "citations": [{"date": "AAAA-MM-DD", "source": str, "start": float | None,
                        "end": float | None, "snippet": str, "url": str}]}
        Los marcadores [n] de answer remiten a citations[n-1]. source identifica el fragmento
        ("ECB press conference, 6 June 2024. Answer by the President"). start y end (segundos)
        solo existen en las ruedas procesadas con vídeo; en el resto valen None y la cita
        enlaza a la transcripción oficial (url). Sin MLX (fuera de Mac) no se genera
        respuesta: se devuelven los fragmentos más relevantes.
    """
    if not pregunta or not pregunta.strip():
        return {"answer": "Please type a question.", "citations": []}

    fragmentos, recuperador = cargar_indice()
    indices, _ = recuperador.buscar([pregunta], k)
    recuperados = fragmentos.iloc[indices[0]].reset_index(drop=True)

    if not mlx_disponible():
        n = min(3, len(recuperados))
        marcas = "".join(f"[{i}]" for i in range(1, n + 1))
        return {"answer": f"Answer generation is not available on this machine. Most relevant excerpts: {marcas}",
                "citations": [cita(recuperados.iloc[i]) for i in range(n)]}

    respuesta = llm.generar(_modelo_llm(), PROMPT_SISTEMA, _mensaje(pregunta, recuperados))
    texto, usados = procesar_citas(respuesta, len(recuperados))
    return {"answer": texto, "citations": [cita(recuperados.iloc[i]) for i in usados]}
