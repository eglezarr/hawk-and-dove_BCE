"""Evaluación de la recuperación del chat (RAG) · Bloque 2, fase 4.

Tarea de referencia: para cada pregunta de un periodista del histórico, el recuperador debe
encontrar, entre los fragmentos de las 55 ruedas, alguno de los fragmentos con los que el
panel respondió a esa misma pregunta.
"""
import time

import numpy as np
import pandas as pd

from src.text.models.retrieval_backends import crear_recuperador
from src.text.models.stance_backends import liberar_memoria
from src.text.rag import MIN_PALABRAS

K_MAX = 10
METRICAS = ["recall@1", "recall@5", "recall@10", "mrr@10"]

# La misma pregunta en inglés y en español, para medir si un modelo entiende preguntas en español
PREGUNTAS_EN_ES = pd.DataFrame([
    ("What did the ECB say about the impact of US tariffs on euro area growth?",
     "¿Qué dijo el BCE sobre el impacto de los aranceles de Estados Unidos en el crecimiento de la zona del euro?"),
    ("Why did the ECB start cutting interest rates in June 2024?",
     "¿Por qué empezó el BCE a bajar los tipos de interés en junio de 2024?"),
    ("How does the ECB view wage growth and its effect on services inflation?",
     "¿Cómo ve el BCE el crecimiento de los salarios y su efecto en la inflación de los servicios?"),
    ("What is the Transmission Protection Instrument and when would it be used?",
     "¿Qué es el Instrumento para la Protección de la Transmisión y cuándo se utilizaría?"),
    ("How is the ECB responding to the energy price shock caused by the conflict in the Middle East?",
     "¿Cómo está respondiendo el BCE al shock de precios de la energía causado por el conflicto en Oriente Medio?"),
    ("What did the ECB say about the pandemic emergency purchase programme?",
     "¿Qué dijo el BCE sobre el programa de compras de emergencia frente a la pandemia?"),
    ("Is the ECB concerned about the appreciation of the euro exchange rate?",
     "¿Le preocupa al BCE la apreciación del tipo de cambio del euro?"),
    ("What does the ECB say about the risks of climate change for monetary policy?",
     "¿Qué dice el BCE sobre los riesgos del cambio climático para la política monetaria?"),
], columns=["en", "es"])


# ---------------------------------------------------------------------------
# Datos de referencia
# ---------------------------------------------------------------------------
def _clave(datos: pd.DataFrame) -> pd.Series:
    """Fecha y turno de pregunta ("2024-06-06#3"); los fragmentos sin turno llevan -1."""
    return datos["date"] + "#" + datos["qa_id"].fillna(-1).astype(int).astype(str)


def consultas_evaluacion(corpus: pd.DataFrame, fragmentos: pd.DataFrame) -> pd.DataFrame:
    """Una consulta por turno de pregunta: todo lo que pregunta el periodista en ese turno.

    Se conservan los turnos con al menos MIN_PALABRAS palabras y con respuesta en el índice
    (algún fragmento con la misma fecha y qa_id).
    """
    preguntas = corpus[corpus["role"] == "question"].sort_values(["date", "sentence_id"])
    consultas = preguntas.groupby(["date", "qa_id"], as_index=False).agg(text=("text", " ".join))
    con_respuesta = consultas[_clave(consultas).isin(set(_clave(fragmentos)))]
    return con_respuesta[con_respuesta["text"].str.split().str.len() >= MIN_PALABRAS].reset_index(drop=True)


def rangos(indices: np.ndarray, fragmentos: pd.DataFrame, consultas: pd.DataFrame) -> np.ndarray:
    """Posición (1 = primero) del primer fragmento de la respuesta correcta; 0 si no está entre los recuperados."""
    acierto = _clave(fragmentos).to_numpy()[indices] == _clave(consultas).to_numpy()[:, None]
    return np.where(acierto.any(axis=1), acierto.argmax(axis=1) + 1, 0)


# ---------------------------------------------------------------------------
# Evaluación de los candidatos
# ---------------------------------------------------------------------------
def _latencia_ms(recuperador, consultas: list[str], n: int = 20) -> float:
    """Mediana del tiempo de una búsqueda con una sola pregunta, como en el chat."""
    recuperador.buscar(consultas[:1], 5)   # calentamiento
    tiempos = []
    for consulta in consultas[:n]:
        t0 = time.perf_counter()
        recuperador.buscar([consulta], 5)
        tiempos.append(time.perf_counter() - t0)
    return 1000 * float(np.median(tiempos))


def coincidencia_idiomas(recuperador, pares: pd.DataFrame = PREGUNTAS_EN_ES, k: int = 5) -> float:
    """Proporción media de fragmentos comunes entre los k recuperados con la pregunta en inglés y en español."""
    en, _ = recuperador.buscar(pares["en"].tolist(), k)
    es, _ = recuperador.buscar(pares["es"].tolist(), k)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(en, es)]))


def evaluar_candidatos(candidatos: list[str], fragmentos: pd.DataFrame, consultas: pd.DataFrame,
                       k: int = K_MAX) -> tuple[dict, pd.DataFrame, dict]:
    """Indexa y busca con cada candidato, uno detrás de otro, liberando la memoria entre modelos.

    Returns:
        posiciones: {modelo: rango de la respuesta correcta en cada consulta (0 = no recuperada)}
        costes: tiempos de carga e indexado, latencia por pregunta, memoria y coincidencia EN-ES
        matrices: {modelo: embeddings de los fragmentos}, para guardar el índice del ganador
                  sin recalcularlo (None en BM25)
    """
    textos = fragmentos["indexed_text"].tolist()
    posiciones, costes, matrices = {}, [], {}
    for modelo_id in candidatos:
        t0 = time.perf_counter()
        recuperador = crear_recuperador(modelo_id)
        carga = time.perf_counter() - t0

        t0 = time.perf_counter()
        recuperador.indexar(textos)
        indexado = time.perf_counter() - t0

        indices, _ = recuperador.buscar(consultas["text"].tolist(), k)
        posiciones[modelo_id] = rangos(indices, fragmentos, consultas)
        matrices[modelo_id] = getattr(recuperador, "matriz", None)
        costes.append({"modelo": modelo_id, "carga_s": carga, "indexado_s": indexado,
                       "latencia_ms": _latencia_ms(recuperador, consultas["text"].tolist()),
                       "memoria_gb": recuperador.memoria_gb(),
                       "coincidencia_en_es": coincidencia_idiomas(recuperador)})
        print(f"{modelo_id}: recall@5 = {np.mean((posiciones[modelo_id] >= 1) & (posiciones[modelo_id] <= 5)):.3f}",
              flush=True)
        del recuperador
        liberar_memoria()
    return posiciones, pd.DataFrame(costes), matrices


# ---------------------------------------------------------------------------
# Métricas e incertidumbre
# ---------------------------------------------------------------------------
def valores_por_consulta(posiciones: np.ndarray) -> pd.DataFrame:
    """Aciertos (0/1) en el top 1, 5 y 10 y rango recíproco de cada consulta."""
    r = np.asarray(posiciones)
    return pd.DataFrame({"recall@1": (r == 1).astype(float),
                         "recall@5": ((r >= 1) & (r <= 5)).astype(float),
                         "recall@10": (r >= 1).astype(float),
                         "mrr@10": np.where(r > 0, 1.0 / np.maximum(r, 1), 0.0)})


def _pesos_bootstrap(fechas: pd.Series, n: int, semilla: int) -> np.ndarray:
    """Remuestreo por conglomerados: se remuestrean ruedas enteras, no preguntas sueltas.

    Las preguntas de una misma rueda comparten contexto y no son independientes. Cada fila
    indica cuántas veces entra cada consulta en una remuestra (todas las de una rueda a la vez).
    """
    rng = np.random.default_rng(semilla)
    ruedas, grupo = np.unique(np.asarray(fechas), return_inverse=True)
    elegidas = rng.integers(0, len(ruedas), size=(n, len(ruedas)))
    veces = np.stack([np.bincount(fila, minlength=len(ruedas)) for fila in elegidas])
    return veces[:, grupo].astype(float)


def _intervalo(pesos: np.ndarray, valores: np.ndarray) -> tuple[float, float]:
    """Intervalo al 95 % de la media de los valores en las remuestras."""
    medias = pesos @ valores / pesos.sum(axis=1)
    return tuple(np.percentile(medias, [2.5, 97.5]))


def tabla_resultados(posiciones: dict, fechas: pd.Series, n: int = 2000, semilla: int = 0) -> pd.DataFrame:
    """Métricas de cada candidato con el intervalo al 95 % del recall@5, ordenadas por recall@5."""
    pesos = _pesos_bootstrap(fechas, n, semilla)
    filas = []
    for modelo, r in posiciones.items():
        valores = valores_por_consulta(r)
        inferior, superior = _intervalo(pesos, valores["recall@5"].to_numpy())
        filas.append({"modelo": modelo, **valores.mean().to_dict(), "r5_ic95_inf": inferior, "r5_ic95_sup": superior})
    columnas = ["modelo", "recall@5", "r5_ic95_inf", "r5_ic95_sup", "recall@1", "recall@10", "mrr@10"]
    return pd.DataFrame(filas)[columnas].sort_values("recall@5", ascending=False).reset_index(drop=True)


def comparar_con_mejor(posiciones: dict, fechas: pd.Series, mejor: str, metrica: str = "recall@5",
                       n: int = 2000, semilla: int = 0) -> pd.DataFrame:
    """Diferencia de cada candidato con el mejor (candidato − mejor), con bootstrap pareado por ruedas.

    Pareado: ambos se evalúan sobre las mismas remuestras, de modo que la dificultad de las
    preguntas, común a los dos, no infla la incertidumbre.
    """
    pesos = _pesos_bootstrap(fechas, n, semilla)
    base = valores_por_consulta(posiciones[mejor])[metrica].to_numpy()
    filas = []
    for modelo, r in posiciones.items():
        if modelo == mejor:
            continue
        diferencia = valores_por_consulta(r)[metrica].to_numpy() - base
        inferior, superior = _intervalo(pesos, diferencia)
        filas.append({"modelo": modelo, "diferencia": diferencia.mean(), "ic95_inf": inferior, "ic95_sup": superior,
                      "empate": inferior <= 0 <= superior})
    return pd.DataFrame(filas).sort_values("diferencia", ascending=False).reset_index(drop=True)


def elegir_ganador(tabla: pd.DataFrame, comparacion: pd.DataFrame, multilingue: str, margen: float) -> tuple[str, str]:
    """Regla de decisión: el mayor recall@5, salvo que el multilingüe no sea peor que él en más de `margen`.

    "No peor" exige que el límite inferior del intervalo al 95 % de la diferencia supere −margen.
    """
    mejor = tabla.iloc[0]["modelo"]
    if mejor == multilingue or multilingue not in set(comparacion["modelo"]):
        return mejor, f"{mejor}: mayor recall@5"
    limite = comparacion.set_index("modelo").loc[multilingue, "ic95_inf"]
    if limite > -margen:
        return multilingue, (f"{multilingue}: no es peor que {mejor} en más de {margen * 100:.0f} puntos "
                             f"(límite inferior de la diferencia: {limite:+.3f})")
    return mejor, (f"{mejor}: mayor recall@5; {multilingue} puede ser peor en más de {margen * 100:.0f} puntos "
                   f"(límite inferior de la diferencia: {limite:+.3f})")
