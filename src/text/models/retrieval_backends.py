"""Conexión con los modelos de recuperación (búsqueda semántica y léxica) · Bloque 2.

Todos los recuperadores exponen la misma interfaz:
    indexar(textos)          -> prepara la búsqueda sobre los fragmentos
    buscar(consultas, k)     -> (índices, puntuaciones) de los k fragmentos más parecidos
    memoria_gb()             -> memoria aproximada que ocupa el modelo
"""
import re

import numpy as np

from src.text.models.stance_backends import _memoria_pesos_gb, dispositivo

# Algunos modelos de embeddings esperan una instrucción delante de la consulta (no de los documentos)
PREFIJOS_CONSULTA = {
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


def _top_k(puntuaciones: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Índices y puntuaciones de los k mayores valores de cada fila, en orden descendente."""
    k = min(k, puntuaciones.shape[1])
    candidatos = np.argpartition(-puntuaciones, k - 1, axis=1)[:, :k]
    orden = np.take_along_axis(-puntuaciones, candidatos, axis=1).argsort(axis=1)
    indices = np.take_along_axis(candidatos, orden, axis=1)
    return indices, np.take_along_axis(puntuaciones, indices, axis=1)


class RecuperadorDenso:
    """Búsqueda semántica: similitud del coseno entre embeddings de consulta y de fragmento."""

    def __init__(self, modelo_id: str):
        from sentence_transformers import SentenceTransformer

        self.modelo_id = modelo_id
        self.encoder = SentenceTransformer(modelo_id, device=dispositivo())
        # Tope de 512 tokens: los fragmentos no llegan, y los modelos de contexto largo (bge-m3,
        # 8192) dispararían la memoria si alguna consulta fuera muy larga
        self.encoder.max_seq_length = min(self.encoder.max_seq_length or 512, 512)
        self.prefijo = PREFIJOS_CONSULTA.get(modelo_id, "")
        self.matriz = None   # embeddings de los fragmentos (normalizados)

    def codificar(self, textos: list[str], consulta: bool = False) -> np.ndarray:
        textos = [self.prefijo + t for t in textos] if consulta else list(textos)
        return self.encoder.encode(textos, batch_size=32, normalize_embeddings=True,
                                   show_progress_bar=False).astype(np.float32)

    def indexar(self, textos: list[str] | None = None, matriz: np.ndarray | None = None) -> None:
        """Calcula los embeddings de los fragmentos, o reutiliza unos ya calculados (matriz)."""
        self.matriz = matriz.astype(np.float32) if matriz is not None else self.codificar(textos)

    def buscar(self, consultas: list[str], k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        return _top_k(self.codificar(consultas, consulta=True) @ self.matriz.T, k)

    def memoria_gb(self) -> float:
        return _memoria_pesos_gb(self.encoder)


class RecuperadorBM25:
    """Búsqueda léxica clásica (BM25): coincidencia de palabras ponderada por su rareza."""

    modelo_id = "bm25"

    @staticmethod
    def _tokens(texto: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", texto.lower())

    def indexar(self, textos: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([self._tokens(t) for t in textos])

    def buscar(self, consultas: list[str], k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        puntuaciones = np.vstack([self.bm25.get_scores(self._tokens(c)) for c in consultas])
        return _top_k(puntuaciones, k)

    def memoria_gb(self) -> float:
        return 0.0


def crear_recuperador(modelo_id: str):
    """Recuperador a partir de su identificador ("bm25" o un modelo de sentence-transformers)."""
    return RecuperadorBM25() if modelo_id == "bm25" else RecuperadorDenso(modelo_id)
