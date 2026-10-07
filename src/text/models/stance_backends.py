"""Conexión con los modelos de postura hawkish/dovish · Bloque 2.

Todos los candidatos exponen la misma interfaz:
    predecir(textos) -> DataFrame con p_hawkish, p_neutral, p_dovish y p_irrelevant
    memoria_gb()     -> memoria aproximada que ocupa el modelo
p_irrelevant vale NaN en los modelos que no contemplan esa clase.
"""
import gc
import importlib.util
import os

# Las operaciones que la GPU de Apple (MPS) no implementa se ejecutan en CPU en lugar de fallar
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd

COLUMNAS = ["p_hawkish", "p_neutral", "p_dovish", "p_irrelevant"]

# Instrucciones de los LLM: la misma guía del etiquetado humano, traducida al inglés
PROMPT_SISTEMA = """You are an expert in central bank communication. Classify the monetary policy stance conveyed by one sentence from a European Central Bank press conference, read in isolation.

Labels:
- Hawkish: points to tighter monetary policy, or to keeping it tight, to contain inflation. Signals: rate hikes or a bias to raise rates; shrinking the balance sheet (ending net purchases or reinvestments); high or rising inflation; upside risks to inflation; strong wages, employment or growth that push up prices; rising energy prices; a weaker euro.
- Dovish: points to looser monetary policy, or to keeping it accommodative. Signals: rate cuts or a bias to cut; asset purchases, TLTROs or other liquidity measures; low or falling inflation; downside risks to growth or inflation; weak activity or rising unemployment; falling energy prices; a stronger euro.
- Neutral: about the economy or monetary policy but with no clear direction: balanced or mixed signals, generic commitments (such as following a data-dependent approach), or descriptions with no implication for prices.
- Irrelevant: no information about monetary policy or the economy (greetings, thanks, logistics, procedure, anecdotes).

Rules: judge only what the sentence says. Economic data count by the pressure they put on prices. If signals point both ways, choose the dominant one, or Neutral if they are balanced. Keeping rates unchanged without further guidance is Neutral.

Answer with exactly one word: Hawkish, Dovish, Neutral or Irrelevant."""

# Hipótesis del clasificador zero-shot (NLI): una por etiqueta, redactadas con la misma guía
HIPOTESIS_NLI = {
    "hawkish": "The sentence signals tighter monetary policy, or keeping it tight, to contain inflation.",
    "dovish": "The sentence signals looser, more accommodative monetary policy.",
    "neutral": "The sentence is about the economy or monetary policy but signals no clear direction.",
    "irrelevant": "The sentence is unrelated to monetary policy and the economy.",
}


# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------
def dispositivo() -> str:
    """GPU disponible para PyTorch: MPS en Mac, CUDA con tarjeta NVIDIA o, si no hay, CPU."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def hay_acceso(modelo_id: str) -> bool:
    """True si el repositorio se puede descargar con el token actual (relevante en los modelos restringidos)."""
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(modelo_id, "config.json")
        return True
    except Exception:
        return False


def mlx_disponible() -> bool:
    """MLX solo existe en Mac con Apple Silicon."""
    return importlib.util.find_spec("mlx_lm") is not None


def liberar_memoria() -> None:
    """Libera la memoria de la GPU entre modelos para no acumularla al cargar el siguiente."""
    gc.collect()
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    if mlx_disponible():
        import mlx.core as mx

        (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()


def _tabla(probs: np.ndarray, etiquetas: list[str]) -> pd.DataFrame:
    """Matriz de probabilidades (frases x etiquetas) -> DataFrame con las cuatro columnas comunes."""
    df = pd.DataFrame(probs, columns=[f"p_{e}" for e in etiquetas])
    for columna in COLUMNAS:
        if columna not in df:
            df[columna] = np.nan
    return df[COLUMNAS]


def _memoria_pesos_gb(modelo) -> float:
    """Memoria que ocupan los pesos de un modelo de PyTorch."""
    return sum(p.numel() * p.element_size() for p in modelo.parameters()) / 1e9


# ---------------------------------------------------------------------------
# Candidatos
# ---------------------------------------------------------------------------
class ClasificadorSecuencia:
    """Modelo de clasificación de frases ya ajustado (los RoBERTa de gtfintechlab)."""

    def __init__(self, modelo_id: str, etiquetas: list[str], minusculas: bool = False, lote: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.etiquetas = etiquetas      # etiqueta de cada salida del modelo, en orden (LABEL_0, LABEL_1...)
        self.minusculas = minusculas    # los modelos entrenados con texto en minúsculas lo reciben igual
        self.lote = lote
        self.device = dispositivo()
        self.tokenizer = AutoTokenizer.from_pretrained(modelo_id)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(modelo_id).to(self.device).eval()
        if self.modelo.config.num_labels != len(etiquetas):
            raise ValueError(f"{modelo_id} tiene {self.modelo.config.num_labels} salidas y se han indicado "
                             f"{len(etiquetas)} etiquetas")

    def predecir(self, textos: list[str]) -> pd.DataFrame:
        torch = self._torch
        if self.minusculas:
            textos = [t.lower() for t in textos]
        partes = []
        with torch.no_grad():
            for i in range(0, len(textos), self.lote):
                entrada = self.tokenizer(textos[i:i + self.lote], padding=True, truncation=True,
                                         max_length=256, return_tensors="pt").to(self.device)
                logits = self.modelo(**entrada).logits.float()
                partes.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return _tabla(np.vstack(partes), self.etiquetas)

    def memoria_gb(self) -> float:
        return _memoria_pesos_gb(self.modelo)


class ClasificadorNLI:
    """Clasificador zero-shot por inferencia de lenguaje natural (NLI).

    Para cada frase evalúa una hipótesis por etiqueta y normaliza entre etiquetas
    (softmax) la puntuación de "implicación" de cada hipótesis.
    """

    def __init__(self, modelo_id: str, hipotesis: dict[str, str] = HIPOTESIS_NLI, lote: int = 8):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.etiquetas = list(hipotesis)
        self.hipotesis = list(hipotesis.values())
        self.lote = lote                # frases por lote (cada una genera tantos pares como hipótesis)
        self.device = dispositivo()
        self.tokenizer = AutoTokenizer.from_pretrained(modelo_id)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(modelo_id).to(self.device).eval()
        # Salida que corresponde a "entailment" (y no a "not_entailment")
        etiquetas_modelo = {int(i): e.lower() for i, e in self.modelo.config.id2label.items()}
        self.idx_implicacion = next(i for i, e in etiquetas_modelo.items()
                                    if "entail" in e and not e.startswith("not"))

    def predecir(self, textos: list[str]) -> pd.DataFrame:
        torch = self._torch
        k = len(self.hipotesis)
        partes = []
        with torch.no_grad():
            for i in range(0, len(textos), self.lote):
                bloque = textos[i:i + self.lote]
                premisas = [t for t in bloque for _ in range(k)]
                hipotesis = self.hipotesis * len(bloque)
                entrada = self.tokenizer(premisas, hipotesis, padding=True, truncation="only_first",
                                         max_length=256, return_tensors="pt").to(self.device)
                implicacion = self.modelo(**entrada).logits.float()[:, self.idx_implicacion]
                partes.append(torch.softmax(implicacion.view(len(bloque), k), dim=-1).cpu().numpy())
        return _tabla(np.vstack(partes), self.etiquetas)

    def memoria_gb(self) -> float:
        return _memoria_pesos_gb(self.modelo)


class ClasificadorEmbeddings:
    """Embeddings preentrenados (sin modificar) más una regresión logística entrenada con frases etiquetadas."""

    def __init__(self, modelo_id: str):
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(modelo_id, device=dispositivo())
        self.clf = None

    def _vectores(self, textos) -> np.ndarray:
        return self.encoder.encode(list(textos), batch_size=32, normalize_embeddings=True,
                                   show_progress_bar=False)

    def entrenar(self, textos_train, y_train, textos_val, y_val,
                 valores_c=(0.01, 0.1, 1.0, 10.0, 100.0)) -> pd.DataFrame:
        """Elige la regularización C con val y reentrena con train + val.

        El criterio es el F1 macro en tres clases (irrelevant cuenta como neutral), la
        métrica principal del benchmark. Los pesos de clase equilibrados evitan que la
        clase mayoritaria domine el ajuste.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score

        def tres_clases(etiquetas):
            return ["neutral" if e == "irrelevant" else e for e in etiquetas]

        x_train, x_val = self._vectores(textos_train), self._vectores(textos_val)
        y_train, y_val = list(y_train), list(y_val)
        resultados = []
        for c in valores_c:
            clf = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(x_train, y_train)
            f1 = f1_score(tres_clases(y_val), tres_clases(clf.predict(x_val)),
                          labels=["hawkish", "neutral", "dovish"], average="macro", zero_division=0)
            resultados.append({"C": c, "f1_macro_val": f1})
        resultados = pd.DataFrame(resultados)
        mejor_c = resultados.loc[resultados["f1_macro_val"].idxmax(), "C"]
        self.clf = LogisticRegression(C=mejor_c, class_weight="balanced", max_iter=5000).fit(
            np.vstack([x_train, x_val]), y_train + y_val)
        resultados["elegido"] = resultados["C"] == mejor_c
        return resultados

    def predecir(self, textos: list[str]) -> pd.DataFrame:
        probs = self.clf.predict_proba(self._vectores(textos))
        return _tabla(probs, list(self.clf.classes_))

    def memoria_gb(self) -> float:
        return _memoria_pesos_gb(self.encoder)


class ClasificadorLLM:
    """LLM generalista en zero-shot, ejecutado con MLX (solo Mac con Apple Silicon).

    No genera texto: lee la probabilidad que el modelo asigna a cada etiqueta como primer
    token de su respuesta y la normaliza entre las cuatro etiquetas. Cada etiqueta suma
    sus variantes con y sin mayúscula ("Hawkish" y "hawkish").

    Las instrucciones (el turno de sistema) son iguales para todas las frases: se procesan
    una sola vez y su caché se reutiliza, de modo que por cada frase solo se procesan sus
    propios tokens. Si el modelo no lo permite, se procesa el prompt completo cada vez.
    """

    ETIQUETAS = ["hawkish", "dovish", "neutral", "irrelevant"]

    def __init__(self, modelo_id: str, prompt_sistema: str = PROMPT_SISTEMA):
        import mlx.core as mx
        from mlx_lm import load

        self._mx = mx
        (getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory)()
        self.modelo, envoltorio = load(modelo_id)
        self.tokenizer = getattr(envoltorio, "_tokenizer", envoltorio)   # tokenizador de Hugging Face
        self.prompt_sistema = prompt_sistema

        # Primer token de cada etiqueta, con y sin mayúscula. Ningún token puede repetirse entre
        # etiquetas: si las variantes en minúscula chocan, se usan solo las capitalizadas.
        for variantes in ((str.capitalize, str.lower), (str.capitalize,)):
            self.ids_por_etiqueta = {
                e: sorted({self.tokenizer.encode(v(e), add_special_tokens=False)[0] for v in variantes})
                for e in self.ETIQUETAS
            }
            todos = [i for ids in self.ids_por_etiqueta.values() for i in ids]
            if len(todos) == len(set(todos)):
                break
        else:
            raise ValueError(f"Dos etiquetas comparten primer token en {modelo_id}: {self.ids_por_etiqueta}")
        self.masa = []   # probabilidad total que el modelo deja en las etiquetas (diagnóstico)

        # Prefijo común (turno de sistema). Solo se reutiliza si tokenizar prefijo y resto por
        # separado da exactamente los mismos tokens que tokenizar el prompt completo.
        sistema = self._texto([{"role": "system", "content": self.prompt_sistema}], generacion=False)
        ejemplo = self._texto(self._mensajes("Inflation rose."), generacion=True)
        self.prefijo = None
        if ejemplo.startswith(sistema):
            prefijo = self.tokenizer.encode(sistema, add_special_tokens=False)
            resto = self.tokenizer.encode(ejemplo[len(sistema):], add_special_tokens=False)
            if prefijo + resto == self.tokenizer.encode(ejemplo, add_special_tokens=False):
                self.prefijo, self._sistema = prefijo, sistema

    def _mensajes(self, texto: str) -> list[dict]:
        return [{"role": "system", "content": self.prompt_sistema},
                {"role": "user", "content": f'Sentence: "{texto}"'}]

    def _texto(self, mensajes: list[dict], generacion: bool) -> str:
        # enable_thinking=False desactiva el razonamiento previo de Qwen3 (otros modelos lo ignoran)
        return self.tokenizer.apply_chat_template(mensajes, add_generation_prompt=generacion, tokenize=False,
                                                  enable_thinking=False)

    def _probabilidades(self, logits) -> np.ndarray:
        """Probabilidad de cada etiqueta, normalizada entre las cuatro, a partir de los logits finales."""
        mx = self._mx
        probs = np.array(mx.softmax(logits.astype(mx.float32), axis=-1))
        p = np.array([probs[ids].sum() for ids in self.ids_por_etiqueta.values()])
        self.masa.append(float(p.sum()))
        return p / p.sum()

    def predecir(self, textos: list[str]) -> pd.DataFrame:
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache

        mx = self._mx
        cache = make_prompt_cache(self.modelo) if self.prefijo else None
        if cache is not None and not can_trim_prompt_cache(cache):
            cache = None
        if cache is not None:
            self.modelo(mx.array(self.prefijo)[None], cache=cache)   # instrucciones, una sola vez
            mx.eval([c.state for c in cache])

        filas = []
        for texto in textos:
            prompt = self._texto(self._mensajes(texto), generacion=True)
            if cache is not None and prompt.startswith(self._sistema):
                resto = self.tokenizer.encode(prompt[len(self._sistema):], add_special_tokens=False)
                logits = self.modelo(mx.array(resto)[None], cache=cache)[0, -1]
                filas.append(self._probabilidades(logits))
                trim_prompt_cache(cache, len(resto))   # vuelve al estado de solo instrucciones
            else:
                tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
                filas.append(self._probabilidades(self.modelo(mx.array(tokens)[None])[0, -1]))
        return _tabla(np.vstack(filas), self.ETIQUETAS)

    def memoria_gb(self) -> float:
        mx = self._mx
        return (getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory)() / 1e9
