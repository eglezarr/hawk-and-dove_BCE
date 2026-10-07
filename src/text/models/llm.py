"""Generación de texto con el LLM local (MLX, solo Mac) · Bloque 2: chat e informe."""
import re
from functools import lru_cache

from src.text.models.stance_backends import mlx_disponible


@lru_cache(maxsize=2)
def cargar(modelo_id: str):
    """Carga el modelo una sola vez por sesión (las llamadas siguientes lo reutilizan)."""
    from mlx_lm import load

    return load(modelo_id)


def liberar() -> None:
    """Descarga los modelos cargados, para comparar candidatos sin acumular memoria."""
    import gc

    import mlx.core as mx

    cargar.cache_clear()
    gc.collect()
    (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()


def generar(modelo_id: str, sistema: str, usuario: str, max_tokens: int = 500) -> str:
    """Respuesta del LLM a un mensaje de usuario con unas instrucciones de sistema.

    Decodificación voraz (determinista): la misma pregunta da siempre la misma respuesta.
    El razonamiento previo de Qwen3 se desactiva (enable_thinking=False).
    """
    if not mlx_disponible():
        raise RuntimeError("La generación local necesita MLX (Mac con Apple Silicon)")
    from mlx_lm import generate

    modelo, tokenizer = cargar(modelo_id)
    mensajes = [{"role": "system", "content": sistema}, {"role": "user", "content": usuario}]
    prompt = tokenizer.apply_chat_template(mensajes, add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False)
    tokens = tokenizer.encode(prompt, add_special_tokens=False)   # la plantilla ya incluye los tokens especiales
    texto = generate(modelo, tokenizer, prompt=tokens, max_tokens=max_tokens)
    return re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()
