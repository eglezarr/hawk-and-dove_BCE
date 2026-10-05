"""Funciones del chat en vivo, acordadas entre bloques (sección 6 de la hoja de ruta).

- responder(pregunta) -> {"answer", "citations"}   · bloque 2 (src/text/rag.py)
- voz_a_texto(audio: bytes) -> str                  · bloque 1 (expuesta en src/audio)
- texto_a_voz(texto: str) -> bytes                  · bloque 1 (expuesta en src/audio)

Si un bloque aún no ha publicado su función, la correspondiente vale None y la app
desactiva esa parte en lugar de fallar.
"""
try:
    from src.text.rag import responder
except ImportError:
    responder = None

try:
    from src.audio import voz_a_texto
except ImportError:
    voz_a_texto = None

try:
    from src.audio import texto_a_voz
except ImportError:
    texto_a_voz = None
