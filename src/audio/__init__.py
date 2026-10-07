"""Bloque 1 · Audio: transcripción, hablantes, tono de voz y voz sintética.

Funciones públicas para el chat en vivo:
    voz_a_texto(audio)  -> str          (transcribe audio del usuario)
    texto_a_voz(texto)  -> bytes        (genera audio hablado)
    preparar()          -> None         (precarga modelos en memoria)

Pipeline completo (procesa una rueda de prensa):
    procesar_evento(event_date, video_path)  -> escribe transcript.csv, voice.csv, briefing.mp3
"""

from src.audio.transcription import voz_a_texto
from src.audio.tts import texto_a_voz

# Modelo ASR y TTS cargados en memoria (se inicializan con preparar())
_modelo_asr = None
_modelo_tts = None


def preparar() -> None:
    """Precarga los modelos ASR y TTS para que el chat en vivo no espere.

    La app debe llamar a esta función al arrancar.
    """
    global _modelo_asr, _modelo_tts

    from src.audio.transcription import cargar_modelo_asr
    from src.audio.tts import cargar_modelo_tts
    from src.config import load_config

    cfg = load_config()

    _modelo_asr = cargar_modelo_asr(cfg["models"].get("asr"))
    _modelo_tts = cargar_modelo_tts(cfg["models"].get("tts"))
