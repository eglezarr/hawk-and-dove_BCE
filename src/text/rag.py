"""Preguntas sobre el histórico de ruedas de prensa (RAG) · Bloque 2.

Contrato con la app (bloque 3): `responder` recibe una pregunta y devuelve la
respuesta con sus citas. Esta versión es provisional: devuelve un ejemplo fijo
para que la app pueda integrar el chat desde ya. La implementación real
(embeddings + LLM) llega en la fase 4 y mantendrá esta misma firma.
"""


def responder(pregunta: str, k: int = 5) -> dict:
    """Responde a una pregunta sobre las ruedas de prensa del histórico.

    Args:
        pregunta: pregunta del usuario en texto (si llega por voz, la transcribe antes el bloque 1).
        k: número de fragmentos del histórico que se recuperan para responder.

    Returns:
        {"answer": str,
         "citations": [{"date": "AAAA-MM-DD", "start": float | None, "end": float | None,
                        "snippet": str, "url": str}]}
        `start` y `end` (segundos) solo existen para las ruedas procesadas con vídeo;
        en el resto valen None y la cita enlaza a la transcripción oficial (`url`).
    """
    # Respuesta fija de ejemplo (valores ficticios)
    return {
        "answer": "[Example answer] The tone on inflation has become more hawkish over the last press conferences.",
        "citations": [
            {"date": "2026-07-23", "start": 465.0, "end": 478.0,
             "snippet": "[example] Answer sentence on inflation.", "url": ""},
            {"date": "2026-09-10", "start": 1421.0, "end": 1432.0,
             "snippet": "[example] Answer sentence on inflation risks.", "url": ""},
        ],
    }
