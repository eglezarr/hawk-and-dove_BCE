# Contrato del bloque 2 · Texto y razonamiento

Ficheros y funciones que el bloque 2 entrega a los demás bloques. Los ficheros de ejemplo tienen **valores ficticios**; lo que vale es el formato. Cualquier cambio de formato se avisa antes de hacerlo.

## `data/events/<fecha>/stance.csv`

Una fila por frase del panel (presidenta y vicepresidente). No incluye las preguntas de los periodistas.

| Columna | Tipo | Descripción |
|---|---|---|
| `start`, `end` | float | Segundos desde el inicio del vídeo |
| `speaker` | str | `presidenta` / `vicepresidente` |
| `section` | str | `statement` (declaración) / `qa` (preguntas y respuestas) |
| `text` | str | Texto de la frase |
| `label` | str | `hawkish` / `neutral` / `dovish` |
| `p_hawkish`, `p_neutral`, `p_dovish` | float | Probabilidades del clasificador (suman 1) |
| `score` | float | `p_hawkish − p_dovish`, en [−1, 1]; positivo = hawkish |
| `relevant` | bool | `False` si la clase más probable es "irrelevant" (saludos, fórmulas); sirve para mostrarla atenuada. En las puntuaciones, cada frase pesa su probabilidad de ser relevante |

## `data/events/<fecha>/signals.csv`

Las mismas filas que `stance.csv`, con las señales de voz y cara agregadas en la ventana de cada frase. Es la tabla que usa la app para la transcripción y la línea temporal.

| Columna | Tipo | Descripción |
|---|---|---|
| `start` … `score` | | Igual que en `stance.csv` |
| `voice_arousal`, `voice_valence` | float | Tono de voz en la ventana de la frase (bloque 1), media ponderada por solapamiento; vacío si no hay dato |
| `face_valence`, `face_arousal` | float | Expresión facial en la ventana de la frase (bloque 3); vacío si no se detecta cara |
| `key_moment` | bool | `True` si la frase es uno de los momentos clave del informe |

## `data/events/<fecha>/summary.json`

Resumen del evento e informe:

- `event_date`: fecha de la rueda de prensa.
- `stance`: `score`, `label`, `score_statement`, `score_qa` y `percentile_vs_history`. `score_statement` y `score_qa` son la media de la puntuación de las frases de cada parte, ponderada por la probabilidad de que cada frase sea relevante; `score` pondera al 50 % ambas partes, para que no dependa de la duración del turno de preguntas.
- `voice`: `arousal_mean` y `arousal_z_vs_history` (desviación respecto a la media histórica de la presidenta).
- `face`: `valence_mean` y `label`.
- `key_moments`: lista de `{start, end, reason}`.
- `briefing`: informe en inglés con marcadores de cita `[mm:ss]`, para mostrarlo en la app.
- `briefing_tts`: el mismo texto sin marcadores, para la voz sintética del bloque 1.
- `citations`: lista de `{label, start, end}`, una por marcador.
- `generated_with`: modelos usados, para trazabilidad.

## `data/history/stance_by_conference.csv`

Una fila por rueda de prensa de la era Lagarde, calculada solo con texto. Alimenta la pestaña Histórico. La postura frase a frase de todo el histórico está en `data/history/stance_sentences.parquet`.

| Columna | Tipo | Descripción |
|---|---|---|
| `date` | str | `AAAA-MM-DD` |
| `score`, `score_statement`, `score_qa` | float | Igual que en `summary.json` |
| `n_sentences` | int | Frases del panel clasificadas |
| `decision` | str | Decisión de tipos de la rueda: `subida` / `bajada` / `mantenimiento` |
| `percentile` | int | Porcentaje de ruedas del histórico con una puntuación igual o inferior |
| `url` | str | Enlace a la transcripción oficial |

## Función `responder` (`src/text/rag.py`)

`responder(pregunta: str, k: int = 5) -> dict` devuelve `{"answer": str, "citations": [{"date", "start", "end", "snippet", "url"}]}`. `start` y `end` solo existen para las ruedas procesadas con vídeo; en el resto valen `None` y la cita enlaza a la transcripción oficial. La versión actual devuelve un ejemplo fijo.

## Lo que el bloque 2 necesita de los demás

- **Bloque 1:** `transcript.csv` a nivel de frase (o segmento corto), con `speaker` y `section`; `voice.csv` con `start` y `end`.
- **Bloque 3:** `face.csv` con `start` y `end`; `projections.csv`.
