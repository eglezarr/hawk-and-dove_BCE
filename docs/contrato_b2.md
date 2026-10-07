# Contrato del bloque 2 · Texto y razonamiento

Ficheros y funciones que el bloque 2 entrega a los demás bloques. Cualquier cambio de formato se avisa antes de hacerlo.

**Estado:** `stance.csv` y `summary.json` son reales (versión solo texto) para las seis ruedas de 2026, de febrero a septiembre. Hasta que las frases se alineen con el vídeo (bloque 1), `start` y `end` están vacíos. `voice` y `face` valen `null` hasta la fase 5. `signals.csv` sigue siendo el ejemplo con valores ficticios.

## `data/events/<fecha>/stance.csv`

Una fila por frase del panel (presidenta y vicepresidente), en orden de aparición. No incluye las preguntas de los periodistas.

| Columna | Tipo | Descripción |
|---|---|---|
| `start`, `end` | float | Segundos desde el inicio del vídeo; vacíos hasta alinear la transcripción con el vídeo |
| `sentence_id`, `paragraph_id` | int | Posición de la frase y de su párrafo en la transcripción oficial; sirven para alinear y para enlazar las citas |
| `speaker` | str | `presidenta` / `vicepresidente` |
| `section` | str | `statement` (declaración) / `qa` (preguntas y respuestas) |
| `text` | str | Texto de la frase |
| `label` | str | `hawkish` / `neutral` / `dovish` |
| `p_hawkish`, `p_neutral`, `p_dovish` | float | Probabilidades del clasificador (suman 1; `p_neutral` incluye la clase "irrelevant") |
| `score` | float | `p_hawkish − p_dovish`, en [−1, 1]; positivo = hawkish |
| `relevant` | bool | `False` si la clase más probable es "irrelevant" (saludos, fórmulas); sirve para mostrarla atenuada |
| `weight` | float | Peso de la frase en las puntuaciones: su probabilidad de ser relevante (1 − p_irrelevant) |

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
- `stance`:
  - `score`, `score_statement`, `score_qa`: `score_statement` y `score_qa` son la media de la puntuación de las frases de cada parte, ponderada por `weight`; `score` pondera al 50 % ambas partes, para que no dependa de la duración del turno de preguntas.
  - `percentile_vs_history`, `percentile_statement`, `percentile_qa`: porcentaje de ruedas con una puntuación igual o inferior, calculado solo con las ruedas hasta la fecha del evento (`history_size` ruedas), sin información posterior.
  - `label`: tono relativo al histórico según el quintil de `percentile_vs_history`: `dovish` (≤ 20), `slightly dovish` (≤ 40), `neutral` (≤ 60), `slightly hawkish` (≤ 80) y `hawkish`.
  - `decision`: `subida` / `bajada` / `mantenimiento`. `previous_date` y `previous_percentile`: rueda anterior y su percentil.
- `voice`: `arousal_mean` y `arousal_z_vs_history` (desviación respecto a la media histórica de la presidenta). `null` hasta la fase 5.
- `face`: `valence_mean` y `label`. `null` hasta la fase 5.
- `key_moments`: lista de `{start, end, sentence_id, speaker, text, score, reason}`: las dos frases relevantes más hawkish y las dos más dovish que no repiten literalmente la rueda anterior, en orden de aparición.
- `briefing`: informe en inglés en tres párrafos. El primero lo genera el código (decisión de tipos, tono y percentiles); los otros dos son cinco frases del LLM (mensajes clave y riesgos), cada una con marcadores `[n]` que remiten a `citations[n-1]`, igual que en el chat.
- `briefing_tts`: el mismo texto sin marcadores y con las abreviaturas desarrolladas, para la voz sintética del bloque 1.
- `citations`: lista de `{date, source, start, end, snippet, url, sentence_id}`: el formato de las citas del chat más la primera frase del fragmento citado. Cuando las frases tengan segundos, la app puede mostrar cada cita como `[mm:ss]` y saltar a ese punto del vídeo.
- `disclaimer`: aviso legal fijo ("For information purposes only. This is not investment advice.").
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

`responder(pregunta: str, k: int = 5) -> dict` responde a una pregunta sobre el histórico de ruedas de prensa, citando los fragmentos de los que sale cada afirmación:

```python
{"answer": "The Governing Council decided to raise the three key ECB interest rates by 25 basis points [1] ...",
 "citations": [{"date": "2026-09-10",
                "source": "ECB press conference, 10 September 2026. Monetary policy statement",
                "start": None, "end": None,
                "snippet": "The Governing Council today decided to raise ...",
                "url": "https://www.ecb.europa.eu/press/press_conference/..."}]}
```

- Los marcadores `[n]` de `answer` remiten a `citations[n-1]`.
- `source` identifica el fragmento citado (fecha y quién habla); sirve de título de la cita.
- `start` y `end` (segundos) solo existen para las ruedas procesadas con vídeo; en el resto valen `None` y la cita enlaza a la transcripción oficial (`url`).
- La respuesta va en el idioma de la pregunta; las citas, en el inglés original.
- Pregunta vacía: `answer` = "Please type a question." y sin citas. Sin MLX (fuera de Mac) no se genera texto: `answer` lo indica y `citations` contiene los 3 fragmentos más relevantes.
- `preparar()` carga el índice y el LLM: la app debe llamarla al arrancar para que la primera pregunta no espere. El índice está en `data/history/rag_index/` y se genera con `benchmarks/b2_04_chat.ipynb`.

## Lo que el bloque 2 necesita de los demás

- **Bloque 1:** `transcript.csv` a nivel de frase (o segmento corto), con `speaker` y `section`; `voice.csv` con `start` y `end`.
- **Bloque 3:** `face.csv` con `start` y `end`; `projections.csv`.
