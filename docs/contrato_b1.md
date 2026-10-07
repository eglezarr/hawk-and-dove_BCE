# Contrato del bloque 1 · Audio

Ficheros y funciones que el bloque 1 entrega a los demás bloques. Cualquier cambio de formato se avisa antes de hacerlo.

**Estado:** pendiente — los ficheros de ejemplo están abajo; los reales se generan al procesar cada rueda de prensa.

## `data/events/<fecha>/transcript.csv`

Una fila por frase o segmento corto del panel (presidenta y vicepresidente), en orden cronológico. Incluye las preguntas de los periodistas solo si el modelo las detecta; se marcan con `speaker=periodista`.

| Columna | Tipo | Descripción |
|---|---|---|
| `start` | float | Segundos desde el inicio del vídeo |
| `end` | float | Segundos desde el inicio del vídeo |
| `speaker` | str | `presidenta` / `vicepresidente` / `periodista` (asignado por diarización) |
| `section` | str | `statement` / `qa` (se infiere por la posición del primer turno de preguntas) |
| `text` | str | Texto transcrito por el modelo ASR |

Notas:
- `start` y `end` vienen del modelo ASR con alineamiento a nivel de palabra (Whisper con `return_timestamps="word"`).
- `speaker` se cruza con la diarización: cada segmento ASR hereda el hablante mayoritario en su ventana temporal.
- `section` se asigna automáticamente: todo lo anterior a la primera pregunta de un periodista es `statement`; lo posterior es `qa`.

## `data/events/<fecha>/voice.csv`

Análisis del tono de voz, una fila por ventana temporal (resolución ≈ 1-2 segundos).

| Columna | Tipo | Descripción |
|---|---|---|
| `start` | float | Segundos desde el inicio del vídeo |
| `end` | float | Segundos desde el inicio del vídeo |
| `arousal` | float | Activación emocional [0, 1] |
| `valence` | float | Valencia emocional [0, 1]: 0 = negativo, 1 = positivo |
| `dominance` | float | Dominancia [0, 1] (disponible solo en algunos modelos; vacío si no) |

Notas:
- La granularidad temporal depende del modelo de emoción; se garantiza que `start` y `end` permiten el cruce con `transcript.csv` por solapamiento.
- Para la agregación a nivel de frase (en `signals.csv`), B2 calcula la media ponderada por solapamiento temporal.

## `data/events/<fecha>/briefing.mp3`

Audio MP3 generado a partir del campo `briefing_tts` de `summary.json` (bloque 2), usando el modelo TTS ganador. Duración aproximada: 30-60 segundos.

## Funciones para el chat en vivo

### `voz_a_texto(audio) → str`

Ubicación: `src/audio/transcription.py`

Transcribe un fragmento de audio del usuario (grabado desde el micrófono del navegador) a texto. Usa el modelo ASR ganador. Entrada: `bytes` o ruta a fichero de audio. Salida: texto transcrito.

```python
from src.audio.transcription import voz_a_texto

texto = voz_a_texto(audio_bytes)
# "What was the main message on inflation?"
```

### `texto_a_voz(texto) → bytes`

Ubicación: `src/audio/tts.py`

Genera audio hablado a partir de texto. Usa el modelo TTS ganador. Entrada: `str`. Salida: `bytes` en formato WAV.

```python
from src.audio.tts import texto_a_voz

audio_wav = texto_a_voz("The Governing Council decided to keep rates unchanged.")
```

### `preparar() → None`

Ubicación: `src/audio/__init__.py`

Precarga los modelos ASR y TTS en memoria. La app debe llamarla al arrancar para que la primera interacción no tenga latencia de carga.

## Lo que el bloque 1 necesita de los demás

- **Bloque 2:** `summary.json` con el campo `briefing_tts` (texto del informe sin marcadores, para generar `briefing.mp3`).
- **Bloque 3:** nada directamente; B3 consume `transcript.csv` y `voice.csv`.
- **Datos externos:** vídeo de la rueda de prensa del BCE (se descarga con `scripts/download_ecb.py`) y transcripción oficial (para el benchmark de ASR).

## Modelos candidatos por etapa

| Etapa | Candidatos | Métrica |
|---|---|---|
| ASR | Whisper large-v3, Whisper large-v3-turbo, distil-whisper-large-v3 | WER vs. transcripción oficial |
| Diarización | pyannote 3.1, NeMo MSDD | DER (Diarization Error Rate) |
| Emoción de voz | wav2vec2-large (audEERING), emotion2vec+ large | Concordancia con anotación manual |
| TTS | Kokoro, Parler-TTS | Evaluación subjetiva (MOS) + latencia |
