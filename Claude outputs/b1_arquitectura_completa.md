# Bloque 1 · Audio — Arquitectura completa

## Estructura de ficheros (16 archivos nuevos)

```
hawk-and-dove/
├── docs/
│   └── contrato_b1.md                          ← Contrato: formatos, API y dependencias
├── scripts/
│   └── download_ecb.py                         ← Descarga vídeos y transcripciones del BCE
├── src/audio/
│   ├── __init__.py                             ← API pública: voz_a_texto, texto_a_voz, preparar
│   ├── transcription.py                        ← ASR: transcribir(audio) → DataFrame
│   ├── diarization.py                          ← Hablantes: diarizar + asignar_hablantes
│   ├── voice_emotion.py                        ← Emoción: analizar_emocion → arousal/valence
│   ├── tts.py                                  ← TTS: texto_a_voz, generar_briefing
│   ├── pipeline.py                             ← Orquestador: procesar_evento (todo junto)
│   └── models/
│       ├── __init__.py
│       ├── asr_backends.py                     ← Whisper large-v3 / turbo / distil (implementado)
│       ├── diarization_backends.py             ← pyannote 3.1 (implementado) + NeMo (placeholder)
│       ├── emotion_backends.py                 ← wav2vec2 audEERING (implementado) + emotion2vec (placeholder)
│       └── tts_backends.py                     ← Kokoro (implementado) + Parler-TTS (placeholder)
├── benchmarks/
│   └── b1_01_asr.ipynb                         ← Benchmark ASR con plantilla del proyecto
└── data/events/2026-09-10/
    ├── transcript_example.csv                  ← Ejemplo del formato de salida
    └── voice_example.csv                       ← Ejemplo del formato de salida
```

## Flujo de datos del pipeline

```
Vídeo (.mp4)
    │
    ▼ ffmpeg (extraer_audio)
Audio (.wav, 16kHz, mono)
    │
    ├──▶ ASR (Whisper)              → segmentos con {start, end, text}
    │
    ├──▶ Diarización (pyannote)     → turnos con {start, end, speaker_id}
    │
    └──▶ Emoción de voz (wav2vec2)  → ventanas con {start, end, arousal, valence}
              │
              ▼
         voice.csv
              │
    ┌─────────┤
    │    Cruzar ASR + diarización
    │    (solapamiento temporal)
    │         │
    │         ▼
    │    transcript.csv
    │         │
    └─────────┤
              │
         summary.json (de B2)
              │
              ▼ TTS (Kokoro)
         briefing.mp3
```

## Patrón de backends (igual que B2)

Cada etapa tiene un fichero en `models/` con:
- **Clase base abstracta** (BackendASR, BackendDiarization, etc.) con la interfaz común
- **Una clase por candidato** que implementa la interfaz
- **Función fábrica** (`crear_backend_*`) que instancia el backend según config.yaml

Esto permite:
1. Cambiar de modelo cambiando solo `config.yaml`
2. Comparar candidatos en los notebooks de benchmark
3. Añadir nuevos modelos sin tocar la lógica

## Modelos implementados vs. placeholder

| Etapa | Implementado (listo para ejecutar) | Placeholder (estructura, falta rellenar) |
|---|---|---|
| ASR | Whisper large-v3, turbo, distil | — |
| Diarización | pyannote 3.1 | NeMo MSDD |
| Emoción | wav2vec2 audEERING | emotion2vec+ |
| TTS | Kokoro | Parler-TTS |

## Dependencias necesarias (a añadir a requirements)

**Para la app** (requirements.txt):
```
torch
torchaudio
transformers
kokoro>=0.8
soundfile
ffmpeg-python
```

**Para benchmarks** (requirements-bench.txt):
```
pyannote.audio
jiwer
yt-dlp
```

**Configuración previa necesaria:**
1. Instalar `ffmpeg` en el sistema
2. Configurar `HF_TOKEN` en `.env`
3. Aceptar licencia de pyannote en HF: https://huggingface.co/pyannote/speaker-diarization-3.1
4. Aceptar licencia de pyannote segmentation: https://huggingface.co/pyannote/segmentation-3.0

## Próximos pasos (por orden de prioridad)

1. **Verificar acceso HF** — ¿Tienes HF_TOKEN? ¿Aceptada la licencia de pyannote?
2. **Descargar datos** — `python scripts/download_ecb.py 2026-09-10`
3. **Probar ASR de extremo a extremo** — ejecutar Whisper turbo sobre el audio → transcript.csv
4. **Probar diarización** — pyannote sobre el audio → asignar hablantes
5. **Probar emoción** — wav2vec2 → voice.csv
6. **Probar TTS** — Kokoro → briefing.mp3
7. **Benchmark ASR** — comparar los 3 candidatos con WER
8. **Procesar las 6-10 ruedas** con los ganadores
9. **Integrar con la app** — conectar funciones de chat
