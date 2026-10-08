# Hawk & Dove · Análisis multimodal de las ruedas de prensa del BCE

Herramienta B2B para mesas de tesorería, renta fija y ALM que analiza las ruedas de prensa del BCE combinando **qué se dice** (texto), **cómo se dice** (voz) y **qué se muestra** (expresión facial). Proyecto del Taller B5-T4 del MIAX.

## Conferencias procesadas

El pipeline cubre 6 ruedas de prensa del BCE en 2026:

| Fecha | Vídeo | Audio (B1) | Texto (B2) | Visión (B3) |
|---|---|---|---|---|
| 2026-02-05 | ✅ | ✅ | ✅ | — |
| 2026-03-19 | ✅ | ✅ | ✅ | — |
| 2026-04-30 | ✅ | ✅ | ✅ | — |
| 2026-06-11 | ✅ | ✅ | ✅ | — |
| 2026-07-23 | ✅ | ✅ | ✅ | — |
| 2026-09-10 | ✅ | ✅ | ✅ | ✅ |

## Arquitectura de bloques

### Bloque 1 · Audio

Extrae información del canal de audio de cada conferencia:

1. **Transcripción (ASR)** — `openai/whisper-large-v3-turbo` vía HuggingFace Transformers. Genera segmentos con timestamps.
2. **Diarización** — `pyannote/speaker-diarization-3.1`. Identifica hablantes y los clasifica como presidenta, vicepresidente o periodista según su tiempo de intervención.
3. **Emoción de voz** — `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`. Calcula arousal, valence y dominance en ventanas deslizantes de 2 s con 50 % de solapamiento.
4. **Briefing TTS** — `edge-tts` (voz `en-GB-SoniaNeural`). Genera un resumen hablado a partir de `summary.json`.

El cruce transcripción–diarización asigna a cada frase un hablante y una sección (`statement` o `qa`). La emoción de voz se agrega a nivel de frase por solapamiento temporal ponderado.

### Bloque 2 · Texto y razonamiento

Análisis de postura (hawkish/dovish), combinación de señales multimodales, generación de informes y chatbot RAG. Modelo de postura: `mlx-community/Qwen3-8B-4bit`. Informe: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`. Embeddings: `BAAI/bge-m3`.

### Bloque 3 · Visión y producto

Detección y análisis de expresión facial, gráficos de proyecciones e interfaz Streamlit.

## Estructura

```
hawk-and-dove_BCE/
├── src/
│   ├── audio/              Bloque 1: transcripción, hablantes, tono de voz, TTS
│   │   └── models/         Backends: asr, diarization, emotion, tts
│   ├── text/               Bloque 2: postura, fusión, informe, RAG
│   │   └── models/         Backends: LLM, stance, retrieval
│   ├── vision/             Bloque 3: expresión facial, proyecciones
│   │   └── models/         Backends: pyfeat, siglip, hsemotion
│   ├── app/                Interfaz Streamlit (solo lee data/)
│   ├── config.py           Rutas comunes y carga de config.yaml
│   └── pipeline.py         Encadena los bloques para procesar una rueda de prensa
├── benchmarks/
│   ├── b1_00_ejecutar_pipeline.ipynb   Pipeline batch del Bloque 1 (6 conferencias)
│   ├── b1_01..04_*.ipynb               Benchmarks individuales de cada etapa B1
│   ├── b2_01..06_*.ipynb               Benchmarks del Bloque 2
│   └── b3_01_expresion_facial.ipynb    Benchmark del Bloque 3
├── scripts/
│   ├── download_videos.py  Descarga vídeos de YouTube con yt-dlp
│   ├── download_ecb.py     Descarga transcripciones oficiales del BCE
│   ├── videos.yaml         URLs de las 6 conferencias
│   └── fusion_eventos.py   Procesado por lotes de fusión de señales
├── data/
│   ├── events/<fecha>/     Resultados de cada rueda de prensa
│   │   ├── transcript_raw.csv     Transcripción ASR cruda
│   │   ├── transcript.csv         Transcripción con hablante y sección
│   │   ├── diarization_raw.csv    Segmentos de diarización
│   │   ├── voice.csv              Arousal, valence, dominance por ventana
│   │   ├── stance.csv             Postura por frase (B2)
│   │   ├── face.csv               Expresión facial (B3, solo 2026-09-10)
│   │   ├── signals.csv            Señales fusionadas (B1+B2+B3)
│   │   ├── summary.json           Resumen generado (B2)
│   │   └── briefing.mp3           Resumen hablado (TTS)
│   └── history/            Transcripciones oficiales e índice del histórico
├── docs/                   Hoja de ruta, diagramas y capturas
├── config.yaml             Evento de referencia y modelo elegido por etapa
├── requirements.txt        Dependencias de la app
└── requirements-bench.txt  Dependencias de los notebooks
```

## Puesta en marcha

Requisitos: Python ≥ 3.11, git y ffmpeg. Se recomienda clonar fuera de carpetas sincronizadas (iCloud, OneDrive, Dropbox).

> **Nota Python 3.13:** el proyecto funciona en 3.13 pero `torchaudio.load` requiere un monkey-patch con `soundfile` por incompatibilidad de `torchcodec`. El notebook `b1_00_ejecutar_pipeline.ipynb` ya lo incluye.

**Mac / Linux**

```bash
git clone https://github.com/eglezarr/hawk-and-dove_BCE.git
cd hawk-and-dove_BCE
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-bench.txt
cp .env.example .env
```

**Windows**

```bat
git clone https://github.com/eglezarr/hawk-and-dove_BCE.git
cd hawk-and-dove_BCE
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install -r requirements-bench.txt
copy .env.example .env
```

Después, rellenad las claves en `.env` (nunca se sube a git) y, en VS Code, seleccionad el intérprete `.venv`.

### Descarga de vídeos

```bash
python scripts/download_videos.py --all
```

Descarga los 6 vídeos a `data/raw/<fecha>.mp4` (≈ 720p, H.264). Requiere `yt-dlp` y `ffmpeg`.

### Ejecución del pipeline B1

Abrir `benchmarks/b1_00_ejecutar_pipeline.ipynb` y ejecutar las 5 celdas en orden. Procesa todas las conferencias que tengan vídeo descargado y aún no tengan `voice.csv`.

## Contrato entre bloques

- Cada bloque escribe sus resultados en CSV o JSON en `data/events/<fecha>/`.
- Todo resultado temporal lleva columnas `start` y `end` en **segundos desde el inicio del vídeo**.
- `signals.csv` fusiona las tres señales (voz, texto, cara) en una tabla unificada, usando `stance.csv` como columna vertebral.

## Modelos seleccionados

| Etapa | Modelo | Bloque |
|---|---|---|
| ASR | `openai/whisper-large-v3-turbo` | B1 |
| Diarización | `pyannote/speaker-diarization-3.1` | B1 |
| Emoción de voz | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | B1 |
| TTS | `edge-tts` (en-GB-SoniaNeural) | B1 |
| Postura | `mlx-community/Qwen3-8B-4bit` | B2 |
| Informe | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | B2 |
| Embeddings | `BAAI/bge-m3` | B2 |

## Forma de trabajo

- **Ramas:** `main` es la versión estable; cada bloque trabaja en la suya (`b1-audio`, `b2-texto`, `b3-vision`) e integra por Pull Request.
- **Benchmarks:** un notebook por etapa, copiando `benchmarks/00_plantilla_benchmark.ipynb`.
- **Hoja de ruta completa:** `docs/hoja_de_ruta.docx`.

## Equipo

| Bloque | Responsable |
|---|---|
| B1 · Audio |  |
| B2 · Texto y razonamiento |  |
| B3 · Visión y producto |  |
