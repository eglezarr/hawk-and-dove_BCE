# Hawk & Dove · Análisis multimodal de las ruedas de prensa del BCE

Herramienta B2B para mesas de tesorería, renta fija y ALM que analiza las ruedas de prensa del BCE combinando **qué se dice** (texto) y **cómo se dice** (voz y expresión facial). Proyecto del Taller B5-T4 del MIAX.

> README en construcción: arquitectura, flujo de datos, capturas, benchmarks y viabilidad se completarán al final del proyecto.

## Estructura

```
hawk-and-dove_BCE/
├── src/
│   ├── audio/          Bloque 1: transcripción, hablantes, tono de voz, voz sintética
│   ├── text/           Bloque 2: postura, combinación de señales, informe, RAG
│   ├── vision/         Bloque 3: expresión facial, gráficos de proyecciones
│   ├── app/            Interfaz Streamlit (solo lee data/)
│   ├── config.py       Rutas comunes y carga de config.yaml
│   └── pipeline.py     Encadena los bloques para procesar una rueda de prensa
├── benchmarks/         Un notebook por etapa (a partir de 00_plantilla_benchmark.ipynb)
├── scripts/            Descargas y procesado por lotes
├── data/
│   ├── events/<fecha>/ Resultados de cada rueda de prensa procesada
│   └── history/        Transcripciones oficiales e índice del histórico
├── docs/               Hoja de ruta, diagramas y capturas
├── config.yaml         Evento de referencia y modelo elegido por etapa
├── requirements.txt    Dependencias de la app
└── requirements-bench.txt  Dependencias de los notebooks
```

Dentro de cada bloque, `models/` contiene la conexión con los modelos (un backend por candidato) y el resto de módulos, la lógica.

## Puesta en marcha

Requisitos: Python 3.11 y git. Clonad el repositorio fuera de carpetas sincronizadas (iCloud, OneDrive, Dropbox).

**Mac / Linux**

```bash
git clone https://github.com/eglezarr/hawk-and-dove_BCE.git
cd hawk-and-dove_BCE
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements-bench.txt
cp .env.example .env
```

**Windows**

```bat
git clone https://github.com/eglezarr/hawk-and-dove_BCE.git
cd hawk-and-dove_BCE
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install -r requirements-bench.txt
copy .env.example .env
```

Después, rellenad las claves en `.env` (nunca se sube a git) y, en VS Code, seleccionad el intérprete `.venv`.

## Forma de trabajo

- **Ramas:** `main` es la versión estable; cada bloque trabaja en la suya (`b1-audio`, `b2-texto`, `b3-vision`) e integra por Pull Request.
- **Contrato entre bloques:** cada bloque escribe sus resultados en CSV o JSON en `data/events/<fecha>/`, y todo resultado lleva `start` y `end` en segundos desde el inicio del vídeo.
- **Benchmarks:** un notebook por etapa, copiando `benchmarks/00_plantilla_benchmark.ipynb`.
- **Hoja de ruta completa:** `docs/hoja_de_ruta.docx`.

## Equipo

| Bloque | Responsable |
|---|---|
| B1 · Audio |  |
| B2 · Texto y razonamiento |  |
| B3 · Visión y producto |  |
