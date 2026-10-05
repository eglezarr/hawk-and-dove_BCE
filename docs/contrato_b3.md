# Contrato del bloque 3 · Visión y producto

Ficheros que el bloque 3 entrega a los demás bloques. Los ficheros de ejemplo tienen **valores ficticios**; lo que vale es el formato. Cualquier cambio de formato se avisa antes de hacerlo. Las columnas y rangos están definidos en código en `src/vision/schema.py`.

## `data/events/<fecha>/face.csv`

Una fila por segundo de vídeo. El vídeo se convierte en fotogramas (1 por segundo) y cada fotograma se analiza como una imagen; solo se analiza la expresión cuando hay una cara del panel en primer plano.

| Columna | Tipo | Descripción |
|---|---|---|
| `start`, `end` | float | Segundos desde el inicio del vídeo (`end = start + 1`) |
| `face_detected` | bool | `True` si hay una cara del panel en primer plano; si es `False`, el resto de columnas va vacío |
| `person` | str | `presidenta` / `vicepresidente`; vacío si no se identifica |
| `shot_type` | str | `closeup` (primer plano) / `wide` (plano general o sin cara) / `other` |
| `valence` | float | En [−1, 1]; negativo = expresión desagradable o tensa, positivo = agradable |
| `arousal` | float | En [−1, 1]; negativo = calmada, positivo = activada |
| `p_neutral`, `p_happiness`, `p_sadness`, `p_surprise`, `p_fear`, `p_anger`, `p_disgust`, `p_contempt` | float | Probabilidad de cada emoción (suman 1) |
| `au04_brow_lowerer` | float | Intensidad en [0, 1] del ceño fruncido (preocupación, concentración) |
| `au12_lip_corner_puller` | float | Intensidad en [0, 1] de la sonrisa |
| `au24_lip_pressor` | float | Intensidad en [0, 1] de los labios apretados (tensión) |

Notas para quien cruce este fichero:

- Una banquera central sale «neutral» casi todo el tiempo, así que las probabilidades y los Action Units son más útiles que la emoción ganadora. Lo más informativo es la **desviación respecto a su propia media** en las ruedas procesadas.
- Para agregar en la ventana de una frase (`signals.csv` del bloque 2), basta con la media de las filas con `face_detected = True` que solapan con la frase; si no hay ninguna, el valor queda vacío.

## `data/events/<fecha>/projections.csv`

Proyecciones macroeconómicas del staff publicadas el día de la rueda. Se leen de los gráficos del documento de proyecciones del BCE (no del vídeo).

| Columna | Tipo | Descripción |
|---|---|---|
| `variable` | str | `hicp` (inflación), `hicp_ex_energy_food` (subyacente), `gdp` (PIB real), `unemployment` (paro) |
| `year` | int | Año proyectado |
| `value` | float | Valor proyectado |
| `unit` | str | `pct` (tasa anual en %; para el paro, % de la población activa) |
| `source` | str | Modelo o fuente que produjo el dato (`example` en el fichero de ejemplo) |

## La app

`src/app/main.py`, arrancada con `run.sh` / `run.bat`. Solo lee `data/events/<fecha>/` y `data/history/`, y usa las funciones del chat cuando existen (`responder` del bloque 2; `voz_a_texto` y `texto_a_voz` del bloque 1). Si falta un fichero, la app muestra el resto y avisa de lo que falta.

El vídeo de cada rueda se busca en `data/raw/<fecha>.mp4` (fuera de git).

## Lo que el bloque 3 necesita de los demás

- **Bloque 1:** `transcript.csv` y `voice.csv` con `start` y `end`; `briefing.mp3`; funciones `voz_a_texto(audio: bytes) -> str` y `texto_a_voz(texto: str) -> bytes` (audio), importables como `from src.audio import voz_a_texto, texto_a_voz` (la app las busca ahí; ver `src/app/live.py`).
- **Bloque 2:** `signals.csv`, `stance.csv` y `summary.json` según `docs/contrato_b2.md`; función `responder`.
