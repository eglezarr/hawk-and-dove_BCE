"""Corpus de transcripciones oficiales de las ruedas de prensa del BCE · Bloque 2, fase 1."""
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import pysbd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.ecb.europa.eu"

# Fichero con el listado de ruedas de prensa de cada año. La ruta cambió con el
# rediseño de la web del BCE, así que probamos la actual y la antigua.
INCLUDE_URLS = [
    BASE_URL + "/press/press_conference/monetary-policy-statement/{year}/html/index_include.en.html",
    BASE_URL + "/press/pressconf/{year}/html/index_include.en.html",
]

# Nombre de cada transcripción, p. ej. "ecb.is260430~f99cb123a8.en.html"
# ("is" = introductory statement; las seis cifras son la fecha AAMMDD)
TRANSCRIPT_RE = re.compile(r"ecb\.is(\d{6})~[0-9a-f]+\.en\.html")

# Nos identificamos en las peticiones (buena práctica al descargar de una web pública)
HEADERS = {"User-Agent": "hawk-and-dove (academic project, MIAX)"}


# ---------------------------------------------------------------------------
# Paso 1: listado de ruedas de prensa
# ---------------------------------------------------------------------------
def listar_urls(year: int) -> dict[str, str]:
    """Devuelve {fecha 'AAAA-MM-DD': url} con las ruedas de prensa de un año."""
    for include_url in INCLUDE_URLS:
        resp = requests.get(include_url.format(year=year), headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            continue  # esta ruta no existe para ese año: probamos la siguiente

        soup = BeautifulSoup(resp.text, "html.parser")
        urls = {}
        for link in soup.find_all("a", href=True):
            match = TRANSCRIPT_RE.search(link["href"])
            if match:
                aammdd = match.group(1)
                fecha = f"20{aammdd[:2]}-{aammdd[2:4]}-{aammdd[4:]}"
                # urljoin resuelve enlaces relativos respecto al fichero del listado
                urls[fecha] = urljoin(resp.url, link["href"])
        if urls:
            return urls
    return {}  # ninguna ruta devolvió transcripciones para ese año


# ---------------------------------------------------------------------------
# Paso 2: descarga del HTML con caché local
# ---------------------------------------------------------------------------
def descargar_html(listado: pd.DataFrame, carpeta: Path, pausa: float = 1.0) -> list[Path]:
    """Descarga el HTML de cada rueda de prensa y lo guarda como <fecha>.html.

    Si el fichero ya existe no se vuelve a descargar: así podemos reprocesar el
    corpus tantas veces como haga falta sin volver a consultar la web del BCE.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = []
    for fecha, url in zip(listado["date"], listado["url"]):
        ruta = carpeta / f"{fecha}.html"
        if not ruta.exists():
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()  # si una descarga falla, que se vea el error
            ruta.write_text(resp.text, encoding="utf-8")
            time.sleep(pausa)        # pausa de cortesía entre peticiones
        rutas.append(ruta)
    return rutas


# ---------------------------------------------------------------------------
# Paso 3: lectura de cada transcripción
# ---------------------------------------------------------------------------
# Párrafo que separa la declaración del turno de preguntas (y, a veces, del cierre final)
SEPARADORES = {"* * *", "***"}

# Etiquetas de hablante al inicio de un párrafo de respuesta. Cambian según el año
# ("Lagarde:", "President Lagarde:", "President:", "De Guindos:", "Vice-President:"...)
# e incluyen a los gobernadores anfitriones de las reuniones fuera de Fráncfort.
# El orden importa: se prueba primero el vicepresidente.
ETIQUETAS_HABLANTE = [
    (re.compile(r"^(?:Vice-President(?:\s+(?:Luis\s+)?[Dd]e\s+Guindos|\s+(?:Boris\s+)?Vuj\w+)?"
                r"|(?:Luis\s+)?[Dd]e\s+Guindos|(?:Boris\s+)?Vuj\w+)\s*:\s*"), "vicepresidente"),
    (re.compile(r"^(?:President(?:\s+Lagarde)?|Lagarde)\s*:\s*"), "presidenta"),
    (re.compile(r"^(?:Governor\s+[A-Z]\w+|Yannis\s+Stournaras)\s*:\s*"), "otro"),
]


def _limpiar(texto: str) -> str:
    """Normaliza espacios (incluidos los no separables) y recorta los extremos."""
    return re.sub(r"\s+", " ", texto).strip()


def _separar_etiqueta(texto: str) -> tuple[str | None, str]:
    """Si el párrafo empieza por una etiqueta de hablante, devuelve (hablante, texto sin etiqueta)."""
    for patron, hablante in ETIQUETAS_HABLANTE:
        match = patron.match(texto)
        if match:
            return hablante, texto[match.end():]
    return None, texto


def _proporcion_negrita(elemento) -> float:
    """Proporción del texto de un párrafo que está en negrita (las preguntas van en negrita)."""
    total = len(_limpiar(elemento.get_text("")))
    if total == 0:
        return 0.0
    negrita = sum(len(_limpiar(b.get_text(""))) for b in elemento.find_all(["strong", "b"]))
    return negrita / total


def leer_transcripcion(html: str) -> list[dict]:
    """Divide una transcripción en párrafos con su sección, rol y hablante.

    Reglas, comprobadas sobre las 55 ruedas de la era Lagarde:
    - El texto está en <main> > div.section. Antes del separador "* * *" va la
      declaración; después, el turno de preguntas.
    - Un párrafo en negrita es una pregunta. Los párrafos en negrita seguidos son
      la misma intervención del periodista (mismo qa_id).
    - Una respuesta es de quien indique su etiqueta ("Lagarde:", "De Guindos:"...).
      Sin etiqueta, sigue hablando el mismo; al empezar cada respuesta, la presidenta.
    - Lo que no responde a ninguna pregunta (palabras antes de la primera pregunta o
      el cierre tras un segundo separador) se guarda como "remarks".
    - Una "pregunta" sin respuesta es el cierre del moderador.
    """
    soup = BeautifulSoup(html, "html.parser")
    seccion = soup.find("main").find("div", class_="section", recursive=False)

    # Quitamos las llamadas a notas al pie ("[1]"); los ordinales ("3rd") se mantienen
    for sup in seccion.find_all("sup"):
        if sup.find("a", href=re.compile(r"^#footnote")):
            sup.decompose()

    parrafos = []
    zona = "statement"          # statement -> qa -> closing (tras un segundo separador)
    subseccion = None           # apartado de la declaración ("Inflation", "Risk assessment"...)
    qa_id = 0                   # número de pregunta dentro de la rueda
    hablante = "presidenta"     # quién está hablando en la respuesta actual
    previo_pregunta = False     # para unir párrafos de pregunta consecutivos

    for el in seccion.find_all(["p", "h2"], recursive=False):
        clases = el.get("class") or []
        texto = _limpiar(el.get_text(""))
        if not texto:
            continue

        if el.name == "h2":
            # El primer h2 es la cabecera con los nombres del panel; el resto, apartados
            if "ecb-pressContentSubtitle" not in clases:
                subseccion = texto
            continue
        if "ecb-publicationDate" in clases:   # lugar y fecha
            continue
        if texto in SEPARADORES:
            zona = "qa" if zona == "statement" else "closing"
            continue

        if zona == "statement":
            parrafos.append(dict(section="statement", subsection=subseccion, qa_id=None,
                                 role="statement", speaker="presidenta", text=texto))
            continue

        etiqueta, texto_sin_etiqueta = _separar_etiqueta(texto)

        # Pregunta: párrafo en negrita que no empieza por una etiqueta de hablante
        if etiqueta is None and _proporcion_negrita(el) >= 0.9:
            if not previo_pregunta:
                qa_id += 1
            parrafos.append(dict(section="qa", subsection=None, qa_id=qa_id,
                                 role="question", speaker="periodista", text=texto))
            previo_pregunta = True
            hablante = "presidenta"   # por defecto, responde la presidenta
            continue

        # Respuesta o intervención fuera de pregunta
        if etiqueta:
            hablante = etiqueta
        es_respuesta = zona == "qa" and qa_id > 0
        parrafos.append(dict(section="qa", subsection=None,
                             qa_id=qa_id if es_respuesta else None,
                             role="answer" if es_respuesta else "remarks",
                             speaker=hablante, text=texto_sin_etiqueta))
        previo_pregunta = False

    # Una "pregunta" sin respuesta es en realidad el cierre del moderador
    # (p. ej., "Thank you, President Lagarde. This brings our press conference to an end")
    con_respuesta = {p["qa_id"] for p in parrafos if p["role"] == "answer"}
    for p in parrafos:
        if p["role"] == "question" and p["qa_id"] not in con_respuesta:
            p.update(role="remarks", speaker="moderador", qa_id=None)

    return parrafos


# ---------------------------------------------------------------------------
# Paso 4: frases, corpus y controles de calidad
# ---------------------------------------------------------------------------
COLUMNAS = ["date", "url", "section", "subsection", "qa_id", "role", "speaker",
            "paragraph_id", "sentence_id", "text"]


def construir_corpus(listado: pd.DataFrame, html_dir: Path) -> pd.DataFrame:
    """Construye el corpus frase a frase de todas las ruedas del listado.

    Cada párrafo se parte en frases con pySBD (segmentador basado en reglas que
    respeta abreviaturas como "Mr" o "i.e." y decimales como "2.6 per cent").
    """
    segmentador = pysbd.Segmenter(language="en", clean=False)
    filas = []
    for fecha, url in zip(listado["date"], listado["url"]):
        html = (html_dir / f"{fecha}.html").read_text(encoding="utf-8")
        sentence_id = 0
        for paragraph_id, parrafo in enumerate(leer_transcripcion(html)):
            for frase in segmentador.segment(parrafo["text"]):
                frase = frase.strip()
                if frase:
                    filas.append({**parrafo, "date": fecha, "url": url, "paragraph_id": paragraph_id,
                                  "sentence_id": sentence_id, "text": frase})
                    sentence_id += 1
    corpus = pd.DataFrame(filas)[COLUMNAS]
    corpus["qa_id"] = corpus["qa_id"].astype("Int64")   # entero que admite vacíos
    return corpus


def resumen_por_rueda(corpus: pd.DataFrame) -> pd.DataFrame:
    """Controles de calidad por rueda de prensa (número de frases por rol y hablante)."""
    return corpus.groupby("date").agg(
        frases_declaracion=("role", lambda r: (r == "statement").sum()),
        preguntas=("qa_id", "nunique"),
        frases_pregunta=("role", lambda r: (r == "question").sum()),
        frases_respuesta=("role", lambda r: (r == "answer").sum()),
        frases_vicepresidente=("speaker", lambda s: (s == "vicepresidente").sum()),
        frases_otros=("speaker", lambda s: s.isin(["moderador", "otro"]).sum()),
    )
