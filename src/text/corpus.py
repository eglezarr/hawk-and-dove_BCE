"""Corpus de transcripciones oficiales de las ruedas de prensa del BCE · Bloque 2, fase 1."""
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
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
