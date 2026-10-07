"""Informe de una rueda de prensa (stance.csv y summary.json) · Bloque 2, fase 6.

Versión solo texto: parte de la transcripción oficial y de la postura de cada frase calculada
en la fase 3. Cuando el bloque 1 alinee las frases con el vídeo, start y end dejarán de estar
vacíos; las señales de voz y cara (voice y face) se añaden en la fase 5.

El briefing tiene dos partes:
- Un primer párrafo generado por código: decisión de tipos, tono y percentiles. Las cifras de
  nuestro modelo no pasan por el LLM, así que no las puede alterar.
- Cinco frases del LLM, una por línea (tres mensajes clave y dos riesgos), cada una con la cita
  del fragmento de la transcripción que la respalda. El código las agrupa en dos párrafos.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pysbd

from src.config import HISTORY_DIR, event_dir
from src.text.models import llm
from src.text.rag import HABLANTES, MESES, cita, construir_fragmentos, fecha_en_ingles, procesar_citas

# Tono relativo al histórico: quintiles de las ruedas de la era Lagarde (límite superior del percentil)
TONOS = [(20, "dovish"), (40, "slightly dovish"), (60, "neutral"), (80, "slightly hawkish"), (100, "hawkish")]
AVISO = "For information purposes only. This is not investment advice."
N_FRASES = 5   # frases del LLM: tres mensajes clave y dos riesgos

# Tamaño del movimiento, tal como lo anuncia la declaración ("... by 25 basis points")
PATRON_PUNTOS = re.compile(r"decided to (?:raise|increase|lower|reduce|cut) the (?:three )?key ECB interest rates "
                           r"by (\d+) basis points", re.IGNORECASE)
PATRON_CITA = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
CORTESIA = re.compile(r"(here (is|are)|below (is|are)|sure|certainly)\b", re.IGNORECASE)
SEGMENTADOR = pysbd.Segmenter(language="en", clean=False)   # el mismo segmentador que el corpus

PROMPT_INFORME = """You summarise a European Central Bank (ECB) monetary policy press conference for treasury and fixed-income professionals. You receive the transcript of the press conference, split into numbered excerpts. The rate decision and the overall tone are reported separately: do not repeat them.

Write exactly five sentences, one per line:
- Sentences 1 to 3: the three key messages on inflation, growth and the monetary policy outlook.
- Sentences 4 and 5: the two main risks the ECB highlighted.

Rules:
- Use only the information in the excerpts. Do not compute or invent figures.
- End every sentence with the number of the excerpt that supports it, in brackets, for example: "... higher energy prices [12]." The journalists' questions are context only and cannot be cited.
- Report what the ECB said, in the third person ("The ECB expects..."). Do not copy the President's words in the first person.
- Do not add forecasts, opinions or investment recommendations.
- At most 30 words per sentence. No headings, bullet points or numbering."""


# ---------------------------------------------------------------------------
# Datos del evento
# ---------------------------------------------------------------------------
def cargar_historico() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Transcripciones, postura de cada frase del panel e índice por rueda del histórico."""
    return (pd.read_parquet(HISTORY_DIR / "transcripts.parquet"),
            pd.read_parquet(HISTORY_DIR / "stance_sentences.parquet"),
            pd.read_csv(HISTORY_DIR / "stance_by_conference.csv"))


def etiqueta_tono(percentil: float) -> str:
    """Tono relativo al histórico según el quintil del percentil."""
    return next(etiqueta for limite, etiqueta in TONOS if percentil <= limite)


def _percentil(serie: pd.Series, valor: float) -> int:
    """Porcentaje de ruedas con una puntuación igual o inferior (mismo criterio que el índice histórico)."""
    return int(round(100 * float((serie <= valor).mean())))


def postura_evento(fecha: str, indice: pd.DataFrame) -> dict:
    """Puntuaciones de la rueda y su posición frente a las ruedas anteriores.

    Los percentiles se calculan solo con las ruedas hasta la fecha del evento (incluida): el
    informe dice lo mismo que habría dicho ese día, sin información posterior.
    """
    hasta = indice[indice["date"] <= fecha].sort_values("date").reset_index(drop=True)
    fila = hasta.iloc[-1]
    if fila["date"] != fecha:
        raise ValueError(f"La rueda del {fecha} no está en el índice histórico")
    percentil = _percentil(hasta["score"], fila["score"])
    postura = {
        "score": round(float(fila["score"]), 3),
        "label": etiqueta_tono(percentil),
        "score_statement": round(float(fila["score_statement"]), 3),
        "score_qa": round(float(fila["score_qa"]), 3),
        "percentile_vs_history": percentil,
        "percentile_statement": _percentil(hasta["score_statement"], fila["score_statement"]),
        "percentile_qa": _percentil(hasta["score_qa"], fila["score_qa"]),
        "history_size": len(hasta),
        "decision": fila["decision"],
        "previous_date": None,
        "previous_percentile": None,
    }
    if len(hasta) > 1:   # percentil de la rueda anterior con los datos disponibles en su día
        anteriores = hasta.iloc[:-1]
        postura["previous_date"] = anteriores["date"].iloc[-1]
        postura["previous_percentile"] = _percentil(anteriores["score"], anteriores["score"].iloc[-1])
    return postura


def normalizar_texto(texto: str) -> str:
    """Minúsculas, solo letras y números: para detectar frases repetidas literalmente."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", texto.lower().replace("’", "'"))).strip()


def momentos_clave(frases: pd.DataFrame, repetidas: set[str], por_lado: int = 2) -> list[dict]:
    """Las frases relevantes más hawkish y más dovish de la rueda, en orden de aparición.

    - Se descartan las frases que repiten literalmente la rueda anterior: alrededor de una
      cuarta parte de la declaración se copia de una reunión a otra y no aporta información.
    - Como mucho una por párrafo, para que no se repita la misma idea.
    - En la fase 5 se podrán reordenar con las señales de voz y cara.
    """
    nuevas = frases[frases["relevant"] & ~frases["text"].map(normalizar_texto).isin(repetidas)]
    lados = [nuevas[nuevas["label"] == "hawkish"].sort_values("score", ascending=False, kind="stable"),
             nuevas[nuevas["label"] == "dovish"].sort_values("score", kind="stable")]
    elegidas = pd.concat([lado.drop_duplicates("paragraph_id").head(por_lado) for lado in lados])
    momentos = []
    for f in elegidas.sort_values("sentence_id").itertuples():
        parte = "statement" if f.section == "statement" else "Q&A"
        momentos.append({"start": None, "end": None, "sentence_id": int(f.sentence_id), "speaker": f.speaker,
                         "text": f.text, "score": round(float(f.score), 3),
                         "reason": f"{f.label.capitalize()} signal in the {parte}"})
    return momentos


def _transcripcion(fragmentos: pd.DataFrame, preguntas: pd.Series) -> str:
    """Fragmentos numerados en orden de aparición, con cada pregunta delante de sus respuestas."""
    lineas, turno, en_preguntas = ["Monetary policy statement:"], None, False
    for n, f in enumerate(fragmentos.itertuples(), start=1):
        if f.role == "statement":
            apartado = f" ({f.subsection})" if isinstance(f.subsection, str) else ""
            lineas.append(f"[{n}] Statement{apartado}: {f.text}")
            continue
        if not en_preguntas:
            lineas.append("\nQuestions and answers:")
            en_preguntas = True
        if f.role == "answer" and f.qa_id != turno:
            turno = f.qa_id
            lineas.append(f"Question {turno} (journalist, context only): {preguntas.get(turno, '')}")
        tipo = "Answer" if f.role == "answer" else "Remarks"
        lineas.append(f"[{n}] {tipo} by {HABLANTES[f.speaker]}: {f.text}")
    return "\n".join(lineas)


def preparar_evento(fecha: str, corpus: pd.DataFrame, frases: pd.DataFrame, indice: pd.DataFrame) -> dict:
    """Todo lo que necesita el informe de una rueda, incluido el mensaje para el LLM."""
    corpus_evento = corpus[corpus["date"] == fecha]
    fragmentos = construir_fragmentos(corpus_evento)
    preguntas = (corpus_evento[corpus_evento["role"] == "question"].sort_values("sentence_id")
                 .groupby("qa_id")["text"].agg(" ".join))
    declaracion = " ".join(corpus_evento[corpus_evento["role"] == "statement"].sort_values("sentence_id")["text"])
    puntos = PATRON_PUNTOS.search(declaracion)
    anteriores = sorted(d for d in corpus["date"].unique() if d < fecha)
    repetidas = (set(corpus.loc[corpus["date"] == anteriores[-1], "text"].map(normalizar_texto))
                 if anteriores else set())
    mensaje = (f"TRANSCRIPT OF THE PRESS CONFERENCE OF {fecha_en_ingles(fecha).upper()}\n\n"
               f"{_transcripcion(fragmentos, preguntas)}\n\n"
               f"Write the five sentences now: one per line, each ending with the number of its excerpt in brackets.")
    return {"fecha": fecha, "postura": postura_evento(fecha, indice), "fragmentos": fragmentos,
            "frases": frases[frases["date"] == fecha].sort_values("sentence_id").reset_index(drop=True),
            "puntos": int(puntos.group(1)) if puntos else None, "repetidas": repetidas,
            "desde": indice["date"].min(), "mensaje": mensaje}


# ---------------------------------------------------------------------------
# Redacción
# ---------------------------------------------------------------------------
def ordinal(n: int) -> str:
    """69 -> '69th', 21 -> '21st'."""
    sufijo = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{sufijo}"


def parrafo_tono(postura: dict, puntos: int | None, desde: str) -> str:
    """Primer párrafo del briefing, generado por código: decisión de tipos, tono y percentiles."""
    verbo = {"subida": "raised", "bajada": "lowered"}.get(postura["decision"])
    if verbo:
        decision = f"The ECB {verbo} its key interest rates" + (f" by {puntos} basis points." if puntos else ".")
    else:
        decision = "The ECB kept its key interest rates unchanged."
    anio, mes, _ = desde.split("-")
    p = postura["percentile_vs_history"]
    tono = (f"The Hawk & Dove stance index rates the press conference as {postura['label']}: it ranks at the "
            f"{ordinal(p)} percentile of the {postura['history_size']} press conferences held since "
            f"{MESES[int(mes) - 1]} {anio}")
    if postura["previous_date"]:
        anterior, fecha_anterior = postura["previous_percentile"], fecha_en_ingles(postura["previous_date"])
        if p == anterior:
            tono += f", the same percentile as at the previous meeting on {fecha_anterior}"
        else:
            tono += (f", {'up' if p > anterior else 'down'} from the {ordinal(anterior)} percentile "
                     f"at the previous meeting on {fecha_anterior}")
    partes = (f"The monetary policy statement ranks at the {ordinal(postura['percentile_statement'])} percentile "
              f"and the answers in the Q&A at the {ordinal(postura['percentile_qa'])}.")
    return f"{decision} {tono}. {partes}"


def frases_del_llm(respuesta: str) -> list[str]:
    """Frases del LLM en orden, aunque las escriba seguidas en un párrafo en lugar de una por línea.

    - Se quitan viñetas, numeraciones, encabezados ("Key messages:") y frases de cortesía sin
      cita ("Here are the five sentences").
    - Cada línea se separa en frases con el segmentador del corpus.
    - Cada frase termina en punto con las citas delante ("... prices [12]."), para que al unirlas
      en párrafos no queden frases pegadas.
    """
    frases = []
    for linea in respuesta.splitlines():
        linea = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", linea).strip()
        linea = re.sub(r"^[A-Z][\w &-]{0,40}:\s+(?=[A-Z])", "", linea)   # encabezado delante de la frase
        if not linea or linea.endswith(":"):
            continue
        linea = re.sub(r"([.!?])\s*((?:\[[\d,;\s]+\])+)", r" \2\1", linea)   # "texto. [3]" -> "texto [3]."
        for frase in SEGMENTADOR.segment(linea):
            frase = frase.strip()
            if not re.search(r"[A-Za-z]", frase) or (CORTESIA.match(frase) and not PATRON_CITA.search(frase)):
                continue
            frases.append(frase if re.search(r"[.!?]$", frase) else frase + ".")
    return frases[:N_FRASES]


def tiene_cita(frase: str, n_extractos: int) -> bool:
    """True si la frase cita al menos un fragmento que existe."""
    return any(1 <= int(n) <= n_extractos for grupo in PATRON_CITA.findall(frase) for n in re.findall(r"\d+", grupo))


def texto_para_voz(texto: str) -> str:
    """Versión para la voz sintética: sin marcadores de cita y con las abreviaturas desarrolladas."""
    limpio = re.sub(r"\s*\[\d+\]", "", texto)
    limpio = re.sub(r"(\d)\s*%", r"\1 per cent", limpio)
    limpio = re.sub(r"\b(\d+)\s*bps?\b", r"\1 basis points", limpio)
    limpio = limpio.replace("Q&A", "Q and A")
    return re.sub(r"[ \t]{2,}", " ", limpio).strip()


def redactar_informe(evento: dict, modelo_id: str, max_tokens: int = 300) -> dict:
    """Briefing completo: párrafo de tono (código) y cinco frases del LLM con sus citas.

    El marcador [n] del texto remite a citations[n-1].
    """
    t0 = time.perf_counter()
    respuesta = llm.generar(modelo_id, PROMPT_INFORME, evento["mensaje"], max_tokens=max_tokens)
    segundos = time.perf_counter() - t0

    fragmentos = evento["fragmentos"]
    frases = frases_del_llm(respuesta)
    parrafos = [" ".join(frases[:3]), " ".join(frases[3:])]
    texto_llm, usados = procesar_citas("\n\n".join(p for p in parrafos if p), len(fragmentos))
    briefing = f"{parrafo_tono(evento['postura'], evento['puntos'], evento['desde'])}\n\n{texto_llm}"
    citas = [{**cita(fragmentos.iloc[i]), "sentence_id": int(fragmentos.iloc[i]["sentence_id"])} for i in usados]
    return {"briefing": briefing, "briefing_tts": texto_para_voz(briefing), "citations": citas,
            "respuesta_llm": respuesta, "texto_llm": texto_llm, "n_frases": len(frases),
            "frases_con_cita": sum(tiene_cita(f, len(fragmentos)) for f in frases), "seconds": segundos}


# ---------------------------------------------------------------------------
# Ficheros del evento
# ---------------------------------------------------------------------------
def tabla_postura(frases: pd.DataFrame) -> pd.DataFrame:
    """Postura de cada frase en el formato de stance.csv.

    p_neutral incluye la probabilidad de "irrelevant", como en la etiqueta: las tres suman 1.
    weight es el peso de la frase en las puntuaciones (1 − p_irrelevant). start y end quedan
    vacíos hasta que las frases se alineen con el vídeo.
    """
    p_irrelevant = frases["p_irrelevant"].fillna(0.0)
    tabla = frases[["sentence_id", "paragraph_id", "speaker", "section", "text", "label", "p_hawkish"]].copy()
    tabla["p_neutral"] = frases["p_neutral"] + p_irrelevant
    tabla["p_dovish"] = frases["p_dovish"]
    tabla["score"] = frases["score"]
    tabla["relevant"] = frases["relevant"]
    tabla["weight"] = 1.0 - p_irrelevant
    tabla.insert(0, "end", np.nan)
    tabla.insert(0, "start", np.nan)
    return tabla


def guardar_informe(evento: dict, informe: dict, modelo_id: str) -> Path:
    """Escribe stance.csv y summary.json en data/events/<fecha>/."""
    carpeta = event_dir(evento["fecha"])
    carpeta.mkdir(parents=True, exist_ok=True)
    tabla_postura(evento["frases"]).to_csv(carpeta / "stance.csv", index=False)
    resumen = {
        "event_date": evento["fecha"],
        "stance": evento["postura"],
        "voice": None,   # fase 5 (bloque 1)
        "face": None,    # fase 5 (bloque 3)
        "key_moments": momentos_clave(evento["frases"], evento["repetidas"]),
        "briefing": informe["briefing"],
        "briefing_tts": informe["briefing_tts"],
        "citations": informe["citations"],
        "disclaimer": AVISO,
        "generated_with": {"stance_model": str(evento["frases"]["model"].iloc[0]), "llm": modelo_id},
    }
    (carpeta / "summary.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    return carpeta
