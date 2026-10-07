"""Comprobaciones automáticas del informe generado por el LLM · Bloque 2, fase 6."""
import re

NUMERO = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")
MARCADOR = re.compile(r"\[\d+\]")
# Cifras que el BCE suele escribir con letras ("two per cent")
NUMEROS_EN_LETRAS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
                     "eight": "8", "nine": "9", "ten": "10", "twelve": "12", "fifteen": "15", "twenty": "20",
                     "fifty": "50", "hundred": "100"}
# Lenguaje de recomendación, que el producto no puede usar (no presta asesoramiento de inversión)
RECOMENDACION = re.compile(r"\b(recommend\w*|advis\w*|investors should|you should|consider (?:buying|selling)|"
                           r"go(?:ing)? (?:long|short)|overweight|underweight)\b", re.IGNORECASE)


def _normalizar(numero: str) -> str:
    """'3.0' y '3' cuentan como la misma cifra."""
    return f"{float(numero.replace(',', '.')):g}"


def cifras(texto: str) -> set[str]:
    return {_normalizar(n) for n in NUMERO.findall(MARCADOR.sub("", texto))}


def cifras_sin_respaldo(texto: str, fuente: str) -> list[str]:
    """Cifras del texto que no aparecen en la información que recibió el LLM (posibles invenciones)."""
    disponibles = cifras(fuente) | {cifra for palabra, cifra in NUMEROS_EN_LETRAS.items()
                                    if re.search(rf"\b{palabra}\b", fuente, re.IGNORECASE)}
    return sorted(cifras(texto) - disponibles, key=float)


def revisar(informe: dict, mensaje: str) -> dict:
    """Comprobaciones de un informe frente al mensaje que recibió el LLM.

    Las cifras se comprueban solo en la parte del LLM: las del párrafo de tono salen del código.
    """
    sin_respaldo = cifras_sin_respaldo(informe["texto_llm"], mensaje)
    return {"palabras": len(MARCADOR.sub("", informe["briefing"]).split()),
            "frases": informe["n_frases"],
            "frases_con_cita": informe["frases_con_cita"],
            "citas": len(informe["citations"]),
            "cifras_sin_respaldo": len(sin_respaldo),
            "detalle_cifras": ", ".join(sin_respaldo),
            "recomendaciones": len(RECOMENDACION.findall(informe["briefing"])),
            "segundos": round(informe["seconds"], 1)}
