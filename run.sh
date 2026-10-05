#!/usr/bin/env bash
# Arranque directo de la app (Mac/Linux). La app la crea el bloque 3 en src/app/main.py.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
streamlit run src/app/main.py
