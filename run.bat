@echo off
REM Arranque directo de la app (Windows). La app la crea el bloque 3 en src\app\main.py.
cd /d %~dp0
call .venv\Scripts\activate
streamlit run src\app\main.py
