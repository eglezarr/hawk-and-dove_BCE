"""Conexión con los modelos del bloque 1 (un backend por candidato, intercambiables).

Cada módulo expone una función crear_backend_*() que devuelve un objeto
con la interfaz común de la etapa. Esto permite comparar candidatos
en los benchmarks y cambiar de modelo sin tocar la lógica.
"""
