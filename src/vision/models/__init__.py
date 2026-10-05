"""Conexión con los modelos del bloque 3: un backend por candidato.

`get_face_backend(nombre)` devuelve el backend pedido; el nombre del ganador se
lee de config.yaml (models.face). Los imports son diferidos para que tener
instalado un candidato no obligue a instalar los demás.
"""


def get_face_backend(name: str):
    if name == "dummy":
        from src.vision.models.dummy import DummyFaceBackend
        return DummyFaceBackend()
    # Candidatos del benchmark (se añaden según se implementan): "hsemotion", "pyfeat", "siglip"
    raise ValueError(f"Backend de cara desconocido: {name!r}")
