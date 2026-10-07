"""Conexión con los modelos del bloque 3: un backend por candidato.

`get_face_backend(nombre)` devuelve el backend pedido; el nombre del ganador se
lee de config.yaml (models.face). Los imports son diferidos para que tener
instalado un candidato no obligue a instalar los demás.
"""


def get_face_backend(name: str):
    if name == "dummy":
        from src.vision.models.dummy import DummyFaceBackend
        return DummyFaceBackend()
    if name == "hsemotion":
        from src.vision.models.hsemotion import HSEmotionBackend
        return HSEmotionBackend()
    if name == "siglip":
        from src.vision.models.siglip import SigLIPBackend
        return SigLIPBackend()
    if name == "pyfeat":
        from src.vision.models.pyfeat import PyFeatBackend
        return PyFeatBackend()
    raise ValueError(f"Backend de cara desconocido: {name!r}")
