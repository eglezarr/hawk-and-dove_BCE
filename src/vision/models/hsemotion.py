"""Backend HSEmotion (Savchenko, 2022): emociones faciales + valence y arousal.

Modelo `enet_b0_8_va_mtl`: EfficientNet-B0 entrenado en AffectNet que predice a la
vez las 8 emociones y valence/arousal. Usamos la versión ONNX (paquete
hsemotion-onnx): va en CPU con onnxruntime y no necesita PyTorch. El modelo
(~16 MB) se descarga la primera vez en ~/.hsemotion/.
No estima Action Units.
"""
import numpy as np

# Orden de salida del modelo de 8 clases → nombres de schema.EMOTIONS
_CLASSES = ["anger", "contempt", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]


class HSEmotionBackend:
    name = "hsemotion"

    def __init__(self, model_name: str = "enet_b0_8_va_mtl"):
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

        self._model = HSEmotionRecognizer(model_name=model_name)

    def analyze(self, face: np.ndarray) -> dict:
        _, scores = self._model.predict_emotions(face, logits=False)  # softmax en las 8 emociones
        probs, valence, arousal = scores[:8], scores[8], scores[9]
        return {
            "valence": float(np.clip(valence, -1, 1)),
            "arousal": float(np.clip(arousal, -1, 1)),
            "emotions": {c: float(p) for c, p in zip(_CLASSES, probs)},
            "action_units": {},
        }
