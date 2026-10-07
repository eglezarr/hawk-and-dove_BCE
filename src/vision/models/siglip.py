"""Backend SigLIP zero-shot (Zhai et al., 2023): expresión facial sin entrenar nada.

SigLIP es un modelo contrastivo texto-imagen (como CLIP, pero con pérdida sigmoide):
convierte imágenes y textos en vectores del mismo espacio, de modo que una imagen
queda cerca de los textos que la describen. Para clasificar la expresión comparamos
el recorte de la cara con frases como «a photo of an angry face»: la clase cuyas
frases quedan más cerca es la predicción. No hay entrenamiento con caras etiquetadas.

- Emociones: softmax sobre los logits de las 8 clases de schema.EMOTIONS. Cada clase
  se describe con varias frases (sinónimos × plantillas) y se promedian sus vectores
  (prompt ensembling, como en el artículo de CLIP).
- Valence y arousal: SigLIP no los estima directamente; los obtenemos también por
  zero-shot comparando dos polos opuestos (agradable/desagradable, activada/calmada):
  valor = P(polo positivo) − P(polo negativo), en [-1, 1].

Modelo `google/siglip-base-patch16-224` (~200 M parámetros, ~800 MB, Apache 2.0) con
transformers + PyTorch en CPU. Se descarga la primera vez en ~/.cache/huggingface/.
No estima Action Units.
"""
import numpy as np

DEFAULT_MODEL = "google/siglip-base-patch16-224"

# Plantillas de frase; {} se rellena con cada descripción (que ya lleva el artículo)
TEMPLATES = [
    "a photo of {} face.",
    "a close-up photo of a person with {} expression.",
    "{} face.",
]

# Descripciones de cada emoción (claves en el orden de schema.EMOTIONS)
EMOTION_PROMPTS = {
    "neutral": ["a neutral", "an expressionless"],
    "happiness": ["a happy", "a smiling"],
    "sadness": ["a sad", "a sorrowful"],
    "surprise": ["a surprised", "an astonished"],
    "fear": ["a fearful", "a scared"],
    "anger": ["an angry", "a furious"],
    "disgust": ["a disgusted", "a repulsed"],
    "contempt": ["a contemptuous", "a scornful"],
}

# Polos de valence y arousal: (descripciones del polo positivo, del polo negativo)
VALENCE_PROMPTS = (["a pleasant", "a warm and friendly"], ["an unpleasant", "a cold and hostile"])
AROUSAL_PROMPTS = (["an excited and agitated", "a tense and alert"], ["a calm and relaxed", "a sleepy and bored"])


class SigLIPBackend:
    name = "siglip"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self._model = AutoModel.from_pretrained(model_name).eval()
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._scale = self._model.logit_scale.exp().item()
        self._bias = self._model.logit_bias.item()
        # Los textos no cambian: sus vectores se calculan una sola vez
        self._emotions = list(EMOTION_PROMPTS)
        self._emotion_emb = self._class_embeddings(EMOTION_PROMPTS.values())
        self._valence_emb = self._class_embeddings(VALENCE_PROMPTS)
        self._arousal_emb = self._class_embeddings(AROUSAL_PROMPTS)

    def _encode_text(self, texts: list[str]) -> np.ndarray:
        # SigLIP se entrenó con los textos rellenados a longitud fija: padding="max_length"
        inputs = self._processor(text=texts, padding="max_length", return_tensors="pt")
        with self._torch.no_grad():
            emb = self._model.get_text_features(**inputs).pooler_output.numpy()
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def _class_embeddings(self, classes) -> np.ndarray:
        # Un vector por clase: media de sus frases (descripciones × plantillas), normalizada
        out = []
        for descriptions in classes:
            emb = self._encode_text([t.format(d) for d in descriptions for t in TEMPLATES]).mean(axis=0)
            out.append(emb / np.linalg.norm(emb))
        return np.stack(out)

    def _encode_image(self, image: np.ndarray) -> np.ndarray:
        inputs = self._processor(images=image, return_tensors="pt")
        with self._torch.no_grad():
            emb = self._model.get_image_features(**inputs).pooler_output.numpy()[0]
        return emb / np.linalg.norm(emb)

    def _probs(self, image_emb: np.ndarray, class_emb: np.ndarray) -> np.ndarray:
        logits = self._scale * class_emb @ image_emb + self._bias
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def analyze(self, face: np.ndarray) -> dict:
        img = self._encode_image(face)
        probs = self._probs(img, self._emotion_emb)
        valence = self._probs(img, self._valence_emb)
        arousal = self._probs(img, self._arousal_emb)
        return {
            "valence": float(valence[0] - valence[1]),
            "arousal": float(arousal[0] - arousal[1]),
            "emotions": {e: float(p) for e, p in zip(self._emotions, probs)},
            "action_units": {},
        }
