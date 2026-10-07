"""Backend py-feat (Cheong et al., 2023): emociones faciales + Action Units.

py-feat es una caja de herramientas de análisis facial. Usamos dos de sus modelos:

- Emociones: ResMaskNet (ResNet con máscara de atención, entrenada en FER2013).
  Da 7 clases: no tiene «contempt», que queda a 0.
- Action Units (FACS): landmarks de la cara (MobileFaceNet) → descriptores HOG →
  un clasificador XGBoost por AU. Da la probabilidad de 20 AU; guardamos las tres
  de schema.ACTION_UNITS (AU04 ceño, AU12 sonrisa, AU24 labios apretados).

No estima valence ni arousal (quedan a None).

Los tres candidatos reciben el mismo recorte (la cara elegida por YuNet). Dentro de
él pasamos el detector de py-feat (RetinaFace) para encuadrar la cara como espera el
modelo de AU: los descriptores HOG son muy sensibles al encuadre y la caja de YuNet es
~16 % más pequeña que la de RetinaFace (con ella AU12 salía muy distinto del pipeline
oficial). Si RetinaFace no encuentra la cara, se usa la caja de YuNet reconstruida
dentro del recorte (está centrada, con el margen detect.CROP_MARGIN).

Pesos (~1,3 GB en total, porque py-feat carga también su detector y su modelo de
pose aunque no se usen) desde Hugging Face la primera vez. Licencia MIT; PyTorch en CPU.
"""
import warnings

import numpy as np

from src.vision.detect import CROP_MARGIN

# Orden de salida de ResMaskNet (feat.utils.FEAT_EMOTION_COLUMNS)
_CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
# Columnas de schema.ACTION_UNITS → nombre de la AU en py-feat
_AUS = {"au04_brow_lowerer": "AU04", "au12_lip_corner_puller": "AU12", "au24_lip_pressor": "AU24"}


class PyFeatBackend:
    name = "pyfeat"

    def __init__(self):
        import torch
        from feat import Detectorv1
        from feat.pretrained import AU_LANDMARK_MAP
        from feat.utils.face_mask import extract_hog_features_batched
        from feat.utils.image_operations import extract_face_from_bbox_torch

        self._torch = torch
        self._extract = extract_face_from_bbox_torch
        self._hog = extract_hog_features_batched
        with warnings.catch_warnings():  # avisa de que no estima la pose de la cabeza (no la usamos)
            warnings.simplefilter("ignore", UserWarning)
            self._det = Detectorv1(face_model="retinaface", landmark_model="mobilefacenet",
                                   au_model="xgb", emotion_model="resmasknet",
                                   identity_model=None, gaze_model=None)
        self._au_idx = {col: AU_LANDMARK_MAP["Feat"].index(au) for col, au in _AUS.items()}

    def _face_box(self, h: int, w: int):
        # Caja de YuNet dentro del recorte: centrada, sin el margen que añadió detect.crop
        side = min(h, w) / (1 + 2 * CROP_MARGIN)
        cx, cy = w / 2, h / 2
        return self._torch.tensor([[cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]])

    def _aligned_faces(self, face: np.ndarray):
        """Recortes de py-feat: 112 px para landmarks y AU, 224 px para emociones."""
        torch = self._torch
        img = torch.from_numpy(np.ascontiguousarray(face)).permute(2, 0, 1)[None].float()
        with torch.inference_mode():
            found = self._det.detect_faces(img)[0]  # RetinaFace; recibe píxeles en [0, 255]
        boxes = found["boxes"]
        if boxes.numel() and not torch.isnan(boxes).any():
            i = int(((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).argmax())
            return found["faces"][i:i + 1], found["resmasknet_faces"][i:i + 1]
        box = self._face_box(*face.shape[:2])
        img = img / 255
        faces, _ = self._extract(img, box, face_size=112)
        emo_faces, _ = self._extract(img, box, expand_bbox=1.1, face_size=224)
        return faces, emo_faces

    def analyze(self, face: np.ndarray) -> dict:
        torch = self._torch
        faces, emo_faces = self._aligned_faces(face)
        with torch.inference_mode():
            landmarks = self._det.landmark_detector.forward(faces)[0]
            probs = torch.softmax(self._det.emotion_detector.forward(emo_faces), 1)[0].numpy()
            hog, lmk = self._hog(faces, landmarks, hog_layer=self._det._hog_layer)
            aus = np.asarray(self._det.au_detector.detect_au(frame=hog, landmarks=[lmk]))[0]
        emotions = {c: float(p) for c, p in zip(_CLASSES, probs)}
        emotions["contempt"] = 0.0
        return {
            "valence": None,
            "arousal": None,
            "emotions": emotions,
            "action_units": {col: float(aus[i]) for col, i in self._au_idx.items()},
        }
