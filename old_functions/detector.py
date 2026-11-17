from typing import Tuple
import numpy as np
from ultralytics import YOLO


class Detector:
    """Simple YOLO detector wrapper returning numpy arrays.

    Usage:
        det = Detector(model_path, conf=0.5)
        xyxy, confs = det.detect(frame)
    """
    def __init__(self, model_path: str, conf: float = 0.5):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame) -> Tuple[np.ndarray, np.ndarray]:
        """Run detection and return (xyxy, confs) as numpy arrays.

        xyxy: shape (N,4) with floats, confs: shape (N,)
        If no detections, returns empty arrays.
        """
        results = self.model(frame, conf=self.conf)
        boxes = results[0].boxes
        # boxes.xyxy and boxes.conf may be tensor-like or lists
        try:
            xy = boxes.xyxy.cpu().numpy()
        except Exception:
            try:
                xy = np.array(boxes.xyxy)
            except Exception:
                xy = np.array([])

        try:
            confs = boxes.conf.cpu().numpy()
        except Exception:
            try:
                confs = np.array(boxes.conf)
            except Exception:
                confs = np.array([])

        return xy, confs
