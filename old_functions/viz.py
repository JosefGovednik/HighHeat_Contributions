import cv2
import numpy as np
from typing import Iterable, Tuple


def draw_raw_bbox(vis: np.ndarray, x: int, y: int, w: int, h: int, color=(0, 0, 255), thickness=1):
    cv2.rectangle(vis, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), color, thickness)
    cv2.circle(vis, (int(x), int(y)), 3, color, -1)


def draw_smoothed_path(vis: np.ndarray, points: Iterable[Tuple[int, int]], color=(0, 255, 0), thickness=2):
    pts = np.array(points, dtype=np.int32)
    if pts.shape[0] > 1:
        cv2.polylines(vis, [pts], False, color, thickness, lineType=cv2.LINE_AA)


def draw_smoothed_box(vis: np.ndarray, x: int, y: int, w: int, h: int, color=(0, 255, 0), thickness=1):
    cv2.rectangle(vis, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), color, thickness)
    cv2.circle(vis, (int(x), int(y)), 4, color, -1)


def draw_status_text(vis: np.ndarray, text: str, pos=(10, 40), color=(0, 200, 200)):
    cv2.putText(vis, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_frame_text(vis: np.ndarray, text: str, pos=(10, 18), color=(255, 255, 255)):
    cv2.putText(vis, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
