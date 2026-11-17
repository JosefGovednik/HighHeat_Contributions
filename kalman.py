"""kalman.py

Shared Kalman utilities used by the ball-tracking visualization scripts.

This module exposes:
  - create_kalman(): returns a configured cv2.KalmanFilter
  - safe_to_numpy(): helper to convert result containers to numpy arrays

Keep this file small and importable from scripts placed in the same folder
(run those scripts with `python functions/script.py` so the script directory is
on sys.path and `import kalman` works).
"""
from typing import Any
import cv2
import numpy as np


def create_kalman() -> cv2.KalmanFilter:
    """Create and return a Kalman filter configured for (x,y,vx,vy) tracking.

    State: [x, y, vx, vy]^T
    Measurement: [x, y]^T
    Tunable noise covariances are set to conservative defaults but can be
    adjusted by callers if required.
    """
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    # Process / measurement noise; these are reasonable defaults for video tracking
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


def safe_to_numpy(tensor_like: Any) -> np.ndarray:
    """Convert PyTorch tensors or other array-like objects to numpy arrays.

    Returns an empty numpy array on failure to convert.
    """
    try:
        return tensor_like.cpu().numpy()
    except Exception:
        try:
            return np.array(tensor_like)
        except Exception:
            return np.array([])
