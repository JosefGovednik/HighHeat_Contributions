import cv2
import numpy as np
G = 32.17
# For safe_to_numpy
from typing import Any


def create_kalman(dt):
    # State: [x, y, z, vx, vy, vz]
    # Transition: x_t = x_(t-1) + vx_(t-1) * dt (will not include acceleration yet)
    # Measurement: [x, y, z]
    kf = cv2.KalmanFilter(6, 3)
    kf.transitionMatrix = np.array(
        [[1, 0, 0, dt, 0, 0],
         [0, 1, 0, 0, dt, 0],
         [0, 0, 1, 0, 0, dt],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1]], 
        dtype=np.float32)
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0]], 
        dtype=np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 5e-2
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    return kf

def init_state(kf, x, y, z, vx=0, vy=0, vz=0):
    kf.statePost = np.array([[x], [y], [z], [vx], [vy], [vz]], dtype=np.float32)

def gravity(kf, dt):
    # y = y0 + vy * t + 0.5 * g * t^2
    # vy = vy0 + g * t
    # Note from this, y will be positive.
    kf.statePre[1] += 0.5 * G * dt**2
    kf.statePre[4] += G * dt

def prediction(kf, dt):
    kf.transitionMatrix[0, 3] = dt
    kf.transitionMatrix[1, 4] = dt
    kf.transitionMatrix[2, 5] = dt
    prediction = kf.predict()
    gravity(kf, dt)
    return prediction

def correct(kf, x, y, z):
    measurement = np.array([[x], [y], [z]], dtype=np.float32)
    return kf.correct(measurement)

def get_state(kf):
    s = kf.statePost.flatten()
    return  {
        'x': s[0],
        'y': s[1],
        'z': s[2],
        'vx': s[3],
        'vy': s[4],
        'vz': s[5]
    }

def get_speed(kf):
    s = kf.statePost.flatten()
    vx = s[3]
    vy = s[4]
    vz = s[5]
    return np.sqrt(vx**2 + vy**2 + vz**2) # Returns speed in ft/s, convert to mph by multiplying by 0.681818

def validate(x, y, z):
    # Can adjust x, y, z constraints (x and y describe ball motion, z is the depth)
    if np.isnan(x) or np.isnan(y) or np.isnan(z):
        return False

    if abs(x) > 25:
        return False

    if y < -5 or y > 25:
        return False

    if z < 0 or z > 65: # 60.5 ft to home plate + some buffer for error
        return False

    return True

def set_state_pre(kf, state):
    #state = [x, y, z, vx, vy, vz]
    kf.statePre = np.array(
        state,
        dtype=np.float32
    ).reshape(6, 1)


def accept_prediction(kf):
    #Use predicted state as current state when no measurement is available.
    kf.statePost = kf.statePre.copy()

# Might not need this, but it was in the old kalman implementation so I included it.
def safe_to_numpy(tensor_like: Any) -> np.ndarray:
    try:
        return tensor_like.cpu().numpy()
    except Exception:
        try:
            return np.array(tensor_like)
        except Exception:
            return np.array([])