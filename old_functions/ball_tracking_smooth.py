from ultralytics import YOLO
import cv2
import csv
import os
import time
import numpy as np
from datetime import datetime

"""
ball_tracking_smooth.py

Detects the ball using the project's YOLO model, smooths the detected
trajectory using a Kalman filter for (x,y) and an exponential moving
average (EMA) for width/height, and writes both raw and smoothed
measurements to a CSV file.

Usage (from repository root):
    python functions/ball_tracking_smooth.py

Outputs:
    - CSV file: ball_path_smoothed_<timestamp>.csv
    - Optional visualization window (press 'q' to quit)

Notes:
    - This builds on the conventions in `functions/ball_tracking.py`.
    - Adjust `VIDEO_PATH`, `MODEL_PATH`, and `CONF_THRESHOLD` below as needed.
"""

# --- User-adjustable params -------------------------------------------------
VIDEO_PATH = "functions/Pitch6-fastball-ok.mp4"
MODEL_PATH = "model/weights.pt"
CONF_THRESHOLD = 0.25
EMA_ALPHA = 0.35   # smoothing factor for width/height
VISUALIZE = True   # show detected + smoothed positions
# ---------------------------------------------------------------------------


def create_kalman():
    # 4 state variables: x, y, vx, vy
    # 2 measurements: x, y
    kf = cv2.KalmanFilter(4, 2)

    # State transition matrix (dt assumed = 1 between frames)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)

    # Measurement matrix maps state to measurement (we measure x and y)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], dtype=np.float32)

    # Reasonable covariances (tune if needed)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    return kf


def main():
    # Load model
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Could not open video '{VIDEO_PATH}'")
        return

    # Prepare CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"ball_path_smoothed_{timestamp}.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'frame',
        'raw_x', 'raw_y', 'raw_w', 'raw_h', 'raw_conf',
        'smooth_x', 'smooth_y', 'smooth_w', 'smooth_h'
    ])

    frame_idx = 0
    kalman = None
    ema_w = None
    ema_h = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optionally resize here if you prefer a fixed size; for now use original
        # frame = cv2.resize(frame, (640, 640))

        # Run detection
        results = model(frame, conf=CONF_THRESHOLD)

        raw_x = raw_y = raw_w = raw_h = raw_conf = ''

        # Select the detection with highest confidence (if any)
        boxes = results[0].boxes
        if boxes is not None and len(boxes.xyxy) > 0:
            # boxes.conf may be a torch tensor; convert safely to numpy
            try:
                confs = boxes.conf.cpu().numpy()
            except Exception:
                confs = np.array(boxes.conf)
            best_idx = int(np.argmax(confs))
            x1, y1, x2, y2 = map(float, boxes.xyxy[best_idx])
            raw_w = float(x2 - x1)
            raw_h = float(y2 - y1)
            raw_x = float(x1 + raw_w / 2.0)
            raw_y = float(y1 + raw_h / 2.0)
            raw_conf = float(confs[best_idx])

        # Initialize Kalman if we have a measurement and no filter yet
        if kalman is None:
            if raw_x != '':
                kalman = create_kalman()
                kalman.statePost = np.array([[raw_x], [raw_y], [0.0], [0.0]], dtype=np.float32)
                # initialize EMA
                ema_w = raw_w
                ema_h = raw_h
        # Predict step (always)
        smooth_x = smooth_y = smooth_w = smooth_h = ''
        if kalman is not None:
            pred = kalman.predict()
            pred_x, pred_y = float(pred[0]), float(pred[1])
            # If we have a measurement, correct the Kalman filter
            if raw_x != '':
                meas = np.array([[np.float32(raw_x)], [np.float32(raw_y)]])
                corrected = kalman.correct(meas)
                smooth_x, smooth_y = float(corrected[0]), float(corrected[1])
            else:
                # no measurement: use prediction as smoothed position
                smooth_x, smooth_y = pred_x, pred_y

            # Smooth width/height using EMA when measurements exist
            if raw_w != '':
                if ema_w is None:
                    ema_w = raw_w
                    ema_h = raw_h
                else:
                    ema_w = EMA_ALPHA * raw_w + (1 - EMA_ALPHA) * ema_w
                    ema_h = EMA_ALPHA * raw_h + (1 - EMA_ALPHA) * ema_h
                smooth_w, smooth_h = float(ema_w), float(ema_h)
            else:
                # if no measurement, keep last EMA values
                smooth_w = float(ema_w) if ema_w is not None else ''
                smooth_h = float(ema_h) if ema_h is not None else ''

        # Write row to CSV
        csv_writer.writerow([
            frame_idx,
            raw_x if raw_x != '' else '',
            raw_y if raw_y != '' else '',
            raw_w if raw_w != '' else '',
            raw_h if raw_h != '' else '',
            raw_conf if raw_conf != '' else '',
            smooth_x if smooth_x != '' else '',
            smooth_y if smooth_y != '' else '',
            smooth_w if smooth_w != '' else '',
            smooth_h if smooth_h != '' else ''
        ])

        # Visualization (optional)
        if VISUALIZE:
            vis = frame.copy()
            # draw raw bbox if present
            if raw_x != '':
                x = int(raw_x)
                y = int(raw_y)
                w = int(raw_w)
                h = int(raw_h)
                cv2.rectangle(vis, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 0, 255), 2)
                cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
            # draw smoothed position
            if smooth_x != '' and smooth_y != '':
                sx = int(round(smooth_x))
                sy = int(round(smooth_y))
                if smooth_w != '' and smooth_h != '':
                    sw = int(round(smooth_w))
                    sh = int(round(smooth_h))
                    cv2.rectangle(vis, (int(sx - sw/2), int(sy - sh/2)), (int(sx + sw/2), int(sy + sh/2)), (0, 255, 0), 2)
                cv2.circle(vis, (sx, sy), 4, (0, 255, 0), -1)

            cv2.putText(vis, f"Frame: {frame_idx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.imshow('Ball Tracking (smoothed)', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    # Cleanup
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()

    print(f"Saved smoothed detections to: {csv_path}")


if __name__ == '__main__':
    main()
