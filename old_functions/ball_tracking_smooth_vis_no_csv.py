from ultralytics import YOLO
import cv2
import numpy as np

"""
ball_tracking_smooth_vis_no_csv.py

Same as `ball_tracking_smooth_vis.py` but WITHOUT CSV output.
Forces frame resize to (640,640), runs YOLO detection, smooths trajectory
with Kalman filter, and draws the smoothed path as a continuous curve (polyline).

Run from repo root:
    python functions/ball_tracking_smooth_vis_no_csv.py

Press 'q' in the window to stop.

"""

# User config
VIDEO_PATH = "Test_videos/20250923_184300.mp4"
MODEL_PATH = "model/weights.pt"
FRAME_SIZE = (640, 640)
CONF_THRESHOLD = 0.52
EMA_ALPHA = 0.35
VISUALIZE = True
SAVE_OUTPUT_VIDEO = False  # set True to save annotated video


def create_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf


def safe_to_numpy(tensor_like):
    """Convert various result container types to numpy arrays."""
    try:
        return tensor_like.cpu().numpy()
    except Exception:
        try:
            return np.array(tensor_like)
        except Exception:
            return np.array([])


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Can't open video: {VIDEO_PATH}")
        return

    # optional output video
    out = None
    if SAVE_OUTPUT_VIDEO:
        import datetime
        t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"ball_vis_{t}.mp4", fourcc, 30.0, FRAME_SIZE)

    kalman = None
    ema_w = None
    ema_h = None
    smooth_path = []  # list of (x,y) smoothed centers for drawing

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # force resize to fixed frame size
        frame = cv2.resize(frame, FRAME_SIZE)

        results = model(frame, conf=CONF_THRESHOLD)
        raw_x = raw_y = raw_w = raw_h = ''

        boxes = results[0].boxes
        # extract numeric arrays safely
        xy = safe_to_numpy(getattr(boxes, 'xyxy', []))
        confs = safe_to_numpy(getattr(boxes, 'conf', []))
        if xy.size > 0 and confs.size > 0:
            best_idx = int(np.argmax(confs))
            x1, y1, x2, y2 = xy[best_idx]
            raw_w = float(x2 - x1)
            raw_h = float(y2 - y1)
            raw_x = float(x1 + raw_w / 2.0)
            raw_y = float(y1 + raw_h / 2.0)

        # initialize filter and ema
        if kalman is None and raw_x != '':
            kalman = create_kalman()
            kalman.statePost = np.array([[raw_x], [raw_y], [0.0], [0.0]], dtype=np.float32)
            ema_w = raw_w
            ema_h = raw_h

        smooth_x = smooth_y = smooth_w = smooth_h = ''
        if kalman is not None:
            pred = kalman.predict()
            pred_x, pred_y = float(pred[0]), float(pred[1])
            if raw_x != '':
                meas = np.array([[np.float32(raw_x)], [np.float32(raw_y)]])
                corr = kalman.correct(meas)
                smooth_x, smooth_y = float(corr[0]), float(corr[1])
            else:
                smooth_x, smooth_y = pred_x, pred_y

            if raw_w != '':
                if ema_w is None:
                    ema_w = raw_w
                    ema_h = raw_h
                else:
                    ema_w = EMA_ALPHA * raw_w + (1 - EMA_ALPHA) * ema_w
                    ema_h = EMA_ALPHA * raw_h + (1 - EMA_ALPHA) * ema_h
                smooth_w, smooth_h = float(ema_w), float(ema_h)
            else:
                smooth_w = float(ema_w) if ema_w is not None else ''
                smooth_h = float(ema_h) if ema_h is not None else ''

        # visualization
        vis = frame.copy()
        if raw_x != '':
            rx, ry, rw, rh = int(round(raw_x)), int(round(raw_y)), int(round(raw_w)), int(round(raw_h))
            cv2.rectangle(vis, (int(rx - rw/2), int(ry - rh/2)), (int(rx + rw/2), int(ry + rh/2)), (0, 0, 255), 1)
            cv2.circle(vis, (rx, ry), 3, (0, 0, 255), -1)

        if smooth_x != '' and smooth_y != '':
            sx, sy = int(round(smooth_x)), int(round(smooth_y))
            smooth_path.append((sx, sy))
            # draw smoothed path as polyline
            if len(smooth_path) > 1:
                pts = np.array(smooth_path, dtype=np.int32)
                cv2.polylines(vis, [pts], False, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            # draw current smoothed bbox if available
            if smooth_w != '' and smooth_h != '':
                sw, sh = int(round(smooth_w)), int(round(smooth_h))
                cv2.rectangle(vis, (int(sx - sw/2), int(sy - sh/2)), (int(sx + sw/2), int(sy + sh/2)), (0, 255, 0), 1)
            cv2.circle(vis, (sx, sy), 4, (0, 255, 0), -1)

        cv2.putText(vis, f"Frame: {frame_idx}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if VISUALIZE:
            cv2.imshow('Ball Tracking Smoothed (vis)', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if out is not None:
            out.write(vis)

        frame_idx += 1

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_idx} frames")


if __name__ == '__main__':
    main()
