from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
from kalman import create_kalman, safe_to_numpy
from distance_utils import estimate_abs_distance_ft, establish_pitch_bounds
import config as cfg

"""
ball_tracking_smooth_vis_kalman.py

Copy of `ball_tracking_smooth_vis_no_csv.py` but imports Kalman helpers
from `kalman.py`. Keeps the original file unmodified and demonstrates
how to reuse the Kalman utilities.

Run:
    python functions/ball_tracking_smooth_vis_kalman.py
"""

# same configuration as the original visualizer
VIDEO_PATH = cfg.VIDEO_PATH
MODEL_PATH = "model/weights.pt"
FRAME_SIZE = (640, 640)
CONF_THRESHOLD = 0.52
EMA_ALPHA = 0.35
VISUALIZE = True
SAVE_OUTPUT_VIDEO = False


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Can't open video: {VIDEO_PATH}")
        return

    # Debug: print the configured stop distance
    print(f"DEBUG: STOP_DISTANCE_FT from config = {cfg.STOP_DISTANCE_FT}")

    # optional output video
    out = None
    if SAVE_OUTPUT_VIDEO:
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"ball_vis_{t}.mp4", fourcc, 30.0, FRAME_SIZE)

    kalman = None
    ema_w = None
    ema_h = None
    smooth_path = []  # list of (x,y) smoothed centers for drawing
    abs_distances = []
    stop_drawing = False
    # Extrapolation / initial velocity params
    INITIAL_VEL_FRAMES = 5  # use first 2-5 frames to compute initial velocity
    initial_distances = []  # distances captured in the initial frames
    initial_slope = [None]    # distance/frame estimated from initial frames (use list so nested fn can modify)
    extrapolation_count = [0]  # count frames since last detection (wrapped in list for mutability)
    MAX_EXTRAPOLATE_FRAMES = 30  # predict up to this many frames after loss of detection

    frames_vis = []  # cached visualization frames for pause/step support
    video_ended = False
    paused = False
    display_idx = -1

    frame_idx = 0

    def process_and_render(frame, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing):
        """Process a single frame: run detection, update kalman/EMA, compute stop flag, and return vis and updated state."""
        # local copies
        raw_x = raw_y = raw_w = raw_h = ''

        results = model(frame, conf=CONF_THRESHOLD)
        boxes = results[0].boxes
        xy = safe_to_numpy(getattr(boxes, 'xyxy', []))
        confs = safe_to_numpy(getattr(boxes, 'conf', []))
        if xy.size > 0 and confs.size > 0:
            best_idx = int(np.argmax(confs))
            x1, y1, x2, y2 = xy[best_idx]
            raw_w = float(x2 - x1)
            raw_h = float(y2 - y1)
            raw_x = float(x1 + raw_w / 2.0)
            raw_y = float(y1 + raw_h / 2.0)

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

        # Distance estimation and stop condition
        if raw_w != '':
            # Ball detected: add its measured distance
            try:
                D = estimate_abs_distance_ft(raw_w, FRAME_SIZE[0])
                abs_distances.append(D)
                # capture initial velocities from the first few detections
                if len(initial_distances) < INITIAL_VEL_FRAMES:
                    initial_distances.append(D)
                    if len(initial_distances) >= 2:
                        # simple mean per-frame slope from the initial window
                        diffs = np.diff(np.array(initial_distances, dtype=np.float32))
                        initial_slope[0] = float(np.mean(diffs))
                        print(f"DEBUG: initial_slope updated = {initial_slope[0]:.3f} ft/frame (from {len(initial_distances)} frames)")
            except Exception:
                pass
        else:
            # detection lost: try to extrapolate distance
            try:
                if initial_slope[0] is not None and len(abs_distances) > 0 and extrapolation_count[0] < MAX_EXTRAPOLATE_FRAMES:
                    extrapolated_dist = abs_distances[-1] + initial_slope[0]
                    abs_distances.append(extrapolated_dist)
                    extrapolation_count[0] += 1
                    print(f"DEBUG: extrapolated using initial_slope -> {extrapolated_dist:.2f} ft (count={extrapolation_count[0]})")
                elif len(abs_distances) >= 3:
                    # fallback: short-term linear fit from the last few measurements
                    recent = abs_distances[-3:]
                    x_data = np.array([0, 1, 2], dtype=np.float32)
                    y_data = np.array(recent, dtype=np.float32)
                    z = np.polyfit(x_data, y_data, 1)
                    slope = z[0]
                    extrapolated_dist = abs_distances[-1] + slope
                    abs_distances.append(extrapolated_dist)
                    extrapolation_count[0] += 1
                    print(f"DEBUG: extrapolated using recent_slope -> {extrapolated_dist:.2f} ft (count={extrapolation_count[0]})")
            except Exception:
                pass

        # Check stop condition
        if len(abs_distances) > 0:
            try:
                norm = establish_pitch_bounds(abs_distances)
                if len(norm) > 0:
                    is_extrapolating = raw_w == ''
                    print(f"DEBUG: norm[-1]={norm[-1]:.2f}, STOP_DISTANCE_FT={cfg.STOP_DISTANCE_FT}, should_stop={norm[-1] >= cfg.STOP_DISTANCE_FT}, extrapolating={is_extrapolating}")
                    if norm[-1] >= cfg.STOP_DISTANCE_FT:
                        stop_drawing = True
            except Exception:
                pass

        vis = frame.copy()
        if raw_x != '':
            rx, ry, rw, rh = int(round(raw_x)), int(round(raw_y)), int(round(raw_w)), int(round(raw_h))
            cv2.rectangle(vis, (int(rx - rw/2), int(ry - rh/2)), (int(rx + rw/2), int(ry + rh/2)), (0, 0, 255), 1)
            cv2.circle(vis, (rx, ry), 3, (0, 0, 255), -1)

        # Always draw the already-collected smooth path so it remains visible
        if smooth_x != '' and smooth_y != '':
            sx, sy = int(round(smooth_x)), int(round(smooth_y))
            # only append new points while not stopped; once stopped, keep existing path
            if not stop_drawing:
                smooth_path.append((sx, sy))

        # draw the smooth path collected so far (remains on screen after stop)
        if len(smooth_path) > 1:
            pts = np.array(smooth_path, dtype=np.int32)
            cv2.polylines(vis, [pts], False, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        # draw current smoothed bbox/marker only when still tracking (not stopped)
        if not stop_drawing and smooth_x != '' and smooth_y != '':
            sx, sy = int(round(smooth_x)), int(round(smooth_y))
            if smooth_w != '' and smooth_h != '':
                sw, sh = int(round(smooth_w)), int(round(smooth_h))
                cv2.rectangle(vis, (int(sx - sw/2), int(sy - sh/2)), (int(sx + sw/2), int(sy + sh/2)), (0, 255, 0), 1)
            cv2.circle(vis, (sx, sy), 4, (0, 255, 0), -1)

        if stop_drawing:
            # indicate that drawing has stopped
            cv2.putText(vis, f"Trajectory stopped (>={cfg.STOP_DISTANCE_FT} ft)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

        cv2.putText(vis, f"Frame: {frame_idx}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return vis, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing

    while True:
        if not paused:
            # read and process next frame in normal playback
            ret, frame = cap.read()
            if not ret:
                video_ended = True
                paused = True
                # if nothing has been processed, break
                if len(frames_vis) == 0:
                    break
            else:
                frame = cv2.resize(frame, FRAME_SIZE)
                vis, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing = process_and_render(
                    frame, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing
                )
                frames_vis.append(vis)
                display_idx = len(frames_vis) - 1
                frame_idx += 1

            # show current frame
            if display_idx >= 0:
                cv2.imshow('Ball Tracking Smoothed (kalman import)', frames_vis[display_idx])
            key = cv2.waitKey(1) & 0xFF
        else:
            # paused: show the selected cached frame and block for keypress
            if display_idx >= 0:
                cv2.imshow('Ball Tracking Smoothed (kalman import)', frames_vis[display_idx])
            key = cv2.waitKey(0) & 0xFF

        # handle keys
        if key == ord('q'):
            break
        if key == ord('s'):
            # toggle pause/play
            paused = True
            # ensure display_idx points to the most recent processed frame
            if len(frames_vis) > 0:
                display_idx = min(display_idx, len(frames_vis) - 1)
            continue
        if paused and key == ord('r'):
            # previous frame
            if display_idx > 0:
                display_idx -= 1
            continue
        if paused and key == ord('t'):
            # next frame: either move within cache or process next frame if available
            if display_idx < len(frames_vis) - 1:
                display_idx += 1
            elif not video_ended:
                # process one more frame on demand
                ret, frame = cap.read()
                if not ret:
                    video_ended = True
                else:
                    frame = cv2.resize(frame, FRAME_SIZE)
                    vis, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing = process_and_render(
                        frame, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing
                    )
                    frames_vis.append(vis)
                    display_idx = len(frames_vis) - 1
                    frame_idx += 1
            continue
        # any other key resumes playback
        if paused:
            paused = False

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_idx} frames")


if __name__ == '__main__':
    main()
