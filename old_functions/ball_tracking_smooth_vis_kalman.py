from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
from kalman import create_kalman, safe_to_numpy
from distance_utils import estimate_abs_distance_ft, establish_pitch_bounds
from detector import Detector
from viz import draw_raw_bbox, draw_smoothed_path, draw_smoothed_box, draw_status_text, draw_frame_text
from playback import PlayerCache
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
VIDEO_PATH = "Test_videos/20250923_183444.mp4"
MODEL_PATH = "model/weights.pt"
FRAME_SIZE = (640, 640)
CONF_THRESHOLD = 0.52
EMA_ALPHA = 0.35
VISUALIZE = True
SAVE_OUTPUT_VIDEO = False


def main():
    # detector wrapper
    det = Detector(cfg.MODEL_PATH, conf=cfg.CONF_THRESHOLD)
    cap = cv2.VideoCapture(cfg.VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Can't open video: {VIDEO_PATH}")
        return

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

    player = PlayerCache()
    video_ended = False
    display_idx = -1

    frame_idx = 0

    def process_and_render(frame, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing):
        """Process a single frame: run detection, update kalman/EMA, compute stop flag, and return vis and updated state."""
        # local copies
        raw_x = raw_y = raw_w = raw_h = ''

        # run detection via Detector wrapper
        xy, confs = det.detect(frame)
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
            # estimate absolute distance (feet) using visible pixel width
            try:
                D = estimate_abs_distance_ft(raw_w, cfg.FRAME_SIZE[0])
                abs_distances.append(D)
                norm = establish_pitch_bounds(abs_distances, plate_distance_ft=cfg.STOP_DISTANCE_FT)
                # if the latest normalized distance reaches or exceeds plate distance,
                # stop drawing the smoothed trajectory (but keep playing video)
                if len(norm) > 0 and norm[-1] >= cfg.STOP_DISTANCE_FT:
                    stop_drawing = True
            except Exception:
                # on any calculation error, don't change drawing behavior
                pass

        vis = frame.copy()
        if raw_x != '':
            rx, ry, rw, rh = int(round(raw_x)), int(round(raw_y)), int(round(raw_w)), int(round(raw_h))
            draw_raw_bbox(vis, rx, ry, rw, rh)

        # Always draw the already-collected smooth path so it remains visible
        if smooth_x != '' and smooth_y != '':
            sx, sy = int(round(smooth_x)), int(round(smooth_y))
            # only append new points while not stopped; once stopped, keep existing path
            if not stop_drawing:
                smooth_path.append((sx, sy))

        # draw the smooth path collected so far (remains on screen after stop)
        if len(smooth_path) > 1:
            draw_smoothed_path(vis, smooth_path)

        # draw current smoothed bbox/marker only when still tracking (not stopped)
        if not stop_drawing and smooth_x != '' and smooth_y != '':
            sx, sy = int(round(smooth_x)), int(round(smooth_y))
            if smooth_w != '' and smooth_h != '':
                sw, sh = int(round(smooth_w)), int(round(smooth_h))
                draw_smoothed_box(vis, sx, sy, sw, sh)
            else:
                draw_smoothed_box(vis, sx, sy, 6, 6)

        if stop_drawing:
            # indicate that drawing has stopped
            draw_status_text(vis, f"Trajectory stopped (>={cfg.STOP_DISTANCE_FT} ft)")

        draw_frame_text(vis, f"Frame: {frame_idx}")

        return vis, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing

    # main loop using PlayerCache
    while True:
        if not player.is_paused():
            # read and process next frame in normal playback
            ret, frame = cap.read()
            if not ret:
                video_ended = True
                player.toggle_pause()
                # if nothing has been processed, break
                if len(player.frames) == 0:
                    break
            else:
                frame = cv2.resize(frame, cfg.FRAME_SIZE)
                vis, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing = process_and_render(
                    frame, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing
                )
                player.append_frame(vis)
                display_idx = len(player.frames) - 1
                frame_idx += 1

            # show current frame
            cur = player.current()
            if cur is not None:
                cv2.imshow('Ball Tracking Smoothed (kalman import)', cur)
            key = cv2.waitKey(1) & 0xFF
        else:
            # paused: show the selected cached frame and block for keypress
            cur = player.current()
            if cur is not None:
                cv2.imshow('Ball Tracking Smoothed (kalman import)', cur)
            key = cv2.waitKey(0) & 0xFF

        # handle keys
        if key == ord('q'):
            break
        if key == ord('s'):
            # pause
            player.toggle_pause()
            # ensure display_idx points to the most recent processed frame
            if len(player.frames) > 0:
                player.idx = min(player.idx, len(player.frames) - 1)
            continue
        if player.is_paused() and key == ord('r'):
            # previous frame
            player.step_prev()
            continue
        if player.is_paused() and key == ord('t'):
            # next frame: either move within cache or process next frame if available
            def process_next_cb():
                nonlocal kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing, frame_idx
                ret2, frame2 = cap.read()
                if not ret2:
                    return None
                frame2 = cv2.resize(frame2, cfg.FRAME_SIZE)
                vis2, kalman2, ema_w2, ema_h2, smooth_path2, abs_distances2, stop_drawing2 = process_and_render(
                    frame2, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing
                )
                # update states
                kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing = kalman2, ema_w2, ema_h2, smooth_path2, abs_distances2, stop_drawing2
                frame_idx += 1
                return vis2

            player.step_next(process_next_cb)
            continue
        # any other key resumes playback
        if player.is_paused():
            player.toggle_pause()

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_idx} frames")


if __name__ == '__main__':
    main()
