from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
from kalman import create_kalman, safe_to_numpy
from distance import estimate_abs_distance_ft, establish_pitch_bounds
import config as cfg
from velo_v3 import compute_velocity_v3, save_velocity_v3
from movement_v3 import compute_movement_v3, save_movement_v3

# configuration (use central config values)
VIDEO_PATH = cfg.VIDEO_PATH
MODEL_PATH = cfg.MODEL_PATH
FRAME_SIZE = cfg.FRAME_SIZE
CONF_THRESHOLD = cfg.CONF_THRESHOLD
EMA_ALPHA = cfg.EMA_ALPHA
VISUALIZE = cfg.VISUALIZE
SAVE_OUTPUT_VIDEO = cfg.SAVE_OUTPUT_VIDEO


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Can't open video: {VIDEO_PATH}")
        return

    print(f"DEBUG: STOP_DISTANCE_FT from config = {cfg.STOP_DISTANCE_FT}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 60.0

    out = None
    video_path_out = None
    if SAVE_OUTPUT_VIDEO:
        t = datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path_out = f"ball_vis_v3_{t}.mp4"
        out = cv2.VideoWriter(video_path_out, fourcc, fps, FRAME_SIZE)
        if not out.isOpened():
            print("WARNING: VideoWriter failed to open; output video will not be saved.")
            out = None

    kalman = None
    ema_w = None
    ema_h = None
    smooth_path = []
    abs_distances = []
    stop_drawing = False
    INITIAL_VEL_FRAMES = 5
    initial_distances = []
    initial_slope = [None]
    extrapolation_count = [0]
    MAX_EXTRAPOLATE_FRAMES = 30

    frames_vis = []
    video_ended = False
    paused = False
    display_idx = -1
    frame_idx = 0

    positions_x_px = []
    positions_y_px = []

    def process_and_render(frame, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing):
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

        if raw_w != '':
            try:
                D = estimate_abs_distance_ft(raw_w, FRAME_SIZE[0])
                abs_distances.append(D)
                if len(initial_distances) < INITIAL_VEL_FRAMES:
                    initial_distances.append(D)
                    if len(initial_distances) >= 2:
                        diffs = np.diff(np.array(initial_distances, dtype=np.float32))
                        initial_slope[0] = float(np.mean(diffs))
                        print(f"DEBUG: initial_slope updated = {initial_slope[0]:.3f} ft/frame (from {len(initial_distances)} frames)")
            except Exception:
                pass
        else:
            try:
                if initial_slope[0] is not None and len(abs_distances) > 0 and extrapolation_count[0] < MAX_EXTRAPOLATE_FRAMES:
                    extrapolated_dist = abs_distances[-1] + initial_slope[0]
                    abs_distances.append(extrapolated_dist)
                    extrapolation_count[0] += 1
                    print(f"DEBUG: extrapolated using initial_slope -> {extrapolated_dist:.2f} ft (count={extrapolation_count[0]})")
                elif len(abs_distances) >= 3:
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

        if smooth_x != '' and smooth_y != '':
            sx, sy = int(round(smooth_x)), int(round(smooth_y))
            if not stop_drawing:
                smooth_path.append((sx, sy))

        if len(smooth_path) > 1:
            pts = np.array(smooth_path, dtype=np.int32)
            cv2.polylines(vis, [pts], False, (0, 255, 0), 2, lineType=cv2.LINE_AA)

        if not stop_drawing and smooth_x != '' and smooth_y != '':
            sx, sy = int(round(smooth_x)), int(round(smooth_y))
            if smooth_w != '' and smooth_h != '':
                sw, sh = int(round(smooth_w)), int(round(smooth_h))
                cv2.rectangle(vis, (int(sx - sw/2), int(sy - sh/2)), (int(sx + sw/2), int(sy + sw/2)), (0, 255, 0), 1)
            cv2.circle(vis, (sx, sy), 4, (0, 255, 0), -1)

        cv2.putText(vis, f"Frame: {frame_idx}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        if smooth_x != '' and smooth_y != '':
            positions_x_px.append(float(smooth_x))
            positions_y_px.append(float(smooth_y))

        return vis, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                video_ended = True
                paused = True
                if len(frames_vis) == 0:
                    break
            else:
                frame = cv2.resize(frame, FRAME_SIZE)
                vis, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing = process_and_render(
                    frame, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing
                )
                frames_vis.append(vis)
                if out is not None:
                    out.write(vis)
                display_idx = len(frames_vis) - 1
                frame_idx += 1

            if display_idx >= 0:
                cv2.imshow('Ball Tracking Smoothed (v3)', frames_vis[display_idx])
            key = cv2.waitKey(1) & 0xFF
        else:
            if display_idx >= 0:
                cv2.imshow('Ball Tracking Smoothed (v3)', frames_vis[display_idx])
            key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        if key == ord('s'):
            paused = True
            if len(frames_vis) > 0:
                display_idx = min(display_idx, len(frames_vis) - 1)
            continue
        if paused and key == ord('r'):
            if display_idx > 0:
                display_idx -= 1
            continue
        if paused and key == ord('t'):
            if display_idx < len(frames_vis) - 1:
                display_idx += 1
            elif not video_ended:
                ret, frame = cap.read()
                if not ret:
                    video_ended = True
                else:
                    frame = cv2.resize(frame, FRAME_SIZE)
                    vis, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing = process_and_render(
                        frame, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing
                    )
                    frames_vis.append(vis)
                    if out is not None:
                        out.write(vis)
                    display_idx = len(frames_vis) - 1
                    frame_idx += 1
            continue
        if paused:
            paused = False

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_idx} frames")

    report_path = "motion_report_v3.txt"
    frame_rate = fps
    try:
        SCALE = cfg.DISTANCE_SCALE_FOR_METRICS
        scaled_distances = [d * SCALE for d in abs_distances]

        # Velocity v3 (3D vector, release angles)
        vel_profile = compute_velocity_v3(positions_x_px, positions_y_px, scaled_distances, frame_rate=frame_rate)
        # Movement v3 (gravity-compensated break) using unscaled distances for physical realism
        mov_profile = compute_movement_v3(positions_x_px, positions_y_px, abs_distances, frame_rate=frame_rate)

        with open(report_path, 'w') as f:
            f.write(f"Velocity (avg): {abs(vel_profile['avg_speed_mph']):.2f} mph\n")
            f.write(f"Release velocity: {abs(vel_profile['release_speed_mph']):.2f} mph\n")
            f.write(f"Release yaw: {vel_profile['yaw_deg']:.2f} deg, pitch: {vel_profile['pitch_deg']:.2f} deg\n")
            f.write(f"Horizontal break: {mov_profile['total_horizontal_break_in']:.2f} in\n")
            f.write(f"Vertical break (beyond gravity): {mov_profile['total_vertical_break_in']:.2f} in\n")
            f.write(f"Observed drop: {mov_profile['observed_vertical_drop_in']:.2f} in (gravity expected: {mov_profile['gravity_drop_in']:.2f} in)\n")

        print(f"V3 Motion report saved to: {report_path}")
    except Exception as e:
        print(f"WARNING: Failed to generate V3 motion report: {e}")
        import traceback
        traceback.print_exc()

    if SAVE_OUTPUT_VIDEO and video_path_out is not None:
        print(f"Output video saved to: {video_path_out}")
        return report_path, video_path_out
    else:
        return report_path, None


if __name__ == '__main__':
    main()
