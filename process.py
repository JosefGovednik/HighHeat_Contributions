import cv2
import numpy as np
from kalman import safe_to_numpy
from distance import estimate_abs_distance_ft, establish_pitch_bounds
import config as cfg


def process_and_render(frame, model, kalman, ema_w, ema_h, smooth_path, abs_distances, 
                       stop_drawing, initial_distances, initial_slope, extrapolation_count,
                       positions_x_px, positions_y_px, frame_idx):

    from kalman import create_kalman
    
    # local copies
    raw_x = raw_y = raw_w = raw_h = ''

    results = model(frame, conf=cfg.CONF_THRESHOLD)
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
                ema_w = cfg.EMA_ALPHA * raw_w + (1 - cfg.EMA_ALPHA) * ema_w
                ema_h = cfg.EMA_ALPHA * raw_h + (1 - cfg.EMA_ALPHA) * ema_h
            smooth_w, smooth_h = float(ema_w), float(ema_h)
        else:
            smooth_w = float(ema_w) if ema_w is not None else ''
            smooth_h = float(ema_h) if ema_h is not None else ''

    # Distance estimation and stop condition
    if raw_w != '':
        # Ball detected: add its measured distance
        try:
            D = estimate_abs_distance_ft(raw_w, cfg.FRAME_SIZE[0])
            abs_distances.append(D)
            # capture initial velocities from the first few detections
            if len(initial_distances) < 5:  # INITIAL_VEL_FRAMES constant
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
            if initial_slope[0] is not None and len(abs_distances) > 0 and extrapolation_count[0] < 30:  # MAX_EXTRAPOLATE_FRAMES constant
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

    cv2.putText(vis, f"Frame: {frame_idx}", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # After smoothing, collect positions for analysis
    if smooth_x != '' and smooth_y != '':
        # Store pixel positions for X,Y movement
        positions_x_px.append(float(smooth_x))
        positions_y_px.append(float(smooth_y))

    return vis, kalman, ema_w, ema_h, smooth_path, abs_distances, stop_drawing
