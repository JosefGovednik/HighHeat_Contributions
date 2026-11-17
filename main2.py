# DO NOT PUSH TO GITHUB. Transfer findings to main.py
from ultralytics import YOLO
import cv2
import numpy as np
from scipy.integrate import solve_ivp
import kalman3d as k3d
import equations
from datetime import datetime # For output file naming
from collections import deque

VIDEO_PATH = "Test_videos/20250923_184300.mp4"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_VIDEO_PATH = f"Test_videos/ball_path_{timestamp}.mp4"
OUTPUT_REPORT_PATH = f"Test_videos/ball_report_{timestamp}.txt"
BALL_MODEL_PATH = "model/weights.pt"
RUBBER_MODEL_PATH = "model/weights2.pt"
CONF = 0.52
VISUALIZE = True
SAVE_OUTPUT_VIDEO = False
FIRST_N = 2 # Frames to use for velocity estimation.
DETECT_SIZE = (720, 720) # Eventually, we want to keep this consistent with the original video. The model works best at 1:1.
STOP_DISTANCE_FT = 60.5
# Get camera and video data from front end. This will be K and fps.
# K, FOCAL_LENGTH_PX given are placeholders.
BASE_SENSOR_FOCAL_LENGTH_PX = 26 / 4.77 / 7.26 * 1920
K = np.array([[BASE_SENSOR_FOCAL_LENGTH_PX, 0, DETECT_SIZE[0] / 2],
              [0, BASE_SENSOR_FOCAL_LENGTH_PX, DETECT_SIZE[1] / 2],
              [0, 0, 1]])

rubber = None # Only detect the rubber once for rubber measurements. If this is not None, skip the detection.
rubber_depth = 0 # Update this when rubber is detected.
ball_deteced = 0 # Use as a switch. 0 = no ball detected ever. 1 = ball detected, 2 = ball lost after detection.
rubber_model = YOLO(RUBBER_MODEL_PATH)
ball_model = YOLO(BALL_MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error opening video file: {VIDEO_PATH}")
    exit()
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 60.0
dt = 1 / fps

# The input width and height are for future implementation. Use DETECT_SIZE for now, uncomment when ready.
#width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video processing
video_writer = None
if SAVE_OUTPUT_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, DETECT_SIZE)
    if not video_writer.isOpened():
        print(f"Warning: could not open output file for writing: {OUTPUT_VIDEO_PATH}")
        video_writer = None

kf = k3d.create_kalman(dt=dt)
initialized = False
real_traj = [] # x, y, z
pix_traj = [] # Draw out ball path
rubber = None
rubber_depth = None
detected_frames = deque(maxlen=FIRST_N) # Store the first N detections for velocity estimation.
release_point = None
plate_point = None
tracking_complete = False
oohv = 0 # Out-of-hand velocity

def physics_predict(kf, dt):
    state = k3d.get_state(kf)

    current_state = np.array([
        state['x'],
        state['y'],
        state['z'],
        state['vx'],
        state['vy'],
        state['vz']
    ], dtype=np.float32)

    sol = solve_ivp(
        equations.derivatives,
        (0, dt),
        current_state,
        method='RK45',
        t_eval=[dt]
    )

    return sol.y[:, -1]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, DETECT_SIZE) #Eventually we can get rid of this when we get the model working outside 1:1 ratios.
    ball = None
    ball_detection = ball_model(frame, imgsz=DETECT_SIZE, conf=CONF)[0]
    rubber_detection = rubber_model(frame, imgsz=DETECT_SIZE, conf=CONF)[0]

    # Rubber detection and depth estimation. Only done once since the rubber doesn't move.
    for box in rubber_detection.boxes:
        name = rubber_model.names[int(box.cls[0])]
        if name == "rubber" and rubber is None:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            rubber = (x1, y1, x2, y2)
            rubber_depth = equations.rubber_depth(rubber, K)
            print(f"Rubber detected at depth: {rubber_depth} ft")
    
    # Ball detection and tracking
    if box in ball_detection.boxes:
        ball_detected = 1 # Ball detected. Use ball data for modeling and calulations.
    elif ball_detected == 1 and box not in ball_detection.boxes:
        ball_detected = 2 # Ball lost. Use existing data to predict trajectory.

    for box in ball_detection.boxes:
        conf = float(box.conf[0])
        if conf < CONF:
            ball_detected = 2 # Register as ball lost.
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        ball = (x1, y1, x2, y2)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        u = center[0]
        v = center[1]
        pix_traj.append((int(u), int(v)))
        for i in range(1, len(pix_traj)):
            cv2.line(
                frame,
                pix_traj[i - 1],
                pix_traj[i],
                (0, 255, 255),  # yellow
                2
            )
        if video_writer is not None:
            video_writer.write(frame)

        if VISUALIZE:
            cv2.imshow("Pitch Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        diameter_px = min(x2 - x1, y2 - y1)
        focal_length_px = K[0, 0]
        ball_depth = equations.ball_depth(focal_length_px, diameter_px)
        z = max(0, ball_depth - rubber_depth)
        coordinates = equations.xyz(u, v, z, K)

        # Temporary for some results. Use Kalman filter and physics to get more accurate results.
        detected_frames.append(coordinates)
        if len(detected_frames) == FIRST_N and oohv == 0:
            # Use equations.velocity to get out-of-hand velocity. Can pivot to a kalman filter estimation
            oohv = equations.velocity(detected_frames, fps * (FIRST_N - 1)) * 0.681818

        x, y, z = coordinates
        if k3d.validate(x, y, z):
            if not initialized:
                k3d.init_state(kf, x, y, z)
                initialized = True
    
            # Physics next-frame estimation and adjustments
            prediction = k3d.prediction(kf, dt)
            ideal = physics_predict(kf, dt)
            k3d.set_state_pre(kf, ideal)
            if ball_detected == 1:
                k3d.correct(kf, x, y, z)
            else:
                k3d.accept_prediction(kf)
            
            state = k3d.get_state(kf)
            if release_point is None:
                release_point = np.array([state['x'], state['y'], state['z']])
            
            if (not tracking_complete and state['z'] >= STOP_DISTANCE_FT):
                plate_point = np.array([state['x'], state['y'], state['z']])
                tracking_complete = True

            real_traj.append((state['x'], state['y'], state['z']))

            # Calculate out-of-hand velocity
            oohv = k3d.get_speed(kf) * 0.681818


horizontal_break = None
vertical_break = None

if (
    release_point is not None
    and plate_point is not None
):
    horizontal_break = (
        plate_point[0]
        - release_point[0]
    ) * 12

    vertical_break = (
        plate_point[1]
        - release_point[1]
    ) * 12

    with open(OUTPUT_REPORT_PATH, "w") as f:
        f.write(f"Release Velocity: {oohv:.2f} mph\n")

        if horizontal_break is not None:
            f.write(
                f"Horizontal Movement: "
                f"{horizontal_break:.2f} in\n")

            f.write(
                f"Vertical Movement: "
                f"{vertical_break:.2f} in\n"
            )
