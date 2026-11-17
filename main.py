from ultralytics import YOLO
import cv2
import numpy as np
from scipy.integrate import solve_ivp
import kalman3d as k3d
import equations
from datetime import datetime # For output file naming

# For now we will hard code files for testing.
VIDEO_PATH = "Test_videos/20250923_184300.mp4"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_VIDEO_PATH = f"Test_videos/ball_path_{timestamp}.mp4"
MODEL_PATH = "model/weights2.pt"
CONF_THRESHOLD = 0.52
VISUALIZE = True
SAVE_OUTPUT_VIDEO = True
STOP_DISTANCE_FT = 60.5
# Get camera and video data from front end. This will be K and fps.
# K, FOCAL_LENGTH_PX given are placeholders.
BASE_SENSOR_FOCAL_LENGTH_PX = 26 / 4.77 / 7.26 * 1920


rubber = None # Only detect the rubber once for rubber measurements. If this is not None, skip the detection.
rubber_depth = 0 # Update this when rubber is detected.
ball_deteced = 0 # Use as a switch. 0 = no ball detected ever. 1 = ball detected, 2 = ball lost after detection.
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error opening video file: {VIDEO_PATH}")
    exit()

input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if input_width <= 0 or input_height <= 0:
    input_width, input_height = 1920, 1080

ORIGINAL_IMG_SIZE = (input_width, input_height)
IMG_SIZE = (int(input_width / 2), int(input_height / 2))
FRAME_WIDTH = IMG_SIZE[0]
FRAME_HEIGHT = IMG_SIZE[1]
FOCAL_LENGTH_PX = BASE_SENSOR_FOCAL_LENGTH_PX * (FRAME_WIDTH / ORIGINAL_IMG_SIZE[0])
K = np.array([[FOCAL_LENGTH_PX, 0, FRAME_WIDTH / 2],
              [0, FOCAL_LENGTH_PX, FRAME_HEIGHT / 2],
              [0, 0, 1]])

video_writer = None
if SAVE_OUTPUT_VIDEO:
    output_fps = cap.get(cv2.CAP_PROP_FPS)
    if output_fps <= 0:
        output_fps = 60.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, output_fps, (FRAME_WIDTH, FRAME_HEIGHT))
    if not video_writer.isOpened():
        print(f"Warning: could not open output file for writing: {OUTPUT_VIDEO_PATH}")
        video_writer = None

fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1 / fps if fps and fps > 0 else 1/60

kf = k3d.create_kalman(dt=dt)
initialized = False
trajectory = [] # real trajectory
trail_pixels = [] # image tracjectory to draw out ball pathi


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    ball = None
    detections = model(frame, imgsz=640)[0]
    for box in detections.boxes:
        cv2.rectangle(frame, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
        id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Confidence for rubber detection: {conf:.2f}")
        if conf < CONF_THRESHOLD:
            continue
        name = model.names[id]
        if name == 'rubber' and rubber is None:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() # Store rubber bbox for depth estimation
            rubber = (x1, y1, x2, y2)
            rubber_depth = equations.rubber_depth(FOCAL_LENGTH_PX, y2-y1)
        elif name == 'ball':
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() # Get ball bbox for depth estimation
            ball = (x1, y1, x2, y2)
            ball_detected = 1
        
    prediction = k3d.prediction(kf, dt)

    if ball is not None:
        x1, y1, x2, y2 = ball
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        u = center[0]
        v = center[1]
        diameter = min(x2 - x1, y2 - y1)
        focal_length = K[0, 0]
        ball_depth = equations.ball_depth(focal_length, diameter)
        z = max(0, ball_depth - rubber_depth)
        coordinates = equations.xyz(u, v, z, K)
        x, y, z = coordinates
        if k3d.validate(x, y, z):
            if not initialized:
                k3d.init_state(kf, x, y, z)
                initialized = True
            else:
                k3d.correct(kf, x, y, z)
            
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        trail_pixels.append((int(u), int(v)))

    for i in range(1, len(trail_pixels)):
        cv2.line(frame, trail_pixels[i-1], trail_pixels[i], (0, 255, 255), 2)

    state = k3d.get_state(kf)
    trajectory.append((state['x'], state['y'], state['z']))

    current_state = np.array([state['x'], state['y'], state['z'],
                                state['vx'], state['vy'], state['vz']], dtype=np.float32)
    solution = solve_ivp(equations.derivatives, (0, 0.5), current_state,
                            method='RK45', t_eval=np.linspace(0, 0.5, 25))
    pred_pos = solution.y[:3, :].T

    velo = k3d.get_speed(kf) * 3600 / 5280 # get_speed returns in ft/s, convert to mph

    if VISUALIZE:
        cv2.imshow("Processing Debug View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if video_writer is not None:
        video_writer.write(frame)

cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()