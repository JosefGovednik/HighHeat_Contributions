from ultralytics import YOLO
import cv2
import numpy as np
from scipy.integrate import solve_ivp
import kalman3d as k3d
import equations
from datetime import datetime

# For now we will hard code files for testing.
VIDEO_PATH = "Test_videos/20250923_184300.mp4"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_VIDEO_PATH = f"Test_videos/ball_path_{timestamp}.mp4"
BALL_MODEL_PATH = "model/weights.pt"
RUBBER_MODEL_PATH = "model/weights2.pt"
CONF_THRESHOLD = 0.52
VISUALIZE = True
SAVE_OUTPUT_VIDEO = True
DETECT_SIZE = (720, 720)
OUTPUT_SIZE = (720, 720)

# Get camera and video data from front end. This will be K and fps.
# K, FOCAL_LENGTH_PX given are placeholders.
BASE_SENSOR_FOCAL_LENGTH_PX = 26 / 4.77 / 7.26 * 1920


rubber = None  # Only detect the rubber once. If this is not None, skip rubber detection.
rubber_depth = 0
ball_detected = 0

ball_model = YOLO(BALL_MODEL_PATH)
rubber_model = YOLO(RUBBER_MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
	print(f"Error opening video file: {VIDEO_PATH}")
	exit()

input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if input_width <= 0 or input_height <= 0:
	input_width, input_height = 1920, 1080

ORIGINAL_IMG_SIZE = (input_width, input_height)
FRAME_WIDTH = DETECT_SIZE[0]
FRAME_HEIGHT = DETECT_SIZE[1]
OUTPUT_WIDTH = OUTPUT_SIZE[0]
OUTPUT_HEIGHT = OUTPUT_SIZE[1]
SCALE_X = OUTPUT_WIDTH / FRAME_WIDTH
SCALE_Y = OUTPUT_HEIGHT / FRAME_HEIGHT
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
	video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, output_fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
	if not video_writer.isOpened():
		print(f"Warning: could not open output file for writing: {OUTPUT_VIDEO_PATH}")
		video_writer = None

fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1 / fps if fps and fps > 0 else 1 / 60

kf = k3d.create_kalman(dt=dt)
initialized = False
trajectory = []
trail_pixels = []

while True:
	ret, frame = cap.read()
	if not ret:
		break

	detect_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
	output_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
	ball = None

	# Detect rubber with the rubber model until locked once.
	if rubber is None:
		rubber_results = rubber_model(detect_frame, imgsz=FRAME_WIDTH)[0]
		best_rubber = None
		for box in rubber_results.boxes:
			cls_id = int(box.cls[0])
			conf = float(box.conf[0])
			if conf < CONF_THRESHOLD:
				continue
			name = str(rubber_model.names[cls_id]).lower()
			if name != "rubber":
				continue
			if best_rubber is None or conf > best_rubber[0]:
				best_rubber = (conf, box)

		if best_rubber is not None:
			_, box = best_rubber
			x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
			rubber = (x1, y1, x2, y2)
			rubber_depth = equations.rubber_depth(FOCAL_LENGTH_PX, y2 - y1)
	cv2.rectangle(output_frame, (int(x1 * SCALE_X), int(y1 * SCALE_Y)), (int(x2 * SCALE_X), int(y2 * SCALE_Y)), (255, 0, 0), 2)

	# Detect ball with the ball model each frame.
	ball_results = ball_model(detect_frame, imgsz=FRAME_WIDTH)[0]
	best_ball = None
	for box in ball_results.boxes:
		cls_id = int(box.cls[0])
		conf = float(box.conf[0])
		if conf < CONF_THRESHOLD:
			continue
		name = str(ball_model.names[cls_id]).lower()
		if name not in {"ball", "baseball"}:
			continue
		if best_ball is None or conf > best_ball[0]:
			best_ball = (conf, box)

	if best_ball is not None:
		_, box = best_ball
		x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
		ball = (x1, y1, x2, y2)
		ball_detected = 1

	prediction = k3d.prediction(kf, dt)

	if ball is not None:
		x1, y1, x2, y2 = ball
		x1_out, y1_out, x2_out, y2_out = (
			int(round(x1 * SCALE_X)),
			int(round(y1 * SCALE_Y)),
			int(round(x2 * SCALE_X)),
			int(round(y2 * SCALE_Y)),
		)
		center = ((x1 + x2) / 2, (y1 + y2) / 2)
		u = center[0]
		v = center[1]
		u_out = int(round(u * SCALE_X))
		v_out = int(round(v * SCALE_Y))
		diameter = max(x2 - x1, y2 - y1)
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

		cv2.rectangle(output_frame, (x1_out, y1_out), (x2_out, y2_out), (0, 255, 0), 2)
		trail_pixels.append((u_out, v_out))

	if len(trail_pixels) > 1:
		pts = np.array(trail_pixels, dtype=np.int32)
		cv2.polylines(output_frame, [pts], False, (0, 255, 255), 3, lineType=cv2.LINE_AA)
	if len(trail_pixels) > 0:
		cv2.circle(output_frame, trail_pixels[-1], 4, (0, 255, 255), -1)

	state = k3d.get_state(kf)
	trajectory.append((state['x'], state['y'], state['z']))

	current_state = np.array([state['x'], state['y'], state['z'],
							  state['vx'], state['vy'], state['vz']], dtype=np.float32)
	solution = solve_ivp(equations.derivatives, (0, 0.5), current_state,
						 method='RK45', t_eval=np.linspace(0, 0.5, 25))
	pred_pos = solution.y[:3, :].T

	velo = k3d.get_speed(kf) * 3600 / 5280  # get_speed returns ft/s; convert to mph

	if VISUALIZE:
		cv2.imshow("Processing Debug View", output_frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	if video_writer is not None:
		video_writer.write(output_frame)

cap.release()
if video_writer is not None:
	video_writer.release()
cv2.destroyAllWindows()
