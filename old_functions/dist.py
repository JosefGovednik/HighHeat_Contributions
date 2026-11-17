from ultralytics import YOLO
import cv2
import numpy as np

# -------------------------------------------------------
# Distance math functions (American units)
# -------------------------------------------------------

def estimate_abs_distance_ft(
    ball_px_width,
    frame_width_px,
    ball_diameter_in=2.9,
    focal_length_in=26 / 25.4,
    sensor_width_in=9.5 / 25.4
):
    frac = ball_px_width / frame_width_px
    ball_sensor_in = frac * sensor_width_in
    distance_in = focal_length_in * (ball_diameter_in / ball_sensor_in)
    return distance_in / 12.0


def establish_pitch_bounds(abs_distances):
    """
    Maps absolute distances to:
    - first frame = 0 ft (release point)
    - last frame = 60.5 ft (home plate)
    """
    if len(abs_distances) < 2:
        return abs_distances

    farthest = abs_distances[0]
    closest  = abs_distances[-1]
    total_travel = farthest - closest

    if total_travel <= 0:
        return abs_distances

    scale_factor = 60.5 / total_travel
    return [(farthest - d) * scale_factor for d in abs_distances]


# -------------------------------------------------------
# Your YOLO tracking code with the updated function name
# -------------------------------------------------------

model = YOLO("model/weights.pt")
video_path = "Test_videos/20250923_183444.mp4"
cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (640, 640))

ball_path = []
abs_distances = []
norm_distances = []

frame_width_px = 640  # due to your 640x640 resize

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 640))
    results = model(frame, conf=0.37)

    detected_ball_width = None

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        ball_path.append((cx, cy))

        ball_px_width = x2 - x1
        detected_ball_width = ball_px_width

        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # Distance calculation
    if detected_ball_width:
        D = estimate_abs_distance_ft(detected_ball_width, frame_width_px)
        abs_distances.append(D)

        # NEW function name used here
        norm_distances = establish_pitch_bounds(abs_distances)

        if len(norm_distances) > 0:
            text = f"Distance: {norm_distances[-1]:.2f} ft"
            cv2.putText(frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Path drawing
    for i in range(1, len(ball_path)):
        cv2.line(frame, ball_path[i - 1], ball_path[i], (0, 255, 0), 2)

    cv2.imshow("Ball Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#print last distance for debug
if len(norm_distances) > 0:
    print(f"Last normalized distance: {norm_distances[-1]:.2f} ft")
if len(abs_distances) > 0:
    print(f"Last absolute distance: {abs_distances[-1]:.2f} ft")

cap.release()
out.release()
cv2.destroyAllWindows()
