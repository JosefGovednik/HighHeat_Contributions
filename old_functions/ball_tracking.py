from ultralytics import YOLO
import cv2

# Load the trained YOLO model
model = YOLO("model/weights.pt")

# Load the video file
video_path = "Test_videos/20250923_184300.mp4"  # Replace with any test video file path
cap = cv2.VideoCapture(video_path)

# Output video writer (save the result)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (640, 640))  # Match resized resolution

# List to store ball coordinates for the path
ball_path = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and reorient the frame
    frame = cv2.resize(frame, (640, 640))

    # Run YOLO detection on the frame
    results = model(frame, conf=0.37)  # Lower confidence threshold to improve detection

    # Process detections
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Store the ball's center coordinates
        ball_path.append((center_x, center_y))

        # Draw a blue dot at the center of the detected ball
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

    # Draw the ball's path
    for i in range(1, len(ball_path)):
        cv2.line(frame, ball_path[i - 1], ball_path[i], (0, 255, 0), 2)  # Draw green lines for the path

    # Show the frame
    cv2.imshow("Ball Detection", frame)

    # Write the frame to the output video
    out.write(frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
#print(len(ball_path)) # Debug: Print the length of the ball path