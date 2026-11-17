from ultralytics import YOLO
import cv2

def process_video(video_path, model_path, output_path, conf_threshold=0.2):
    """
    Processes a video to detect and track a ball, and returns the output video path.

    Args:
        video_path (str): Path to the input video file.
        model_path (str): Path to the YOLO model weights file.
        output_path (str): Path to save the output video.
        conf_threshold (float): Confidence threshold for YOLO detection.

    Returns:
        str: Path to the output video.
    """
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 360))  # Match resized resolution

    # List to store ball coordinates for the path
    ball_path = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and reorient the frame
        frame = cv2.resize(frame, (640, 360))
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Run YOLO detection on the frame
        results = model(frame, conf=conf_threshold)  # Use the confidence threshold

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

        # Write the frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Return the path to the output video
    return output_path