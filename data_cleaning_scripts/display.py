def extrapolate_and_display(video_path, ball_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 360))  # Match resized resolution

    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and reorient the frame
        frame = cv2.resize(frame, (640, 360))
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Draw the ball's path
        for i in range(1, len(ball_path)):
            cv2.line(frame, ball_path[i - 1], ball_path[i], (0, 255, 0), 2)  # Draw green lines for the path

        # Extrapolate the ball's path to the catcher
        if len(ball_path) > 2:
            # Use the last 3 points to calculate the trajectory
            x_coords = [p[0] for p in ball_path[-3:]]
            y_coords = [p[1] for p in ball_path[-3:]]
            poly = np.polyfit(y_coords, x_coords, 1)  # Fit a linear model (y = mx + b)
            poly_fn = np.poly1d(poly)

            # Extrapolate to the bottom of the frame (catcher's position)
            catcher_y = 360  # Assuming the catcher is at the bottom of the frame
            catcher_x = int(poly_fn(catcher_y))
            cv2.circle(frame, (catcher_x, catcher_y), 8, (0, 0, 255), -1)  # Draw a red dot for the catcher

        # Show the frame
        cv2.imshow("Ball Path and Extrapolation", frame)

        # Write the frame to the output video
        out.write(frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit the program
            break

        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()