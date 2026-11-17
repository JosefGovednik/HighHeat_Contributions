import numpy as np

def compute_movement_profile(x_positions, y_positions, distances, 
                            frame_width_px=640.0,
                            focal_length_in=4.32/25.4,
                            sensor_width_in=7.26/25.4):
    """
    Computes horizontal (x) and vertical (y) movement in inches per frame,
    using proper pinhole camera geometry scaled by depth.

    Parameters:
        x_positions: list of floats (pixel x positions)
        y_positions: list of floats (pixel y positions)
        distances: list of floats, absolute depth distances (ft)
        frame_width_px: frame width in pixels (default 640)
        focal_length_in: lens focal length in inches
        sensor_width_in: sensor physical width in inches

    Returns:
        movements: list of tuples (delta_x_inches, delta_y_inches)
        total_displacement: tuple (total_dx_inches, total_dy_inches)
    """
    movements = []

    for i in range(1, len(x_positions)):
        delta_x_px = x_positions[i] - x_positions[i-1]
        delta_y_px = y_positions[i] - y_positions[i-1]

        # Calculate pixel-to-physical scale at this depth
        # At distance D: 1 pixel = (D * sensor_width) / (focal_length * frame_width)
        if i < len(distances):
            avg_dist_ft = (distances[i] + distances[i-1]) / 2
            avg_dist_in = avg_dist_ft * 12.0
            # pixels to inches at this distance
            px_to_in = (avg_dist_in * sensor_width_in) / (focal_length_in * frame_width_px)
        else:
            # fallback if distances not available
            px_to_in = 0.5  # rough estimate

        delta_x_in = delta_x_px * px_to_in
        delta_y_in = delta_y_px * px_to_in
        movements.append((delta_x_in, delta_y_in))

    total_dx = sum([dx for dx, dy in movements])
    total_dy = sum([dy for dx, dy in movements])
    return movements, (total_dx, total_dy)


def save_movement_to_file(movements, total_displacement, file_path="movement_profile.txt"):
    with open(file_path, 'w') as f:
        f.write("Frame\tDelta_X(inches)\tDelta_Y(inches)\n")
        for idx, (dx, dy) in enumerate(movements, start=1):
            f.write(f"{idx}\t{dx:.2f}\t{dy:.2f}\n")
        f.write(f"\nTotal Displacement: X={total_displacement[0]:.2f} inches, Y={total_displacement[1]:.2f} inches\n")
