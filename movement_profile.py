# movement_profile.py
import numpy as np

def compute_movement_profile(x_positions, y_positions, distances):
    """
    Computes horizontal and vertical displacement per frame in inches, accounting for depth.
    
    Parameters:
        x_positions: list of floats (pixel x positions)
        y_positions: list of floats (pixel y positions)
        distances: list of floats (normalized distances in feet, accounting for depth)
    
    Returns:
        movements: list of tuples (delta_x_inches, delta_y_inches) per frame
        total_displacement: tuple (total_dx_inches, total_dy_inches)
    """
    movements = []
    
    # Use distances to compute a scaling factor for each frame
    # The farther away (higher distance), the more a pixel represents in real space
    for i in range(1, len(x_positions)):
        # Pixel deltas
        delta_x_px = x_positions[i] - x_positions[i-1]
        delta_y_px = y_positions[i] - y_positions[i-1]
        
        # Use average distance between frames for scaling
        if i < len(distances):
            avg_dist = (distances[i] + distances[i-1]) / 2 if i > 0 else distances[i]
            # Rough approximation: scale factor increases with distance
            # This is a simplified model; for precise results, use camera calibration
            scale_factor = avg_dist / 60.0  # normalize to typical pitch distance
            delta_x_ft = delta_x_px * scale_factor * 0.1  # rough px to ft conversion
            delta_y_ft = delta_y_px * scale_factor * 0.1
        else:
            delta_x_ft = 0.0
            delta_y_ft = 0.0
        
        # Convert feet to inches
        delta_x_inches = delta_x_ft * 12.0
        delta_y_inches = delta_y_ft * 12.0
        movements.append((delta_x_inches, delta_y_inches))
    
    total_dx = sum([dx for dx, dy in movements])
    total_dy = sum([dy for dx, dy in movements])
    return movements, (total_dx, total_dy)

def save_movement_to_file(movements, total_displacement, file_path="movement_profile.txt"):
    with open(file_path, 'w') as f:
        f.write("Frame\tDelta_X(inches)\tDelta_Y(inches)\n")
        for idx, (dx, dy) in enumerate(movements, start=1):
            f.write(f"{idx}\t{dx:.2f}\t{dy:.2f}\n")
        f.write(f"\nTotal Displacement: X={total_displacement[0]:.2f} inches, Y={total_displacement[1]:.2f} inches\n")
