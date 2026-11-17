# velo.py
import numpy as np

def compute_velocity(distances, frame_rate=60.0):
    """
    Computes per-frame velocity (mph) from a list of normalized distances (ft).
    
    Parameters:
        distances: list of floats, distances per frame (normalized, in feet)
        frame_rate: video frames per second
    
    Returns:
        velocities: list of floats, velocity per frame in ft/sec
        avg_velocity: float, average velocity over the pitch
    """
    velocities = []
    for i in range(1, len(distances)):
        delta_d = distances[i] - distances[i-1]  # ft
        delta_t = 1.0 / frame_rate  # sec per frame
        velocity_ft_s = delta_d / delta_t  # ft/s
        velocity_mph = velocity_ft_s * 0.681818  # convert to mph (1 ft/s = 0.681818 mph)
        velocities.append(velocity_mph)
    
    avg_velocity = float(np.mean(velocities)) if velocities else 0.0
    return velocities, avg_velocity

def save_velocity_to_file(velocities, avg_velocity, file_path="velocity_profile.txt"):
    with open(file_path, 'w') as f:
        f.write("Frame\tVelocity(mph)\n")
        for idx, v in enumerate(velocities, start=1):
            f.write(f"{idx}\t{v:.2f}\n")
        f.write(f"\nAverage Velocity: {avg_velocity:.2f} mph\n")
