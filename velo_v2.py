import numpy as np

def compute_velocity_from_depth(distances, frame_rate=60.0):
    """
    Computes per-frame velocity (mph) along the depth axis (z) from camera to ball.

    Parameters:
        distances: list of floats, absolute distances along the depth axis (ft)
        frame_rate: frames per second of the video

    Returns:
        velocities: list of floats, per-frame velocity (mph)
        avg_velocity: float, average velocity over the pitch (mph)
    """
    velocities = []
    for i in range(1, len(distances)):
        delta_z = distances[i-1] - distances[i]  # ball moving toward home plate
        delta_t = 1.0 / frame_rate
        v_ft_s = delta_z / delta_t
        v_mph = v_ft_s * 0.681818
        velocities.append(v_mph)
    
    avg_velocity = float(np.mean(velocities)) if velocities else 0.0
    return velocities, avg_velocity


def save_velocity_to_file(velocities, avg_velocity, file_path="velocity_profile.txt"):
    with open(file_path, 'w') as f:
        f.write("Frame\tVelocity(mph)\n")
        for idx, v in enumerate(velocities, start=1):
            f.write(f"{idx}\t{v:.2f}\n")
        f.write(f"\nAverage Velocity: {avg_velocity:.2f} mph\n")
