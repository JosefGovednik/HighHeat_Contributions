import numpy as np
from typing import List, Tuple, Dict

# Conversion constant
FTS_TO_MPH = 0.681818  # ft/s -> mph

DEF_FIRST_N = 6  # frames used to estimate release angle & initial velocity


def _pixel_to_inches(distance_ft: float, frame_width_px: float = 640.0, focal_length_in: float = 4.32/25.4, sensor_width_in: float = 7.26/25.4) -> float:
    """Return how many inches one pixel spans at the given depth (feet)."""
    dist_in = distance_ft * 12.0
    return (dist_in * sensor_width_in) / (focal_length_in * frame_width_px)


def _to_physical_series(x_px: List[float], y_px: List[float], z_ft: List[float], frame_width_px: float = 640.0,
                        focal_length_in: float = 4.32/25.4, sensor_width_in: float = 7.26/25.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert pixel x,y plus depth (feet) to physical coordinates (feet) relative to release point.

    We treat camera optical axis as +Z toward the pitcher (or away from plate). We build X,Y in feet using pixel scale at
    each depth. Coordinates are shifted so the first frame is (0,0, z0)."""
    x_ft = []
    y_ft = []
    for i in range(len(x_px)):
        scale_in = _pixel_to_inches(z_ft[i], frame_width_px, focal_length_in, sensor_width_in)
        scale_ft = scale_in / 12.0
        x_ft.append((x_px[i] - x_px[0]) * scale_ft)
        y_ft.append((y_px[i] - y_px[0]) * scale_ft)
    return np.array(x_ft, dtype=np.float32), np.array(y_ft, dtype=np.float32), np.array(z_ft, dtype=np.float32)


def compute_velocity_v3(x_px: List[float], y_px: List[float], distances_ft: List[float], frame_rate: float = 60.0,
                        frame_width_px: float = 640.0, focal_length_in: float = 4.32/25.4, sensor_width_in: float = 7.26/25.4,
                        release_window: int = DEF_FIRST_N) -> Dict[str, object]:
    """Compute full 3D velocity profile + release angles.

    Args:
        x_px, y_px: pixel center positions per frame.
        distances_ft: depth distance camera->ball in feet per frame (same length as x_px).
        frame_rate: frames per second.
        frame_width_px, focal_length_in, sensor_width_in: camera model params.
        release_window: number of initial frames to estimate release direction.

    Returns dict with keys:
        speeds_mph: list per-frame 3D speed.
        avg_speed_mph: average of speeds.
        release_speed_mph: speed computed over first window.
        yaw_deg: horizontal angle (left/right) relative to camera axis.
        pitch_deg: vertical angle (up/down) relative to camera axis.
        components_ft_s: list of (vx, vy, vz) ft/s per frame (length n-1).
    """
    n = min(len(x_px), len(y_px), len(distances_ft))
    if n < 2:
        return {"speeds_mph": [], "avg_speed_mph": 0.0, "release_speed_mph": 0.0, "yaw_deg": 0.0, "pitch_deg": 0.0, "components_ft_s": []}

    x_ft, y_ft, z_ft = _to_physical_series(x_px[:n], y_px[:n], distances_ft[:n], frame_width_px, focal_length_in, sensor_width_in)

    dt = 1.0 / frame_rate
    speeds = []
    components = []
    for i in range(1, n):
        dx = x_ft[i] - x_ft[i-1]
        dy = y_ft[i] - y_ft[i-1]
        dz = z_ft[i-1] - z_ft[i]  # toward plate => positive forward velocity
        vx = dx / dt
        vy = dy / dt
        vz = dz / dt
        v_mag = np.sqrt(vx*vx + vy*vy + vz*vz)
        speeds.append(v_mag * FTS_TO_MPH)
        components.append((vx, vy, vz))

    avg_speed = float(np.mean(speeds)) if speeds else 0.0

    # Release direction estimation using initial window vector (from frame0 to frame_release_window)
    rw = min(release_window, n-1)
    dx_rel = x_ft[rw] - x_ft[0]
    dy_rel = y_ft[rw] - y_ft[0]
    dz_rel = z_ft[0] - z_ft[rw]
    horiz_mag = np.sqrt(dx_rel*dx_rel + dz_rel*dz_rel) + 1e-8
    total_mag = np.sqrt(dx_rel*dx_rel + dy_rel*dy_rel + dz_rel*dz_rel) + 1e-8
    yaw_deg = np.degrees(np.arctan2(dx_rel, dz_rel))  # left/right
    pitch_deg = np.degrees(np.arcsin(dy_rel / total_mag))  # up/down angle

    release_speed_ft_s = total_mag / (rw * dt) if rw > 0 else 0.0
    release_speed_mph = release_speed_ft_s * FTS_TO_MPH

    return {
        "speeds_mph": speeds,
        "avg_speed_mph": avg_speed,
        "release_speed_mph": release_speed_mph,
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "components_ft_s": components,
    }


def save_velocity_v3(profile: Dict[str, object], path: str = "velocity_v3.txt") -> None:
    with open(path, 'w') as f:
        f.write("Frame\tSpeed(mph)\n")
        for i, v in enumerate(profile.get("speeds_mph", []), start=1):
            f.write(f"{i}\t{v:.2f}\n")
        f.write("\nAverage Speed: {:.2f} mph\n".format(profile.get("avg_speed_mph", 0.0)))
        f.write("Release Speed: {:.2f} mph\n".format(profile.get("release_speed_mph", 0.0)))
        f.write("Release Yaw: {:.2f} deg\n".format(profile.get("yaw_deg", 0.0)))
        f.write("Release Pitch: {:.2f} deg\n".format(profile.get("pitch_deg", 0.0)))
