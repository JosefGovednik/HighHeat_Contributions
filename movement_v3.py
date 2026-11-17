import numpy as np
from typing import List, Tuple, Dict

GRAVITY_FTPS2 = 32.174  # ft/s^2


def _pixel_to_inches(distance_ft: float, frame_width_px: float = 640.0, focal_length_in: float = 4.32/25.4, sensor_width_in: float = 7.26/25.4) -> float:
    dist_in = distance_ft * 12.0
    return (dist_in * sensor_width_in) / (focal_length_in * frame_width_px)


def _to_physical(x_px: List[float], y_px: List[float], z_ft: List[float], frame_width_px: float = 640.0,
                 focal_length_in: float = 4.32/25.4, sensor_width_in: float = 7.26/25.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_ft = []
    y_ft = []
    for i in range(len(x_px)):
        scale_in = _pixel_to_inches(z_ft[i], frame_width_px, focal_length_in, sensor_width_in)
        scale_ft = scale_in / 12.0
        x_ft.append((x_px[i] - x_px[0]) * scale_ft)
        y_ft.append((y_px[i] - y_px[0]) * scale_ft)
    return np.array(x_ft, dtype=np.float32), np.array(y_ft, dtype=np.float32), np.array(z_ft, dtype=np.float32)


def compute_movement_v3(x_px: List[float], y_px: List[float], distances_ft: List[float], frame_rate: float = 60.0,
                        frame_width_px: float = 640.0, focal_length_in: float = 4.32/25.4, sensor_width_in: float = 7.26/25.4) -> Dict[str, object]:
    """Compute movement with gravity compensation and break decomposition.

    Returns dict with:
        total_horizontal_break_in: net lateral break (inches) relative to initial direction.
        total_vertical_break_in: net vertical break beyond gravity (inches).
        gravity_drop_in: expected gravity drop (inches).
        observed_vertical_drop_in: total observed vertical drop (inches).
        per_frame_break: list of (h_break_in, v_break_in) per frame.
    """
    n = min(len(x_px), len(y_px), len(distances_ft))
    if n < 3:
        return {"total_horizontal_break_in": 0.0, "total_vertical_break_in": 0.0, "gravity_drop_in": 0.0,
                "observed_vertical_drop_in": 0.0, "per_frame_break": []}

    x_ft, y_ft, z_ft = _to_physical(x_px[:n], y_px[:n], distances_ft[:n], frame_width_px, focal_length_in, sensor_width_in)

    dt = 1.0 / frame_rate
    t = np.arange(n, dtype=np.float32) * dt

    # Fit quadratic to vertical motion: y = a*t^2 + b*t + c ; gravity term expected a ≈ -0.5*g
    coeffs = np.polyfit(t, y_ft, 2)
    a, b, c = coeffs

    # Gravity-only path using fitted initial vertical velocity (b) but enforcing a_grav = -0.5*G
    grav_a = -0.5 * GRAVITY_FTPS2
    grav_path = grav_a * t**2 + b * t + c

    observed_vertical_drop_ft = y_ft[0] - y_ft[-1]
    gravity_drop_ft = grav_path[0] - grav_path[-1]
    vertical_break_ft = (y_ft - grav_path)  # positive = stayed up relative to gravity-only

    # Horizontal break: Remove component along initial horizontal direction (x vs z ignored; use x axis displacement)
    # Simpler: net horizontal break = final x_ft (since we center release at 0)
    total_horizontal_break_ft = x_ft[-1]

    # Convert to inches
    total_horizontal_break_in = total_horizontal_break_ft * 12.0
    total_vertical_break_in = (vertical_break_ft[-1] - vertical_break_ft[0]) * 12.0
    gravity_drop_in = gravity_drop_ft * 12.0
    observed_vertical_drop_in = observed_vertical_drop_ft * 12.0

    per_frame_break = [(vertical_break_ft[i] * 12.0, ) for i in range(n)]  # vertical break series only

    return {
        "total_horizontal_break_in": float(total_horizontal_break_in),
        "total_vertical_break_in": float(total_vertical_break_in),
        "gravity_drop_in": float(gravity_drop_in),
        "observed_vertical_drop_in": float(observed_vertical_drop_in),
        "per_frame_break": per_frame_break,
    }


def save_movement_v3(profile: Dict[str, object], path: str = "movement_v3.txt") -> None:
    with open(path, 'w') as f:
        f.write("Total Horizontal Break: {:.2f} in\n".format(profile.get("total_horizontal_break_in", 0.0)))
        f.write("Total Vertical Break (beyond gravity): {:.2f} in\n".format(profile.get("total_vertical_break_in", 0.0)))
        f.write("Observed Vertical Drop: {:.2f} in\n".format(profile.get("observed_vertical_drop_in", 0.0)))
        f.write("Gravity Drop (expected): {:.2f} in\n".format(profile.get("gravity_drop_in", 0.0)))
