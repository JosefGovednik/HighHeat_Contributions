import config as cfg
from typing import Iterable, List

def estimate_abs_distance_ft(
    ball_px_width: float,
    frame_width_px: float,
    ball_diameter_in: float = 2.9,
    focal_length_in: float = 4.32 / 25.4,
    sensor_width_in: float = 7.26 / 25.4,
    distance_scale: float = 1.0,  # No scaling here - keep for path drawing
) -> float:
    """Estimate absolute distance from camera to ball in feet.

    Uses a simple pinhole-camera approximation:
        distance = focal_length * (ball_diameter / ball_size_on_sensor) * distance_scale

    The fraction of the frame occupied by the ball in pixels is mapped to
    sensor width to get the ball size on the sensor. The distance_scale parameter
    allows calibration adjustment if theoretical calculations don't match real-world distances.

    Args:
        ball_px_width: ball bounding-box width in pixels.
        frame_width_px: frame width in pixels (e.g., 640 when resizing to 640x640).
        ball_diameter_in: assumed physical ball diameter in inches (default 2.9).
        focal_length_in: lens focal length in inches (default approx 4.32 mm).
        sensor_width_in: sensor physical width in inches (default approx 7.26 mm).
        distance_scale: calibration multiplier to adjust distance estimates (default 2.5).

    Returns:
        Distance in feet (float).
    """
    if frame_width_px <= 0 or ball_px_width <= 0:
        raise ValueError("frame_width_px and ball_px_width must be positive")

    frac = float(ball_px_width) / float(frame_width_px)
    ball_sensor_in = frac * sensor_width_in
    # avoid division by zero
    if ball_sensor_in <= 0:
        raise ValueError("Computed ball size on sensor is zero or negative")

    distance_in = focal_length_in * (ball_diameter_in / ball_sensor_in) * distance_scale
    return distance_in / 12.0


def establish_pitch_bounds(abs_distances: Iterable[float], plate_distance_ft: float = None) -> List[float]:
    """Map absolute distances (camera->ball) to normalized pitch distances.

    The function assumes the first element in ``abs_distances`` is the farthest
    (release) and the last element is the closest (near home). It linearly
    rescales the measured distances so that the travel from farthest->closest
    maps to 0->plate_distance_ft (default from config.STOP_DISTANCE_FT).

    If the input list has fewer than 2 elements or has non-positive travel,
    the input list is returned as-is.

    Args:
        abs_distances: iterable of absolute distances (feet) ordered along the pitch.
        plate_distance_ft: the physical plate distance to map to (default from config.STOP_DISTANCE_FT).

    Returns:
        A list of normalized distances in feet.
    """
    if plate_distance_ft is None:
        plate_distance_ft = cfg.STOP_DISTANCE_FT
        print(f"DEBUG distance_utils: Using cfg.STOP_DISTANCE_FT = {plate_distance_ft}")

    abs_list = list(abs_distances)
    if len(abs_list) < 2:
        return abs_list

    farthest = abs_list[0]
    closest = abs_list[-1]
    total_travel = farthest - closest

    if total_travel <= 0:
        # cannot normalize if distances not monotonic; return copy
        return abs_list

    scale_factor = float(plate_distance_ft) / total_travel
    return [(farthest - d) * scale_factor for d in abs_list]


__all__ = [
    'estimate_abs_distance_ft',
    'establish_pitch_bounds',
]