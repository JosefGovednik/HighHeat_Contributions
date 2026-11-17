"""distance_utils.py

Reusable distance-approximation utilities extracted from the project's
ball-tracking script. Import this module to reuse the functions without
copying the whole script.

Exports:
  - estimate_abs_distance_ft(ball_px_width, frame_width_px, ...)
  - establish_pitch_bounds(abs_distances, plate_distance_ft=60.5)

Example:
    from distance_utils import estimate_abs_distance_ft, establish_pitch_bounds
    d = estimate_abs_distance_ft(24, 640)
    norm = establish_pitch_bounds([d, ...])

"""
from typing import Iterable, List

def estimate_abs_distance_ft(
    ball_px_width: float,
    frame_width_px: float,
    ball_diameter_in: float = 2.9,
    focal_length_in: float = 26 / 25.4,
    sensor_width_in: float = 9.5 / 25.4,
) -> float:
    """Estimate absolute distance from camera to ball in feet.

    Uses a simple pinhole-camera approximation:
        distance = focal_length * (ball_diameter / ball_size_on_sensor)

    The fraction of the frame occupied by the ball in pixels is mapped to
    sensor width to get the ball size on the sensor.

    Args:
        ball_px_width: ball bounding-box width in pixels.
        frame_width_px: frame width in pixels (e.g., 640 when resizing to 640x640).
        ball_diameter_in: assumed physical ball diameter in inches (default 2.9).
        focal_length_in: lens focal length in inches (default approx 26 mm).
        sensor_width_in: sensor physical width in inches (default approx 9.5 mm).

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

    distance_in = focal_length_in * (ball_diameter_in / ball_sensor_in)
    return distance_in / 12.0


def establish_pitch_bounds(abs_distances: Iterable[float], plate_distance_ft: float = 60.5) -> List[float]:
    """Map absolute distances (camera->ball) to normalized pitch distances.

    The function assumes the first element in ``abs_distances`` is the farthest
    (release) and the last element is the closest (near home). It linearly
    rescales the measured distances so that the travel from farthest->closest
    maps to 0->plate_distance_ft (default 60.5 ft).

    If the input list has fewer than 2 elements or has non-positive travel,
    the input list is returned as-is.

    Args:
        abs_distances: iterable of absolute distances (feet) ordered along the pitch.
        plate_distance_ft: the physical plate distance to map to (default 60.5).

    Returns:
        A list of normalized distances in feet.
    """
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
