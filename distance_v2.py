# import config as cfg
# cfg file has nonrobust scalars, avoid implementing for now. Only use camera data (focal length, sensor size)
from typing import Iterable, List

# Distance estimation based on pinhole camera model
# New model used: model detects ball, rubber, and catcher chest
# Strategy: Use front of rubber as d = 0 and ratios between known ball diameter and rubber length to estimate distance.
# This should be more robust than using ball size alone, especially at longer distances where the ball may be very small in the frame.
# Virtualize a point 60.5ft away and use the ratio of the rubber length in pixels to the known physical length to estimate distance.
# Extra: See if we can generate a strike zone grid based on the known dimensions of the strike zone and the estimated distance.

BALL_DIAMETER_INCHES = 2.9
RUBBER_LENGTH_INCHES = 24
RATIO_INIT = BALL_DIAMETER_INCHES/RUBBER_LENGTH_INCHES #Ratio at d=0

#Hard-code camera parameters for now, but these should ideally be calibrated for each camera setup
CROP_FACTOR = 4.77
FOCAL_LENGTH_35 = 26
FOCAL_LENGTH = FOCAL_LENGTH_35 / CROP_FACTOR
SENSOR_SIZE = 7.26
FOCAL_LENGTH_PX = FOCAL_LENGTH / SENSOR_SIZE * 1920 # Video was 1080x1920
# Assumption: ratio will hold regardless of units (in vs pixels)
# Personal note: Avoid conversions from pixels to in/ft until end of calculation

def get_rubber_length(rx1, ry1, rx2, ry2):
    rubber_pixel_length = max(rx2 - rx1, ry2 - ry1)
    if rubber_pixel_length == 0:
        return float('inf')  # Return infinity and try another frame to find rubber
    return rubber_pixel_length

def estimate_distance(rubber_pixel_length, ball_bbox: List[int], rubber_bbox: List[int]) -> float:
    """
    Estimate the distance of the ball from the camera using the bounding boxes of the ball and the rubber.

    Args:
        ball_bbox (List[int]): Bounding box of the ball [x1, y1, x2, y2].
        rubber_bbox (List[int]): Bounding box of the rubber [x1, y1, x2, y2].

    Returns:
        float: Estimated distance in feet.
    """
    # Calculate pixel lengths
    ball_pixel_diameter = min(ball_bbox[2] - ball_bbox[0], ball_bbox[3] - ball_bbox[1])
    ball_center = ((ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2)
    #rubber_pixel_length = max(rubber_bbox[2] - rubber_bbox[0], rubber_bbox[3] - rubber_bbox[1])
    rubber_front_center = ((rubber_bbox[0] + rubber_bbox[2]) / 2, rubber_bbox[1])  # Assuming front of rubber is at the top of the bbox

    # Avoid division by zero
    if rubber_pixel_length == 0:
        return float('inf')

    # Calculate ratio of rubber and ball at d=d
    ball_ratio = ball_pixel_diameter / rubber_pixel_length
    

    # Use monocular depth estimation based on the ratio of the ball size to the rubber size
    init_ball_pixel_diameter = RATIO_INIT * rubber_pixel_length  # Expected ball size at d=0
    z_ball = FOCAL_LENGTH_PX * (BALL_DIAMETER_INCHES / ball_pixel_diameter)
    z_rubber = FOCAL_LENGTH_PX * (RUBBER_LENGTH_INCHES / rubber_pixel_length)
    distance = z_ball - z_rubber

    return distance / 12.0 #returns distance in feet between ball and rubber