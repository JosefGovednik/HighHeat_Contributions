"""Configuration constants for ball-tracking visualizers."""

VIDEO_PATH = "Test_videos/20250923_184300.mp4"
# Working videos w/ results:
# - "Test_videos/20250923_183444.mp4": 2-seam fastball away, good velo, decent movement
# - "Test_videos/20250923_183457.mp4": 2-seam fastball middle, overshot velo, movement a bit wonky
# - "Test_videos/20250923_184300.mp4": curveball middle, good velo, good movement
MODEL_PATH = "model/weights.pt"
FRAME_SIZE = (640, 640)
CONF_THRESHOLD = 0.52
EMA_ALPHA = 0.35
VISUALIZE = True
SAVE_OUTPUT_VIDEO = False
STOP_DISTANCE_FT = 16  # set this to desired stop distance (ft)

# Velocity/movement calibration: scale factor to convert camera distances to real-world
# This adjusts velocity and movement calculations without affecting the drawn path

# THIS IS A PATCH FIX. Further iteration on velo and movement are needed to account for release angle.
DISTANCE_SCALE_FOR_METRICS = (60.5 / STOP_DISTANCE_FT) * 1.27 # 1.27 works best with v2, v3 works best without a multiplier

# Movement calibration: reduce movement values if they're too large
# Adjust this down if horizontal/vertical movement seems exaggerated
MOVEMENT_SCALE = 1 / DISTANCE_SCALE_FOR_METRICS  # Multiply movement by this factor
