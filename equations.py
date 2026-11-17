import numpy as np
G = 32.17
MASS = 0.00972 # 5 oz, but in slugs
RADIUS = 2.9 / 24 # radius in feet
DIAMETER = 2 * RADIUS 
RHO = 0.0023769 # Air density (keep in between 1.1 and 1.225 kg/m^3, convert to slugs/ft^3).
DRAG_COEFF = 0.3 # Guesstimate
CSA = np.pi * RADIUS**2 # Cross-sectional area of the ball in ft^2
RUBBER_LENGTH = 24/12
RUBBER_WIDTH = 6/12 # I left the inches values in in case we want to work in inches instead of feet.

#===========================================
#                 PHYSICS
#===========================================

# Forces
def F_g():
    # F_g = g * m
    return [0, G * MASS, 0]

def F_d(v):
    # F_d = 0.5 * rho * |v| * v * C_d * A
    mag = np.linalg.norm(v) # speed
    if mag < 1e-3:
        return np.zeros_like(v) # drag undetectable, adjust if this is a camera angle issue
    return -0.5 * RHO * v * mag * DRAG_COEFF * CSA

# Magnus force is for spin, but we don't have spin data. Skip for now.

# Kinematics
def acceleration(v):
    # A = F / m
    return (F_g() + F_d(v)) / MASS

# Velocities and positions are updated in the ball_tracking algorithm due to frame-by-frame processing.
# However, if we wanted to calculate velocity here, I have included this function:
def velocity(positions, fps):
    if len(positions) < 2:
        return np.zeros(3) # Two positions are required for velocity calculation.
    return (positions[-1] - positions[0]) * fps
    
# If you want this to be returned to the user as their out-of-hand velocity, get |v| and convert to mph.

def derivatives(t, state):
    # state = [x, y, z, vx, vy, vz]
    pos = state[:3]
    vel = state[3:]
    acc = acceleration(vel)
    return np.concatenate((vel, acc))

#===========================================
#             DEPTH ESTIMATION
#===========================================

# Pinhole equations and depth estimations for z cooridnate
# Note: focal_length and ball_diameter or rubber_length should be in pixels (the return value will be in ft).
def ball_depth(focal_length, ball_diameter):
    # z = f * real_diameter / image_diameter
    if ball_diameter < 1e-3:
        return float('inf') # Ball is too small to detect, treat as infinitely far away
    return focal_length * DIAMETER / ball_diameter

def rubber_depth(focal_length, rubber_length):
    # z = f * real_length / image_length
    return focal_length * RUBBER_LENGTH / rubber_length

def xyz(px, py, z, K):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    x = (px - cx) * z / fx
    y = (py - cy) * z / fy
    return np.array([x, y, z])