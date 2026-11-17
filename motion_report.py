from velo import compute_velocity
from movement_profile import compute_movement_profile

def generate_motion_report(distances, x_positions, y_positions, frame_rate, out_path):
    velocities, _ = compute_velocity(distances, frame_rate)
    out_of_hand_velocity = velocities[0] if velocities else 0.0
    _, (total_dx, total_dy) = compute_movement_profile(x_positions, y_positions, distances)
    
    with open(out_path, "w") as f:
        f.write("PITCH REPORT\n")
        f.write("====================\n\n")
        f.write(f"Out-of-hand velocity: {out_of_hand_velocity:.2f} mph\n")
        f.write(f"Total horizontal movement: {total_dx:.2f} inches\n")
        f.write(f"Total vertical movement: {total_dy:.2f} inches\n")
