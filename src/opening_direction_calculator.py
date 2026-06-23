import numpy as np

def calculate_opening_direction(arc_points, hinge, arc_start, arc_end):
    """
    Computes the door opening direction (CW or CCW) from arc coordinates.
    Ensures points are evaluated from arc_start (closed) to arc_end (open).
    """
    if not arc_points or not hinge or not arc_start or not arc_end:
        return "N/A"
        
    # Ensure points are ordered from arc_start (closed) to arc_end (open)
    p_first = arc_points[0]
    p_last = arc_points[-1]
    
    d_first_start = np.hypot(p_first[0] - arc_start[0], p_first[1] - arc_start[1])
    d_last_start = np.hypot(p_last[0] - arc_start[0], p_last[1] - arc_start[1])
    
    ordered_pts = list(arc_points)
    if d_first_start > d_last_start:
        ordered_pts = list(reversed(ordered_pts))
        
    # Calculate angular differences relative to the hinge
    hx, hy = hinge
    angles = []
    for pt in ordered_pts:
        dx = pt[0] - hx
        dy = pt[1] - hy
        angles.append(np.arctan2(dy, dx))
        
    if len(angles) < 2:
        return "N/A"
        
    total_diff = 0.0
    for i in range(len(angles) - 1):
        diff = angles[i+1] - angles[i]
        # Normalize to [-pi, pi]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        total_diff += diff
        
    if total_diff > 0.01:
        return "CW"
    elif total_diff < -0.01:
        return "CCW"
        
    return "N/A"
