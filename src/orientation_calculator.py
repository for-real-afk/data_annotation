import numpy as np

def calculate_orientation(hinge, leaf_end):
    """
    Computes the door orientation based on the open leaf vector relative to the hinge.
    Normalizes the rotation to 0, 90, 180, or 270 degrees.
    """
    if not hinge or not leaf_end:
        return 0
        
    hx, hy = hinge
    lex, ley = leaf_end
    
    dx = lex - hx
    dy = ley - hy
    
    # Calculate angle in degrees
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    # Normalize to [0, 360)
    angle_deg = angle_deg % 360
    
    # Round to nearest 90-degree multiple
    orientation = int(round(angle_deg / 90.0) * 90) % 360
    
    return orientation
