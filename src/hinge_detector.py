import numpy as np

def detect_hinge(threshold_bbox, path_start, leaf_start=None):
    """
    Geometrically determines the door hinge point.
    - If leaf_start is explicitly provided by the leaf detector, that is chosen.
    - Otherwise, it resolves the hinge as the corner of the threshold opposite to the path start.
    """
    if leaf_start is not None:
        return leaf_start
        
    if not threshold_bbox or not path_start:
        return path_start
        
    xmin, ymin, xmax, ymax = threshold_bbox
    w = xmax - xmin
    h = ymax - ymin
    
    px, py = path_start
    
    if w >= h:  # Horizontal threshold (opening is horizontal)
        # If path start is closer to left, hinge is on the right
        hx = xmax if abs(px - xmin) < abs(px - xmax) else xmin
        # Match the y coordinate (top or bottom edge)
        hy = ymin if abs(py - ymin) < abs(py - ymax) else ymax
    else:  # Vertical threshold (opening is vertical)
        # Match the x coordinate (left or right edge)
        hx = xmin if abs(px - xmin) < abs(px - xmax) else xmax
        # If path start is closer to top, hinge is at the bottom
        hy = ymax if abs(py - ymin) < abs(py - ymax) else ymin
        
    return (float(hx), float(hy))
