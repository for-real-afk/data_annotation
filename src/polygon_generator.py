import numpy as np

def generate_polygon(hinge, leaf_end, arc_points):
    """
    Assembles hinge, leaf_end, and arc_points into a closed wedge polygon.
    Returns a list of (x, y) coordinates.
    """
    if not hinge or not leaf_end:
        return []
        
    poly = []
    # Start at Hinge
    poly.append(hinge)
    # Go to Leaf End
    poly.append(leaf_end)
    
    # Add arc points. We want the arc points to go from leaf_end to arc_start.
    if arc_points:
        p_first = arc_points[0]
        p_last = arc_points[-1]
        
        d_first = np.hypot(p_first[0] - leaf_end[0], p_first[1] - leaf_end[1])
        d_last = np.hypot(p_last[0] - leaf_end[0], p_last[1] - leaf_end[1])
        
        ordered_arc = list(arc_points)
        if d_first > d_last:
            ordered_arc = list(reversed(ordered_arc))
            
        # Add arc points to polygon (excluding leaf_end if it's already close to the first point)
        for pt in ordered_arc:
            d_prev = np.hypot(pt[0] - poly[-1][0], pt[1] - poly[-1][1])
            if d_prev > 1e-3:
                poly.append(pt)
                
    return poly
