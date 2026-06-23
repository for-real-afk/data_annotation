import numpy as np
from svgpathtools import Line

def detect_leaf(segments, threshold_bbox, hinge, arc_end=None, arc_start=None):
    """
    Detects the door leaf segment: (leaf_start, leaf_end, leaf_length).
    - Checks line segments connecting the hinge to either of the arc endpoints.
    - Falls back to threshold opening dimension and orientation if no line segment is found.
    """
    leaf_start = None
    leaf_end = None
    
    targets = []
    if arc_end is not None:
        targets.append(arc_end)
    if arc_start is not None:
        targets.append(arc_start)
        
    # 1. Search for a Line segment connecting the hinge to either arc endpoint
    if segments:
        for seg in segments:
            if isinstance(seg, Line):
                p1 = (float(seg.start.real), float(seg.start.imag))
                p2 = (float(seg.end.real), float(seg.end.imag))
                
                # Check orientation 1: p1 near hinge, p2 near any arc target
                if hinge is not None:
                    d_hinge1 = np.hypot(p1[0] - hinge[0], p1[1] - hinge[1])
                    if d_hinge1 < 15.0:
                        for target in targets:
                            d_arc = np.hypot(p2[0] - target[0], p2[1] - target[1])
                            if d_arc < 15.0:
                                return p1, p2, float(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
                                
                # Check orientation 2: p2 near hinge, p1 near any arc target
                if hinge is not None:
                    d_hinge2 = np.hypot(p2[0] - hinge[0], p2[1] - hinge[1])
                    if d_hinge2 < 15.0:
                        for target in targets:
                            d_arc = np.hypot(p1[0] - target[0], p1[1] - target[1])
                            if d_arc < 15.0:
                                return p2, p1, float(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
                                
        # Fallback 1b: Search for a Line segment connecting to any arc endpoint (hinge is loose)
        for target in targets:
            for seg in segments:
                if isinstance(seg, Line):
                    p1 = (float(seg.start.real), float(seg.start.imag))
                    p2 = (float(seg.end.real), float(seg.end.imag))
                    
                    d1 = np.hypot(p1[0] - target[0], p1[1] - target[1])
                    d2 = np.hypot(p2[0] - target[0], p2[1] - target[1])
                    
                    if d1 < 10.0:
                        return p2, p1, float(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
                    elif d2 < 10.0:
                        return p1, p2, float(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))
                        
    # 2. Synthetic leaf fallback: Connect hinge directly to the open arc endpoint
    # (Works perfectly for diagonal/angled doors)
    if hinge and arc_end is not None:
        return hinge, arc_end, float(np.hypot(arc_end[0] - hinge[0], arc_end[1] - hinge[1]))
    if hinge and arc_start is not None:
        return hinge, arc_start, float(np.hypot(arc_start[0] - hinge[0], arc_start[1] - hinge[1]))
        
    # 3. Geometric fallback using threshold dimensions and hinge (last resort for axis-aligned assumptions)
    if threshold_bbox and hinge:
        xmin, ymin, xmax, ymax = threshold_bbox
        w = xmax - xmin
        h = ymax - ymin
        
        hx, hy = hinge
        
        if w >= h:  # Horizontal threshold, leaf is vertical (length is w)
            length = w
            if arc_end is not None:
                dy = arc_end[1] - hy
                direction = -1.0 if dy < 0 else 1.0
            else:
                direction = -1.0  # Default to up
            leaf_start = (float(hx), float(hy))
            leaf_end = (float(hx), float(hy + direction * length))
        else:  # Vertical threshold, leaf is horizontal (length is h)
            length = h
            if arc_end is not None:
                dx = arc_end[0] - hx
                direction = -1.0 if dx < 0 else 1.0
            else:
                direction = 1.0  # Default to right
            leaf_start = (float(hx), float(hy))
            leaf_end = (float(hx + direction * length), float(hy))
            
        return leaf_start, leaf_end, float(length)
        
    if hinge:
        return hinge, hinge, 0.0
    return (0.0, 0.0), (0.0, 0.0), 0.0
