import numpy as np
from svgpathtools import QuadraticBezier, CubicBezier, Arc

def detect_arc(segments, hinge, threshold_bbox, leaf_end=None, samples=20, specific_curve=None, threshold_coords=None):
    """
    Extracts the swing arc points and start/end coordinates.
    - Samples points along any Bezier curve or Arc segment found in the path.
    - If no curve exists, synthetically constructs a quarter-circle arc between the threshold corner and the leaf end.
    """
    arc_points = []
    arc_start = None
    arc_end = None
    curve_seg = None
    
    # 1. Look for a curve segment in the path
    if specific_curve is not None:
        curve_seg = specific_curve
    elif segments:
        for seg in segments:
            if isinstance(seg, (QuadraticBezier, CubicBezier, Arc)):
                curve_seg = seg
                break
                
    if curve_seg is not None:
        arc_start = (float(curve_seg.start.real), float(curve_seg.start.imag))
        arc_end = (float(curve_seg.end.real), float(curve_seg.end.imag))
        
        # Sample points
        for t in np.linspace(0, 1, samples):
            p = curve_seg.point(t)
            arc_points.append((float(p.real), float(p.imag)))
            
        return arc_start, arc_end, arc_points, curve_seg
        
    # 2. Synthetic arc fallback using hinge and leaf_end
    if hinge and leaf_end:
        hx, hy = hinge
        lex, ley = leaf_end
        
        # Construct synthetic arc by determining the closed door position (arc_start)
        # The closed position lies along the threshold direction.
        radius = np.hypot(lex - hx, ley - hy)
        
        # Determine direction of threshold
        tx, ty = None, None
        
        # If threshold_coords are available, find the point furthest from hinge to get threshold vector
        if threshold_coords:
            best_dist = -1.0
            furthest_pt = None
            for pt in threshold_coords:
                dist = np.hypot(pt[0] - hx, pt[1] - hy)
                if dist > best_dist:
                    best_dist = dist
                    furthest_pt = pt
            if furthest_pt is not None:
                tx, ty = furthest_pt[0] - hx, furthest_pt[1] - hy
                
        # If no threshold_coords, fallback to threshold_bbox corners
        if (tx is None or ty is None) and threshold_bbox:
            xmin, ymin, xmax, ymax = threshold_bbox
            corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
            best_dist = -1.0
            furthest_pt = None
            for pt in corners:
                dist = np.hypot(pt[0] - hx, pt[1] - hy)
                if dist > best_dist:
                    best_dist = dist
                    furthest_pt = pt
            if furthest_pt is not None:
                tx, ty = furthest_pt[0] - hx, furthest_pt[1] - hy
                
        # If still not found, default to axis aligned based on dx/dy
        if tx is None or ty is None:
            tx, ty = 1.0, 0.0
            
        dist_t = np.hypot(tx, ty)
        if dist_t > 0:
            tx, ty = tx / dist_t, ty / dist_t
        else:
            tx, ty = 1.0, 0.0
            
        # Arc start (closed position) is along the threshold direction at door radius
        arc_start_x = hx + radius * tx
        arc_start_y = hy + radius * ty
        arc_start = (float(arc_start_x), float(arc_start_y))
        arc_end = (float(lex), float(ley))
        
        # Generate quarter-circle arc centered at hinge
        theta_start = np.arctan2(arc_start[1] - hy, arc_start[0] - hx)
        theta_end = np.arctan2(arc_end[1] - hy, arc_end[0] - hx)
        
        # Normalize angular difference to [-pi, pi]
        diff = theta_end - theta_start
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        
        # Sample angles
        for t in np.linspace(0, 1, samples):
            theta = theta_start + t * diff
            x = hx + radius * np.cos(theta)
            y = hy + radius * np.sin(theta)
            arc_points.append((float(x), float(y)))
            
        return arc_start, arc_end, arc_points, None
        
    return None, None, [], None
