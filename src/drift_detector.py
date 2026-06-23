import numpy as np

def detect_drift(door, img_mask, img_shape):
    """
    Checks if a door annotation has drifted.
    Returns a dict with drift status and message.
    """
    h, w = img_shape[:2]
    bbox = door["bbox"]
    if not bbox:
        return {"status": "misaligned", "reason": "empty_bbox"}
        
    xmin, ymin, xmax, ymax = bbox
    
    # 1. Check if it's too close to the image borders (white margins)
    margin_w = int(w * 0.03)
    margin_h = int(h * 0.03)
    if xmin < margin_w or xmax > w - margin_w or ymin < margin_h or ymax > h - margin_h:
        return {"status": "misaligned", "reason": "falls_in_margins"}
        
    # 2. Check if it falls completely in white space (doesn't overlap or touch any walls)
    # A door should be near walls. Let's crop a region around the door and check wall pixel density.
    # We expand the door bbox by 10 pixels to see if it touches walls
    pad = 10
    ixmin = max(0, int(xmin) - pad)
    iymin = max(0, int(ymin) - pad)
    ixmax = min(w, int(xmax) + pad)
    iymax = min(h, int(ymax) + pad)
    
    door_region = img_mask[iymin:iymax, ixmin:ixmax]
    if np.sum(door_region > 0) == 0:
        return {"status": "misaligned", "reason": "floating_outside_walls"}
        
    return {"status": "aligned"}
