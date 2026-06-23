import cv2
import numpy as np
import os

def generate_alignment_validation(svg_mask, img_mask, doors_info, M_refine, output_path):
    """
    Generates validation visualization:
    - Layer 1: Red = Warped SVG Walls
    - Layer 2: Blue = Image Walls
    - Layer 3: Green = Aligned Door Geometry
    """
    h, w = img_mask.shape[:2]
    
    # Create a 3-channel image (BGR format)
    # Red is channel 2, Green is channel 1, Blue is channel 0
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Warp SVG mask using refinement matrix
    svg_warped = cv2.warpAffine(svg_mask, M_refine[:2], (w, h), flags=cv2.INTER_NEAREST)
    
    # Set channels
    # Image walls -> Blue channel (index 0)
    vis[img_mask > 0, 0] = 255
    # SVG walls -> Red channel (index 2)
    vis[svg_warped > 0, 2] = 255
    
    # Draw aligned doors in Green channel (index 1)
    for door in doors_info:
        if not door:
            continue
        poly = door["polygon"]
        if len(poly) > 0:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            cv2.fillPoly(vis, [pts], (0, 100, 0))
            
    # Save the visualization
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis)
