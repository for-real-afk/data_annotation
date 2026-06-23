import os
import cv2
import numpy as np

def generate_qa_overlay(doors_info, png_file, output_qa_path):
    if not os.path.exists(png_file):
        return
        
    img = cv2.imread(png_file)
    if img is None:
        return
        
    overlay = img.copy()
    
    for door in doors_info:
        if not door:
            continue
            
        # 1. Draw Mask (Yellow Polygon) - translucent
        poly = door.get("polygon", [])
        if poly and len(poly) > 0:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 255, 255)) # Yellow in BGR is (0, 255, 255)
            
        # 2. Draw Door Leaf (Blue Line)
        leaf = door.get("leaf")
        if leaf and len(leaf) == 2:
            p1 = tuple(map(int, leaf[0]))
            p2 = tuple(map(int, leaf[1]))
            cv2.line(img, p1, p2, (255, 0, 0), 2) # Blue in BGR is (255, 0, 0)
            
        # 3. Draw Swing Arc (Green Curve)
        arc = door.get("arc", [])
        if arc and len(arc) > 1:
            pts_arc = np.array(arc, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts_arc], False, (0, 255, 0), 2) # Green in BGR is (0, 255, 0)
            
        # 4. Draw Hinge (Red Dot)
        hinge = door.get("hinge")
        if hinge:
            pt_hinge = tuple(map(int, hinge))
            cv2.circle(img, pt_hinge, 4, (0, 0, 255), -1) # Red in BGR is (0, 0, 255)
            
        # 5. Draw labels
        bbox = door.get("bbox")
        if bbox:
            xmin, ymin, xmax, ymax = bbox
            orient = door.get("orientation", 0)
            direction = door.get("opening_direction", "N/A")
            label_text = f"{door['door_type']} ({orient}deg, {direction})"
            cv2.putText(img, label_text, (int(xmin), max(int(ymin) - 5, 15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Blend translucent polygon overlay (Yellow)
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    os.makedirs(os.path.dirname(output_qa_path), exist_ok=True)
    cv2.imwrite(output_qa_path, img)
