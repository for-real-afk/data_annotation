import os
# pyrefly: ignore [missing-import]
import cv2
import numpy as np

IMG_PATH = "F1_original (1).png"
BBOX_PATH = os.path.join("output", "model.txt")
SEG_PATH = os.path.join("output", "model_seg.txt")
OUTPUT_DIR = "output"

def visualize_bboxes():
    if not os.path.exists(BBOX_PATH):
        print(f"BBox file {BBOX_PATH} not found.")
        return
        
    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"Image {IMG_PATH} not found.")
        return
        
    h, w = img.shape[:2]
    
    with open(BBOX_PATH, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cls, xc, yc, bw, bh = map(float, line.split())
        
        # Calculate pixel coordinates
        x1 = int((xc - bw/2.0) * w)
        y1 = int((yc - bh/2.0) * h)
        x2 = int((xc + bw/2.0) * w)
        y2 = int((yc + bh/2.0) * h)
        
        # Green for Door (0), Blue for Window (1)
        color = (0, 255, 0) if int(cls) == 0 else (255, 0, 0)
        label = "door" if int(cls) == 0 else "window"
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
    out_path = os.path.join(OUTPUT_DIR, "debug_bbox.png")
    cv2.imwrite(out_path, img)
    print(f"BBox visualization saved to {out_path}")


def visualize_segmentation():
    if not os.path.exists(SEG_PATH):
        print(f"Segmentation file {SEG_PATH} not found.")
        return
        
    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"Image {IMG_PATH} not found.")
        return
        
    h, w = img.shape[:2]
    overlay = img.copy()
    
    with open(SEG_PATH, "r") as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        cls = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        
        # Convert pairs of normalized coords to pixel coords
        pts = []
        for i in range(0, len(coords), 2):
            px = int(coords[i] * w)
            py = int(coords[i+1] * h)
            pts.append([px, py])
            
        pts = np.array(pts, dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Green for Door (0), Blue for Window (1)
        color = (0, 255, 0) if cls == 0 else (255, 0, 0)
        
        # Draw transparent filled polygon
        cv2.fillPoly(overlay, [pts], color)
        # Draw outline
        cv2.polylines(img, [pts], True, color, 2)
        
    # Combine original image with the transparent overlay
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    out_path = os.path.join(OUTPUT_DIR, "debug_seg.png")
    cv2.imwrite(out_path, img)
    print(f"Segmentation visualization saved to {out_path}")


if __name__ == "__main__":
    visualize_bboxes()
    visualize_segmentation()
