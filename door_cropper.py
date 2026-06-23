import os
import cv2

def crop_and_save_doors(doors_info, png_file, svg_name, crops_dir, start_id):
    metadata = []
    if not os.path.exists(png_file):
        return metadata, start_id
        
    img = cv2.imread(png_file)
    if img is None:
        return metadata, start_id
        
    h, w = img.shape[:2]
    
    current_id = start_id
    for door in doors_info:
        if not door:
            continue
            
        bbox = door["bbox"]
        xmin, ymin, xmax, ymax = bbox
        
        # Convert to integer pixel coordinates and pad slightly
        pad = 5
        ixmin = max(0, int(xmin) - pad)
        iymin = max(0, int(ymin) - pad)
        ixmax = min(w, int(xmax) + pad)
        iymax = min(h, int(ymax) + pad)
        
        # Check valid crop size
        if ixmax - ixmin <= 0 or iymax - iymin <= 0:
            continue
            
        crop = img[iymin:iymax, ixmin:ixmax]
        
        crop_filename = f"door_{current_id:05d}.png"
        crop_path = os.path.join(crops_dir, crop_filename)
        cv2.imwrite(crop_path, crop)
        
        metadata.append({
            "door_id": current_id,
            "door_type": door["door_type"],
            "orientation": door["orientation"],
            "opening_direction": door["opening_direction"],
            "svg": svg_name,
            # Future compatibility fields (placeholders/estimates)
            "width_mm": 900,   # Standard default door width placeholder
            "height_mm": 2100  # Standard default door height placeholder
        })
        current_id += 1
        
    return metadata, current_id
