def generate_yolo_labels(doors_info, png_w, png_h):
    yolo_bbox_lines = []
    yolo_seg_lines = []
    
    # Class ID for doors is 0
    class_id = 0
    
    for door in doors_info:
        if not door:
            continue
            
        # 1. Generate BBox line
        bbox = door["bbox"]
        xmin, ymin, xmax, ymax = bbox
        
        x_center = ((xmin + xmax) / 2.0) / png_w
        y_center = ((ymin + ymax) / 2.0) / png_h
        w_norm = (xmax - xmin) / png_w
        h_norm = (ymax - ymin) / png_h
        
        # Clip to [0.0, 1.0] range
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))
        
        yolo_bbox_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        
        # 2. Generate Segmentation line
        poly_pts = door["polygon"]
        if len(poly_pts) >= 3:
            normalized_coords = []
            for pt in poly_pts:
                x_norm = pt[0] / png_w
                y_norm = pt[1] / png_h
                x_norm = max(0.0, min(1.0, x_norm))
                y_norm = max(0.0, min(1.0, y_norm))
                normalized_coords.append(f"{x_norm:.6f} {y_norm:.6f}")
            
            yolo_seg_lines.append(f"{class_id} " + " ".join(normalized_coords))
            
    return yolo_bbox_lines, yolo_seg_lines
