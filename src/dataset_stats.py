import json
import numpy as np

def compile_global_stats(all_processed_data, total_svgs, output_dir):
    total_doors = 0
    doors_per_plan = []
    
    widths = []
    heights = []
    
    door_types = {}
    schema_variants = {}
    anomalies = []
    
    for plan in all_processed_data:
        svg_name = plan["svg"]
        doors = plan["doors"]
        
        num_doors = len(doors)
        total_doors += num_doors
        doors_per_plan.append(num_doors)
        
        for door in doors:
            dtype = door["door_type"]
            door_types[dtype] = door_types.get(dtype, 0) + 1
            
            # Record schema variant
            variant = plan["schema_variant"]
            schema_variants[variant] = schema_variants.get(variant, 0) + 1
            
            # Calculate width and height in pixels
            bbox = door.get("bbox")
            if not bbox:
                continue
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            widths.append(w)
            heights.append(h)
            
            # Anomaly checks (arbitrary pixel size limits based on typical floorplans)
            # e.g., doors under 10 pixels or over 500 pixels wide/high, or aspect ratios > 10
            aspect_ratio = w / h if h > 0 else 0
            if w < 10 or h < 10:
                anomalies.append({
                    "svg": svg_name,
                    "reason": f"Extremely small door bbox: {w:.1f}x{h:.1f} pixels"
                })
            elif w > 500 or h > 500:
                anomalies.append({
                    "svg": svg_name,
                    "reason": f"Extremely large door bbox: {w:.1f}x{h:.1f} pixels"
                })
            elif aspect_ratio > 10 or (aspect_ratio > 0 and 1/aspect_ratio > 10):
                anomalies.append({
                    "svg": svg_name,
                    "reason": f"Anomalous door aspect ratio: {w:.1f}x{h:.1f} pixels"
                })
                
    # Calculate statistics
    avg_doors = float(np.mean(doors_per_plan)) if len(doors_per_plan) > 0 else 0.0
    min_doors = int(np.min(doors_per_plan)) if len(doors_per_plan) > 0 else 0
    max_doors = int(np.max(doors_per_plan)) if len(doors_per_plan) > 0 else 0
    
    w_mean = float(np.mean(widths)) if len(widths) > 0 else 0.0
    h_mean = float(np.mean(heights)) if len(heights) > 0 else 0.0
    w_std = float(np.std(widths)) if len(widths) > 0 else 0.0
    h_std = float(np.std(heights)) if len(heights) > 0 else 0.0
    
    stats = {
        "total_svgs": total_svgs,
        "total_doors": total_doors,
        "avg_doors_per_plan": avg_doors,
        "min_doors_per_plan": min_doors,
        "max_doors_per_plan": max_doors,
        "door_bbox_width_mean": w_mean,
        "door_bbox_height_mean": h_mean,
        "door_bbox_width_std": w_std,
        "door_bbox_height_std": h_std,
        "anomalies_detected": anomalies,
        "schema_variants": schema_variants
    }
    
    return stats, door_types
