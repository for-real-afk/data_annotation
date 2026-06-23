import os
import sys
import cv2
import numpy as np
from bs4 import BeautifulSoup

# Add current and src directories to path
curr_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(curr_dir, "src")
for path in [curr_dir, src_dir]:
    if path not in sys.path:
        sys.path.append(path)

from svg_parser import get_svg_dimensions, calculate_alignment
from door_extractor import is_furniture
from transform_resolver import get_global_transform
from coordinate_mapper import CoordinateMapper
from wall_extractor import extract_svg_walls
from image_wall_detector import detect_image_walls
from svg_image_registrar import register_masks, compute_alignment_score
from door_anatomy_extractor import extract_door_info_cad, extract_door_svg_geometry
from drift_detector import detect_drift

def debug_plan(svg_path, png_path):
    print(f"Debugging SVG: {svg_path}")
    print(f"Debugging PNG: {png_path}")
    
    img = cv2.imread(png_path)
    if img is None:
        print("Error: Could not read image.")
        return
        
    with open(svg_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "xml")
        
    svg_w, svg_h = get_svg_dimensions(soup)
    print(f"SVG Dimensions: {svg_w}x{svg_h}")
    print(f"Image Dimensions: {img.shape[1]}x{img.shape[0]}")
    
    scale_x_coarse, scale_y_coarse, dx_coarse, dy_coarse, png_w, png_h = calculate_alignment(soup, img, svg_w, svg_h)
    print(f"Coarse alignment: scale_x={scale_x_coarse:.4f}, scale_y={scale_y_coarse:.4f}, dx={dx_coarse:.2f}, dy={dy_coarse:.2f}")
    
    M_coarse = np.array([
        [scale_x_coarse, 0, dx_coarse],
        [0, scale_y_coarse, dy_coarse],
        [0, 0, 1]
    ], dtype=float)
    
    image_wall_mask_thresh = detect_image_walls(img)
    img_inv_coarse = cv2.bitwise_not(image_wall_mask_thresh)
    dist_map_coarse = cv2.distanceTransform(img_inv_coarse, cv2.DIST_L2, 5)
    
    svg_wall_mask = extract_svg_walls(soup, scale_x_coarse, scale_y_coarse, dx_coarse, dy_coarse, png_w, png_h, dist_map=dist_map_coarse)
    print(f"SVG wall mask active pixels: {np.sum(svg_wall_mask > 0)}")
    print(f"Image wall mask active pixels: {np.sum(image_wall_mask_thresh > 0)}")
    
    M_refine = register_masks(svg_wall_mask, image_wall_mask_thresh)
    
    score_coarse = compute_alignment_score(svg_wall_mask, image_wall_mask_thresh, np.eye(3))
    score_refined = compute_alignment_score(svg_wall_mask, image_wall_mask_thresh, M_refine)
    print(f"Alignment Score: Coarse = {score_coarse:.4f}, Refined = {score_refined:.4f}")
    print(f"M_coarse:\n{M_coarse}")
    print(f"M_refine:\n{M_refine}")
    M_total = M_refine @ M_coarse
    print(f"M_total:\n{M_total}")
    mapper = CoordinateMapper(M_total)
    
    # Collect door tags
    import re
    door_tags = []
    for tag in soup.find_all():
        tag_id = str(tag.get("id", ""))
        tag_classes = tag.get("class", [])
        tag_classes_str = " ".join(tag_classes) if isinstance(tag_classes, list) else str(tag_classes)
        tag_text = (tag_id + " " + tag_classes_str).lower()
        
        if is_furniture(tag):
            continue
            
        if re.search(r"\bdoor\b", tag_text):
            if any(p in door_tags for p in tag.parents):
                continue
            door_tags.append(tag)
            
    print(f"Found {len(door_tags)} door tags.")
    for idx, tag in enumerate(door_tags):
        print(f"\n--- Door {idx+1} ---")
        print(f"Tag: <{tag.name} id='{tag.get('id')}' class='{tag.get('class')}'>")
        print(f"Raw XML: {tag.prettify()[:1000]}")
        door_attr = extract_door_info_cad(tag, mapper)
        
        if not door_attr:
            print("  No door anatomy extracted.")
            continue
            
        if isinstance(door_attr, list):
            for sub_idx, sub_attr in enumerate(door_attr):
                print(f"  Sub-Door {sub_idx+1}:")
                print_door_details(sub_attr, image_wall_mask_thresh, img.shape)
        else:
            print_door_details(door_attr, image_wall_mask_thresh, img.shape)
            if idx == 2:  # Door 3
                # Print the segments extracted for this door
                segments, threshold_bbox, path_start, threshold_coords = extract_door_svg_geometry(tag)
                print(f"    Raw geometry segments count: {len(segments)}")
                for s_idx, seg in enumerate(segments):
                    print(f"      Seg {s_idx+1}: {type(seg).__name__} (start={seg.start}, end={seg.end})")

def print_door_details(door_attr, image_wall_mask, img_shape):
    drift_res = detect_drift(door_attr, image_wall_mask, img_shape)
    print(f"    Door Type: {door_attr['door_type']}")
    print(f"    BBox: {door_attr['bbox']}")
    print(f"    Polygon points: {len(door_attr['polygon']) if door_attr['polygon'] else 0}")
    print(f"    Hinge: {door_attr['hinge']}")
    print(f"    Leaf segments: {door_attr['leaf']}")
    print(f"    Arc points: {len(door_attr['arc']) if door_attr['arc'] else 0}")
    print(f"    Drift status: {drift_res['status']}")
    if "reason" in drift_res:
        print(f"    Drift reason: {drift_res['reason']}")

if __name__ == "__main__":
    svg_path = r"C:\Users\deepa\.cache\kagglehub\datasets\qmarva\cubicasa5k\versions\4\cubicasa5k\cubicasa5k\colorful\10469\model.svg"
    png_path = r"C:\Users\deepa\.cache\kagglehub\datasets\qmarva\cubicasa5k\versions\4\cubicasa5k\cubicasa5k\colorful\10469\F1_original.png"
    debug_plan(svg_path, png_path)
