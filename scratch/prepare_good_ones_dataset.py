import os
import re
import cv2
import numpy as np
import random
import shutil
import zipfile
import sys
from bs4 import BeautifulSoup

# Setup import path for src modules
sys.path.append(r"d:\projects\data_annotations\src")
from svg_parser import get_svg_dimensions, calculate_alignment
from transform_resolver import get_global_transform
from coordinate_mapper import CoordinateMapper
from wall_extractor import extract_svg_walls
from image_wall_detector import detect_image_walls
from svg_image_registrar import register_masks, compute_alignment_score
from door_anatomy_extractor import extract_door_info_cad
from drift_detector import detect_drift
from segmentation_generator import generate_yolo_labels
from door_extractor import is_furniture

download_dir = r"C:\Users\deepa\Downloads\good ones"
accepted_dir = r"d:\projects\data_annotations\dataset\accepted"
base_dataset_dir = r"d:\projects\data_annotations\good_ones_dataset"

# Recreate folders
shutil.rmtree(base_dataset_dir, ignore_errors=True)
splits = ["train", "val"]
for split in splits:
    os.makedirs(os.path.join(base_dataset_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(base_dataset_dir, "labels", split), exist_ok=True)

# Get files
png_files = [f for f in os.listdir(download_dir) if f.endswith(".png")]

plans_info = []
for f in png_files:
    m = re.match(r"sample_\d+_(cubicasa5k_cubicasa5k_.*)\.png", f)
    if m:
        plan_name = m.group(1)
    else:
        plan_name = os.path.splitext(f)[0]
    plans_info.append({"original_file": f, "plan_name": plan_name})

random.seed(42)
random.shuffle(plans_info)
split_idx = int(len(plans_info) * 0.8)
train_plans = plans_info[:split_idx]
val_plans = plans_info[split_idx:]

def get_svg_path(clean_plan):
    plan_id = clean_plan.split('_')[-1]
    base_dir = r"C:\Users\deepa\.cache\kagglehub\datasets\qmarva\cubicasa5k\versions\4\cubicasa5k\cubicasa5k"
    
    for type_dir in ['colorful', 'high_quality', 'high_quality_architectural']:
        if type_dir in clean_plan:
            svg_path = os.path.join(base_dir, type_dir, plan_id, "model.svg")
            if os.path.exists(svg_path):
                return svg_path
            svg_path = os.path.join(base_dir, type_dir, plan_id, "model1.svg")
            if os.path.exists(svg_path):
                return svg_path
    return None

def force_generate_label(svg_path, png_path):
    img = cv2.imread(png_path)
    if img is None:
        return []
    with open(svg_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "xml")
        
    svg_w, svg_h = get_svg_dimensions(soup)
    try:
        scale_x_coarse, scale_y_coarse, dx_coarse, dy_coarse, png_w, png_h = calculate_alignment(soup, img, svg_w, svg_h)
    except Exception as e:
        print(f"    Coarse alignment error: {e}")
        return []
        
    M_coarse = np.array([
        [scale_x_coarse, 0, dx_coarse],
        [0, scale_y_coarse, dy_coarse],
        [0, 0, 1]
    ], dtype=float)
    
    image_wall_mask_thresh = detect_image_walls(img)
    img_inv_coarse = cv2.bitwise_not(image_wall_mask_thresh)
    dist_map_coarse = cv2.distanceTransform(img_inv_coarse, cv2.DIST_L2, 5)
    
    svg_wall_mask = extract_svg_walls(soup, scale_x_coarse, scale_y_coarse, dx_coarse, dy_coarse, png_w, png_h, dist_map=dist_map_coarse)
    M_refine = register_masks(svg_wall_mask, image_wall_mask_thresh)
    
    M_total = M_refine @ M_coarse
    mapper = CoordinateMapper(M_total)
    
    door_tags = []
    for tag in soup.find_all():
        tag_id = str(tag.get("id", ""))
        tag_classes = tag.get("class", [])
        tag_classes_str = " ".join(tag_classes) if isinstance(tag_classes, list) else str(tag_classes)
        tag_text = (tag_id + " " + tag_classes_str).lower()
        
        if not re.search(r"\bdoor\b", tag_text):
            continue
        if is_furniture(tag):
            continue
        if any(p in door_tags for p in tag.parents):
            continue
        door_tags.append(tag)
        
    doors_info = []
    for tag in door_tags:
        door_attr = extract_door_info_cad(tag, mapper)
        if door_attr:
            if isinstance(door_attr, list):
                for sub_attr in door_attr:
                    drift_res = detect_drift(sub_attr, image_wall_mask_thresh, img.shape)
                    sub_attr["drift_status"] = drift_res["status"]
                    doors_info.append(sub_attr)
            else:
                drift_res = detect_drift(door_attr, image_wall_mask_thresh, img.shape)
                door_attr["drift_status"] = drift_res["status"]
                doors_info.append(door_attr)
                
    aligned_doors = [d for d in doors_info if d.get("drift_status") == "aligned"]
    doors_to_use = aligned_doors if aligned_doors else doors_info
    
    yolo_bbox, yolo_seg = generate_yolo_labels(doors_to_use, png_w, png_h)
    return yolo_seg

def process_split_plans(plans_list, split_name):
    print(f"\nProcessing {split_name} split...")
    count = 0
    for idx, plan in enumerate(plans_list):
        orig_file = plan["original_file"]
        plan_name = plan["plan_name"]
        orig_img_path = os.path.join(download_dir, orig_file)
        
        new_name = f"{plan_name}"
        if "- Copy" in plan_name:
            clean_name = plan_name.replace(" - Copy (2)", "_copy2").replace(" - Copy (3)", "_copy3").replace(" - Copy", "_copy1")
            new_name = clean_name
            
        dst_img_path = os.path.join(base_dataset_dir, "images", split_name, f"{new_name}.png")
        dst_lbl_path = os.path.join(base_dataset_dir, "labels", split_name, f"{new_name}.txt")
        
        shutil.copy2(orig_img_path, dst_img_path)
        
        base_plan = plan_name
        for suffix in [" - Copy (2)", " - Copy (3)", " - Copy"]:
            if suffix in base_plan:
                base_plan = base_plan.replace(suffix, "")
                break
                
        src_lbl_file = f"{base_plan}.txt"
        src_lbl_path = os.path.join(accepted_dir, src_lbl_file)
        
        label_lines = []
        if os.path.exists(src_lbl_path):
            with open(src_lbl_path, "r") as lf:
                label_lines = lf.readlines()
        else:
            print(f" [{idx+1}/{len(plans_list)}] Label missing for: {base_plan}. Force generating from SVG...")
            svg_path = get_svg_path(base_plan)
            if svg_path:
                label_lines = force_generate_label(svg_path, orig_img_path)
                print(f"    Successfully generated {len(label_lines)} labels.")
            else:
                print(f"    Warning: SVG not found in cache for {base_plan}")
                
        with open(dst_lbl_path, "w") as f:
            for line in label_lines:
                f.write(line.strip() + "\n")
                
        count += 1
        if count % 20 == 0:
            print(f"  Processed {count}/{len(plans_list)} plans")

process_split_plans(train_plans, "train")
process_split_plans(val_plans, "val")

yaml_content = f"""path: /content/good_ones_dataset
train: images/train
val: images/val

names:
  0: door
"""
with open(os.path.join(base_dataset_dir, "dataset.yaml"), "w") as f:
    f.write(yaml_content)

zip_path = os.path.join(os.path.dirname(base_dataset_dir), "good_ones_dataset.zip")
if os.path.exists(zip_path):
    os.remove(zip_path)
shutil.make_archive(
    base_name=os.path.join(os.path.dirname(base_dataset_dir), "good_ones_dataset"),
    format="zip",
    root_dir=base_dataset_dir
)
print("\nDataset packaging complete! Zip archive created successfully.")
