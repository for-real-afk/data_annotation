import os
import re
import sys
import shutil
import argparse
import random
import json
import cv2
import numpy as np
from bs4 import BeautifulSoup
import kagglehub
import multiprocessing

# Add current and src directories to path for multiprocessing and submodule imports
curr_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(curr_dir, "src")
for path in [curr_dir, src_dir]:
    if path not in sys.path:
        sys.path.append(path)

# Import local modules
from svg_parser import get_svg_dimensions, calculate_alignment
from door_extractor import is_furniture
from segmentation_generator import generate_yolo_labels
from qa_visualizer import generate_qa_overlay
from dataset_stats import compile_global_stats
from transform_resolver import get_global_transform
from coordinate_mapper import CoordinateMapper
from wall_extractor import extract_svg_walls
from image_wall_detector import detect_image_walls
from svg_image_registrar import register_masks, compute_alignment_score
from alignment_validator import generate_alignment_validation
from door_anatomy_extractor import extract_door_info_cad
from drift_detector import detect_drift

def process_plan_worker(args_tuple):
    pair, args, data_dir, images_dir, labels_dir, segments_dir, crops_dir, qa_alignment_dir = args_tuple
    svg_path = pair["svg_path"]
    png_path = pair["png_path"]
    
    # Create unique plan name from parent folders to avoid naming collisions
    rel_root = os.path.relpath(pair["root"], data_dir)
    plan_name = rel_root.replace(os.sep, "_")
    
    try:
        # Load image once
        img = cv2.imread(png_path)
        if img is None:
            return None, f"Error: Image not found or invalid: {png_path}"
            
        with open(svg_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "xml")
            
        svg_w, svg_h = get_svg_dimensions(soup)
        scale_x_coarse, scale_y_coarse, dx_coarse, dy_coarse, png_w, png_h = calculate_alignment(soup, img, svg_w, svg_h)
        
        # Build coarse transform matrix (non-uniform scaling)
        M_coarse = np.array([
            [scale_x_coarse, 0, dx_coarse],
            [0, scale_y_coarse, dy_coarse],
            [0, 0, 1]
        ], dtype=float)
        
        # Detect walls in the raster image (simple threshold for DT score)
        image_wall_mask_thresh = detect_image_walls(img)
        
        # Compute distance transform on coarse space for outlier filtering
        img_inv_coarse = cv2.bitwise_not(image_wall_mask_thresh)
        dist_map_coarse = cv2.distanceTransform(img_inv_coarse, cv2.DIST_L2, 5)
        
        # Render SVG walls in coarse space (outlines), filtering outlier elements
        svg_wall_mask = extract_svg_walls(soup, scale_x_coarse, scale_y_coarse, dx_coarse, dy_coarse, png_w, png_h, dist_map=dist_map_coarse)
        
        # Register SVG wall mask with image wall mask using outlines
        M_refine = register_masks(svg_wall_mask, image_wall_mask_thresh)
        
        # Safety check: Compare score before and after refinement, fallback to Identity if refinement is worse
        score_coarse = compute_alignment_score(svg_wall_mask, image_wall_mask_thresh, np.eye(3))
        score_refined = compute_alignment_score(svg_wall_mask, image_wall_mask_thresh, M_refine)
        
        if score_coarse > score_refined:
            M_refine = np.eye(3, dtype=np.float32)
            alignment_score = score_coarse
        else:
            alignment_score = score_refined
            
        # Total transform matrix: SVG -> Coarse -> Refined (Image space)
        M_total = M_refine @ M_coarse
        
        # Extract scale, translation, rotation from M_total for logging
        scale_x = np.hypot(M_total[0, 0], M_total[1, 0])
        scale_y = np.hypot(M_total[0, 1], M_total[1, 1])
        tx = M_total[0, 2]
        ty = M_total[1, 2]
        rotation = np.degrees(np.arctan2(M_total[1, 0], M_total[0, 0])) % 360
        
        align_meta = {
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "tx": float(tx),
            "ty": float(ty),
            "rotation": float(rotation),
            "alignment_score": alignment_score
        }
        
        # Check alignment score threshold
        status = "aligned"
        if alignment_score < 0.85:
            status = "rejected"
        elif alignment_score < 0.95:
            status = "review"
            
        align_meta["status"] = status
        
        # Generate validation visualization (val_plan_name.png)
        val_filename = f"val_{plan_name}.png"
        val_path = os.path.join(qa_alignment_dir, val_filename)
        
        if status == "rejected":
            # Save validation overlay anyway for review, but do not export dataset
            generate_alignment_validation(svg_wall_mask, image_wall_mask_thresh, [], M_refine, val_path)
            return {
                "plan_name": plan_name,
                "status": "rejected",
                "alignment_score": alignment_score,
                "doors_info": []
            }, f"Plan {plan_name} rejected: Alignment score {alignment_score:.3f} < 0.85"
            
        # Create Coordinate Mapper using cumulative transformation matrix
        try:
            mapper = CoordinateMapper(M_total)
        except np.linalg.LinAlgError:
            return {
                "plan_name": plan_name,
                "status": "rejected",
                "alignment_score": 0.0,
                "doors_info": []
            }, f"Plan {plan_name} rejected: Singular alignment transformation matrix"
        
        doors_info = []
        schema_variant = "standard"
        has_id_door = False
        has_class_door = False
        has_id_doors = False
        
        # Collect all door element tags
        door_tags = []
        for tag in soup.find_all():
            tag_id = str(tag.get("id", ""))
            tag_classes = tag.get("class", [])
            tag_classes_str = " ".join(tag_classes) if isinstance(tag_classes, list) else str(tag_classes)
            tag_text = (tag_id + " " + tag_classes_str).lower()
            
            if tag.get("id") == "Door":
                has_id_door = True
            if tag.get("id") == "Doors":
                has_id_doors = True
            if "door" in tag_classes_str.lower():
                has_class_door = True

            class_id = None
            if re.search(r"\bdoor\b", tag_text):
                class_id = 0
            
            if class_id is None:
                continue
                
            if is_furniture(tag):
                continue
                
            # Avoid duplicate nested tags
            if any(p in door_tags for p in tag.parents):
                continue
                
            door_tags.append(tag)
            
        # Extract CAD/BIM anatomy for each door tag
        for tag in door_tags:
            door_attr = extract_door_info_cad(tag, mapper)
            if door_attr:
                if isinstance(door_attr, list):
                    for sub_attr in door_attr:
                        drift_res = detect_drift(sub_attr, image_wall_mask_thresh, img.shape)
                        sub_attr["drift_status"] = drift_res["status"]
                        if "reason" in drift_res:
                            sub_attr["drift_reason"] = drift_res["reason"]
                        doors_info.append(sub_attr)
                else:
                    drift_res = detect_drift(door_attr, image_wall_mask_thresh, img.shape)
                    door_attr["drift_status"] = drift_res["status"]
                    if "reason" in drift_res:
                        door_attr["drift_reason"] = drift_res["reason"]
                    doors_info.append(door_attr)
                
        # Generate validation visualization with mapped doors
        generate_alignment_validation(svg_wall_mask, image_wall_mask_thresh, doors_info, M_refine, val_path)
        
        if has_id_doors:
            schema_variant = "id_doors_plural"
        elif has_id_door and has_class_door:
            schema_variant = "id_door_and_class"
        elif has_id_door:
            schema_variant = "id_door_only"
        elif has_class_door:
            schema_variant = "class_door_only"
        else:
            schema_variant = "custom"
            
        # Filter doors that didn't drift
        aligned_doors = [d for d in doors_info if d.get("drift_status") == "aligned"]
        
        # Copy image to dataset/images/
        output_png_filename = f"{plan_name}.png"
        output_png_path = os.path.join(images_dir, output_png_filename)
        shutil.copy(png_path, output_png_path)
        
        # Generate YOLO detection and segmentation labels
        yolo_bbox, yolo_seg = generate_yolo_labels(aligned_doors, png_w, png_h)
        
        # Save YOLO segmentation labels
        label_filename = f"{plan_name}.txt"
        label_path = os.path.join(labels_dir, label_filename)
        with open(label_path, "w") as lf:
            lf.write("\n".join(yolo_seg) + "\n")
            
        # Save raw segments metadata JSON
        segment_filename = f"{plan_name}.json"
        segment_path = os.path.join(segments_dir, segment_filename)
        with open(segment_path, "w") as sf:
            json.dump({
                "plan_name": plan_name,
                "alignment_metadata": align_meta,
                "doors": doors_info
            }, sf, indent=2, default=lambda o: list(o) if isinstance(o, (set, np.ndarray)) else o)
            
        # Save crops with temporary names
        crops_meta = []
        for door_idx, door in enumerate(aligned_doors):
            bbox = door["bbox"]
            if not bbox:
                continue
            xmin, ymin, xmax, ymax = bbox
            pad = 5
            ixmin = max(0, int(xmin) - pad)
            iymin = max(0, int(ymin) - pad)
            ixmax = min(png_w, int(xmax) + pad)
            iymax = min(png_h, int(ymax) + pad)
            if ixmax - ixmin <= 0 or iymax - iymin <= 0:
                continue
                
            crop = img[iymin:iymax, ixmin:ixmax]
            temp_crop_filename = f"temp_{plan_name}_{door_idx}.png"
            temp_crop_path = os.path.join(crops_dir, temp_crop_filename)
            cv2.imwrite(temp_crop_path, crop)
            
            crops_meta.append({
                "temp_filename": temp_crop_filename,
                "door_type": door["door_type"],
                "orientation": door["orientation"],
                "opening_direction": door["opening_direction"],
                "svg": f"{plan_name}.svg",
                "width_mm": 900,
                "height_mm": 2100
            })
            
        result = {
            "plan_name": plan_name,
            "status": status,
            "alignment_score": alignment_score,
            "schema_variant": schema_variant,
            "doors_info": doors_info,
            "crops_meta": crops_meta
        }
        return result, None
    except Exception as e:
        import traceback
        return None, f"Error processing plan {plan_name}: {e}\n{traceback.format_exc()}"

def main():
    parser = argparse.ArgumentParser(description="SVG Door Dataset Generator")
    parser.add_argument("--data_dir", type=str, default="", help="Path to the Cubicasa5k dataset")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output directory for generated dataset")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of plans to process (0 for all)")
    parser.add_argument("--qa_limit", type=int, default=100, help="Number of plans to generate QA overlays for")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes (default: 4)")
    args = parser.parse_args()

    # Automatically locate or download dataset via kagglehub if data_dir not provided
    if not args.data_dir:
        print("Locating or downloading cubicasa5k dataset via kagglehub...")
        try:
            args.data_dir = kagglehub.dataset_download("qmarva/cubicasa5k")
            print(f"Dataset path: {args.data_dir}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            sys.exit(1)

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist!")
        sys.exit(1)

    # Setup directories
    images_dir = os.path.join(args.output_dir, "images")
    labels_dir = os.path.join(args.output_dir, "labels")
    segments_dir = os.path.join(args.output_dir, "segments")
    crops_dir = os.path.join(args.output_dir, "crops")
    qa_dir = os.path.join(args.output_dir, "qa")
    qa_alignment_dir = os.path.join(args.output_dir, "qa_alignment")

    for d in [images_dir, labels_dir, segments_dir, crops_dir, qa_dir, qa_alignment_dir]:
        os.makedirs(d, exist_ok=True)

    print("Scanning dataset directory recursively for plan pairs...")
    plan_pairs = []
    for root, dirs, files in os.walk(args.data_dir):
        svg_files = [f for f in files if f.endswith(".svg")]
        if not svg_files:
            continue
            
        for svg_name in svg_files:
            svg_path = os.path.join(root, svg_name)
            # Find corresponding PNG
            png_files = [f for f in files if f.endswith(".png") and "original" in f.lower()]
            if not png_files:
                png_files = [f for f in files if f.endswith(".png")]
            if not png_files:
                png_files = [f for f in files if f.endswith(".jpg")]
                
            if png_files:
                plan_pairs.append({
                    "svg_path": svg_path,
                    "png_path": os.path.join(root, png_files[0]),
                    "root": root
                })

    total_available = len(plan_pairs)
    print(f"Found {total_available} plan pairs (SVG + PNG).")

    if args.limit > 0:
        plan_pairs = plan_pairs[:args.limit]
        print(f"Limited processing to {len(plan_pairs)} plans.")

    # Prepare worker args
    worker_args = [
        (pair, args, args.data_dir, images_dir, labels_dir, segments_dir, crops_dir, qa_alignment_dir)
        for pair in plan_pairs
    ]

    pool_workers = min(args.workers, multiprocessing.cpu_count())
    print(f"Starting multiprocessing pool with {pool_workers} workers...")
    pool = multiprocessing.Pool(processes=pool_workers)

    all_processed_data = []
    crop_metadata_list = []
    schema_variants_set = set()
    crop_id = 1
    processed_count = 0
    rejected_count = 0
    accepted_count = 0

    results = pool.imap_unordered(process_plan_worker, worker_args)

    for res, err in results:
        processed_count += 1
        if err:
            print(f"[{processed_count}/{len(plan_pairs)}] {err}")
            rejected_count += 1
            continue
            
        plan_name = res["plan_name"]
        status = res["status"]
        alignment_score = res["alignment_score"]
        
        if status == "rejected":
            rejected_count += 1
            continue
            
        accepted_count += 1
        schema_variant = res["schema_variant"]
        doors_info = res["doors_info"]
        crops_meta = res["crops_meta"]
        
        schema_variants_set.add(schema_variant)
        
        # Rename crops sequentially in parent process
        for cm in crops_meta:
            temp_path = os.path.join(crops_dir, cm["temp_filename"])
            final_filename = f"door_{crop_id:05d}.png"
            final_path = os.path.join(crops_dir, final_filename)
            
            if os.path.exists(temp_path):
                try:
                    os.replace(temp_path, final_path)
                except Exception as ex:
                    print(f"Warning: Failed to rename {temp_path} to {final_path}: {ex}")
                
            crop_metadata_list.append({
                "door_id": crop_id,
                "door_type": cm["door_type"],
                "orientation": cm["orientation"],
                "opening_direction": cm["opening_direction"],
                "svg": cm["svg"],
                "width_mm": cm["width_mm"],
                "height_mm": cm["height_mm"]
            })
            crop_id += 1
            
        all_processed_data.append({
            "svg": f"{plan_name}.svg",
            "schema_variant": schema_variant,
            "doors": doors_info
        })
        
        if processed_count % 10 == 0 or processed_count == len(plan_pairs):
            print(f"[{processed_count}/{len(plan_pairs)}] Processed. Total doors extracted: {crop_id - 1}")

    pool.close()
    pool.join()

    # Create crops metadata index file
    crop_meta_path = os.path.join(crops_dir, "metadata.json")
    with open(crop_meta_path, "w") as f:
        json.dump(crop_metadata_list, f, indent=2)
    print(f"Saved crops metadata index to: {crop_meta_path}")

    # QA Visualization overlay generation
    print("Generating QA visualizations on a subset of plans...")
    sample_size = min(accepted_count, args.qa_limit)
    if sample_size > 0:
        # Filter only accepted plans for QA samples
        accepted_plan_names = {x["svg"].replace(".svg", "") for x in all_processed_data}
        accepted_plan_pairs = [p for p in plan_pairs if (os.path.relpath(p["root"], args.data_dir).replace(os.sep, "_")) in accepted_plan_names]
        qa_samples = random.sample(accepted_plan_pairs, sample_size)
        
        for idx, pair in enumerate(qa_samples):
            rel_root = os.path.relpath(pair["root"], args.data_dir)
            plan_name = rel_root.replace(os.sep, "_")
            
            # Load processed door info
            segment_path = os.path.join(segments_dir, f"{plan_name}.json")
            if os.path.exists(segment_path):
                with open(segment_path, "r") as sf:
                    plan_data = json.load(sf)
                    doors_info = plan_data.get("doors", [])
                    
                qa_output_path = os.path.join(qa_dir, f"sample_{idx+1:03d}_{plan_name}.png")
                # Filter to only draw aligned doors to verify what went to training
                aligned_doors = [d for d in doors_info if d.get("drift_status") == "aligned"]
                generate_qa_overlay(aligned_doors, pair["png_path"], qa_output_path)
                
        print(f"Generated {sample_size} QA visualization overlays in: {qa_dir}")

    # Compile global dataset statistics & taxonomies
    print("Compiling global statistics and taxonomies...")
    stats, door_types = compile_global_stats(all_processed_data, len(plan_pairs), args.output_dir)
    
    # Save statistics JSON
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as sf:
        json.dump(stats, sf, indent=2)
        
    # Save door types JSON
    types_path = os.path.join(args.output_dir, "door_types.json")
    with open(types_path, "w") as tf:
        json.dump(door_types, tf, indent=2)
        
    print(f"Saved dataset statistics report to: {stats_path}")
    print(f"Saved door taxonomy distribution to: {types_path}")
    
    print("\nDataset Generation Pipeline completed successfully!")
    print(f"Summary: Total plans processed = {processed_count}, Accepted = {accepted_count}, Rejected = {rejected_count}, Total doors extracted = {stats.get('total_doors', 0)}")

if __name__ == "__main__":
    main()
