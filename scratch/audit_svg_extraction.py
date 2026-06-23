import os
import sys
import glob
import random
import json
import re
import cv2
import numpy as np
from bs4 import BeautifulSoup

# Ensure correct path alignment for imports
curr_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(curr_dir)
src_dir = os.path.join(project_dir, "src")
for path in [project_dir, src_dir]:
    if path not in sys.path:
        sys.path.append(path)

from svg_parser import get_svg_dimensions, parse_points
from transform_resolver import get_global_transform
from coordinate_mapper import CoordinateMapper
from door_anatomy_extractor import extract_door_info_cad
from door_extractor import is_furniture

def extract_windows_svg(soup, mapper, source_svg_file):
    """
    Extracts window polygons from SVG geometry nodes directly.
    """
    windows = []
    window_idx = 0
    for tag in soup.find_all(['rect', 'polygon', 'path']):
        parent_id = str(tag.parent.get('id', '')).lower() if tag.parent else ""
        parent_class = str(tag.parent.get('class', '')).lower() if tag.parent else ""
        tag_id = str(tag.get('id', '')).lower()
        tag_class = str(tag.get('class', '')).lower()
        
        is_window = any(k in parent_id or k in parent_class or k in tag_id or k in tag_class for k in ['window', 'glass'])
        if not is_window: continue
        
        m_glob = get_global_transform(tag)
        pts = []
        if tag.name == 'path':
            d = tag.get('d', '')
            try:
                from svgpathtools import parse_path
                path_obj = parse_path(d)
                for segment in path_obj:
                    for t in np.linspace(0, 1, 5):
                        pt = segment.point(t)
                        pt_glob = m_glob @ np.array([pt.real, pt.imag, 1.0])
                        pts.append((pt_glob[0], pt_glob[1]))
            except: pass
        elif tag.name == 'polygon':
            coords = parse_points(tag.get('points', ''))
            for pt in coords:
                pt_glob = m_glob @ np.array([pt[0], pt[1], 1.0])
                pts.append((pt_glob[0], pt_glob[1]))
        elif tag.name == 'rect':
            try:
                rx = float(tag.get('x', 0))
                ry = float(tag.get('y', 0))
                rw = float(tag.get('width', 0))
                rh = float(tag.get('height', 0))
                coords = [(rx, ry), (rx+rw, ry), (rx+rw, ry+rh), (rx, ry+rh)]
                for pt in coords:
                    pt_glob = m_glob @ np.array([pt[0], pt[1], 1.0])
                    pts.append((pt_glob[0], pt_glob[1]))
            except: pass
            
        if len(pts) >= 3:
            pts_png = [mapper.svg_to_image(pt[0], pt[1]) for pt in pts]
            xs = [p[0] for p in pts_png]
            ys = [p[1] for p in pts_png]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            width_px = float(max(xs) - min(xs))
            
            parent_group = tag.parent.name if tag.parent else "svg"
            parent_id_str = tag.parent.get("id", "") if tag.parent else ""
            source_svg_group = f"{parent_group}#{parent_id_str}" if parent_id_str else parent_group
            
            windows.append({
                "window_id": f"w_{window_idx}",
                "bbox": bbox,
                "polygon": pts_png,
                "width_px": width_px,
                "scale_available": False,
                "source_class": "window",
                "source_svg_group": source_svg_group,
                "source_svg_file": source_svg_file
            })
            window_idx += 1
            
    return windows

def main():
    print("="*60)
    print("SVG Extraction Quality Assurance Audit Tool")
    print("="*60)
    
    # 1. Gather SVG files
    svg_files = []
    # Search locally first
    local_search = os.path.join(project_dir, "local_test_data", "**", "*.svg")
    svg_files.extend(glob.glob(local_search, recursive=True))
    
    # Try CubiCasa5K cache via kagglehub if available
    try:
        import kagglehub
        raw_dataset_path = kagglehub.dataset_download("qmarva/cubicasa5k")
        kaggle_search = os.path.join(raw_dataset_path, "**", "model*.svg")
        svg_files.extend(glob.glob(kaggle_search, recursive=True))
    except Exception as e:
        print(f"Info: Kagglehub check skipped/failed ({e})")
        
    svg_files = list(set(svg_files))
    if not svg_files:
        print("ERROR: No SVG files found to audit! Place some .svg files inside local_test_data/.")
        sys.exit(1)
        
    print(f"Found total {len(svg_files)} SVGs. Sampling up to 100 for audit...")
    random.seed(42)
    random.shuffle(svg_files)
    sample_svgs = svg_files[:min(100, len(svg_files))]
    
    qa_samples_dir = os.path.join(project_dir, "qa_samples")
    os.makedirs(qa_samples_dir, exist_ok=True)
    
    # Check if CairoSVG is available for rendering background
    has_cairosvg = False
    try:
        import cairosvg
        # Trigger an internal check to make sure cairo-2 dll is loadable
        cairosvg.svg2png(bytestring=b'<svg></svg>')
        has_cairosvg = True
        print("CairoSVG is available. Rendered backgrounds will be high-fidelity floorplans.")
    except Exception as e:
        print(f"CairoSVG initialization failed (DLLs may be missing locally): {e}. Falling back to blank white canvas layouts.")
        
    audit_results = []
    
    for i, svg_path in enumerate(sample_svgs):
        print(f"[{i+1}/{len(sample_svgs)}] Auditing: {os.path.basename(svg_path)}")
        basename = os.path.splitext(os.path.basename(svg_path))[0]
        plan_dir_name = os.path.basename(os.path.dirname(svg_path))
        plan_id = f"{plan_dir_name}_{basename}"
        
        try:
            with open(svg_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'xml')
            
            svg_w, svg_h = get_svg_dimensions(soup)
            
            # Setup CoordinateMapper from SVG to 1024x1024 space
            M_scale = np.array([
                [1024.0 / svg_w, 0, 0],
                [0, 1024.0 / svg_h, 0],
                [0, 0, 1]
            ])
            mapper = CoordinateMapper(M_scale)
            
            # Extract doors
            doors = []
            door_idx = 0
            for tag in soup.find_all():
                tag_id = str(tag.get("id", ""))
                tag_classes = tag.get("class", [])
                tag_classes_str = " ".join(tag_classes) if isinstance(tag_classes, list) else str(tag_classes)
                tag_text = (tag_id + " " + tag_classes_str).lower()
                
                if not re.search(r"\bdoor\b", tag_text): continue
                if is_furniture(tag): continue
                # Parent door group check to prevent double parsing
                if any(p in [t for t in soup.find_all() if re.search(r"\bdoor\b", (str(t.get("id","")) + " " + str(t.get("class",""))).lower())] for p in tag.parents):
                    continue
                
                door_res = extract_door_info_cad(tag, mapper, source_svg_file=os.path.basename(svg_path), door_idx=door_idx)
                if door_res:
                    if isinstance(door_res, list):
                        doors.extend(door_res)
                        door_idx += len(door_res)
                    else:
                        doors.append(door_res)
                        door_idx += 1
                        
            # Extract windows
            windows = extract_windows_svg(soup, mapper, os.path.basename(svg_path))
            
            # Draw overlay image
            # Initialize 1024x1024 canvas
            canvas_path = os.path.join(qa_samples_dir, f"temp_{plan_id}.png")
            if has_cairosvg:
                try:
                    cairosvg.svg2png(url=svg_path, write_to=canvas_path, output_width=1024, output_height=1024)
                    canvas = cv2.imread(canvas_path)
                    os.remove(canvas_path)
                except Exception as ex:
                    print(f"  Warning: CairoSVG failed to render background for {basename}: {ex}")
                    canvas = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
            else:
                canvas = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
                
            # Draw door polygons (Green)
            for d in doors:
                poly = np.array(d["polygon"], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [poly], isClosed=True, color=(0, 200, 0), thickness=2)
                # Draw small circle on Hinge
                if "hinge" in d and d["hinge"] is not None:
                    # Some versions return hinge coords
                    pass
                    
            # Draw window polygons (Blue)
            for w in windows:
                poly = np.array(w["polygon"], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [poly], isClosed=True, color=(220, 50, 50), thickness=2)
                
            # Save visual audit
            out_img_path = os.path.join(qa_samples_dir, f"audit_{plan_id}.png")
            cv2.imwrite(out_img_path, canvas)
            
            audit_results.append({
                "plan_id": plan_id,
                "svg_dimensions": [svg_w, svg_h],
                "doors_count": len(doors),
                "windows_count": len(windows),
                "visual_overlay": out_img_path
            })
            
        except Exception as e:
            print(f"  Failed to process SVG {svg_path}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save log report
    audit_report_path = os.path.join(qa_samples_dir, "audit_report.json")
    with open(audit_report_path, "w") as f:
        json.dump(audit_results, f, indent=2)
        
    print("\n" + "="*40)
    print("Audit Stage Complete!")
    print(f"Processed plans: {len(audit_results)}")
    print(f"Audit overlays saved under: {qa_samples_dir}")
    print(f"Detailed audit log available at: {audit_report_path}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
