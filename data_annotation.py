import os
import re
import cv2
import numpy as np
from bs4 import BeautifulSoup
from svgpathtools import parse_path
from scipy.spatial import ConvexHull

SVG_FILE = "./model1.svg"
PNG_FILE = "./F1_original (1).png"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_points(points_str):
    coords = []
    nums = re.findall(r"[-+]?\d*\.?\d+", points_str)
    for i in range(0, len(nums), 2):
        try:
            x = float(nums[i])
            y = float(nums[i + 1])
            coords.append((x, y))
        except:
            pass
    return coords


def bbox_from_coords(coords):
    if len(coords) == 0:
        return None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    )


def extract_path_points(path_d, samples=200):
    try:
        path = parse_path(path_d)
        pts = []
        for seg in path:
            for t in np.linspace(0, 1, samples):
                p = seg.point(t)
                pts.append((p.real, p.imag))
        return pts
    except Exception:
        return []


def collect_geometry(node):
    coords = []
    polygons = node.find_all(["polygon", "polyline"])
    for poly in polygons:
        pts = parse_points(poly.get("points", ""))
        coords.extend(pts)

    paths = node.find_all("path")
    for p in paths:
        pts = extract_path_points(p.get("d", ""))
        coords.extend(pts)

    lines = node.find_all("line")
    for l in lines:
        try:
            x1 = float(l.get("x1"))
            y1 = float(l.get("y1"))
            x2 = float(l.get("x2"))
            y2 = float(l.get("y2"))
            coords.extend([(x1, y1), (x2, y2)])
        except:
            pass

    rects = node.find_all("rect")
    for r in rects:
        try:
            x = float(r.get("x"))
            y = float(r.get("y"))
            w = float(r.get("width"))
            h = float(r.get("height"))
            coords.extend(
                [
                    (x, y),
                    (x + w, y),
                    (x + w, y + h),
                    (x, y + h),
                ]
            )
        except:
            pass
    return coords


def collect_direct_geometry(node):
    coords = []
    polygons = node.find_all(["polygon", "polyline"], recursive=False)
    for poly in polygons:
        pts = parse_points(poly.get("points", ""))
        coords.extend(pts)
        
    rects = node.find_all("rect", recursive=False)
    for r in rects:
        try:
            x = float(r.get("x"))
            y = float(r.get("y"))
            w = float(r.get("width"))
            h = float(r.get("height"))
            coords.extend([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        except:
            pass
            
    lines = node.find_all("line", recursive=False)
    for l in lines:
        try:
            coords.extend([(float(l.get("x1")), float(l.get("y1"))), (float(l.get("x2")), float(l.get("y2")))])
        except:
            pass
    return coords


def get_polygon_coords(coords):
    unique_coords = []
    seen = set()
    for pt in coords:
        rounded = (round(pt[0], 4), round(pt[1], 4))
        if rounded not in seen:
            seen.add(rounded)
            unique_coords.append(pt)
            
    if len(unique_coords) < 3:
        bbox = bbox_from_coords(coords)
        if not bbox:
            return []
        xmin, ymin, xmax, ymax = bbox
        return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        
    try:
        hull = ConvexHull(unique_coords)
        return [unique_coords[idx] for idx in hull.vertices]
    except Exception:
        bbox = bbox_from_coords(coords)
        if not bbox:
            return []
        xmin, ymin, xmax, ymax = bbox
        return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def is_furniture(tag):
    for p in [tag] + list(tag.parents):
        if p.name == '[document]':
            continue
        p_text = (str(p.get("id", "")) + " " + str(p.get("class", ""))).lower()
        if any(x in p_text for x in ["furniture", "appliance", "cabinet", "bench"]):
            return True
    return False


# Load SVG
if not os.path.exists(SVG_FILE):
    print(f"Error: {SVG_FILE} not found!")
    exit(1)

with open(SVG_FILE, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "xml")

# Get SVG dimensions
svg_tag = soup.find("svg")
if not svg_tag:
    print("Error: No <svg> tag found in file!")
    exit(1)

try:
    svg_w = float(svg_tag.get("width"))
    svg_h = float(svg_tag.get("height"))
except (ValueError, TypeError):
    viewbox = svg_tag.get("viewBox", "").split()
    if len(viewbox) == 4:
        svg_w = float(viewbox[2])
        svg_h = float(viewbox[3])
    else:
        print("Error: Could not extract width and height from SVG tag or viewBox!")
        exit(1)

# Get PNG dimensions (for fallback)
if os.path.exists(PNG_FILE):
    img_png = cv2.imread(PNG_FILE)
    png_h, png_w = img_png.shape[:2]
else:
    png_w, png_h = svg_w, svg_h

# --- DYNAMIC ALIGNMENT ---
# 1. Find boundaries of the structural walls/railings in the SVG
structural_svg_coords = []
for tag in soup.find_all():
    tag_text = (str(tag.get("id", "")) + " " + str(tag.get("class", ""))).lower()
    if any(x in tag_text for x in ["wall", "railing"]):
        # Collect only direct child geometry to avoid nesting issues
        structural_svg_coords.extend(collect_direct_geometry(tag))

if len(structural_svg_coords) > 0:
    xmin_svg, ymin_svg, xmax_svg, ymax_svg = bbox_from_coords(structural_svg_coords)
else:
    xmin_svg, ymin_svg, xmax_svg, ymax_svg = 0.0, 0.0, svg_w, svg_h

# 2. Find boundaries of dark pixels (black lines of building border) in the PNG
if os.path.exists(PNG_FILE):
    img_gray = cv2.imread(PNG_FILE, cv2.IMREAD_GRAYSCALE)
    dark_pixels = np.where(img_gray < 50)
    if len(dark_pixels[0]) > 0:
        ymin_png = float(dark_pixels[0].min())
        ymax_png = float(dark_pixels[0].max())
        xmin_png = float(dark_pixels[1].min())
        xmax_png = float(dark_pixels[1].max())
    else:
        xmin_png, ymin_png, xmax_png, ymax_png = 0.0, 0.0, float(png_w), float(png_h)
else:
    xmin_png, ymin_png, xmax_png, ymax_png = xmin_svg, ymin_svg, xmax_svg, ymax_svg

# 3. Calculate dynamic scale and translation
w_svg = xmax_svg - xmin_svg
w_png = xmax_png - xmin_png
scale = w_png / w_svg

dx = xmin_png - xmin_svg * scale
dy = ymin_png - ymin_svg * scale

print(f"SVG Dimensions: Width = {svg_w}, Height = {svg_h}")
print(f"PNG Dimensions: Width = {png_w}, Height = {png_h}")
print(f"Structural SVG Bounds: xmin={xmin_svg:.2f}, ymin={ymin_svg:.2f}, xmax={xmax_svg:.2f}, ymax={ymax_svg:.2f}")
print(f"Structural PNG Bounds: xmin={xmin_png:.2f}, ymin={ymin_png:.2f}, xmax={xmax_png:.2f}, ymax={ymax_png:.2f}")
print(f"Calculated Alignment: Scale = {scale:.6f}, dx = {dx:.2f}, dy = {dy:.2f}")

yolo_bbox_lines = []
yolo_seg_lines = []

for tag in soup.find_all():
    tag_text = (
        str(tag.get("id", "")) +
        " " +
        str(tag.get("class", ""))
    ).lower()

    # Filter out furniture
    if is_furniture(tag):
        continue

    # Class Mapping: 0 for door, 1 for window
    class_id = None
    if re.search(r"\bdoor", tag_text):
        class_id = 0
    elif re.search(r"\bwindow", tag_text):
        class_id = 1

    if class_id is None:
        continue

    # Extract all points in SVG coordinate space
    coords = collect_geometry(tag)
    if len(coords) == 0:
        continue

    # Map the points to PNG pixel coordinates using dynamic alignment parameters
    transformed_coords = [(pt[0] * scale + dx, pt[1] * scale + dy) for pt in coords]

    # 1. BBox Label Generation (relative to PNG dimensions)
    bbox = bbox_from_coords(transformed_coords)
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        
        # Calculate normalized center and sizes relative to PNG
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

    # 2. Segmentation Label Generation (relative to PNG dimensions)
    poly_pts = get_polygon_coords(transformed_coords)
    if len(poly_pts) > 0:
        normalized_coords = []
        for pt in poly_pts:
            x_norm = pt[0] / png_w
            y_norm = pt[1] / png_h
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            normalized_coords.append(f"{x_norm:.6f} {y_norm:.6f}")
        
        yolo_seg_lines.append(f"{class_id} " + " ".join(normalized_coords))

# Write files
bbox_output_file = os.path.join(OUTPUT_DIR, "model.txt")
with open(bbox_output_file, "w") as f:
    f.write("\n".join(yolo_bbox_lines) + "\n")

seg_output_file = os.path.join(OUTPUT_DIR, "model_seg.txt")
with open(seg_output_file, "w") as f:
    f.write("\n".join(yolo_seg_lines) + "\n")

print(f"\nSaved {len(yolo_bbox_lines)} YOLO Detection BBox annotations to: {bbox_output_file}")
print(f"Saved {len(yolo_seg_lines)} YOLO Segmentation Polygon annotations to: {seg_output_file}")