import os
import re
import numpy as np
from bs4 import BeautifulSoup
from scipy.spatial import ConvexHull
from svg_parser import collect_geometry, bbox_from_coords, extract_path_points

def is_furniture(tag):
    for p in [tag] + list(tag.parents):
        if p.name == '[document]':
            continue
        p_text = (str(p.get("id", "")) + " " + str(p.get("class", ""))).lower()
        if any(x in p_text for x in ["furniture", "appliance", "cabinet", "bench", "bath", "sink", "toilet", "stove", "counter", "closet", "wardrobe", "cupboard", "kitchen", "shower", "tub"]):
            return True
    return False


def get_path_start(path_tag):
    d = path_tag.get("d", "")
    match = re.search(r"M\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)", d, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
    nums = re.findall(r"[-+]?\d*\.?\d+", d)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None


def get_hinge_point(threshold_bbox, path_start):
    if not threshold_bbox or not path_start:
        return None
    xmin, ymin, xmax, ymax = threshold_bbox
    corners = [(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)]
    min_dist = float('inf')
    hinge = corners[0]
    for c in corners:
        dist = (c[0] - path_start[0])**2 + (c[1] - path_start[1])**2
        if dist < min_dist:
            min_dist = dist
            hinge = c
    return hinge


def detect_swing_direction(path_pts, hinge):
    if not path_pts or not hinge:
        return "none"
    
    angles = []
    for pt in path_pts:
        dx = pt[0] - hinge[0]
        dy = pt[1] - hinge[1]
        angles.append(np.arctan2(dy, dx))
        
    if len(angles) < 2:
        return "none"
        
    total_diff = 0.0
    for i in range(len(angles) - 1):
        diff = angles[i+1] - angles[i]
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        total_diff += diff
        
    if total_diff > 0.01:
        return "clockwise"
    elif total_diff < -0.01:
        return "counter_clockwise"
    return "none"


def detect_orientation(threshold_bbox, path_pts, hinge):
    if not threshold_bbox:
        return 0
    xmin, ymin, xmax, ymax = threshold_bbox
    w_t = xmax - xmin
    h_t = ymax - ymin
    
    if not path_pts or not hinge:
        return 90 if h_t > w_t else 0

    mean_dx = np.mean([pt[0] - hinge[0] for pt in path_pts])
    mean_dy = np.mean([pt[1] - hinge[1] for pt in path_pts])
    
    if h_t > w_t:  # Vertical door
        return 90 if mean_dx >= 0 else 270
    else:  # Horizontal door
        return 180 if mean_dy >= 0 else 0


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


def extract_door_info(tag, svg_w, svg_h, scale, dx, dy, png_w, png_h):
    tag_text = (str(tag.get("id", "")) + " " + str(tag.get("class", ""))).lower()
    
    door_type = tag.get("class")
    if not door_type:
        door_type = "Door"
    elif isinstance(door_type, list):
        door_type = " ".join(door_type)
    else:
        door_type = str(door_type)

    svg_coords = collect_geometry(tag)
    if len(svg_coords) == 0:
        return None

    # Extract threshold bbox
    threshold_node = tag.find(lambda t: t.name in ["g", "polygon", "rect"] and any(x in str(t.get("class", "")).lower() or x in str(t.get("id", "")).lower() for x in ["threshold"]))
    if threshold_node:
        threshold_coords = collect_geometry(threshold_node)
    else:
        polys = tag.find_all(["polygon", "rect"])
        threshold_coords = []
        min_area = float('inf')
        for poly in polys:
            pts = collect_geometry(poly)
            bbox = bbox_from_coords(pts)
            if bbox:
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area < min_area:
                    min_area = area
                    threshold_coords = pts
                    
    threshold_bbox = bbox_from_coords(threshold_coords) if len(threshold_coords) > 0 else None

    # Get path start point for hinge estimation
    paths = tag.find_all("path")
    path_start = None
    path_pts = []
    for path in paths:
        if not path.parent or "threshold" not in str(path.parent.get("id", "")).lower() and "threshold" not in str(path.parent.get("class", "")).lower():
            start = get_path_start(path)
            if start:
                path_start = start
                path_pts.extend(extract_path_points(path.get("d", ""), samples=20))

    hinge = get_hinge_point(threshold_bbox, path_start)
    opening_direction = detect_swing_direction(path_pts, hinge)
    orientation = detect_orientation(threshold_bbox, path_pts, hinge)

    transformed_coords = [(pt[0] * scale + dx, pt[1] * scale + dy) for pt in svg_coords]
    
    bbox_png = bbox_from_coords(transformed_coords)
    if not bbox_png:
        return None

    polygon_png = get_polygon_coords(transformed_coords)
    hinge_png = (hinge[0] * scale + dx, hinge[1] * scale + dy) if hinge else None

    return {
        "door_type": door_type,
        "bbox": bbox_png,
        "polygon": polygon_png,
        "orientation": orientation,
        "opening_direction": opening_direction,
        "hinge": hinge_png,
        "raw_geometry_count": len(svg_coords)
    }
