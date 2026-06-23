import re
import numpy as np
import cv2
from transform_resolver import get_global_transform
from svg_parser import parse_points, extract_path_points, collect_direct_geometry
from door_extractor import is_furniture

def extract_svg_walls(soup, scale_x, scale_y, dx, dy, width, height, dist_map=None):
    """
    Renders SVG wall outlines into a binary mask of size (height, width).
    Walls are extracted from elements with class matching Wall External, Wall Internal, Wall.
    The points are resolved to global SVG space, then coarsely scaled/translated to PNG space.
    If dist_map is provided, elements with very low coarse overlap (< 25%) are filtered out as outliers.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    candidates = []
    
    # We walk all geometry tags: polygon, polyline, path, rect, line
    for tag in soup.find_all(["polygon", "polyline", "path", "rect", "line"]):
        # Check if the tag itself or any of its parents is excluded (furniture, doors, windows, etc.)
        is_excluded = False
        for p in [tag] + list(tag.parents):
            if p.name == '[document]':
                continue
            p_text = (str(p.get("id", "")) + " " + str(p.get("class", ""))).lower()
            if any(x in p_text for x in ["door", "window", "glass", "furniture", "appliance", "cabinet", "bench", "bath", "sink", "toilet", "stove", "counter", "closet", "wardrobe", "cupboard", "kitchen", "shower", "tub"]):
                is_excluded = True
                break
                
        if is_excluded:
            continue
            
        # Check if the tag or any of its parents is a wall/railing
        is_wall_element = False
        curr = tag
        while curr is not None and curr.name != '[document]':
            t_id = str(curr.get("id", ""))
            t_class = curr.get("class", [])
            t_class_str = " ".join(t_class) if isinstance(t_class, list) else str(t_class)
            t_text = (t_id + " " + t_class_str).lower()
            if re.search(r"\bwall\b|\brailing\b", t_text):
                is_wall_element = True
                break
            curr = curr.parent
            
        if not is_wall_element:
            continue
            
        # Get global transform matrix
        m_global = get_global_transform(tag)
        
        # Collect local points for this specific element
        local_points = []
        if tag.name in ["polygon", "polyline"]:
            local_points = parse_points(tag.get("points", ""))
        elif tag.name == "rect":
            try:
                x = float(tag.get("x", 0))
                y = float(tag.get("y", 0))
                w = float(tag.get("width", 0))
                h = float(tag.get("height", 0))
                local_points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            except:
                pass
        elif tag.name == "line":
            try:
                x1 = float(tag.get("x1", 0))
                y1 = float(tag.get("y1", 0))
                x2 = float(tag.get("x2", 0))
                y2 = float(tag.get("y2", 0))
                local_points = [(x1, y1), (x2, y2)]
            except:
                pass
        elif tag.name == "path":
            local_points = extract_path_points(tag.get("d", ""), samples=20)
            
        if not local_points:
            continue
            
        # Transform local points to global SVG space, then map to image space
        img_points = []
        for pt in local_points:
            # Global SVG space
            pt_glob = m_global @ np.array([pt[0], pt[1], 1.0])
            # Coarse image space
            x_img = pt_glob[0] * scale_x + dx
            y_img = pt_glob[1] * scale_y + dy
            img_points.append([x_img, y_img])
            
        img_points = np.array(img_points, dtype=np.int32)
        if len(img_points) >= 2:
            candidates.append(img_points)
            
    # Draw elements, filtering by dist_map if provided
    for pts in candidates:
        if dist_map is not None:
            # Create a temporary mask to count inliers for this element
            temp_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.polylines(temp_mask, [pts], isClosed=True, color=255, thickness=2)
            active_pts = np.argwhere(temp_mask > 0)
            if len(active_pts) == 0:
                continue
            
            # Lookup distances in dist_map.
            # Coarse alignment can be slightly off, so we use a search radius of 25 pixels
            # to check if the element has ANY correspondence in the image walls.
            distances = dist_map[active_pts[:, 0], active_pts[:, 1]]
            inliers = np.sum(distances <= 25.0)
            inlier_ratio = inliers / len(active_pts)
            
            # Filter out wall elements with extremely low coarse overlap
            if inlier_ratio < 0.25:
                continue
                
        cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=2)
        
    return mask
