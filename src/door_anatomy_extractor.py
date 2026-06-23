import re
import numpy as np
from svgpathtools import parse_path, Line, QuadraticBezier, CubicBezier, Arc

from transform_resolver import get_global_transform
from coordinate_mapper import CoordinateMapper
from hinge_detector import detect_hinge
from leaf_detector import detect_leaf
from arc_detector import detect_arc
from opening_direction_calculator import calculate_opening_direction
from orientation_calculator import calculate_orientation
from polygon_generator import generate_polygon
from svg_parser import collect_direct_geometry, bbox_from_coords, extract_path_points, parse_points

def get_path_start(path_tag):
    d = path_tag.get("d", "")
    match = re.search(r"M\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)", d, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
    nums = re.findall(r"[-+]?\d*\.?\d+", d)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None

def transform_point_complex(pt, matrix):
    x = pt.real
    y = pt.imag
    res = matrix @ np.array([x, y, 1.0])
    return complex(res[0], res[1])

def transform_segment(seg, matrix):
    if isinstance(seg, Line):
        return Line(
            transform_point_complex(seg.start, matrix),
            transform_point_complex(seg.end, matrix)
        )
    elif isinstance(seg, QuadraticBezier):
        return QuadraticBezier(
            transform_point_complex(seg.start, matrix),
            transform_point_complex(seg.control, matrix),
            transform_point_complex(seg.end, matrix)
        )
    elif isinstance(seg, CubicBezier):
        return CubicBezier(
            transform_point_complex(seg.start, matrix),
            transform_point_complex(seg.control1, matrix),
            transform_point_complex(seg.control2, matrix),
            transform_point_complex(seg.end, matrix)
        )
    elif isinstance(seg, Arc):
        scale_x = np.hypot(matrix[0, 0], matrix[0, 1])
        new_radius = seg.radius * scale_x
        return Arc(
            transform_point_complex(seg.start, matrix),
            new_radius,
            seg.rotation,
            seg.large_arc,
            seg.sweep,
            transform_point_complex(seg.end, matrix)
        )
    return seg

def extract_door_svg_geometry(tag):
    """
    Finds threshold bbox and swing paths/lines/polygons, and returns them in global SVG space.
    """
    threshold_coords = []
    threshold_ids = set()
    
    # 1. Identify threshold node/elements
    threshold_node = tag.find(lambda t: t.name in ["g", "polygon", "rect"] and any(
        x in str(t.get("class", "")).lower() or x in str(t.get("id", "")).lower() 
        for x in ["threshold"]
    ))
    
    if threshold_node:
        threshold_elements = [threshold_node] if threshold_node.name in ["polygon", "polyline", "rect", "line", "path"] else []
        threshold_elements.extend(threshold_node.find_all(["polygon", "polyline", "rect", "line", "path"], recursive=True))
        for child in threshold_elements:
            threshold_ids.add(id(child))
            m_glob = get_global_transform(child)
            coords = collect_direct_geometry(child) if child.name != "path" else extract_path_points(child.get("d", ""), samples=20)
            for pt in coords:
                pt_glob = m_glob @ np.array([pt[0], pt[1], 1.0])
                threshold_coords.append((pt_glob[0], pt_glob[1]))
    else:
        # Fallback search for threshold class/id in child tags
        for child in tag.find_all(["polygon", "rect"]):
            t_id = str(child.get("id", "")).lower()
            t_class = str(child.get("class", "")).lower()
            if "threshold" in t_id or "threshold" in t_class:
                threshold_ids.add(id(child))
                m_glob = get_global_transform(child)
                coords = collect_direct_geometry(child)
                for pt in coords:
                    pt_glob = m_glob @ np.array([pt[0], pt[1], 1.0])
                    threshold_coords.append((pt_glob[0], pt_glob[1]))
                    
        if not threshold_coords:
            # Absolute fallback to all polygons and rects in group if no explicit threshold is found
            for child in tag.find_all(["polygon", "rect"]):
                m_glob = get_global_transform(child)
                coords = collect_direct_geometry(child)
                bbox = bbox_from_coords(coords)
                if bbox:
                    xmin, ymin, xmax, ymax = bbox
                    pts = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                    pts_glob = [(m_glob @ np.array([pt[0], pt[1], 1.0]))[:2] for pt in pts]
                    threshold_coords.extend(pts_glob)
                    threshold_ids.add(id(child))
                    
    threshold_bbox = bbox_from_coords(threshold_coords) if threshold_coords else None
    
    segments = []
    path_start = None
    
    # 2. Iterate over all geometry tags in the group
    for geom_tag in tag.find_all(["path", "line", "rect", "polygon", "polyline"]):
        # Skip if it is marked as a threshold element
        if id(geom_tag) in threshold_ids:
            continue
            
        is_threshold = False
        curr = geom_tag.parent
        while curr is not None and curr != tag:
            t_id = str(curr.get("id", "")).lower()
            t_class = str(curr.get("class", "")).lower()
            if "threshold" in t_id or "threshold" in t_class or "panelarea" in t_id or "panelarea" in t_class:
                is_threshold = True
                break
            curr = curr.parent
            
        if is_threshold:
            continue
            
        # Filter duplicate threshold polygon/rect
        if geom_tag.name in ["polygon", "rect"] and threshold_bbox:
            coords = parse_points(geom_tag.get("points", "")) if geom_tag.name == "polygon" else []
            if geom_tag.name == "rect":
                try:
                    rx = float(geom_tag.get("x", 0))
                    ry = float(geom_tag.get("y", 0))
                    rw = float(geom_tag.get("width", 0))
                    rh = float(geom_tag.get("height", 0))
                    coords = [(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)]
                except:
                    pass
            if coords:
                m_glob = get_global_transform(geom_tag)
                coords_glob = [(m_glob @ np.array([pt[0], pt[1], 1.0]))[:2] for pt in coords]
                geom_bbox = bbox_from_coords(coords_glob)
                if geom_bbox:
                    if (abs(geom_bbox[0] - threshold_bbox[0]) < 1e-1 and
                        abs(geom_bbox[1] - threshold_bbox[1]) < 1e-1 and
                        abs(geom_bbox[2] - threshold_bbox[2]) < 1e-1 and
                        abs(geom_bbox[3] - threshold_bbox[3]) < 1e-1):
                        continue
            
        # Get global transform matrix
        m_glob = get_global_transform(geom_tag)
        
        if geom_tag.name == "path":
            d = geom_tag.get("d", "")
            if d:
                try:
                    parsed_path = parse_path(d)
                    for seg in parsed_path:
                        segments.append(transform_segment(seg, m_glob))
                    
                    local_start = get_path_start(geom_tag)
                    if local_start:
                        pt_glob = m_glob @ np.array([local_start[0], local_start[1], 1.0])
                        path_start = (pt_glob[0], pt_glob[1])
                except Exception:
                    pass
        elif geom_tag.name == "line":
            try:
                x1 = float(geom_tag.get("x1", 0))
                y1 = float(geom_tag.get("y1", 0))
                x2 = float(geom_tag.get("x2", 0))
                y2 = float(geom_tag.get("y2", 0))
                seg = Line(complex(x1, y1), complex(x2, y2))
                segments.append(transform_segment(seg, m_glob))
            except Exception:
                pass
        elif geom_tag.name == "rect":
            try:
                x = float(geom_tag.get("x", 0))
                y = float(geom_tag.get("y", 0))
                w = float(geom_tag.get("width", 0))
                h = float(geom_tag.get("height", 0))
                p1 = complex(x, y)
                p2 = complex(x + w, y)
                p3 = complex(x + w, y + h)
                p4 = complex(x, y + h)
                segments.append(transform_segment(Line(p1, p2), m_glob))
                segments.append(transform_segment(Line(p2, p3), m_glob))
                segments.append(transform_segment(Line(p3, p4), m_glob))
                segments.append(transform_segment(Line(p4, p1), m_glob))
            except Exception:
                pass
        elif geom_tag.name in ["polygon", "polyline"]:
            try:
                coords = parse_points(geom_tag.get("points", ""))
                if len(coords) >= 2:
                    for i in range(len(coords) - 1):
                        p1 = complex(coords[i][0], coords[i][1])
                        p2 = complex(coords[i+1][0], coords[i+1][1])
                        segments.append(transform_segment(Line(p1, p2), m_glob))
                    if geom_tag.name == "polygon":
                        p1 = complex(coords[-1][0], coords[-1][1])
                        p2 = complex(coords[0][0], coords[0][1])
                        segments.append(transform_segment(Line(p1, p2), m_glob))
            except Exception:
                pass
                
    return segments, threshold_bbox, path_start, threshold_coords

def extract_single_door_anatomy(tag, segments, threshold_bbox, path_start, threshold_coords, mapper, specific_curve=None, source_svg_file="model.svg", door_idx=0):
    """
    Extracts door anatomy for a single door curve/leaf set.
    """
    # 2. Sequential resolution of anatomy
    # First, find arc endpoints and points without hinge/leaf info
    arc_start_svg, arc_end_svg, arc_points_svg, curve_seg = detect_arc(
        segments, None, threshold_bbox, None, specific_curve=specific_curve, threshold_coords=threshold_coords
    )
    
    # Get initial hinge
    hinge_svg = detect_hinge(threshold_bbox, path_start, None)
    
    # Detect door leaf
    leaf_start_svg, leaf_end_svg, leaf_length_svg = detect_leaf(
        segments, threshold_bbox, hinge_svg, arc_end_svg, arc_start_svg
    )
    
    # Refine hinge with leaf start
    hinge_svg = detect_hinge(threshold_bbox, path_start, leaf_start_svg)
    
    # Refine arc with refined hinge and leaf_end
    arc_start_svg, arc_end_svg, arc_points_svg, curve_seg = detect_arc(
        segments, hinge_svg, threshold_bbox, leaf_end_svg, specific_curve=specific_curve, threshold_coords=threshold_coords
    )
    
    # 3. Calculate orientation & opening direction in global SVG space
    orientation = calculate_orientation(hinge_svg, leaf_end_svg)
    opening_direction = calculate_opening_direction(
        arc_points_svg, hinge_svg, arc_start_svg, arc_end_svg
    )
    
    # 4. Generate tight polygon in global SVG space
    poly_svg = generate_polygon(hinge_svg, leaf_end_svg, arc_points_svg)
    
    # 5. Map all coordinates to PNG image space using mapper
    if hinge_svg:
        hinge_png = mapper.svg_to_image(hinge_svg[0], hinge_svg[1])
    else:
        hinge_png = None
        
    if leaf_start_svg and leaf_end_svg:
        leaf_start_png = mapper.svg_to_image(leaf_start_svg[0], leaf_start_svg[1])
        leaf_end_png = mapper.svg_to_image(leaf_end_svg[0], leaf_end_svg[1])
        leaf_png = [leaf_start_png, leaf_end_png]
    else:
        leaf_png = None
        
    arc_points_png = [mapper.svg_to_image(pt[0], pt[1]) for pt in arc_points_svg] if arc_points_svg else []
    poly_png = [mapper.svg_to_image(pt[0], pt[1]) for pt in poly_svg] if poly_svg else []
    
    # Compute bounding box of the mapped polygon in PNG space
    if poly_png:
        xs = [pt[0] for pt in poly_png]
        ys = [pt[1] for pt in poly_png]
        bbox_png = [min(xs), min(ys), max(xs), max(ys)]
    else:
        bbox_png = None
        
    door_type = tag.get("class")
    if not door_type:
        door_type = "Door"
    elif isinstance(door_type, list):
        door_type = " ".join(door_type)
    else:
        door_type = str(door_type)
        
    # Calculate width_px and height_px (Hook 8 & Hook 3)
    width_px = 0.0
    if leaf_start_png and leaf_end_png:
        width_px = float(np.linalg.norm(np.array(leaf_end_png) - np.array(leaf_start_png)))
    # Standard height estimation on 2D view (usually standard CAD heights like 210cm)
    # We set a standard CAD scaling multiplier (2.69 x width_px) or default to 210.0px
    height_px = float(width_px * 2.69) if width_px > 0 else 210.0

    parent_group = tag.parent.name if tag.parent else "svg"
    parent_id = tag.parent.get("id", "") if tag.parent else ""
    source_svg_group = f"{parent_group}#{parent_id}" if parent_id else parent_group
    
    return {
        "door_id": f"d_{door_idx}",
        "door_type": door_type,
        "bbox": bbox_png,
        "polygon": poly_png,
        "orientation": orientation,
        "opening_direction": opening_direction,
        "hinge": hinge_png,
        "leaf": leaf_png,
        "arc": arc_points_png,
        "width_px": width_px,
        "height_px": height_px,
        "scale_available": False,
        "source_class": door_type,
        "source_svg_group": source_svg_group,
        "source_svg_file": source_svg_file,
        "raw_geometry_count": len(segments)
    }

def extract_door_info_cad(tag, mapper, source_svg_file="model.svg", door_idx=0):
    """
    Extracts door anatomy components (Hinge, Leaf, Arc) in global space,
    maps them to PNG space using mapper, and computes orientation, opening direction, etc.
    Supports splitting double doors (multiple swing curves).
    """
    # 1. Extract SVG geometry components in global SVG space
    segments, threshold_bbox, path_start, threshold_coords = extract_door_svg_geometry(tag)
    
    # 2. Identify if this is a double door (multiple swing curves)
    curves = [seg for seg in segments if isinstance(seg, (QuadraticBezier, CubicBezier, Arc))]
    
    if len(curves) > 1:
        # Multi-door extraction (double doors)
        results = []
        for i, curve in enumerate(curves):
            res = extract_single_door_anatomy(tag, segments, threshold_bbox, path_start, threshold_coords, mapper, 
                                              specific_curve=curve, source_svg_file=source_svg_file, door_idx=door_idx + i)
            if res:
                results.append(res)
        return results
    else:
        # Single door extraction
        curve = curves[0] if len(curves) == 1 else None
        return extract_single_door_anatomy(tag, segments, threshold_bbox, path_start, threshold_coords, mapper, 
                                          specific_curve=curve, source_svg_file=source_svg_file, door_idx=door_idx)
