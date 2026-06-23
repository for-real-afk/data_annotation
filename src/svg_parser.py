import os
import re
import cv2
import numpy as np
from bs4 import BeautifulSoup
from svgpathtools import parse_path

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


def extract_path_points(path_d, samples=20):
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
    if node.name in ["polygon", "polyline"]:
        return parse_points(node.get("points", ""))
    elif node.name == "rect":
        try:
            x = float(node.get("x", 0))
            y = float(node.get("y", 0))
            w = float(node.get("width", 0))
            h = float(node.get("height", 0))
            return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        except:
            return []
    elif node.name == "line":
        try:
            return [(float(node.get("x1", 0)), float(node.get("y1", 0))), 
                    (float(node.get("x2", 0)), float(node.get("y2", 0)))]
        except:
            return []
            
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


def get_svg_dimensions(soup):
    svg_tag = soup.find("svg")
    if not svg_tag:
        raise ValueError("No <svg> tag found in file!")
    
    try:
        svg_w = float(svg_tag.get("width"))
        svg_h = float(svg_tag.get("height"))
    except (ValueError, TypeError):
        viewbox = svg_tag.get("viewBox", "").split()
        if len(viewbox) == 4:
            svg_w = float(viewbox[2])
            svg_h = float(viewbox[3])
        else:
            raise ValueError("Could not extract width and height from SVG tag or viewBox!")
    return svg_w, svg_h


def calculate_alignment(soup, img_png, svg_w, svg_h):
    if img_png is not None:
        png_h, png_w = img_png.shape[:2]
    else:
        png_w, png_h = svg_w, svg_h

    # --- DYNAMIC ALIGNMENT ---
    from transform_resolver import get_global_transform
    structural_svg_coords = []
    for tag in soup.find_all():
        tag_text = (str(tag.get("id", "")) + " " + str(tag.get("class", ""))).lower()
        if any(x in tag_text for x in ["wall", "railing"]):
            # Include tag itself if it's a geometry element, plus all geometry descendants
            elements = [tag] if tag.name in ["polygon", "polyline", "rect", "line", "path"] else []
            elements.extend(tag.find_all(["polygon", "polyline", "rect", "line", "path"], recursive=True))
            
            for child in elements:
                coords = []
                if child.name in ["polygon", "polyline"]:
                    coords = parse_points(child.get("points", ""))
                elif child.name == "rect":
                    try:
                        x = float(child.get("x", 0))
                        y = float(child.get("y", 0))
                        w = float(child.get("width", 0))
                        h = float(child.get("height", 0))
                        coords = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                    except:
                        pass
                elif child.name == "line":
                    try:
                        coords = [(float(child.get("x1", 0)), float(child.get("y1", 0))), 
                                  (float(child.get("x2", 0)), float(child.get("y2", 0)))]
                    except:
                        pass
                elif child.name == "path":
                    coords = extract_path_points(child.get("d", ""), samples=20)
                
                if coords:
                    m_glob = get_global_transform(child)
                    for pt in coords:
                        pt_glob = m_glob @ np.array([pt[0], pt[1], 1.0])
                        structural_svg_coords.append((pt_glob[0], pt_glob[1]))

    if len(structural_svg_coords) > 0:
        xmin_svg, ymin_svg, xmax_svg, ymax_svg = bbox_from_coords(structural_svg_coords)
    else:
        xmin_svg, ymin_svg, xmax_svg, ymax_svg = 0.0, 0.0, svg_w, svg_h

    if img_png is not None:
        # Downscale image to 10% size to calculate dark pixel boundaries extremely fast
        h, w = img_png.shape[:2]
        small_h, small_w = max(1, h // 10), max(1, w // 10)
        img_small = cv2.resize(img_png, (small_w, small_h))
        img_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        
        # Threshold dark pixels in small space
        _, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Remove border frame components that touch the image boundaries
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
        margin_x = int(small_w * 0.02)
        margin_y = int(small_h * 0.02)
        
        clean_thresh = thresh.copy()
        for i in range(1, num_labels):
            cx = stats[i, cv2.CC_STAT_LEFT]
            cy = stats[i, cv2.CC_STAT_TOP]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            if cx < margin_x or cy < margin_y or (cx + cw) > (small_w - margin_x) or (cy + ch) > (small_h - margin_y):
                clean_thresh[labels == i] = 0
                
        dark_pixels = np.where(clean_thresh > 0)
        if len(dark_pixels[0]) > 0:
            ymin_png = float(dark_pixels[0].min()) * (h / small_h)
            ymax_png = float(dark_pixels[0].max()) * (h / small_h)
            xmin_png = float(dark_pixels[1].min()) * (w / small_w)
            xmax_png = float(dark_pixels[1].max()) * (w / small_w)
        else:
            xmin_png, ymin_png, xmax_png, ymax_png = 0.0, 0.0, float(png_w), float(png_h)
    else:
        xmin_png, ymin_png, xmax_png, ymax_png = xmin_svg, ymin_svg, xmax_svg, ymax_svg

    w_svg = xmax_svg - xmin_svg
    h_svg = ymax_svg - ymin_svg
    w_png = xmax_png - xmin_png
    h_png = ymax_png - ymin_png
    
    if w_png <= 5.0:
        w_png = float(png_w)
        xmin_png = 0.0
    if h_png <= 5.0:
        h_png = float(png_h)
        ymin_png = 0.0
    
    scale_x = w_png / w_svg if w_svg > 0 else 1.0
    scale_y = h_png / h_svg if h_svg > 0 else 1.0

    dx = xmin_png - xmin_svg * scale_x
    dy = ymin_png - ymin_svg * scale_y

    return scale_x, scale_y, dx, dy, png_w, png_h


def transform_point(pt, scale, dx, dy):
    return pt[0] * scale + dx, pt[1] * scale + dy
