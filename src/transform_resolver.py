import re
import numpy as np

def parse_transform_string(transform_str):
    """
    Parses an SVG transform attribute string and returns a 3x3 transformation matrix.
    Supports matrix, translate, scale, rotate, skewX, skewY.
    """
    m = np.identity(3, dtype=float)
    if not transform_str:
        return m
        
    # Find all commands like matrix(...), translate(...), etc.
    pattern = r"([a-zA-Z]+)\s*\(([^)]+)\)"
    commands = re.findall(pattern, transform_str)
    
    for cmd_name, args_str in commands:
        # Split by comma or whitespace, convert to floats
        args = [float(x) for x in re.split(r"[\s,]+", args_str.strip()) if x]
        if not args:
            continue
            
        m_cmd = np.identity(3, dtype=float)
        if cmd_name == "matrix":
            if len(args) == 6:
                a, b, c, d, e, f = args
                m_cmd = np.array([
                    [a, c, e],
                    [b, d, f],
                    [0, 0, 1]
                ], dtype=float)
        elif cmd_name == "translate":
            tx = args[0]
            ty = args[1] if len(args) > 1 else 0.0
            m_cmd = np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ], dtype=float)
        elif cmd_name == "scale":
            sx = args[0]
            sy = args[1] if len(args) > 1 else sx
            m_cmd = np.array([
                [sx, 0, 0],
                [0, sy, 0],
                [0, 0, 1]
            ], dtype=float)
        elif cmd_name == "rotate":
            angle = np.radians(args[0])
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            if len(args) == 3:
                cx, cy = args[1], args[2]
                # translate(cx, cy) * rotate(angle) * translate(-cx, -cy)
                t1 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=float)
                r = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=float)
                t2 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=float)
                m_cmd = t1 @ r @ t2
            else:
                m_cmd = np.array([
                    [cos_a, -sin_a, 0],
                    [sin_a, cos_a, 0],
                    [0, 0, 1]
                ], dtype=float)
        elif cmd_name == "skewX":
            angle = np.radians(args[0])
            m_cmd = np.array([
                [1, np.tan(angle), 0],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=float)
        elif cmd_name == "skewY":
            angle = np.radians(args[0])
            m_cmd = np.array([
                [1, 0, 0],
                [np.tan(angle), 1, 0],
                [0, 0, 1]
            ], dtype=float)
            
        m = m @ m_cmd
        
    return m

def get_global_transform(node):
    """
    Recursively resolves parent transformations from leaf element to the root element.
    Returns a cumulative 3x3 transformation matrix.
    """
    m_global = np.identity(3, dtype=float)
    
    # Collect all ancestors from current node up to the root
    ancestors = []
    curr = node
    while curr is not None and curr.name != '[document]':
        ancestors.append(curr)
        curr = curr.parent
        
    # Multiply from root to leaf
    for ancestor in reversed(ancestors):
        transform_str = ancestor.get("transform", "")
        if transform_str:
            m_local = parse_transform_string(transform_str)
            m_global = m_global @ m_local
            
    return m_global
