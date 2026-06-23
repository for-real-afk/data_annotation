import cv2
import numpy as np
from scipy.optimize import minimize

def register_masks(svg_mask, img_mask):
    """
    Registers the coarsely aligned outline svg_mask with the img_mask (thresholded image walls).
    Uses a coarse-to-fine translation grid search followed by multi-start local Nelder-Mead optimization
    with independent scaling in x and y to support aspect-ratio/phone-photo lens distortions.
    Runs at full resolution for maximum accuracy.
    """
    h, w = img_mask.shape[:2]
    
    # Ensure binary masks are 0 and 255
    svg_mask = ((svg_mask > 127) * 255).astype(np.uint8)
    img_mask = ((img_mask > 127) * 255).astype(np.uint8)
    
    # Compute Distance Transform on original inverted image mask
    img_inv = cv2.bitwise_not(img_mask)
    dist_map = cv2.distanceTransform(img_inv, cv2.DIST_L2, 5)
    
    # Get active SVG wall points at original resolution
    svg_pts = np.argwhere(svg_mask > 0) # nx2 array of (y, x)
    if len(svg_pts) == 0:
        return np.eye(3, dtype=np.float32)
        
    # Swap to (x, y) coordinates for transform math
    svg_pts_xy = svg_pts[:, [1, 0]].astype(np.float32)
    
    # Sub-sample points for performance
    if len(svg_pts_xy) > 1500:
        indices = np.random.choice(len(svg_pts_xy), 1500, replace=False)
        svg_pts_sampled = svg_pts_xy[indices]
    else:
        svg_pts_sampled = svg_pts_xy
        
    # --- STAGE 1: Coarse-to-Fine Grid Search for Translation ---
    best_tx_c = 0.0
    best_ty_c = 0.0
    min_loss_c = float('inf')
    
    tx_range_c = np.arange(-120, 121, 10)
    ty_range_c = np.arange(-120, 121, 10)
    
    for tx in tx_range_c:
        for ty in ty_range_c:
            pts_shifted = svg_pts_sampled + np.array([tx, ty], dtype=np.float32)
            out_of_bounds = (pts_shifted[:, 0] < 0) | (pts_shifted[:, 0] >= w) | (pts_shifted[:, 1] < 0) | (pts_shifted[:, 1] >= h)
            xs = np.clip(pts_shifted[:, 0], 0, w - 1).astype(np.int32)
            ys = np.clip(pts_shifted[:, 1], 0, h - 1).astype(np.int32)
            # Use robust clipped loss to prevent distant outliers from pulling alignment off
            dists = np.minimum(dist_map[ys, xs], 20.0)
            dists[out_of_bounds] = 20.0
            loss = np.mean(dists)
            if loss < min_loss_c:
                min_loss_c = loss
                best_tx_c = tx
                best_ty_c = ty
                
    best_tx_f = best_tx_c
    best_ty_f = best_ty_c
    min_loss_f = min_loss_c
    
    tx_range_f = np.arange(best_tx_c - 10, best_tx_c + 11, 2)
    ty_range_f = np.arange(best_ty_c - 10, best_ty_c + 11, 2)
    
    for tx in tx_range_f:
        for ty in ty_range_f:
            pts_shifted = svg_pts_sampled + np.array([tx, ty], dtype=np.float32)
            out_of_bounds = (pts_shifted[:, 0] < 0) | (pts_shifted[:, 0] >= w) | (pts_shifted[:, 1] < 0) | (pts_shifted[:, 1] >= h)
            xs = np.clip(pts_shifted[:, 0], 0, w - 1).astype(np.int32)
            ys = np.clip(pts_shifted[:, 1], 0, h - 1).astype(np.int32)
            dists = np.minimum(dist_map[ys, xs], 20.0)
            dists[out_of_bounds] = 20.0
            loss = np.mean(dists)
            if loss < min_loss_f:
                min_loss_f = loss
                best_tx_f = tx
                best_ty_f = ty
                
    # --- STAGE 2: Multi-Start Local Optimization ---
    # We test different initializations of scale refinement [sx, sy] to prevent getting stuck in local scale minima.
    scale_choices = [0.94, 1.0, 1.06]
    best_opt_params = None
    best_opt_loss = float('inf')
    
    def loss_func(params):
        sx, sy, tx, ty, theta = params
        
        # Constrain parameters to prevent unreasonable scaling/rotation
        if not (0.80 <= sx <= 1.20 and 0.80 <= sy <= 1.20 and 
                best_tx_f - 30 <= tx <= best_tx_f + 30 and 
                best_ty_f - 30 <= ty <= best_ty_f + 30 and 
                -5.0 <= theta <= 5.0):
            return 1e6
            
        rad = np.radians(theta)
        cos_t = np.cos(rad)
        sin_t = np.sin(rad)
        
        # Map points using refinement parameters
        x_new = sx * cos_t * svg_pts_sampled[:, 0] - sy * sin_t * svg_pts_sampled[:, 1] + tx
        y_new = sx * sin_t * svg_pts_sampled[:, 0] + sy * cos_t * svg_pts_sampled[:, 1] + ty
        
        # Check out of bounds
        out_of_bounds = (x_new < 0) | (x_new >= w) | (y_new < 0) | (y_new >= h)
        
        # Clamp to bounds
        xs = np.clip(x_new, 0, w - 1).astype(np.int32)
        ys = np.clip(y_new, 0, h - 1).astype(np.int32)
        
        dists = np.minimum(dist_map[ys, xs], 20.0)
        dists[out_of_bounds] = 20.0
        return np.mean(dists)
        
    for sx_i in scale_choices:
        for sy_i in scale_choices:
            init_params = [sx_i, sy_i, float(best_tx_f), float(best_ty_f), 0.0]
            res = minimize(loss_func, init_params, method='Nelder-Mead', options={'maxiter': 150})
            if res.success and res.fun < best_opt_loss:
                best_opt_loss = res.fun
                best_opt_params = res.x
                
    if best_opt_params is None:
        best_opt_params = [1.0, 1.0, float(best_tx_f), float(best_ty_f), 0.0]
        
    sx_opt, sy_opt, tx_opt, ty_opt, theta_opt = best_opt_params
    
    # Construct M_refine refinement matrix
    rad = np.radians(theta_opt)
    cos_t = np.cos(rad)
    sin_t = np.sin(rad)
    
    M_refine = np.eye(3, dtype=np.float32)
    M_refine[0, 0] = sx_opt * cos_t
    M_refine[0, 1] = -sy_opt * sin_t
    M_refine[0, 2] = tx_opt
    M_refine[1, 0] = sx_opt * sin_t
    M_refine[1, 1] = sy_opt * cos_t
    M_refine[1, 2] = ty_opt
    
    return M_refine

def compute_alignment_score(svg_mask, img_mask, M_refine, max_dist=7):
    """
    Computes the alignment score using the distance transform of the simple threshold image mask.
    This checks how close the SVG wall pixels are to the dark wall regions in the PNG.
    """
    h, w = img_mask.shape[:2]
    # Warp SVG mask using refinement matrix
    svg_warped = cv2.warpAffine(svg_mask, M_refine[:2], (w, h), flags=cv2.INTER_NEAREST)
    
    # Compute Distance Transform on the inverted image mask
    img_inv = cv2.bitwise_not(img_mask)
    dist_map = cv2.distanceTransform(img_inv, cv2.DIST_L2, 5)
    
    # Denominator must be the original count of active SVG wall pixels before warping
    # to penalize models that shift alignment outside the image boundaries.
    orig_svg_pts_count = np.sum(svg_mask > 0)
    if orig_svg_pts_count == 0:
        return 0.0
        
    # Find active SVG wall pixels in the warped image
    svg_pts = np.argwhere(svg_warped > 0)
    if len(svg_pts) == 0:
        return 0.0
        
    distances = dist_map[svg_pts[:, 0], svg_pts[:, 1]]
    inliers = np.sum(distances <= max_dist)
    score = inliers / orig_svg_pts_count
    
    return float(max(0.0, min(1.0, score)))
