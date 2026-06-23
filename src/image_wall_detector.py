import cv2
import numpy as np

def detect_image_walls(img):
    """
    Extracts solid wall structures from a floorplan PNG image.
    Standard floorplans have white background and dark/filled walls.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    # Threshold for dark wall pixels (usually < 200)
    _, wall_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological opening to remove small texts, furniture, and grids
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kernel_open)
    
    # Morphological closing to seal small gaps within walls
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_close)
    
    return clean
