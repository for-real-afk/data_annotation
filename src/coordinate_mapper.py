import numpy as np

class CoordinateMapper:
    def __init__(self, alignment_matrix_3x3):
        self.matrix = np.array(alignment_matrix_3x3, dtype=float)
        self.inv_matrix = np.linalg.inv(self.matrix)
        
    def svg_to_image(self, x, y):
        pt = np.array([x, y, 1.0])
        res = self.matrix @ pt
        return float(res[0]), float(res[1])
        
    def image_to_svg(self, x, y):
        pt = np.array([x, y, 1.0])
        res = self.inv_matrix @ pt
        return float(res[0]), float(res[1])
