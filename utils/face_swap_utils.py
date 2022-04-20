import cv2
import numpy as np
from utils import triangle_utils


def triangle_segment_pipeline(points, image):
    triangle = triangle_utils.build_triangle(points[0], points[1], points[2])
    rectangle = cv2.boundingRect(triangle)
    (x, y, w, h) = rectangle
    cropped_triangle = image[y:y+h, x:x+w]
    cropped_triangle_mask = np.zeros((h, w), np.uint8)

    coords = np.array([
        [points[0][0] - x, points[0][1] - y],
        [points[1][0] - x, points[1][1] - y],
        [points[2][0] - x, points[2][1] - y]
    ])

    cv2.fillConvexPoly(cropped_triangle_mask, coords, (255, 255, 255))
    cropped_triangle = cv2.bitwise_and(cropped_triangle,
                                       cropped_triangle,
                                       mask=cropped_triangle_mask)

    return coords, cropped_triangle, cropped_triangle_mask, rectangle