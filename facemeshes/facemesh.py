from facemeshes import triangle_utils as tu

import numpy as np
import cv2


def generate_fitted_mesh(image: np.ndarray,
                         swap: np.ndarray,
                         landmarks: np.ndarray,
                         swap_landmarks: np.ndarray,
                         swap_triangles: np.ndarray):
    """

    :param image:
    :param swap:
    :param landmarks:
    :param swap_landmarks:
    :param swap_triangles:
    :return:
    """

    constructed_mesh = np.zeros_like(image, np.uint8)
    for triangle in swap_triangles:
        # Get the positions of each triangle in the swap image.
        sw_t_1, sw_t_2, sw_t_3 = tu.extract_triangle_coordinates(swap_landmarks, triangle)
        swap_triangle_points = np.array([sw_t_1, sw_t_2, sw_t_3], dtype=np.int32)
        swap_points, swap_cropped_triangle, _, __ = segment_triangle(swap_triangle_points, swap)

        # Get the positions of each triangle in our face.
        i_t_1, i_t_2, i_t_3 = tu.extract_triangle_coordinates(landmarks, triangle)
        # From here, we need points to get our transform matrix. This is to transform our
        # x and y to some new coordinates using the affine_transform later on.
        # The cropped triangle, and mask are used to get the area which we want to swap out.
        # The rectangle coordinates are used to get the height and width of these triangles we
        # want to warp to.
        image_triangle_points = np.array([i_t_1, i_t_2, i_t_3], dtype=np.int32)
        image_points, \
            image_cropped_triangle, \
            image_cropped_triangle_mask, \
            image_rectangle = segment_triangle(image_triangle_points, image)
        (x, y, w, h) = image_rectangle

        image_points = np.float32(image_points)
        swap_points = np.float32(swap_points)

        # Get the transformation matrix to get from swap_points to points.
        transform_matrix = cv2.getAffineTransform(swap_points, image_points)
        # Warp the cropped swap triangle to the shape of our face.
        warped_cropped = cv2.warpAffine(swap_cropped_triangle, transform_matrix, (w, h))
        # Bitwise AND to overlay the empty mask we got to the colors in the swap's face.
        warped_cropped = cv2.bitwise_and(warped_cropped, warped_cropped, mask=image_cropped_triangle_mask)

        # Area of the triangle to reconstruct
        area = constructed_mesh[y:y + h, x:x + w]
        # This helps to get rid of the white lines otherwise generated with this method.
        area_gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        _, triangle_mask = cv2.threshold(area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_cropped = cv2.bitwise_and(warped_cropped, warped_cropped, mask=triangle_mask)

        # Add our RGB values to that cropped and warped triangle we just processed, and puts it into the new_face mask.
        area = cv2.add(warped_cropped, area)
        constructed_mesh[y:y + h, x:x + w] = area

    kernel = np.ones((2, 2), np.uint8)
    constructed_mesh = cv2.dilate(constructed_mesh, kernel, iterations=1)
    return constructed_mesh


def segment_triangle(points: np.ndarray, image: np.ndarray):
    """

    :param points:
    :param image:
    :return:
    """

    rectangle = cv2.boundingRect(points)
    (x, y, w, h) = rectangle
    cropped_triangle = image[y:y + h, x:x + w]
    cropped_triangle_mask = np.zeros((h, w), np.uint8)

    coords = np.array([
        [points[0][0] - x, points[0][1] - y],
        [points[1][0] - x, points[1][1] - y],
        [points[2][0] - x, points[2][1] - y]
    ])

    cv2.fillConvexPoly(cropped_triangle_mask, coords, (255, 255, 255))
    try:
        cropped_triangle = cv2.bitwise_and(cropped_triangle,
                                           cropped_triangle,
                                           mask=cropped_triangle_mask)
    except cv2.error:
        print("Unable to crop face area! Please move further from your webcam.")
        exit(1)
        # return None, None, None, None, False

    return coords, cropped_triangle, cropped_triangle_mask, rectangle
