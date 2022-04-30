import numpy as np
import cv2


def get_delaunay_triangles_by_index(landmarks: np.ndarray) -> np.ndarray:
    """

    :param landmarks:
    :return:
    """

    triangles = get_delaunay_triangulation(landmarks)
    return get_triangles_by_vertex(landmarks, triangles)


def get_delaunay_triangulation(landmarks: np.ndarray) -> np.ndarray:
    """
    Returns the points of all delaunay triangles.

    :param landmarks: Landmarks on the face to triangulate
    :return: Numpy array of triangles
    """

    hull = cv2.convexHull(landmarks)
    bounding_rect = cv2.boundingRect(hull)
    sub_divider = cv2.Subdiv2D(bounding_rect)
    for x, y in landmarks:
        sub_divider.insert((int(x), int(y)))
    triangles = sub_divider.getTriangleList()
    return np.array(triangles, dtype=np.int32)


def get_triangles_by_vertex(landmarks: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    triangle_points = []
    for triangle in triangles:
        pt1, pt2, pt3 = extract_triangle_vertices(triangle)
        p_pt1, p_pt2, p_pt3 = extract_indices(landmarks, [pt1, pt2, pt3])
        triangle_points.append([p_pt1, p_pt2, p_pt3])
    return np.array(triangle_points, dtype=np.uint32)


def extract_triangle_vertices(triangle: np.ndarray) -> (int, int, int):
    """
    Retrieves the three points from our delaunay triangulation for one triangle.

    :param triangle: Triangle to parse
    :return: Set of three points corresponding to the triangle
    """

    pt1 = (triangle[0], triangle[1])
    pt2 = (triangle[2], triangle[3])
    pt3 = (triangle[4], triangle[5])
    return pt1, pt2, pt3


def extract_indices(landmarks: np.ndarray, triangle_coordinates: list) -> (int, int, int):
    """

    :param landmarks:
    :param triangle_coordinates:
    :return:
    """

    p_pt1 = np.where((landmarks == triangle_coordinates[0]).all(axis=1))
    p_pt2 = np.where((landmarks == triangle_coordinates[1]).all(axis=1))
    p_pt3 = np.where((landmarks == triangle_coordinates[2]).all(axis=1))
    return p_pt1[0][0], p_pt2[0][0], p_pt3[0][0]


def extract_triangle_coordinates(landmarks: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    pt1 = landmarks[triangle[0]]
    pt2 = landmarks[triangle[1]]
    pt3 = landmarks[triangle[2]]
    return np.array([pt1, pt2, pt3], dtype=np.int32)


def draw_triangle(image, pt1, pt2, pt3):
    cv2.line(image, pt1, pt2, (200, 200, 200, 0.5), None)
    cv2.line(image, pt2, pt3, (200, 200, 200, 0.5), None)
    cv2.line(image, pt1, pt3, (200, 200, 200, 0.5), None)


def draw_all_triangles(image, landmarks, triangle_indices):
    for triangle in triangle_indices:
        pt1 = landmarks[triangle[0]]
        pt2 = landmarks[triangle[1]]
        pt3 = landmarks[triangle[2]]
        draw_triangle(image, pt1, pt2, pt3)
