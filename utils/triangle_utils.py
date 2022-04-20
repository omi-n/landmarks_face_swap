import numpy as np
import cv2


def find_and_extract_index(landmarks, point):
    # NOTE: FIND A WAY TO OPTIMIZE THIS!!
    index = np.where((landmarks == point).all(axis=1))
    i = index[0][0] if index[0][0] else None
    return i


def get_all_triangle_points(triangle):
    triangle = np.array(triangle, dtype=np.int32)
    pt1 = (triangle[0], triangle[1])
    pt2 = (triangle[2], triangle[3])
    pt3 = (triangle[4], triangle[5])
    return pt1, pt2, pt3


def build_triangle(index_pt1, index_pt2, index_pt3):
    t = [index_pt1, index_pt2, index_pt3]
    return np.array(t)


def draw_triangle(image, pt1, pt2, pt3):
    cv2.line(image, pt1, pt2, (200, 200, 200, 0.5), None)
    cv2.line(image, pt2, pt3, (200, 200, 200, 0.5), None)
    cv2.line(image, pt1, pt3, (200, 200, 200, 0.5), None)


def get_delaunay_triangles(landmarks, hull):
    face_detect_rect = cv2.boundingRect(hull)
    subdiv = cv2.Subdiv2D(face_detect_rect)
    for x, y in landmarks:
        subdiv.insert((int(x), int(y)))
    triangles = subdiv.getTriangleList()
    return triangles


def get_all_triangles_by_index(landmarks, triangles):
    all_triangle_indices = []
    for triangle in triangles:
        # NOTE: This is the most computationally expensive part.
        # Likely because we have 468 landmarks. Maybe find a smaller model?
        # Maybe we can do this once per face and keep the triangle indices.
        pt1, pt2, pt3 = get_all_triangle_points(triangle)

        index_pt1 = find_and_extract_index(landmarks, pt1)
        index_pt2 = find_and_extract_index(landmarks, pt2)
        index_pt3 = find_and_extract_index(landmarks, pt3)

        if index_pt1 and index_pt2 and index_pt3:
            all_triangle_indices.append(build_triangle(index_pt1, index_pt2, index_pt3))

    return np.array(all_triangle_indices, np.uint32)


def draw_all_triangles(image, landmarks, triangle_indices):
    for triangle in triangle_indices:
        pt1 = landmarks[triangle[0]]
        pt2 = landmarks[triangle[1]]
        pt3 = landmarks[triangle[2]]
        draw_triangle(image, pt1, pt2, pt3)


def full_triangle_index_pipeline(landmarks, hull):
    triangles = get_delaunay_triangles(landmarks, hull)
    return get_all_triangles_by_index(landmarks, triangles)


def get_triangle_positions(landmarks, triangle):
    pt1 = landmarks[triangle[0]]
    pt2 = landmarks[triangle[1]]
    pt3 = landmarks[triangle[2]]
    return pt1, pt2, pt3