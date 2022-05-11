from facemeshes import cv_utils as cu
import math
import numpy as np
import cv2


def face_orientation(image, landmarks):
    """
    SOURCES:

    https://github.com/jerryhouuu/Face-Yaw-Roll-Pitch-from-Pose-Estimation-using-OpenCV

    https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

    Adapted to work with mediapipe.

    :param image:
    :param landmarks:
    :return:
    """

    size = image.shape
    image_points = np.array([
        landmarks[4],
        landmarks[199],
        landmarks[130],
        landmarks[263],
        landmarks[62],
        landmarks[308]
    ], dtype="double")

    world = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-165.0, 170.0, -135.0),
        (165.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype="double")

    center = (size[1] / 2, size[0] / 2)
    focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(world, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    axis = np.float32([[500, 0, 0],
                       [0, 500, 0],
                       [0, 0, 500]])

    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    worldpts, jac2 = cv2.projectPoints(world, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    imgpts = np.array(imgpts, dtype=int)

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return imgpts, worldpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), landmarks[4]


def draw_pose_degrees(image, rotate_degree):
    """

    :param image:
    :param rotate_degree:
    :return:
    """

    cu.draw_text_on_image(image, f"ROLL: {rotate_degree[0]}", (0, 80))
    cu.draw_text_on_image(image, f"TILT: {rotate_degree[1]}", (0, 110))
    cu.draw_text_on_image(image, f"YAW: {rotate_degree[2]}", (0, 140))


def draw_pose_lines(image, point, image_pts):
    """

    :param image:
    :param point:
    :param image_pts:
    :return:
    """

    cv2.line(image, point, tuple(image_pts[1].ravel()), (0, 255, 0), 3)
    cv2.line(image, point, tuple(image_pts[0].ravel()), (255, 0, 0), 3)
    cv2.line(image, point, tuple(image_pts[2].ravel()), (0, 0, 255), 3)
