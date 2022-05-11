from facemeshes import cv_utils as cu
import math
import numpy as np
import cv2


def face_orientation(image, landmarks):
    """
    SOURCES:

    https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

    Adapted to work with mediapipe.

    :param image:
    :param landmarks:
    :return:
    """

    size = image.shape
    # Part 1: 2D coordinates of a few points
    # Important landmarks for tracking pose according to linked article
    image_points = np.array([
        landmarks[4],    # Nose Tip
        landmarks[199],  # Chin
        landmarks[130],  # Left Corner, Left Eye
        landmarks[263],  # Right Corner, Right Eye
        landmarks[62],   # Left Corner, Mouth
        landmarks[308]   # Right Corner, Mouth
    ], dtype="double")

    # Part 2: 3D coordinates of the same points
    # World points for some generic 3d model. Don't need anything fancy
    # Note: THESE ARE COMPLETELY ARBITRARY!
    # They don't mean anything.
    world = np.array([
        (0.0, 0.0, 0.0),            # Nose Tip
        (0.0, -330.0, -65.0),       # Chin
        (-165.0, 170.0, -135.0),    # Left Corner, Left Eye
        (165.0, 170.0, -135.0),     # Right Corner, Right Eye
        (-150.0, -150.0, -125.0),   # Left Corner, Mouth
        (150.0, -150.0, -125.0)     # Right Corner, Mouth
    ], dtype="double")

    # Part 3: Intrinsic parameters of the camera
    # Approximating Focal Length: https://learnopencv.com/approximate-focal-length-for-webcams-and-cell-phone-cameras/
    center_x, center_y = (size[1] / 2, size[0] / 2)
    focal_length = center_x / np.tan(60 / 2 * np.pi / 180)

    # Direct linear transform: with 0 distortion (no GoPro)
    # [x] = [focal  0  center_x] = [X]
    # [y] = [0  focal  center_y] = [Y]
    # [1] = [0      0         1] = [Z]
    camera_matrix = np.array(
        [[focal_length, 0, center_x],
         [0, focal_length, center_y],
         [0, 0, 1]], dtype="double"
    )

    # solvePnPRansac should be used instead of solvePnP if the underlying data is noisy
    # distortion_coefficients being 0 means that we assume there is no distortion
    # according to the article, this is not an issue unless we are using a gopro camera.
    distortion_coefficients = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(world,
                                                                  image_points,
                                                                  camera_matrix,
                                                                  distortion_coefficients,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)

    axis = np.float32([[500, 0, 0],
                       [0, 500, 0],
                       [0, 0, 500]])

    # computes projections of 3D points to the image plane
    image_points, _ = cv2.projectPoints(axis,
                                        rotation_vector,
                                        translation_vector,
                                        camera_matrix,
                                        distortion_coefficients)
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]

    # np.hstack stacks arrays per column (concat along second axis)
    projection_matrix = np.hstack((rotation_matrix, translation_vector))

    # decomposes a projection matrix into a rotation matrix
    # 6th element in the output are the euler angles. the first 5 are:
    # camera matrix, rotation matrix, translation vector, rotation matrix (x, y, z)
    euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)[6]

    # euler angles is in degrees
    pitch, yaw, roll = [i for i in euler_angles]

    # the angle decompose returns is a bit wierd, going from -180 to 180 in the middle.
    # we can fix this by modifying the function. see the graph below.
    # normally this function would be from 0 to -180 fom the bottom. this is because the 0 angle would be
    # the straight line going "up" and "down". we can rotate this by 90* by doing asin(sin(x)) where x is pitch.
    # see the graph below.
    # https://www.desmos.com/calculator/4894emd0fr
    pitch = math.degrees(math.asin(math.sin(math.radians(pitch))))

    return np.array(image_points, dtype=int), (str(int(pitch)), str(int(yaw)), str(int(roll)))


def draw_pose_degrees(image, rotate_degree):
    """

    :param image:
    :param rotate_degree:
    :return:
    """

    cu.draw_text_on_image(image, f"TILT/PITCH: {rotate_degree[0]}", (0, 80))
    cu.draw_text_on_image(image, f"YAW: {rotate_degree[1]}", (0, 110))
    cu.draw_text_on_image(image, f"ROLL: {rotate_degree[2]}", (0, 140))


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
