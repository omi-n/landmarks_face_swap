from facemeshes import cv_utils as cu
from facemeshes import mp_utils as mu
from facemeshes import triangle_utils as tu
from facemeshes import facemesh as fm
from facemeshes import rotation_utils as ru

import numpy as np
import timeit
import cv2
import argparse

parser = argparse.ArgumentParser(description="Face Swap")

parser.add_argument("--image", dest="image", type=str,
                    help="Path to the image file", required=True)

parser.add_argument("--clone", dest="clone", type=int,
                    help="0 = No seamless clone 1 = Normal clone 2 = Mixed clone", default=2)

parser.add_argument("--threads", dest="threads", type=int,
                    help="Number of jobs for triangle morphing", default=4)

parser.add_argument("--mesh-only", dest="mesh_only", type=str,
                    help="Enable if you don't care about face swaps.", default="false")

parser.add_argument("--debug", dest="debug", type=str,
                    help="Enable for advanced debug information such as FPS and pose.", default="false")

args = parser.parse_args()
args.mesh_only = args.mesh_only.lower()
args.debug = args.debug.lower()

holistic = mu.load_holistic_model()

swap = cv2.imread(args.image)
swap_landmarks = mu.get_face_landmark_list(holistic, swap)
swap_triangles = tu.get_delaunay_triangles_by_index(swap_landmarks)

capture = cv2.VideoCapture(-1)
while capture.isOpened():
    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

    start = timeit.default_timer()
    success, image = capture.read()
    image = cv2.flip(image, 1)

    if not success:
        print("error reading frame!")
        continue

    landmarks = mu.get_face_landmark_list(holistic, image)
    landmarks_depth = mu.get_landmark_list_with_depth(holistic, image)

    if not landmarks.size:
        continue

    face_mesh = fm.generate_fitted_mesh(image, swap, landmarks, swap_landmarks, swap_triangles)

    if args.mesh_only == "true":
        if args.debug == "true":
            img_pts, model_pts, rotate_degree, nose = ru.face_orientation(face_mesh, landmarks)
            fm.draw_debug(face_mesh, nose, img_pts, rotate_degree, start)

        cv2.imshow("constructed mesh", face_mesh)
    else:
        result, image_head_mask, hull = fm.place_mesh_on_image(image, landmarks, face_mesh)

        (x, y, w, h) = cv2.boundingRect(hull)
        center = (int((x + x + w) / 2), int((y + y + h) / 2))
        if args.clone == 1:
            result = cv2.seamlessClone(result, image, image_head_mask, center, cv2.NORMAL_CLONE)
        elif args.clone == 2:
            result = cv2.seamlessClone(result, image, image_head_mask, center, cv2.MIXED_CLONE)

        if args.debug == "true":
            img_pts, model_pts, rotate_degree, nose = ru.face_orientation(result, landmarks)
            fm.draw_debug(result, nose, img_pts, rotate_degree, start)

        cv2.imshow("Face Swap", result)

capture.release()
