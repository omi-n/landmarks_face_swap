from facemeshes import cv_utils as cu
from facemeshes import mp_utils as mu
from facemeshes import triangle_utils as tu
from facemeshes import facemesh as fm

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

parser.add_argument("--mesh-only", dest="mesh_only", type=bool,
                    help="Enable if you don't care about face swaps.", default=False)

args = parser.parse_args()


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

    if not success:
        print("error reading frame!")
        continue

    landmarks = mu.get_face_landmark_list(holistic, image)

    if not landmarks.size:
        continue

    face_mesh = fm.generate_fitted_mesh(image, swap, landmarks, swap_landmarks, swap_triangles)

    if args.mesh_only:
        fps = 1 / (timeit.default_timer() - start)
        face_mesh = cv2.flip(face_mesh, 1)
        cu.draw_fps_on_image(face_mesh, fps)
        cv2.imshow("constructed mesh", face_mesh)
    else:
        hull = cv2.convexHull(landmarks)
        # This section takes the image, and puts a big black area where our face was.
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_mask = np.zeros_like(image_gray)
        # Fill the convex hull with white
        image_head_mask = cv2.fillConvexPoly(image_mask, hull, (255, 255, 255))
        # Flip all the bits so our face is black and everything else is white.
        image_mask = cv2.bitwise_not(image_mask)
        # Remove our face from our image. 0 AND anything = 0, and all is 0 on our face.
        image_no_face = cv2.bitwise_and(image, image, mask=image_mask)
        # Put the generated face from the swap in. This leaves some black squares where we missed some spots.
        result = cv2.add(image_no_face, face_mesh)

        # seamlessClone section
        (x, y, w, h) = cv2.boundingRect(hull)
        center = (int((x + x + w) / 2), int((y + y + h) / 2))
        if args.clone == 1:
            result = cv2.seamlessClone(result, image, image_head_mask, center, cv2.NORMAL_CLONE)
        elif args.clone == 2:
            result = cv2.seamlessClone(result, image, image_head_mask, center, cv2.MIXED_CLONE)

        fps = 1 / (timeit.default_timer() - start)
        face_mesh = cv2.flip(result, 1)
        cu.draw_fps_on_image(result, fps)
        cv2.imshow("Face Swap", result)

capture.release()
