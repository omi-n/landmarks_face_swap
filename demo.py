from facemeshes import cv_utils as cu
from facemeshes import mp_utils as mu
from facemeshes import triangle_utils as tu
from facemeshes import facemesh as fm

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

    fps = 1 / (timeit.default_timer() - start)
    face_mesh = cv2.flip(face_mesh, 1)
    cu.draw_fps_on_image(face_mesh, fps)

    cv2.imshow("constructed mesh", face_mesh)

capture.release()
