import cv2
import mediapipe as mp
import numpy as np
from utils import triangle_utils as triangle_util
from utils import landmark_utils as landmark_util
from utils import face_swap_utils as face_swap_util
import timeit
import sys

# Use the holistic model, as it supports noise filtering so we don't need our own.
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    smooth_landmarks=True
)

# This is the image we want to swap our face with.
try:
    sys.argv[1]
except IndexError:
    print("Please enter the file path.")
    exit(1)

swap = cv2.imread(sys.argv[1])

# Get the landmarks of the face to swap with. Holistic likes RGB images.
try:
    swap = cv2.cvtColor(swap, cv2.COLOR_BGR2RGB)
except cv2.error:
    print("File path not found!")
    exit(1)

results2 = holistic.process(swap)

# Convert back to cv2's format BGR
swap = cv2.cvtColor(swap, cv2.COLOR_RGB2BGR)

# Get the landmark list, and the convex hull. Take those and get all the
#   delaunay triangles.
swap_landmarks = landmark_util.get_landmark_list(results2.face_landmarks.landmark, swap.shape)
#landmark_util.draw_landmarks(swap, swap_landmarks)
swap_hull = cv2.convexHull(swap_landmarks)
all_triangle_indices = triangle_util.full_triangle_index_pipeline(swap_landmarks, swap_hull)

# We want to swap in realtime with our own camera.
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while capture.isOpened():
    # Get the FPS for debug purposes
    start = timeit.default_timer()
    success, image = capture.read()

    if not success:
        print("Reading camera frame failed!")
        continue

    # Process our own face with the mediapipe mesh
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # This is to not crash if we dont find landmarks.
    if not results.face_landmarks:
        continue

    # Generate list of landmarks as it's easier to work with.
    landmark_points = landmark_util.get_landmark_list(results.face_landmarks.landmark, image.shape)

    # This will be the face to swap into ours.
    new_face = np.zeros_like(image, np.uint8)
    # Go through all triangles in the to swap image.
    # Possible improvements:
    # multithreading
    # changing everything possible to numpy
    # porting to cpp (NOT RECOMMENDED)
    for triangle in all_triangle_indices:
        # Get the positions of each triangle in the image.
        swap_1, swap_2, swap_3 = triangle_util.get_triangle_positions(swap_landmarks, triangle)
        # Actually, from this section we ONLY need swap_points and the triangle crop.
        # See the next section for more information.
        swap_points, swap_cropped_triangle, _, __ = face_swap_util.triangle_segment_pipeline(
                    [swap_1, swap_2, swap_3], swap
            )

        # Get the positions of each triangle in our face.
        image_1, image_2, image_3 = triangle_util.get_triangle_positions(landmark_points, triangle)
        # From here, we need points to get our transform matrix. This is to transform our
        # x and y to some new coordinates using the affine_transform later on.
        # The cropped triangle, and mask are used to get the area which we want to swap out.
        # The rectangle coordinates are used to get the height and width of these triangles we
        # want to warp to.
        points,\
            image_cropped_triangle,\
            image_cropped_triangle_mask,\
            image_rect = face_swap_util.triangle_segment_pipeline(
                [image_1, image_2, image_3], image
            )
        (x, y, w, h) = image_rect


        # Warping Masks
        points = np.float32(points)
        swap_points = np.float32(swap_points)

        # Get the transformation matrix to get from swap_points to points.
        transform_matrix = cv2.getAffineTransform(swap_points, points)
        # Warp the cropped swap triangle to the shape of our face.
        warped_cropped_triangle = cv2.warpAffine(swap_cropped_triangle, transform_matrix, (w, h))
        # Bitwise AND to overlay the empty mask we got to the colors in the swap's face.
        warped_cropped_triangle = cv2.bitwise_and(warped_cropped_triangle, warped_cropped_triangle,
                                                  mask=image_cropped_triangle_mask)

        # This is the area we want to change in our face.
        area = new_face[y:y + h, x:x + w]
        # This changes that area to gray for processing in the threshold
        area_gray = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        # This helps to get rid of the white lines otherwise generated with this method.
        _, triangle_mask = cv2.threshold(area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_cropped_triangle = cv2.bitwise_and(warped_cropped_triangle, warped_cropped_triangle,
                                                  mask=triangle_mask)
        # Add our RGB values to that cropped and warped triangle we just processed, and puts it into the new_face mask.
        area = cv2.add(warped_cropped_triangle, area)
        new_face[y:y + h, x:x + w] = area

    # Apply 1 iteration of dilation with 2x2 kernel to get rid of hard lines
    kernel = np.ones((2, 2), np.uint8)
    new_face = cv2.dilate(new_face, kernel, iterations=1)

    # Uncomment to show reconstructed face
    cv2.imshow("new_face", new_face)

    # Get the convex hull for the head mask
    hull = cv2.convexHull(landmark_points)

    # This section takes the image, and puts a big black area where our face was.
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_mask = np.zeros_like(image_gray)
    # Fill the convex hull with white
    image_head_mask = cv2.fillConvexPoly(image_mask, hull, (255,255,255))
    # Flip all the bits so our face is black and everything else is white.
    image_mask = cv2.bitwise_not(image_mask)

    # Remove our face from our image. 0 AND anything = 0, and all is 0 on our face.
    image_noface = cv2.bitwise_and(image, image, mask=image_mask)

    # Put the generated face from the swap in. This leaves some black squares where we missed some spots.
    result = cv2.add(image_noface, new_face)

    # UNCOMMENT THIS SECTION IF YOU THINK SEAMLESS CLONE WILL WORK
    # it doesnt work with paul mccartney
    (x, y, w, h) = cv2.boundingRect(hull)
    center = (int((x + x + w) / 2), int((y + y + h) / 2))
    result = cv2.seamlessClone(result, image, image_head_mask, center, cv2.MIXED_CLONE)

    # FPS counter, showing image.
    end = timeit.default_timer() - start
    fps = 1 / end
    cv2.putText(result, f"FPS: {fps:.1f}", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA, False)
    cv2.imshow("after swap", result)
    if cv2.waitKey(5) & 0xFF == ord('d'):
        break
capture.release()