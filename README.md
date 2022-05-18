# [Face Swapping Using Meshes](https://github.com/rquit/landmarks_face_swap)
To use this project, you will need at least python 3.6. This project has been tested and used
extensively on python 3.10.4

## Instructions for Demo
1. Install all dependencies with: `pip install -r requirements.txt`
2. Run the application with `python demo.py --image <image_to_swap_with> --<flag> <arg>`

You can get help by running `python demo.py --help`

## Arguments for Demo
`--image <path_to_image: string>`
* (REQUIRED) The path to the image you wish to create a face mesh to swap your face with.

`--clone <clone_type: int>`
* (Default 2) The type of seamlessClone you wish to use.
  * 0: None
  * 1: Normal
  * 2: Mixed

`--threads <num_threads: int>`
* (Default 4) The amount of threads for triangle warping.

`--mesh-only <true/false>`
* (Default false) If you only care about the mesh output.

`--debug <true/false>`
* (Default false) If you want to see data and visualizations of pose, and framerate.

## Using `facemeshes/` as a library
There is included documentation for each of the files in their definitions. To generate
a warped face mesh from some passed in image fitted to your face dimensions, you can use
the sample code below.
```python
from facemeshes import mp_utils as mu
from facemeshes import triangle_utils as tu
from facemeshes import facemesh as fm
import cv2

holistic = mu.load_holistic_model()

swap = cv2.imread("path to image")
swap_landmarks = mu.get_face_landmark_list(holistic, swap)
swap_triangles = tu.get_delaunay_triangles_by_index(swap_landmarks)

image = cv2.imread("path to another image")
landmarks = mu.get_face_landmark_list(holistic, image)
face_mesh = fm.generate_fitted_mesh(image,
                                    swap,
                                    landmarks,
                                    swap_landmarks,
                                    swap_triangles)
face_mesh = cv2.flip(face_mesh, 1)

cv2.imshow("Generated Mesh", face_mesh)
```

If you want to put that image on top of a face, you can use the `place_mesh_on_image` function, as shown below.
```python
# assume previous segment
fm.place_mesh_on_image(image, landmarks, face_mesh)
```

You can also show all data with the `--debug true` flag, but it's also possible to use `rotation_utils`.
For example, this following segment gets the rotation angles of the face.
```python
# assume all previous segments
from facemeshes import rotation_utils as ru

img_pts, model_pts, rotate_degrees, nose = ru.face_orientation(image, landmarks)
ru.draw_pose_lines(image, nose, img_pts)
ru.draw_pose_degrees(image, rotate_degrees)
```

## Common Issues
If you are having some error relating to the text overlay on the image, make sure you
run with `python3` and not `python` on your computer.

Same applies to `pip` and `pip3`.

Please check the issues section to see if your bug has been submitted before submitting one!
