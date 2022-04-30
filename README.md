# Face Swapping Using Meshes
To use this project, you will need the latest version of pip and
some fairly up-to-date python version. I have personally
tested this application using python 3.10, but 3.8 and above should
work fine.


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

## Common Issues
If you are having some error relating to the text overlay on the image, make sure you
run with `python3` and not `python` on your computer.

Same applies to `pip` and `pip3`.

Please check the issues section to see if your bug has been submitted before submitting one!
