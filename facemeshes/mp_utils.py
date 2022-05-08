import mediapipe as mp
import numpy as np
import cv2

holistic_model_type = type(mp.solutions.holistic.Holistic)


def load_holistic_model(smooth_landmarks: bool = True, refine_face_landmarks: bool = False) -> holistic_model_type:
    """
    Returns a holistic model from the mediapipe library.
    More info: https://google.github.io/mediapipe/solutions/holistic.html

    :param smooth_landmarks: Toggle for the smooth landmarks option
    :param refine_face_landmarks: Toggle for the refined face landmarks option
    :return: Mediapipe holistic model
    """

    return mp.solutions.holistic.Holistic(
        smooth_landmarks=smooth_landmarks,
        refine_face_landmarks=refine_face_landmarks
    )


def get_face_landmark_list(holistic_model: holistic_model_type, image: np.ndarray) -> np.ndarray:
    """
    Retrieves the facial landmark list in the form of a numpy array from the passed in holistic model and image.

    :param holistic_model: Mediapipe holistic model. If you still need one, use load_holistic_model().
    :param image: The image in the form of a numpy array retrieved by cv2.imread(path)
    :return: A numpy array with landmarks in the form [(x0, y0), (x1, y1), ... , (xn, yn)].
    """

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed = holistic_model.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if not processed.face_landmarks:
        return np.array([])

    landmarks_list_normalized = processed.face_landmarks.landmark
    landmark_list = []

    width, height = image.shape[1], image.shape[0]
    for landmark in landmarks_list_normalized:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        z = int(landmark.z)
        landmark_list.append((x, y))

    return np.array(landmark_list, dtype=np.int32)


def get_landmark_list_with_depth(holistic_model: holistic_model_type, image: np.ndarray)  -> np.ndarray:
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed = holistic_model.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if not processed.face_landmarks:
        return np.array([])

    landmarks_list_normalized = processed.face_landmarks.landmark
    landmark_list = []

    width, height = image.shape[1], image.shape[0]
    for landmark in landmarks_list_normalized:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        z = int(landmark.z)
        landmark_list.append((x, y, z))

    return np.array(landmark_list, dtype=np.int32)


def debug_draw_landmarks(image: np.ndarray, landmark_list: np.ndarray) -> None:
    """
    Draws landmarks on a copy of the passed in image

    :param image: Image to draw landmarks on
    :param landmark_list: List to draw landmarks from
    :return: Landmark annotated image
    """

    for landmark in landmark_list:
        x, y = landmark
        cv2.circle(image, (x, y), 1, (255, 255, 255))
