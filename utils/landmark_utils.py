import cv2
import numpy as np


def draw_landmarks(image, landmarks):
    for landmark in landmarks:
        x, y = landmark
        cv2.circle(image, (x, y), 2, (255, 255, 255))


def get_landmark_list(normalized_list, image_dimensions):
    length = image_dimensions[1]
    height = image_dimensions[0]
    landmark_list = []
    for landmark in normalized_list:
        x = int(landmark.x * length)
        y = int(landmark.y * height)
        landmark_list.append((x, y))
    return np.array(landmark_list, dtype=np.int32)
