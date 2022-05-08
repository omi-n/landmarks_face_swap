import cv2
import numpy as np


def draw_fps_on_image(image: np.ndarray, fps: float) -> None:
    """
    Draws the frames per second calculated on the image.

    :param image:
    :param fps:result
    :return:
    """

    cv2.putText(image,
                f"FPS: {fps:.1f}",
                (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
                False)


def draw_text_on_image(image: np.ndarray, text: str, position: tuple) -> None:
    """

    :param image:
    :param text:
    :return:
    """

    cv2.putText(image,
                text,
                position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
                False)
