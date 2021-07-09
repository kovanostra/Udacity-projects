import cv2
import numpy as np

from src.domain.logger import get_logger
from src.infrastructure.parameters import *


class FrameBinarizer:
    def __init__(self) -> None:
        pass

    def binarize(self, frame: np.ndarray) -> np.ndarray:
        s_channel = self._to_hls(frame)[:, :, 2]
        scaled_gradient_x = self._get_scaled_gradient_x(s_channel)
        sobel_binary = self._apply_threshold(scaled_gradient_x)
        get_logger().info("Computed frame gradients and binarized frame")
        return sobel_binary

    @staticmethod
    def _get_scaled_gradient_x(channel: np.ndarray) -> np.ndarray:
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
        scaled_sobel = np.uint8(255 * np.absolute(sobelx) / np.max(np.absolute(sobelx)))
        return scaled_sobel

    @staticmethod
    def _apply_threshold(frame: np.ndarray) -> np.ndarray:
        frame_binarized = np.zeros_like(frame)
        frame_binarized[(frame >= MAX_THRESHOLD[0]) & (frame <= MAX_THRESHOLD[1]) |
                        (frame >= MIN_THRESHOLD[0]) & (frame <= MIN_THRESHOLD[1])] = 1
        return frame_binarized

    @staticmethod
    def _to_hls(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
