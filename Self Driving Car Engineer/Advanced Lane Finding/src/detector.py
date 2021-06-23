import os
from typing import Tuple

import cv2
import numpy as np
from matplotlib import image as mpimg, pyplot as plt

from src.logger import get_logger


class Detector:
    def __init__(self) -> None:
        self.images = []
        self.calibration_images = []
        self.output_directory = None
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.transform_matrix = None

    def build(self, images_directory: str, calibration_directory: str, output_directory: str) -> None:
        self.output_directory = output_directory
        self._load_images(calibration_directory, images_directory)
        self._calibrate_camera()
        self._get_perspective_transform()

    def start(self) -> None:
        for image in self.images:
            image_undistorted = self._undistort_image(image)
            image_gradients = self._get_gradients(image_undistorted)
            image_transformed = self._apply_perspective_transform(image_gradients)
            plt.imshow(image_transformed)
            plt.show()

    def _load_images(self, calibration_directory: str, images_directory: str):
        for image_name in os.listdir(images_directory):
            self.images.append(mpimg.imread(os.path.join(images_directory, image_name)))
        for image_name in os.listdir(calibration_directory):
            self.calibration_images.append(mpimg.imread(os.path.join(calibration_directory, image_name)))
        get_logger().info("Loaded images from ./{}".format(images_directory))

    def _calibrate_camera(self, nx: int = 9, ny: int = 6):
        get_logger().info("Starting camera calibration")
        all_object_points = np.zeros(shape=(ny * nx, 3), dtype=np.float32)
        all_object_points[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        image_points_2d = []
        object_points_3d = []
        image_shape = self.calibration_images[0].shape[:2]
        for image in self.calibration_images:
            gray = self._to_gray(image)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret:
                object_points_3d.append(all_object_points)
                image_points_2d.append(corners)
        if image_points_2d:
            _, self.camera_matrix, self.distortion_coefficients, _, _ = cv2.calibrateCamera(object_points_3d,
                                                                                            image_points_2d,
                                                                                            image_shape,
                                                                                            None,
                                                                                            None)
            get_logger().info("Calibration parameters calculated")
        else:
            raise RuntimeError("Couldn't detect correctly the chessboard corners. "
                               "Impossible to calculate calibration parameters.")

    def _get_perspective_transform(self):
        source_points = np.float32([[560, 480],
                                    [256, 690],
                                    [740, 480],
                                    [1053, 690]])
        destination_points = np.float32([[int(self.images[0].shape[1] * 0.25), 0],
                                         [int(self.images[0].shape[1] * 0.25), self.images[0].shape[0]],
                                         [int(self.images[0].shape[1] * 0.75), 0],
                                         [int(self.images[0].shape[1] * 0.75), self.images[0].shape[0]]])
        self.transform_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        get_logger().info("Calculated the perspective transform matrix")

    def _undistort_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.undistort(np.copy(image),
                             self.camera_matrix,
                             self.distortion_coefficients,
                             None,
                             self.camera_matrix)

    def _get_gradients(self,
                       image: np.ndarray,
                       s_thresh: Tuple[int, int] = (130, 200),
                       sx_thresh: Tuple[int, int] = (50, 150)) -> np.ndarray:
        s_channel = self._to_hls(image)[:, :, 2]
        scaled_gradient_x = self._get_scaled_gradient_x(s_channel)
        sobel_binary = self._apply_threshold(scaled_gradient_x, sx_thresh)
        s_channel_binary = self._apply_threshold(s_channel, s_thresh)
        color_binary = np.dstack((np.zeros_like(sobel_binary), sobel_binary, s_channel_binary)) * 255
        get_logger().info("Computed image gradients and binarized image")
        return color_binary

    @staticmethod
    def _get_scaled_gradient_x(s_channel: np.ndarray) -> np.ndarray:
        sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=9)
        scaled_sobel = np.uint8(255 * np.absolute(sobelx) / np.max(np.absolute(sobelx)))
        return scaled_sobel

    @staticmethod
    def _apply_threshold(image: np.ndarray, threshold: Tuple[int, int]) -> np.ndarray:
        image_binarized = np.zeros_like(image)
        image_binarized[(image >= threshold[0]) & (image <= threshold[1])] = 1
        return image_binarized

    def _apply_perspective_transform(self, image_gradients: np.ndarray) -> np.ndarray:
        img_size = (image_gradients.shape[1], image_gradients.shape[0])
        image_transformed = cv2.warpPerspective(image_gradients, self.transform_matrix, img_size)
        return image_transformed

    @staticmethod
    def _to_hls(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2GRAY)
