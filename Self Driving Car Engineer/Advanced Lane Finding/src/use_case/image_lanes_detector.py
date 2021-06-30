import os
from typing import Tuple

import cv2
import numpy as np
from matplotlib import image as mpimg, pyplot as plt

from src.domain.lane import Lane
from src.domain.lane_finder import LaneFinder
from src.domain.logger import get_logger
from src.infrastructure.parameters import *


class ImageLanesDetector:
    def __init__(self) -> None:
        self.images = []
        self.calibration_images = []
        self.output_directory = None
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.transform_matrix = None
        self.transform_matrix_inverse = None

    def build(self, images_directory: str, calibration_directory: str, output_directory: str) -> None:
        self.output_directory = output_directory
        self._load_images(calibration_directory, images_directory)
        self._calibrate_camera()
        self._get_perspective_transform()

    def start(self) -> None:
        for image in self.images:
            image_undistorted = self._undistort_image(np.copy(image))
            image_gradients = self._get_gradients(image_undistorted)
            image_transformed = self._apply_perspective_transform(image_gradients)
            road_lanes = self._find_road_lanes(image_transformed)
            road_lanes_reverted = self._apply_inverse_perspective_transform(road_lanes)
            final_image = (np.copy(image_undistorted) + road_lanes_reverted.astype(int)) // 2
            plt.imshow(final_image)
            plt.show()

    def _load_images(self, calibration_directory: str, images_directory: str):
        for image_name in os.listdir(images_directory):
            self.images.append(mpimg.imread(os.path.join(images_directory, image_name)))
        for image_name in os.listdir(calibration_directory):
            self.calibration_images.append(mpimg.imread(os.path.join(calibration_directory, image_name)))
        get_logger().info("Loaded images from ./{}".format(images_directory))

    def _calibrate_camera(self, nx: int = NX, ny: int = NY):
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
        source_points = np.float32(SOURCE_POINTS)
        destination_points = np.float32([[int(self.images[0].shape[1] * DESTINATION_X_MIN_PERCENTAGE), 0],
                                         [int(self.images[0].shape[1] * DESTINATION_X_MIN_PERCENTAGE),
                                          self.images[0].shape[0]],
                                         [int(self.images[0].shape[1] * DESTINATION_X_MAX_PERCENTAGE), 0],
                                         [int(self.images[0].shape[1] * DESTINATION_X_MAX_PERCENTAGE),
                                          self.images[0].shape[0]]])
        self.transform_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        self.transform_matrix_inverse = cv2.getPerspectiveTransform(destination_points, source_points)
        get_logger().info("Calculated the perspective transform matrices")

    def _undistort_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.undistort(image,
                             self.camera_matrix,
                             self.distortion_coefficients,
                             None,
                             self.camera_matrix)

    def _get_gradients(self, image: np.ndarray) -> np.ndarray:
        s_channel = self._to_hls(image)[:, :, 2]
        scaled_gradient_x = self._get_scaled_gradient_x(s_channel)
        sobel_binary = self._apply_threshold(scaled_gradient_x, SX_THRESH)
        get_logger().info("Computed image gradients and binarized image")
        return sobel_binary

    @staticmethod
    def _get_scaled_gradient_x(s_channel: np.ndarray) -> np.ndarray:
        sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
        scaled_sobel = np.uint8(255 * np.absolute(sobelx) / np.max(np.absolute(sobelx)))
        return scaled_sobel

    @staticmethod
    def _apply_threshold(image: np.ndarray, threshold: Tuple[int, int]) -> np.ndarray:
        image_binarized = np.zeros_like(image)
        image_binarized[(image >= threshold[0]) & (image <= threshold[1])] = 1
        return image_binarized

    def _apply_perspective_transform(self, image: np.ndarray) -> np.ndarray:
        img_size = (image.shape[1], image.shape[0])
        image_transformed = cv2.warpPerspective(image, self.transform_matrix, img_size)
        return image_transformed

    def _apply_inverse_perspective_transform(self, image: np.ndarray) -> np.ndarray:
        img_size = (image.shape[1], image.shape[0])
        image_transformed = cv2.warpPerspective(image, self.transform_matrix_inverse, img_size)
        return image_transformed

    def _find_road_lanes(self, image: np.ndarray) -> np.ndarray:
        left_lane, right_lane = self._find_lane_pixels(image)
        left_fit_x, right_fit_x = self._fit_polynomial(image, left_lane, right_lane)
        output_image = self._draw_lanes(image, left_fit_x, right_fit_x)
        return output_image

    @staticmethod
    def _draw_lanes(image: np.ndarray, left_fit_x: np.ndarray, right_fit_x: np.ndarray) -> np.ndarray:
        output_image = np.zeros(shape=(image.shape[0], image.shape[1], 3))
        left_y = np.linspace(0, len(left_fit_x) - 1, len(left_fit_x)).astype(int)
        right_y = np.linspace(0, len(right_fit_x) - 1, len(right_fit_x)).astype(int)
        lane_area_points = list(zip(left_fit_x, left_y)) + list(zip(right_fit_x, right_y))
        lane_area_points_sorted = np.array(sorted(lane_area_points, key=lambda x: x[1]), dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(output_image, [lane_area_points_sorted], GREEN)
        return output_image

    @staticmethod
    def _find_lane_pixels(image: np.ndarray) -> Tuple[Lane, Lane]:
        histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
        histogram_midpoint = np.int(histogram.shape[0] // 2)
        left_lane_finder = LaneFinder(image=image, base=np.argmax(histogram[:histogram_midpoint]))
        left_lane = left_lane_finder.search_lane_points()
        right_lane_finder = LaneFinder(image=image, base=np.argmax(histogram[histogram_midpoint:]) + histogram_midpoint)
        right_lane = right_lane_finder.search_lane_points()
        return left_lane, right_lane

    def _fit_polynomial(self, image: np.ndarray, left_lane: Lane, right_lane: Lane) -> Tuple[np.ndarray, np.ndarray]:
        left_parameters = np.polyfit(left_lane.y, left_lane.x, deg=2)
        right_parameters = np.polyfit(right_lane.y, right_lane.x, deg=2)

        polynomial_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_polynomial_x = self._get_polynomial_x_coordinates(left_parameters, polynomial_y)
        right_polynomial_x = self._get_polynomial_x_coordinates(right_parameters, polynomial_y)

        left_polynomial_x_filtered = np.array([int(x) for x in left_polynomial_x if x > 0])
        right_polynomial_x_filtered = np.array([int(x) for x in right_polynomial_x if x <= image.shape[1]])
        return left_polynomial_x_filtered, right_polynomial_x_filtered

    @staticmethod
    def _get_polynomial_x_coordinates(parameters: tuple, y: np.ndarray) -> np.ndarray:
        a, b, c = parameters
        x = (a * y ** 2) + (b * y) + c
        return x

    @staticmethod
    def _to_hls(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2GRAY)
