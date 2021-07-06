from abc import ABCMeta, abstractmethod
from typing import Tuple, List

import cv2
import numpy as np

from src.domain.lane import Lane
from src.domain.lane_finder import LaneFinder
from src.domain.logger import get_logger
from src.infrastructure.parameters import *


class LanesDetector(metaclass=ABCMeta):
    def __init__(self):
        self.distortion_coefficients = None
        self.transform_matrix = None
        self.calibration_images = []
        self.camera_matrix = None
        self.transform_matrix_inverse = None
        self.left_lane = []
        self.right_lane = []
        self.left_fit_parameters = None
        self.right_fit_parameters = None

    @abstractmethod
    def build(self, images_directory: str, calibration_directory: str, output_directory: str) -> None:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

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

    def _undistort_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.undistort(image,
                             self.camera_matrix,
                             self.distortion_coefficients,
                             None,
                             self.camera_matrix)

    def _region_of_interest(self, image: np.ndarray) -> np.ndarray:
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(image)
        vertices = self._get_vertices(image)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(image.shape) > 2:
            channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    @staticmethod
    def _get_vertices(image: np.ndarray) -> np.ndarray:
        height_limit = 0.6
        width_limit_bottom_min = 0.2
        width_limit_bottom_max = 0.95
        width_limit_middle_min = 0.45
        width_limit_middle_max = 0.55

        return np.array([[(int(image.shape[0]*width_limit_bottom_min), image.shape[0]),
                          (int(image.shape[1]*width_limit_middle_min), int(image.shape[0]*height_limit)),
                          (int(image.shape[1]*width_limit_middle_max), int(image.shape[0]*height_limit)),
                          (int(image.shape[1]*width_limit_bottom_max), image.shape[0])]], dtype=np.int32)

    def _get_gradients(self, image: np.ndarray) -> np.ndarray:
        l_channel = self._to_hls(image)[:, :, 1]
        s_channel = self._to_hls(image)[:, :, 2]
        scaled_gradient_x = self._get_scaled_gradient_x(s_channel)
        sobel_binary = self._apply_threshold(scaled_gradient_x)
        # s_binary = np.zeros_like(s_channel)
        # s_binary[(s_channel >= S_THRESH[0]) & (s_channel <= S_THRESH[1])] = 1
        # Stack each channel
        # color_binary = (sobel_binary + s_binary) // 2
        get_logger().info("Computed image gradients and binarized image")
        return sobel_binary

    @staticmethod
    def _get_scaled_gradient_x(channel: np.ndarray) -> np.ndarray:
        sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
        scaled_sobel = np.uint8(255 * np.absolute(sobelx) / np.max(np.absolute(sobelx)))
        return scaled_sobel

    @staticmethod
    def _apply_threshold(image: np.ndarray) -> np.ndarray:
        image_binarized = np.zeros_like(image)
        image_binarized[(image >= SX_THRESH[0]) & (image <= SX_THRESH[1]) |
                        (image >= S_THRESH[0]) & (image <= S_THRESH[1])] = 1
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
        self._find_lane_pixels(image)
        left_fit_x, right_fit_x = self._fit_polynomial(image)
        output_image = self._draw_lanes(image, left_fit_x, right_fit_x)
        return output_image

    @staticmethod
    def _draw_lanes(image: np.ndarray, left_fit_x: np.ndarray, right_fit_x: np.ndarray) -> np.ndarray:
        output_image = np.zeros(shape=(image.shape[0], image.shape[1], 3))
        left_y = np.linspace(0, len(left_fit_x) - 1, len(left_fit_x)).astype(int)
        right_y = np.linspace(0, len(right_fit_x) - 1, len(right_fit_x)).astype(int)
        lane_area_points = list(zip(left_fit_x, left_y)) + list(zip(right_fit_x, right_y))
        lane_area_points_sorted = np.array(sorted(lane_area_points, key=lambda x: x[1]), dtype=np.int32).reshape(
            (-1, 1, 2))
        cv2.fillPoly(output_image, [lane_area_points_sorted], GREEN)
        return output_image

    def _find_lane_pixels(self, image: np.ndarray) -> None:
        histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
        histogram_midpoint = np.int(histogram.shape[0] // 2)
        left_lane_finder = LaneFinder(image=image,
                                      base=np.argmax(histogram[:histogram_midpoint]),
                                      fit_parameters=self.left_fit_parameters)
        self.left_lane.append(left_lane_finder.search_lane_points())
        right_lane_finder = LaneFinder(image=image,
                                       base=np.argmax(histogram[histogram_midpoint:]) + histogram_midpoint,
                                       fit_parameters=self.right_fit_parameters)
        self.right_lane.append(right_lane_finder.search_lane_points())

    def _fit_polynomial(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.left_lane) > LANES_MEMORY:
            self.left_lane.pop(0)
        if len(self.right_lane) > LANES_MEMORY:
            self.right_lane.pop(0)
        left_x = np.concatenate([lane.x for lane in self.left_lane])
        left_y = np.concatenate([lane.y for lane in self.left_lane])
        self.left_fit_parameters = np.polyfit(left_y, left_x, deg=2)
        right_x = np.concatenate([lane.x for lane in self.right_lane])
        right_y = np.concatenate([lane.y for lane in self.right_lane])
        self.right_fit_parameters = np.polyfit(right_y, right_x, deg=2)

        polynomial_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_polynomial_x = self._get_polynomial_x_coordinates(self.left_fit_parameters, polynomial_y)
        right_polynomial_x = self._get_polynomial_x_coordinates(self.right_fit_parameters, polynomial_y)

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
