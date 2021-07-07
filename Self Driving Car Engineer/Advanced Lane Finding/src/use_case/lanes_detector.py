from abc import ABCMeta, abstractmethod
from typing import Tuple, List

import cv2
import matplotlib
import numpy as np

from src.domain.lane import Lane

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from src.domain.lane_finder import LaneFinder
from src.domain.logger import get_logger
from src.infrastructure.parameters import *


class LanesDetector(metaclass=ABCMeta):
    def __init__(self):
        self.record_all_layers = False
        self.distortion_coefficients = None
        self.calibration_images = []
        self.camera_matrix = None
        self.transform_matrix = None
        self.transform_matrix_inverse = None
        self.left_lanes = []
        self.right_lanes = []
        self.left_fit_parameters = None
        self.right_fit_parameters = None
        self.left_curvature = []
        self.right_curvature = []
        self.distance_from_centre = []

    @abstractmethod
    def build(self,
              source_path: str,
              calibration_directory: str,
              output_directory: str,
              record_all_layers: bool) -> None:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    def _apply_pipeline(self, image: np.ndarray) -> np.ndarray:
        image_undistorted = self._undistort_image(image)
        image_gradients = self._get_gradients(image_undistorted)
        image_region_of_interest = self._region_of_interest(image_gradients)
        image_transformed = self._apply_perspective_transform(image_region_of_interest)
        road_lanes = self._find_road_lanes(image_transformed)
        road_lanes_reverted = self._apply_inverse_perspective_transform(road_lanes)
        self._add_text_to_image(road_lanes_reverted)
        final_image = self._add_lanes_to_undistorted_image(image_undistorted, road_lanes_reverted)
        if self.record_all_layers:
            all_images = [image_gradients, image_region_of_interest, image_transformed,
                          road_lanes, road_lanes_reverted, final_image]
            final_image = self._record_all_layers(all_images)
        return final_image

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
            get_logger().error("Couldn't detect correctly the chessboard corners. "
                               "Impossible to calculate calibration parameters.")
            raise RuntimeError

    def _undistort_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.undistort(image,
                             self.camera_matrix,
                             self.distortion_coefficients,
                             None,
                             self.camera_matrix)

    def _region_of_interest(self, image: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(image)
        vertices = self._get_vertices(image)
        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    @staticmethod
    def _get_vertices(image: np.ndarray) -> np.ndarray:
        image_height, image_width = image.shape
        roi_x_bottom_min_pixel = int(image_width * ROI_WIDTH_LIMIT_BOTTOM_MIN)
        roi_x_bottom_max_pixel = int(image_width * ROI_WIDTH_LIMIT_BOTTOM_MAX)
        roi_x_top_min_pixel = int(image_width * ROI_WIDTH_LIMIT_TOP_MIN)
        poi_x_top_max_pixel = int(image_width * ROI_WIDTH_LIMIT_TOP_MAX)
        roi_y_top_pixel = int(image_height * ROI_HEIGHT_LIMIT)
        return np.array([[(roi_x_bottom_min_pixel, image_height),
                          (roi_x_top_min_pixel, roi_y_top_pixel),
                          (poi_x_top_max_pixel, roi_y_top_pixel),
                          (roi_x_bottom_max_pixel, image_height)]], dtype=np.int32)

    def _get_gradients(self, image: np.ndarray) -> np.ndarray:
        s_channel = self._to_hls(image)[:, :, 2]
        scaled_gradient_x = self._get_scaled_gradient_x(s_channel)
        sobel_binary = self._apply_threshold(scaled_gradient_x)
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
        image_binarized[(image >= MAX_THRESHOLD[0]) & (image <= MAX_THRESHOLD[1]) |
                        (image >= MIN_THRESHOLD[0]) & (image <= MIN_THRESHOLD[1])] = 1
        return image_binarized

    def _apply_perspective_transform(self, image: np.ndarray) -> np.ndarray:
        image_size = (image.shape[1], image.shape[0])
        image_transformed = cv2.warpPerspective(image, self.transform_matrix, image_size)
        return image_transformed

    def _apply_inverse_perspective_transform(self, image: np.ndarray) -> np.ndarray:
        image_size = (image.shape[1], image.shape[0])
        image_transformed = cv2.warpPerspective(image, self.transform_matrix_inverse, image_size)
        return image_transformed

    def _find_road_lanes(self, image: np.ndarray) -> np.ndarray:
        self._find_lane_pixels(image)
        self._ensure_memory_time_span()
        left_fit_x, right_fit_x = self._fit_polynomial(image)
        output_image = self._draw_lanes(image, left_fit_x, right_fit_x)
        return output_image

    def _find_lane_pixels(self, image: np.ndarray) -> None:
        histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
        histogram_midpoint = np.int(histogram.shape[0] // 2)
        current_left_lane = self._search_for_left_lane_points(image, histogram, histogram_midpoint)
        current_right_lane = self._search_for_right_lane(image, histogram, histogram_midpoint)
        self._update_distance_from_centre(image, current_left_lane, current_right_lane)

    def _search_for_right_lane(self, image: np.ndarray, histogram: np.ndarray, histogram_midpoint: int) -> Lane:
        right_highest_histogram_point = np.argmax(histogram[histogram_midpoint:]) + histogram_midpoint
        lane_finder = LaneFinder(image=image,
                                 base=right_highest_histogram_point,
                                 fit_parameters=self.right_fit_parameters)
        current_lane = lane_finder.search_lane_points()
        if any(current_lane.x < histogram_midpoint):
            current_lane = self._start_histogram_lane_search(lane_finder)
        if not any(current_lane.x < histogram_midpoint):
            self.right_lanes.append(current_lane)
        return current_lane

    def _search_for_left_lane_points(self, image: np.ndarray, histogram: np.ndarray, histogram_midpoint: int) -> Lane:
        left_highest_histogram_point = np.argmax(histogram[:histogram_midpoint])
        lane_finder = LaneFinder(image=image,
                                 base=left_highest_histogram_point,
                                 fit_parameters=self.left_fit_parameters)
        current_lane = lane_finder.search_lane_points()
        if any(current_lane.x > histogram_midpoint):
            current_lane = self._start_histogram_lane_search(lane_finder)
        if not any(current_lane.x > histogram_midpoint):
            self.left_lanes.append(current_lane)
        return current_lane

    def _start_histogram_lane_search(self, lane_finder: LaneFinder) -> Lane:
        lane_finder.reset_state_with(fit_parameters=None)
        current_lane = lane_finder.search_lane_points()
        self.left_lanes = self.left_lanes[-FORGET_FRAMES:-1]
        return current_lane

    def _fit_polynomial(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.left_fit_parameters = self._get_polynomial_fit_parameters(self.left_lanes)
        self.right_fit_parameters = self._get_polynomial_fit_parameters(self.right_lanes)

        polynomial_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_polynomial_x = self._get_polynomial_x_coordinates(self.left_fit_parameters, polynomial_y)
        left_polynomial_x_filtered = np.array([int(x) for x in left_polynomial_x if x > 0])
        left_fit_cr = np.polyfit(polynomial_y * Y_TO_METERS_PER_PIXEL, left_polynomial_x * X_TO_METERS_PER_PIXEL, 2)

        right_polynomial_x = self._get_polynomial_x_coordinates(self.right_fit_parameters, polynomial_y)
        right_polynomial_x_filtered = np.array([int(x) for x in right_polynomial_x if x <= image.shape[1]])
        right_fit_cr = np.polyfit(polynomial_y * Y_TO_METERS_PER_PIXEL, right_polynomial_x * X_TO_METERS_PER_PIXEL, 2)

        y_eval = np.max(polynomial_y)

        self.left_curvature.append(self._get_curvature(left_fit_cr, y_eval))
        self.right_curvature.append(self._get_curvature(right_fit_cr, y_eval))

        return left_polynomial_x_filtered, right_polynomial_x_filtered

    @staticmethod
    def _get_polynomial_fit_parameters(lanes: List[Lane]) -> Tuple:
        x = np.concatenate([lane.x for lane in lanes])
        y = np.concatenate([lane.y for lane in lanes])
        return np.polyfit(y, x, deg=2)

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

    @staticmethod
    def _get_curvature(fit_curvature: Tuple, y: float) -> np.ndarray:
        a, b, _ = fit_curvature
        curvature = ((1 + (2 * a * y * Y_TO_METERS_PER_PIXEL + b) ** 2) ** (3 / 2)) / abs(2 * a)
        return curvature

    @staticmethod
    def _get_polynomial_x_coordinates(parameters: tuple, y: np.ndarray) -> np.ndarray:
        a, b, c = parameters
        x = (a * y ** 2) + (b * y) + c
        return x

    def _ensure_memory_time_span(self) -> None:
        variables_with_memory = [self.left_lanes, self.right_lanes, self.left_curvature, self.right_curvature,
                                 self.distance_from_centre]
        for variable in variables_with_memory:
            if len(variable) > MEMORY_TIME_SPAN:
                variable.pop(0)

    def _add_text_to_image(self, image: np.ndarray) -> None:
        cv2.putText(image, "Radius of left curvature {}m".format(round(np.average(self.left_curvature))), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 3)
        cv2.putText(image, "Radius of right curvature {}m".format(round(np.average(self.right_curvature))), (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 3)
        cv2.putText(image, "Distance from centre {}m".format(round(np.average(self.distance_from_centre), 2)),
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 3)

    @staticmethod
    def _to_hls(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _record_all_layers(images: List[np.ndarray]) -> np.ndarray:
        figure, axes = plt.subplots(2, 3)
        axes[0, 0].imshow(images[0])
        axes[0, 0].set_title("Gradients")
        axes[0, 1].imshow(images[1])
        axes[0, 1].set_title("ROI")
        axes[0, 2].imshow(images[2])
        axes[0, 2].set_title("ROI transformed")
        axes[1, 0].imshow(images[3])
        axes[1, 0].set_title("Lanes transformed")
        axes[1, 1].imshow(images[4])
        axes[1, 1].set_title("Lanes original")
        axes[1, 2].imshow(images[5])
        axes[1, 2].set_title("Final image")
        figure.canvas.draw()
        data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        return data

    @staticmethod
    def _add_lanes_to_undistorted_image(image_undistorted: np.ndarray, road_lanes_reverted: np.ndarray) -> np.ndarray:
        return (image_undistorted + road_lanes_reverted.astype(int)) // 2

    def _update_distance_from_centre(self,
                                     image: np.ndarray,
                                     current_left_lane_points: Lane,
                                     current_right_lane_points: Lane) -> None:
        if self._both_lanes_have_detected_points(current_left_lane_points, current_right_lane_points):
            lanes_centre = (np.average(current_right_lane_points.x) - np.average(current_left_lane_points.x))
            image_centre = (image.shape[1] / 2)
            self.distance_from_centre.append((lanes_centre - image_centre) * X_TO_METERS_PER_PIXEL)

    @staticmethod
    def _both_lanes_have_detected_points(current_left_lane_points: Lane, current_right_lane_points: Lane) -> bool:
        return len(current_right_lane_points.x) > 0 and len(current_left_lane_points.x)
