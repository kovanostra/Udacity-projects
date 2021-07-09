from typing import List

import cv2
import numpy as np

from src.domain.logger import get_logger
from src.infrastructure.parameters import *


class FrameTransformer:
    def __init__(self):
        self.frame_points_2d = []
        self.object_points_3d = []
        self.camera_matrix = None
        self.distortion_coefficients = None
        self.transform_matrix = None
        self.transform_matrix_inverse = None

    def calibrate_camera(self, calibration_frames: List[np.ndarray]) -> None:
        self._detect_chessboard_corners(calibration_frames)
        self._apply_camera_calibration(calibration_frames)

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.undistort(frame,
                             self.camera_matrix,
                             self.distortion_coefficients,
                             None,
                             self.camera_matrix)

    def get_perspective_transform_parameters(self, frame: np.ndarray) -> None:
        source_points = np.float32(SOURCE_POINTS)
        frame_height, frame_width, _ = frame.shape
        x_min_pixel = int(frame_width * DESTINATION_X_MIN_PERCENTAGE)
        x_max_pixel = int(frame_width * DESTINATION_X_MAX_PERCENTAGE)
        destination_points = np.float32([[x_min_pixel, 0],
                                         [x_min_pixel, frame_height],
                                         [x_max_pixel, 0],
                                         [x_max_pixel, frame_height]])
        self.transform_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        self.transform_matrix_inverse = cv2.getPerspectiveTransform(destination_points, source_points)
        get_logger().info("Calculated the perspective transform matrices")

    def apply_perspective_transform(self, frame: np.ndarray) -> np.ndarray:
        frame_size = (frame.shape[1], frame.shape[0])
        frame_transformed = cv2.warpPerspective(frame, self.transform_matrix, frame_size)
        return frame_transformed

    def apply_inverse_perspective_transform(self, frame: np.ndarray) -> np.ndarray:
        frame_size = (frame.shape[1], frame.shape[0])
        frame_transformed = cv2.warpPerspective(frame, self.transform_matrix_inverse, frame_size)
        return frame_transformed

    def isolate_region_of_interest(self, frame: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(frame)
        vertices = self._get_vertices(frame)
        if len(frame.shape) > 2:
            channel_count = frame.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_frame = cv2.bitwise_and(frame, mask)
        return masked_frame

    def _detect_chessboard_corners(self, calibration_frames: List[np.ndarray]) -> None:
        all_object_points = np.zeros(shape=(NY * NX, 3), dtype=np.float32)
        all_object_points[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2)
        for frame in calibration_frames:
            gray = self._to_gray(frame)
            ret, corners = cv2.findChessboardCorners(gray, (NX, NY), None)
            if ret:
                self.object_points_3d.append(all_object_points)
                self.frame_points_2d.append(corners)

    def _apply_camera_calibration(self, calibration_frames: List[np.ndarray]) -> None:
        if self.frame_points_2d:
            frame_shape = calibration_frames[0].shape[:2]
            _, self.camera_matrix, self.distortion_coefficients, _, _ = cv2.calibrateCamera(self.object_points_3d,
                                                                                            self.frame_points_2d,
                                                                                            frame_shape,
                                                                                            None,
                                                                                            None)
            get_logger().info("Calibration parameters calculated")
        else:
            get_logger().error("Couldn't detect correctly the chessboard corners. "
                               "Impossible to calculate calibration parameters.")
            raise RuntimeError

    @staticmethod
    def _get_vertices(frame: np.ndarray) -> np.ndarray:
        frame_height, frame_width = frame.shape
        roi_x_bottom_min_pixel = int(frame_width * ROI_WIDTH_LIMIT_BOTTOM_MIN)
        roi_x_bottom_max_pixel = int(frame_width * ROI_WIDTH_LIMIT_BOTTOM_MAX)
        roi_x_top_min_pixel = int(frame_width * ROI_WIDTH_LIMIT_TOP_MIN)
        poi_x_top_max_pixel = int(frame_width * ROI_WIDTH_LIMIT_TOP_MAX)
        roi_y_top_pixel = int(frame_height * ROI_HEIGHT_LIMIT)
        return np.array([[(roi_x_bottom_min_pixel, frame_height),
                          (roi_x_top_min_pixel, roi_y_top_pixel),
                          (poi_x_top_max_pixel, roi_y_top_pixel),
                          (roi_x_bottom_max_pixel, frame_height)]], dtype=np.int32)

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(np.copy(frame), cv2.COLOR_RGB2GRAY)
