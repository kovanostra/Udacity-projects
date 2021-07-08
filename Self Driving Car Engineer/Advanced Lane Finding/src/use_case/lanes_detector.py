from abc import ABCMeta, abstractmethod
from typing import Tuple, List

import cv2
import numpy as np

from src.domain.frame_binarizer import FrameBinarizer
from src.domain.frame_layers_recorder import FrameLayersRecorder
from src.domain.frame_transformer import FrameTransformer
from src.domain.lane import Lane
from src.domain.lane_finder import LaneFinder
from src.infrastructure.parameters import *


class LanesDetector(metaclass=ABCMeta):
    def __init__(self,
                 frame_transformer: FrameTransformer,
                 frame_binarizer: FrameBinarizer,
                 frame_layers_recorder: FrameLayersRecorder):
        self.frame_transformer = frame_transformer
        self.frame_binarizer = frame_binarizer
        self.frame_layers_recorder = frame_layers_recorder
        self.frame_layers = {"undistorted": np.ndarray([]),
                             "binarized": np.ndarray([]),
                             "region_of_interest": np.ndarray([]),
                             "road_lanes": np.ndarray([]),
                             "road_lanes_reverted": np.ndarray([]),
                             "final": np.ndarray([])}
        self.record_all_layers = False
        self.calibration_frames = []
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

    def _apply_pipeline(self, frame: np.ndarray) -> np.ndarray:
        self.frame_layers["undistorted"] = self.frame_transformer.undistort_frame(frame)
        self.frame_layers["binarized"] = self.frame_binarizer.binarize(self.frame_layers["undistorted"])
        self.frame_layers["region_of_interest"] = self.frame_transformer.isolate_region_of_interest(
            self.frame_layers["binarized"])
        self.frame_layers["transformed"] = self.frame_transformer.apply_perspective_transform(
            self.frame_layers["region_of_interest"])
        self.frame_layers["road_lanes"] = self._find_road_lanes(self.frame_layers["transformed"])
        self.frame_layers["road_lanes_reverted"] = self.frame_transformer.apply_inverse_perspective_transform(
            self.frame_layers["road_lanes"])
        self._add_text_to_frame(self.frame_layers["road_lanes_reverted"])
        self.frame_layers["final"] = self._add_lanes_to_undistorted_frame(self.frame_layers["undistorted"],
                                                                          self.frame_layers["road_lanes_reverted"])
        if self.record_all_layers:
            self.frame_layers["final"] = self.frame_layers_recorder.record_all_layers(self.frame_layers)
        return self.frame_layers["final"]

    def _find_road_lanes(self, frame: np.ndarray) -> np.ndarray:
        self._find_lane_pixels(frame)
        self._ensure_memory_time_span()
        left_fit_x, right_fit_x = self._fit_polynomial(frame)
        output_frame = self._draw_lanes(frame, left_fit_x, right_fit_x)
        return output_frame

    def _find_lane_pixels(self, frame: np.ndarray) -> None:
        histogram = np.sum(frame[frame.shape[0] // 2:, :], axis=0)
        histogram_midpoint = np.int(histogram.shape[0] // 2)
        current_left_lane = self._search_for_left_lane_points(frame, histogram, histogram_midpoint)
        current_right_lane = self._search_for_right_lane(frame, histogram, histogram_midpoint)
        self._update_distance_from_centre(frame, current_left_lane, current_right_lane)

    def _search_for_right_lane(self, frame: np.ndarray, histogram: np.ndarray, histogram_midpoint: int) -> Lane:
        right_highest_histogram_point = np.argmax(histogram[histogram_midpoint:]) + histogram_midpoint
        lane_finder = LaneFinder(frame=frame,
                                 base=right_highest_histogram_point,
                                 fit_parameters=self.right_fit_parameters)
        current_lane = lane_finder.search_lane_points()
        if any(current_lane.x < histogram_midpoint):
            current_lane = self._start_histogram_lane_search(lane_finder)
        if not any(current_lane.x < histogram_midpoint):
            self.right_lanes.append(current_lane)
        return current_lane

    def _search_for_left_lane_points(self, frame: np.ndarray, histogram: np.ndarray, histogram_midpoint: int) -> Lane:
        left_highest_histogram_point = np.argmax(histogram[:histogram_midpoint])
        lane_finder = LaneFinder(frame=frame,
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

    def _fit_polynomial(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.left_fit_parameters = self._get_polynomial_fit_parameters(self.left_lanes)
        self.right_fit_parameters = self._get_polynomial_fit_parameters(self.right_lanes)

        polynomial_y = np.linspace(0, frame.shape[0] - 1, frame.shape[0])
        left_polynomial_x = self._get_polynomial_x_coordinates(self.left_fit_parameters, polynomial_y)
        left_polynomial_x_filtered = np.array([int(x) for x in left_polynomial_x if x > 0])
        left_fit_cr = np.polyfit(polynomial_y * Y_TO_METERS_PER_PIXEL, left_polynomial_x * X_TO_METERS_PER_PIXEL, 2)

        right_polynomial_x = self._get_polynomial_x_coordinates(self.right_fit_parameters, polynomial_y)
        right_polynomial_x_filtered = np.array([int(x) for x in right_polynomial_x if x <= frame.shape[1]])
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
    def _draw_lanes(frame: np.ndarray, left_fit_x: np.ndarray, right_fit_x: np.ndarray) -> np.ndarray:
        output_frame = np.zeros(shape=(frame.shape[0], frame.shape[1], 3))
        left_y = np.linspace(0, len(left_fit_x) - 1, len(left_fit_x)).astype(int)
        right_y = np.linspace(0, len(right_fit_x) - 1, len(right_fit_x)).astype(int)
        lane_area_points = list(zip(left_fit_x, left_y)) + list(zip(right_fit_x, right_y))
        lane_area_points_sorted = np.array(sorted(lane_area_points, key=lambda x: x[1]), dtype=np.int32).reshape(
            (-1, 1, 2))
        cv2.fillPoly(output_frame, [lane_area_points_sorted], GREEN)
        return output_frame

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

    def _add_text_to_frame(self, frame: np.ndarray) -> None:
        cv2.putText(frame, "Radius of left curvature {}m".format(round(np.average(self.left_curvature))), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 3)
        cv2.putText(frame, "Radius of right curvature {}m".format(round(np.average(self.right_curvature))), (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 3)
        cv2.putText(frame, "Distance from centre {}m".format(round(np.average(self.distance_from_centre), 2)),
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 3)

    @staticmethod
    def _add_lanes_to_undistorted_frame(frame_undistorted: np.ndarray, road_lanes_reverted: np.ndarray) -> np.ndarray:
        return (frame_undistorted + road_lanes_reverted.astype(int)) // 2

    def _update_distance_from_centre(self,
                                     frame: np.ndarray,
                                     current_left_lane_points: Lane,
                                     current_right_lane_points: Lane) -> None:
        if self._both_lanes_have_detected_points(current_left_lane_points, current_right_lane_points):
            lanes_centre = (np.average(current_right_lane_points.x) - np.average(current_left_lane_points.x))
            frame_centre = (frame.shape[1] / 2)
            self.distance_from_centre.append((lanes_centre - frame_centre) * X_TO_METERS_PER_PIXEL)

    @staticmethod
    def _both_lanes_have_detected_points(current_left_lane_points: Lane, current_right_lane_points: Lane) -> bool:
        return len(current_right_lane_points.x) > 0 and len(current_left_lane_points.x)
