from typing import Tuple, List, Dict

import numpy as np

from src.domain.lane import Lane
from src.domain.window import Window
from src.infrastructure.parameters import *


class LaneFinder:
    def __init__(self) -> None:
        self.frame = None
        self.histogram = None
        self.histogram_midpoint = None
        self.current = None
        self.nonzero_y, self.nonzero_x = None, None
        self.window_height = None
        self.lane_indices = None
        self.distance_from_centre = []
        self.lanes = {LEFT: [], RIGHT: []}

    def detect_lanes(self, frame: np.ndarray, fit_parameters: Dict[str, tuple]) -> None:
        self._ensure_memory_time_span()
        self.frame = frame
        self.nonzero_y, self.nonzero_x = self._set_nonzero()
        self.window_height = self._set_window_height()
        self.histogram = np.sum(frame[frame.shape[0] // 2:, :], axis=0)
        self.histogram_midpoint = np.int(self.histogram.shape[0] // 2)
        self._find_lane_pixels(fit_parameters)
        self._update_distance_from_centre(frame)

    def _find_lane_pixels(self, fit_parameters: Dict[str, tuple]) -> None:
        self._search_for_left_lane_points(fit_parameters[LEFT])
        self._search_for_right_lane_points(fit_parameters[RIGHT])

    def _search_for_left_lane_points(self, fit_parameters: Tuple[float, float, float]) -> None:
        self.current = np.argmax(self.histogram[:self.histogram_midpoint])
        if len(fit_parameters) > 0:
            histogram_search = False
        else:
            histogram_search = True
        current_lane = self._search_lane_points(histogram_search=histogram_search, fit_parameters=fit_parameters)
        if any(current_lane.x > self.histogram_midpoint):
            current_lane = self._search_lane_points(histogram_search=True, fit_parameters=fit_parameters)
            self.lanes[LEFT] = self.lanes[LEFT][-FORGET_FRAMES:-1]
        if not any(current_lane.x > self.histogram_midpoint):
            self.lanes[LEFT].append(current_lane)

    def _search_for_right_lane_points(self, fit_parameters: Tuple[float, float, float]) -> None:
        self.current = np.argmax(self.histogram[self.histogram_midpoint:]) + self.histogram_midpoint
        if len(fit_parameters) > 0:
            histogram_search = False
        else:
            histogram_search = True
        current_lane = self._search_lane_points(histogram_search=histogram_search, fit_parameters=fit_parameters)
        if any(current_lane.x < self.histogram_midpoint):
            current_lane = self._search_lane_points(histogram_search=True, fit_parameters=fit_parameters)
            self.lanes[RIGHT] = self.lanes[RIGHT][-FORGET_FRAMES:-1]
        if not any(current_lane.x < self.histogram_midpoint):
            self.lanes[RIGHT].append(current_lane)

    def _search_lane_points(self, histogram_search: bool, fit_parameters: Tuple[float, float, float]) -> Lane:
        if histogram_search:
            all_lane_indices = self._apply_sliding_window_search()
            self.lane_indices = np.concatenate(all_lane_indices)
        else:
            self.lane_indices = self._apply_search_on_previous_frame_results(fit_parameters)
        return Lane(x=self.nonzero_x[self.lane_indices], y=self.nonzero_y[self.lane_indices])

    def _apply_search_on_previous_frame_results(self, fit_parameters: Tuple[float, float, float]) -> List[bool]:
        y_fitted_line = self._get_y_fitted_line(fit_parameters)
        return (self.nonzero_x > (y_fitted_line - MARGIN)) & (self.nonzero_x < (y_fitted_line + MARGIN))

    def _get_y_fitted_line(self, fit_parameters: Tuple[float, float, float]) -> np.ndarray:
        a, b, c = fit_parameters
        return a * (self.nonzero_y ** 2) + b * self.nonzero_y + c

    def _apply_sliding_window_search(self) -> List:
        all_lane_indices = []
        for window_number in range(NUMBER_OF_WINDOWS):
            current_window = self._get_current_window(MARGIN, window_number)
            good_indices = self._get_nonzero_indices_within_the_current_window(current_window)
            all_lane_indices.append(good_indices)
        return all_lane_indices

    def _recenter_next_window(self, current_window: Window, frame_window: np.ndarray) -> None:
        if np.nonzero(frame_window)[0].shape[0] > MIN_PIXELS:
            self.current = current_window.x_low + int(np.average(np.nonzero(frame_window)[1]))

    def _get_nonzero_indices_within_the_current_window(self, current_window: Window) -> Tuple:
        return (self._get_nonzero_indices_in_y_dimension(current_window) &
                self._get_nonzero_indices_in_x_dimension(current_window)).nonzero()[0]

    def _get_nonzero_indices_in_y_dimension(self, current_window: Window) -> List[bool]:
        return (self.nonzero_y >= current_window.y_low) & (self.nonzero_y < current_window.y_high)

    def _get_nonzero_indices_in_x_dimension(self, current_window: Window) -> List[bool]:
        return (self.nonzero_x >= current_window.x_low) & (self.nonzero_x < current_window.x_high)

    def _get_current_window(self, margin: int, window_number: int) -> Window:
        current_window = Window(margin, window_number, self.window_height)
        current_window.set_dimensions(self.frame, self.current)
        return current_window

    def _set_window_height(self) -> int:
        return np.int(self.frame.shape[0] // NUMBER_OF_WINDOWS)

    def _set_nonzero(self) -> Tuple[np.ndarray, np.ndarray]:
        frame_nonzero = self.frame.nonzero()
        return frame_nonzero[0], frame_nonzero[1]

    def _ensure_memory_time_span(self) -> None:
        for variable in [self.lanes[LEFT], self.lanes[RIGHT], self.distance_from_centre]:
            if len(variable) > MEMORY_TIME_SPAN:
                variable.pop(0)

    def _update_distance_from_centre(self, frame: np.ndarray) -> None:
        current_left_lane = self.lanes[LEFT][-1]
        current_right_lane = self.lanes[RIGHT][-1]
        if self._both_lanes_have_detected_points(current_left_lane, current_right_lane):
            lanes_centre = (np.average(current_right_lane.x) - np.average(current_left_lane.x))
            frame_centre = (frame.shape[1] / 2)
            self.distance_from_centre.append((lanes_centre - frame_centre) * X_TO_METERS_PER_PIXEL)

    @staticmethod
    def _both_lanes_have_detected_points(current_left_lane_points: Lane, current_right_lane_points: Lane) -> bool:
        return len(current_right_lane_points.x) > 0 and len(current_left_lane_points.x)
