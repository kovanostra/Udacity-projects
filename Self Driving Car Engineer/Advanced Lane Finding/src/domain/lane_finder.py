from typing import Tuple, List

import numpy as np

from src.domain.lane import Lane
from src.domain.window import Window
from src.infrastructure.parameters import NUMBER_OF_WINDOWS, MARGIN, MIN_PIXELS


class LaneFinder:
    def __init__(self, image: np.ndarray, base: np.ndarray, fit_parameters: np.ndarray) -> None:
        self.image = image
        self.base = base
        self.fit_parameters = fit_parameters
        self.current = self.base
        self.nonzero_y, self.nonzero_x = self._set_nonzero()
        self.window_height = self._set_window_height()
        self.windows = []
        self.lane_indices = None

    def search_lane_points(self) -> Lane:
        if self.fit_parameters is None:
            all_lane_indices = self._apply_sliding_window_search()
            self.lane_indices = np.concatenate(all_lane_indices)
        else:
            self.lane_indices = self._apply_search_on_previous_frame_results()
        return Lane(x=self.nonzero_x[self.lane_indices], y=self.nonzero_y[self.lane_indices])

    def _apply_search_on_previous_frame_results(self) -> List[int]:
        return ((self.nonzero_x > (
                self.fit_parameters[0] * (self.nonzero_y ** 2) + self.fit_parameters[1] * self.nonzero_y +
                self.fit_parameters[2] - MARGIN))
                & (self.nonzero_x < (
                        self.fit_parameters[0] * (self.nonzero_y ** 2) + self.fit_parameters[1] * self.nonzero_y +
                        self.fit_parameters[2] + MARGIN)))

    def _apply_sliding_window_search(self) -> List:
        all_lane_indices = []
        for window_number in range(NUMBER_OF_WINDOWS):
            current_window = self._get_current_window(MARGIN, window_number)
            self.windows.append(current_window)
            image_window = self.image[current_window.y_low:current_window.y_high,
                           current_window.x_low:current_window.x_high]

            good_indices = self._get_nonzero_indices_within_the_current_window(current_window)
            all_lane_indices.append(good_indices)
            # self._recenter_next_window(current_window, image_window)
        return all_lane_indices

    def _recenter_next_window(self, current_window: Window, image_window: np.ndarray) -> None:
        if np.nonzero(image_window)[0].shape[0] > MIN_PIXELS:
            self.current = current_window.x_low + int(np.average(np.nonzero(image_window)[1]))

    def _get_nonzero_indices_within_the_current_window(self, current_window: Window) -> Tuple:
        return (self._get_nonzero_indices_in_y_dimension(current_window) &
                self._get_nonzero_indices_in_x_dimension(current_window)).nonzero()[0]

    def _get_nonzero_indices_in_y_dimension(self, current_window: Window) -> np.ndarray:
        return (self.nonzero_y >= current_window.y_low) & (self.nonzero_y < current_window.y_high)

    def _get_nonzero_indices_in_x_dimension(self, current_window: Window) -> np.ndarray:
        return (self.nonzero_x >= current_window.x_low) & (self.nonzero_x < current_window.x_high)

    def _get_current_window(self, margin: int, window_number: int) -> Window:
        current_window = Window(margin, window_number, self.window_height)
        current_window.set_dimensions(self.image, self.current)
        return current_window

    def _set_window_height(self) -> int:
        return np.int(self.image.shape[0] // NUMBER_OF_WINDOWS)

    def _set_nonzero(self) -> Tuple[np.ndarray, np.ndarray]:
        image_nonzero = self.image.nonzero()
        return image_nonzero[0], image_nonzero[1]
