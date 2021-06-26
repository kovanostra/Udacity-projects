from typing import Tuple, List

import numpy as np

from src.window import Window


class LaneFinder:
    def __init__(self,
                 image: np.ndarray,
                 base: np.ndarray,
                 number_of_windows: int) -> None:
        self.image = image
        self.base = base
        self.current = base
        self.nonzero_y, self.nonzero_x = self._set_nonzero()
        self.number_of_windows = number_of_windows
        self.window_height = self._set_window_height()
        self.windows = []
        self.lane_indices = None
        self.y_positions = None
        self.x_positions = None

    def search_lane_points(self, margin: int, min_pixels: int) -> None:
        all_lane_indices = []
        for window_number in range(self.number_of_windows):
            current_window = self._get_current_window_dimensions(margin, window_number)
            self.windows.append(current_window)
            image_window = self.image[current_window.y_low:current_window.y_high,
                                      current_window.x_low:current_window.x_high]

            good_indices = self._get_nonzero_indices_within_the_current_window(current_window)
            all_lane_indices.append(good_indices)
            self._recenter_next_window(current_window, image_window, min_pixels)

        self.lane_indices = np.concatenate(all_lane_indices)

        self.x_positions = self.nonzero_x[self.lane_indices]
        self.y_positions = self.nonzero_y[self.lane_indices]

    def _recenter_next_window(self, current_window: Window, image_window: np.ndarray, min_pixels: int) -> None:
        if np.nonzero(image_window)[0].shape[0] > min_pixels:
            self.current = current_window.x_low + int(np.average(np.nonzero(image_window)[1]))

    def _get_nonzero_indices_within_the_current_window(self, current_window: Window) -> Tuple:
        return ((self.nonzero_y >= current_window.y_low) & (self.nonzero_y < current_window.y_high) &
                (self.nonzero_x >= current_window.x_low) & (self.nonzero_x < current_window.x_high)).nonzero()[0]

    def _get_current_window_dimensions(self, margin: int, window_number: int) -> Window:
        current_window = Window(margin, window_number, self.window_height)
        current_window.set_dimensions(self.image, self.current)
        return current_window

    def _set_window_height(self) -> int:
        return np.int(self.image.shape[0] // self.number_of_windows)

    def _set_nonzero(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array(self.image.nonzero()[0]), np.array(self.image.nonzero()[1])
