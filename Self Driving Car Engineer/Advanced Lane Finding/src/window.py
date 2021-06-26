import numpy as np


class Window:
    def __init__(self, margin: int, window_number: int, window_height: int) -> None:
        self.margin = margin
        self.window_number = window_number
        self.window_height = window_height
        self.y_low = None
        self.y_high = None
        self.x_low = None
        self.x_high = None

    def set_dimensions(self, image: np.ndarray, current_point: np.ndarray) -> None:
        self.y_low = image.shape[0] - (self.window_number + 1) * self.window_height
        self.y_high = image.shape[0] - self.window_number * self.window_height
        self.x_low = current_point - self.margin // 2
        self.x_high = current_point + self.margin // 2
