from typing import Tuple, Dict, List

import numpy as np

from src.domain.lane import Lane
from src.infrastructure.parameters import *


class LineFitter:
    def __init__(self) -> None:
        self.curvature = {LEFT: [], RIGHT: []}
        self.fit_parameters = {LEFT: (), RIGHT: ()}
        self.polynomial = {LEFT: {X: np.array([])},
                           RIGHT: {X: np.array([])},
                           Y: np.array([])}

    def fit_second_order_polynomial(self, frame: np.ndarray, lanes: Dict[str, List[Lane]]) -> None:
        self._ensure_memory_time_span()

        self.polynomial[Y] = np.linspace(0, frame.shape[0] - 1, frame.shape[0])
        y_eval = np.max(self.polynomial[Y])

        self._get_left_polynomial(lanes)
        self._get_right_polynomial(frame, lanes)

        self._get_left_curvature(y_eval)
        self._get_right_curvature(y_eval)

    def _get_left_polynomial(self, lanes: Dict[str, List[Lane]]) -> None:
        self.fit_parameters[LEFT] = self._get_polynomial_fit_parameters(lanes[LEFT])
        left_polynomial_x = self._get_polynomial_x_coordinates(self.fit_parameters[LEFT], self.polynomial[Y])
        self.polynomial[LEFT][X] = np.array([int(x) for x in left_polynomial_x if x > 0])

    def _get_right_polynomial(self, frame: np.ndarray, lanes: Dict[str, List[Lane]]) -> None:
        self.fit_parameters[RIGHT] = self._get_polynomial_fit_parameters(lanes[RIGHT])
        right_polynomial_x = self._get_polynomial_x_coordinates(self.fit_parameters[RIGHT], self.polynomial[Y])
        self.polynomial[RIGHT][X] = np.array([int(x) for x in right_polynomial_x if x <= frame.shape[1]])

    def _get_left_curvature(self, y_eval: int) -> None:
        left_fit_cr = np.polyfit(self.polynomial[Y] * Y_TO_METERS_PER_PIXEL,
                                 self.polynomial[LEFT][X] * X_TO_METERS_PER_PIXEL, 2)
        self.curvature[LEFT].append(self._get_curvature(left_fit_cr, y_eval))

    def _get_right_curvature(self, y_eval: int) -> None:
        y_in_meters = self.polynomial[Y] * Y_TO_METERS_PER_PIXEL
        x_in_meters = self.polynomial[RIGHT][X] * X_TO_METERS_PER_PIXEL
        right_fit_cr = np.polyfit(y_in_meters, x_in_meters, 2)
        self.curvature[RIGHT].append(self._get_curvature(right_fit_cr, y_eval))

    @staticmethod
    def _get_polynomial_fit_parameters(lanes: List[Lane]) -> Tuple:
        x = np.concatenate([lane.x for lane in lanes])
        y = np.concatenate([lane.y for lane in lanes])
        return np.polyfit(y, x, deg=2)

    @staticmethod
    def _get_polynomial_x_coordinates(parameters: tuple, y: np.ndarray) -> np.ndarray:
        a, b, c = parameters
        x = (a * y ** 2) + (b * y) + c
        return x

    @staticmethod
    def _get_curvature(fit_curvature: Tuple, y: float) -> np.ndarray:
        a, b, _ = fit_curvature
        curvature = ((1 + (2 * a * y * Y_TO_METERS_PER_PIXEL + b) ** 2) ** (3 / 2)) / abs(2 * a)
        return curvature

    def _ensure_memory_time_span(self) -> None:
        variables_with_memory = [self.curvature[LEFT], self.curvature[RIGHT]]
        for variable in variables_with_memory:
            if len(variable) > MEMORY_TIME_SPAN:
                variable.pop(0)
