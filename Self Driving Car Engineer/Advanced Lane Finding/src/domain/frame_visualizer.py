from typing import List

import cv2
import numpy as np

from src.infrastructure.parameters import *


class FrameVisualizer:
    def __init__(self) -> None:
        pass

    @staticmethod
    def draw_lanes(frame: np.ndarray, polynomial: np.ndarray) -> np.ndarray:
        output_frame = np.zeros(shape=(frame.shape[0], frame.shape[1], 3))
        left_y = np.linspace(0, len(polynomial[LEFT][X]) - 1, len(polynomial[LEFT][X])).astype(int)
        right_y = np.linspace(0, len(polynomial[RIGHT][X]) - 1, len(polynomial[RIGHT][X])).astype(int)
        lane_area_points = list(zip(polynomial[LEFT][X], left_y)) + list(zip(polynomial[RIGHT][X], right_y))
        lane_area_points_sorted = np.array(sorted(lane_area_points, key=lambda x: x[1]),
                                           dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(output_frame, [lane_area_points_sorted], GREEN)
        return output_frame

    @staticmethod
    def add_text_to_frame(frame: np.ndarray, curvature: List[float], distance_from_centre: List[float]) -> None:
        cv2.putText(frame, "Radius of left curvature {}m".format(round(np.average(curvature[LEFT]))), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 3)
        cv2.putText(frame, "Radius of right curvature {}m".format(round(np.average(curvature[RIGHT]))), (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 3)
        cv2.putText(frame, "Distance from centre {}m".format(round(np.average(distance_from_centre), 2)), (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 3)
