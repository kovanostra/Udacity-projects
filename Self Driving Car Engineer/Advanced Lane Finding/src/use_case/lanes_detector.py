from abc import ABCMeta, abstractmethod

import numpy as np

from src.domain.frame_binarizer import FrameBinarizer
from src.domain.frame_layers_recorder import FrameLayersRecorder
from src.domain.frame_transformer import FrameTransformer
from src.domain.frame_visualizer import FrameVisualizer
from src.domain.lane_finder import LaneFinder
from src.domain.line_fitter import LineFitter
from src.infrastructure.parameters import *


class LanesDetector(metaclass=ABCMeta):
    def __init__(self,
                 frame_transformer: FrameTransformer,
                 frame_binarizer: FrameBinarizer,
                 lane_finder: LaneFinder,
                 line_fitter: LineFitter,
                 frame_visualizer: FrameVisualizer,
                 frame_layers_recorder: FrameLayersRecorder) -> None:
        self.frame_transformer = frame_transformer
        self.frame_binarizer = frame_binarizer
        self.lane_finder = lane_finder
        self.line_fitter = line_fitter
        self.frame_visualizer = frame_visualizer
        self.frame_layers_recorder = frame_layers_recorder
        self.record_all_layers = False
        self.calibration_frames = []
        self.frame = {UNDISTORTED: np.ndarray([]),
                      BINARIZED: np.ndarray([]),
                      ROI: np.ndarray([]),
                      TRANSFORMED: np.ndarray([]),
                      LANES: np.ndarray([]),
                      LANES_REVERTED: np.ndarray([]),
                      FINAL: np.ndarray([])}

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

    def _apply_pipeline(self, frame_raw: np.ndarray) -> np.ndarray:
        self.frame[UNDISTORTED] = self.frame_transformer.undistort_frame(frame_raw)
        self.frame[BINARIZED] = self.frame_binarizer.binarize(self.frame[UNDISTORTED])
        self.frame[ROI] = self.frame_transformer.isolate_region_of_interest(self.frame[BINARIZED])
        self.frame[TRANSFORMED] = self.frame_transformer.apply_perspective_transform(self.frame[ROI])
        self.frame[LANES] = self._find_road_lanes(self.frame[TRANSFORMED])
        self._inverse_perspective_transform()
        self._finalize_frame()
        return self.frame[FINAL]

    def _find_road_lanes(self, frame: np.ndarray) -> np.ndarray:
        self.lane_finder.detect_lanes(frame, self.line_fitter.fit_parameters)
        self.line_fitter.fit_second_order_polynomial(frame, self.lane_finder.lanes)
        output_frame = self.frame_visualizer.draw_lanes(frame, self.line_fitter.polynomial)
        return output_frame

    def _inverse_perspective_transform(self) -> None:
        self.frame[LANES_REVERTED] = self.frame_transformer.apply_inverse_perspective_transform(self.frame[LANES])
        self.frame_visualizer.add_text_to_frame(self.frame[LANES_REVERTED],
                                                self.line_fitter.curvature,
                                                self.lane_finder.distance_from_centre)

    def _finalize_frame(self) -> None:
        self.frame[FINAL] = self._add_lanes_to_undistorted_frame()
        if self.record_all_layers:
            self.frame[FINAL] = self.frame_layers_recorder.record_all_layers(self.frame)

    def _add_lanes_to_undistorted_frame(self) -> np.ndarray:
        return (self.frame[UNDISTORTED] + self.frame[LANES_REVERTED].astype(int)) // 2
