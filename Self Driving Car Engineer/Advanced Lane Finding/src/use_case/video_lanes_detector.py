import os
from pathlib import Path

import cv2
from matplotlib import image as mpimg
from moviepy.video.io.VideoFileClip import VideoFileClip

from src.domain.frame_binarizer import FrameBinarizer
from src.domain.frame_layers_recorder import FrameLayersRecorder
from src.domain.frame_transformer import FrameTransformer
from src.domain.frame_visualizer import FrameVisualizer
from src.domain.lane_finder import LaneFinder
from src.domain.line_fitter import LineFitter
from src.domain.logger import get_logger
from src.use_case.lanes_detector import LanesDetector


class VideoLanesDetector(LanesDetector):
    def __init__(self,
                 frame_transformer: FrameTransformer,
                 frame_binarizer: FrameBinarizer,
                 lane_finder: LaneFinder,
                 line_fitter: LineFitter,
                 frame_visualizer: FrameVisualizer,
                 frame_layers_recorder: FrameLayersRecorder) -> None:
        super().__init__(frame_transformer,
                         frame_binarizer,
                         lane_finder,
                         line_fitter,
                         frame_visualizer,
                         frame_layers_recorder)
        self.video = None
        self.video_frame = None
        self.output_path = None

    def build(self,
              video_path: str,
              calibration_directory: str,
              output_directory: str,
              record_all_layers: bool) -> None:
        self.record_all_layers = record_all_layers
        self._set_output_path(output_directory, video_path)
        self._load_video(calibration_directory, video_path)
        self.frame_transformer.calibrate_camera(self.calibration_frames)
        self.frame_transformer.get_perspective_transform_parameters(self.video_frame)

    def start(self) -> None:
        get_logger().info("Started processing video {}".format(self.output_path.split("/")[-1]))
        white_clip = self.video.fl_image(self._apply_pipeline)
        white_clip.write_videofile(self.output_path, audio=False)
        get_logger().info("Successfully saved video at {}".format(self.output_path))

    def _set_output_path(self, output_directory: str, video_path: str) -> None:
        Path.mkdir(Path(output_directory), exist_ok=True)
        self.output_path = os.path.join(output_directory, video_path.split("/")[-1])

    def _load_video(self, calibration_directory: str, video_path: str) -> None:
        self.video = VideoFileClip(video_path)
        self._extract_a_video_frame(video_path)
        for image_name in os.listdir(calibration_directory):
            self.calibration_frames.append(mpimg.imread(os.path.join(calibration_directory, image_name)))
        get_logger().info("Loaded video from ./{}".format(video_path))

    def _extract_a_video_frame(self, video_path: str) -> None:
        video_capture = cv2.VideoCapture(video_path)
        success, image = video_capture.read()
        while success:
            self.video_frame = image
            break
