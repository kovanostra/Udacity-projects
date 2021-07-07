import os
from pathlib import Path

import cv2
import numpy as np
from matplotlib import image as mpimg, pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip

from src.domain.logger import get_logger
from src.infrastructure.parameters import SOURCE_POINTS, DESTINATION_X_MIN_PERCENTAGE, DESTINATION_X_MAX_PERCENTAGE
from src.use_case.lanes_detector import LanesDetector


class VideoLanesDetector(LanesDetector):
    def __init__(self) -> None:
        super().__init__()
        self.video = None
        self.video_frame = None
        self.output_path = None

    def build(self, video_path: str, calibration_directory: str, output_directory: str, record_all_layers: bool) -> None:
        self.record_all_layers = record_all_layers
        self._set_output_path(output_directory, video_path)
        self._load_video(calibration_directory, video_path)
        self._calibrate_camera()
        self._get_perspective_transform()

    def start(self) -> None:
        get_logger().info("Started processing video {}".format(self.output_path.split("/")[-1]))
        white_clip = self.video.fl_image(self._apply_pipeline)
        white_clip.write_videofile(self.output_path, audio=False)
        get_logger().info("Successfully saved video at {}".format(self.output_path))

    def _set_output_path(self, output_directory: str, video_path: str) -> None:
        Path.mkdir(Path(output_directory), exist_ok=True)
        self.output_path = os.path.join(output_directory, video_path.split("/")[-1])

    def _load_video(self, calibration_directory: str, video_path: str):
        self.video = VideoFileClip(video_path)
        self._extract_a_video_frame(video_path)
        for image_name in os.listdir(calibration_directory):
            self.calibration_images.append(mpimg.imread(os.path.join(calibration_directory, image_name)))
        get_logger().info("Loaded video from ./{}".format(video_path))

    def _extract_a_video_frame(self, video_path: str) -> None:
        video_capture = cv2.VideoCapture(video_path)
        success, image = video_capture.read()
        while success:
            self.video_frame = image
            break

    def _get_perspective_transform(self):
        source_points = np.float32(SOURCE_POINTS)
        video_frame_height, video_frame_width, _ = self.video_frame.shape
        x_min_pixel = int(video_frame_width * DESTINATION_X_MIN_PERCENTAGE)
        x_max_pixel = int(video_frame_width * DESTINATION_X_MAX_PERCENTAGE)
        destination_points = np.float32([[x_min_pixel, 0],
                                         [x_min_pixel, video_frame_height],
                                         [x_max_pixel, 0],
                                         [x_max_pixel, video_frame_height]])
        self.transform_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        self.transform_matrix_inverse = cv2.getPerspectiveTransform(destination_points, source_points)
        get_logger().info("Calculated the perspective transform matrices")
