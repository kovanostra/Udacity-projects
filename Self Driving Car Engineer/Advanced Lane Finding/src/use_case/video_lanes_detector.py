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

    def build(self, video_path: str, calibration_directory: str, output_directory: str) -> None:
        Path.mkdir(Path(output_directory), exist_ok=True)
        self.output_path = os.path.join(output_directory, video_path.split("/")[-1])
        self._load_video(calibration_directory, video_path)
        self._calibrate_camera()
        self._get_perspective_transform()

    def start(self) -> None:
        white_clip = self.video.fl_image(self._process_frame)
        white_clip.write_videofile(self.output_path, audio=False)

    def _process_frame(self, image: np.ndarray) -> np.ndarray:
        image_undistorted = self._undistort_image(np.copy(image))
        image_gradients = self._get_gradients(image_undistorted)
        image_region = self._region_of_interest(image_gradients)
        image_transformed = self._apply_perspective_transform(image_region)
        road_lanes = self._find_road_lanes(image_transformed)
        road_lanes_reverted = self._apply_inverse_perspective_transform(road_lanes)
        final_image = (np.copy(image_undistorted) + road_lanes_reverted.astype(int)) // 2

        # f, axarr = plt.subplots(2, 3)
        # axarr[0, 0].imshow(image_gradients)
        # axarr[0, 1].imshow(image_region)
        # axarr[0, 2].imshow(image_transformed)
        # axarr[1, 0].imshow(road_lanes)
        # axarr[1, 1].imshow(road_lanes_reverted)
        # axarr[1, 2].imshow(final_image)
        # plt.show()
        return final_image

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
        destination_points = np.float32([[int(self.video_frame.shape[1] * DESTINATION_X_MIN_PERCENTAGE), 0],
                                         [int(self.video_frame.shape[1] * DESTINATION_X_MIN_PERCENTAGE),
                                          self.video_frame.shape[0]],
                                         [int(self.video_frame.shape[1] * DESTINATION_X_MAX_PERCENTAGE), 0],
                                         [int(self.video_frame.shape[1] * DESTINATION_X_MAX_PERCENTAGE),
                                          self.video_frame.shape[0]]])
        self.transform_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        self.transform_matrix_inverse = cv2.getPerspectiveTransform(destination_points, source_points)
        get_logger().info("Calculated the perspective transform matrices")
