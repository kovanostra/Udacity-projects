import os

import cv2
import numpy as np
from matplotlib import image as mpimg, pyplot as plt

from src.domain.logger import get_logger
from src.infrastructure.parameters import SOURCE_POINTS, DESTINATION_X_MAX_PERCENTAGE, DESTINATION_X_MIN_PERCENTAGE
from src.use_case.lanes_detector import LanesDetector


class ImageLanesDetector(LanesDetector):
    def __init__(self) -> None:
        super().__init__()
        self.images = []
        self.output_directory = None

    def build(self, images_directory: str, calibration_directory: str, output_directory: str, record_all_layers: bool) -> None:
        self.output_directory = output_directory
        self.record_all_layers = record_all_layers
        self._load_images(calibration_directory, images_directory)
        self._calibrate_camera()
        self._get_perspective_transform()

    def start(self) -> None:
        for image, image_name in self.images:
            self._reset_state()
            final_image = self._apply_pipeline(image)
            cv2.imwrite(os.path.join(self.output_directory, image_name), cv2.cvtColor(np.float32(final_image), cv2.COLOR_RGB2BGR))

    def _reset_state(self) -> None:
        self.left_parameters = None
        self.right_parameters = None
        self.left_lane = []
        self.right_lane = []

    def _load_images(self, calibration_directory: str, images_directory: str):
        for image_name in os.listdir(images_directory):
            self.images.append([mpimg.imread(os.path.join(images_directory, image_name)), image_name])
        for image_name in os.listdir(calibration_directory):
            self.calibration_images.append(mpimg.imread(os.path.join(calibration_directory, image_name)))
        get_logger().info("Loaded images from ./{}".format(images_directory))

    def _get_perspective_transform(self):
        source_points = np.float32(SOURCE_POINTS)
        destination_points = np.float32([[int(self.images[0][0].shape[1] * DESTINATION_X_MIN_PERCENTAGE), 0],
                                         [int(self.images[0][0].shape[1] * DESTINATION_X_MIN_PERCENTAGE),
                                          self.images[0][0].shape[0]],
                                         [int(self.images[0][0].shape[1] * DESTINATION_X_MAX_PERCENTAGE), 0],
                                         [int(self.images[0][0].shape[1] * DESTINATION_X_MAX_PERCENTAGE),
                                          self.images[0][0].shape[0]]])
        self.transform_matrix = cv2.getPerspectiveTransform(source_points, destination_points)
        self.transform_matrix_inverse = cv2.getPerspectiveTransform(destination_points, source_points)
        get_logger().info("Calculated the perspective transform matrices")
