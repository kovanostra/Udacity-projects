import os

import cv2
import numpy as np
from matplotlib import image as mpimg

from src.domain.frame_binarizer import FrameBinarizer
from src.domain.frame_layers_recorder import FrameLayersRecorder
from src.domain.frame_transformer import FrameTransformer
from src.domain.logger import get_logger
from src.use_case.lanes_detector import LanesDetector


class ImageLanesDetector(LanesDetector):
    def __init__(self,
                 frame_transformer: FrameTransformer,
                 frame_binarizer: FrameBinarizer,
                 frame_layers_recorder: FrameLayersRecorder) -> None:
        super().__init__(frame_transformer, frame_binarizer, frame_layers_recorder)
        self.images = []
        self.output_directory = None

    def build(self,
              images_directory: str,
              calibration_directory: str,
              output_directory: str,
              record_all_layers: bool) -> None:
        self.record_all_layers = record_all_layers
        self.output_directory = output_directory
        self._load_images(calibration_directory, images_directory)
        self.frame_transformer.calibrate_camera(self.calibration_frames)
        self.frame_transformer.get_perspective_transform_parameters(self.images[0][0])

    def start(self) -> None:
        for image, image_name in self.images:
            get_logger().info("Started processing image {}".format(image_name))
            self._reset_state()
            final_image = self._apply_pipeline(image)
            self._save_final_image(final_image, image_name)

    def _reset_state(self) -> None:
        self.left_parameters = None
        self.right_parameters = None
        self.left_lane = []
        self.right_lane = []

    def _load_images(self, calibration_directory: str, images_directory: str):
        for image_name in os.listdir(images_directory):
            self.images.append([mpimg.imread(os.path.join(images_directory, image_name)), image_name])
        for image_name in os.listdir(calibration_directory):
            self.calibration_frames.append(mpimg.imread(os.path.join(calibration_directory, image_name)))
        get_logger().info("Loaded images from ./{}".format(images_directory))

    def _save_final_image(self, image: np.ndarray, image_name: str) -> None:
        output_path = os.path.join(self.output_directory, image_name)
        image_post_processed = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_post_processed)
        get_logger().info("Successfully saved image at {}".format(output_path))
