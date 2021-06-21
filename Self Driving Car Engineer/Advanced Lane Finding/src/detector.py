import os

import cv2
import numpy as np
from matplotlib import image as mpimg

from src.logger import get_logger


class Detector:
    def __init__(self) -> None:
        self.images = []
        self.calibration_images = []
        self.output_directory = None
        self.camera_matrix = None
        self.distortion_coefficients = None

    def build(self, images_directory: str, calibration_directory: str, output_directory: str) -> None:
        for image_name in os.listdir(images_directory):
            self.images.append(mpimg.imread(os.path.join(images_directory, image_name)))
        for image_name in os.listdir(calibration_directory):
            self.calibration_images.append(mpimg.imread(os.path.join(calibration_directory, image_name)))
        self.output_directory = output_directory
        get_logger().info("Loaded images from ./{}".format(images_directory))
        self._calibrate_camera()

    def start(self) -> None:
        pass

    def _calibrate_camera(self, nx: int = 9, ny: int = 6):
        get_logger().info("Starting camera calibration")
        all_object_points = np.zeros(shape=(ny * nx, 3), dtype=np.float32)
        all_object_points[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        image_points_2d = []
        object_points_3d = []
        image_shape = self.calibration_images[0].shape[:2]
        for image in self.calibration_images:
            gray = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret:
                object_points_3d.append(all_object_points)
                image_points_2d.append(corners)
        if image_points_2d:
            _, self.camera_matrix, self.distortion_coefficients, _, _ = cv2.calibrateCamera(object_points_3d,
                                                                                            image_points_2d,
                                                                                            image_shape,
                                                                                            None,
                                                                                            None)
            get_logger().info("Calibration parameters calculated")
        else:
            raise RuntimeError("Couldn't detect correctly the chessboard corners. "
                               "Impossible to calculate calibration parameters.")
