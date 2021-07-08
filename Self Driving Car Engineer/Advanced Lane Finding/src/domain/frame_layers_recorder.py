from typing import Dict

import matplotlib
import numpy as np

from src.infrastructure.parameters import *

matplotlib.use('Agg')
from matplotlib import pyplot as plt


class FrameLayersRecorder:
    def __init__(self) -> None:
        pass

    @staticmethod
    def record_all_layers(frames: Dict[str, np.ndarray]) -> np.ndarray:
        figure, axes = plt.subplots(2, 3)
        axes[0, 0].imshow(frames[BINARIZED])
        axes[0, 0].set_title("Gradients")
        axes[0, 1].imshow(frames[ROI])
        axes[0, 1].set_title("ROI")
        axes[0, 2].imshow(frames[TRANSFORMED])
        axes[0, 2].set_title("ROI transformed")
        axes[1, 0].imshow(frames[ROAD_LANES])
        axes[1, 0].set_title("Lanes transformed")
        axes[1, 1].imshow(frames[ROAD_LANES_REVERTED])
        axes[1, 1].set_title("Lanes original")
        axes[1, 2].imshow(frames[FINAL])
        axes[1, 2].set_title("Final frame")
        figure.canvas.draw()
        data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        return data
