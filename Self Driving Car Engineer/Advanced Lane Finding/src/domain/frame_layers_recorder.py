from typing import List, Dict

import matplotlib
import numpy as np

matplotlib.use('Agg')
from matplotlib import pyplot as plt


class FrameLayersRecorder:
    def __init__(self) -> None:
        pass

    @staticmethod
    def record_all_layers(frames: Dict[str, np.ndarray]) -> np.ndarray:
        figure, axes = plt.subplots(2, 3)
        axes[0, 0].imshow(frames["binarized"])
        axes[0, 0].set_title("Gradients")
        axes[0, 1].imshow(frames["region_of_interest"])
        axes[0, 1].set_title("ROI")
        axes[0, 2].imshow(frames["transformed"])
        axes[0, 2].set_title("ROI transformed")
        axes[1, 0].imshow(frames["road_lanes"])
        axes[1, 0].set_title("Lanes transformed")
        axes[1, 1].imshow(frames["road_lanes_reverted"])
        axes[1, 1].set_title("Lanes original")
        axes[1, 2].imshow(frames["final"])
        axes[1, 2].set_title("Final frame")
        figure.canvas.draw()
        data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        return data
