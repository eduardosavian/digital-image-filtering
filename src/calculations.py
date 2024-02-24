import numpy as np
import matplotlib.pyplot as plt
from utils import write_plot_image

import numpy as np

import matplotlib.pyplot as plt


def calculate_mse(image_without_noise: np.ndarray, image_with_noise: np.ndarray) -> float:
    n, m = image_without_noise.shape[:2]

    mse = np.sum((image_without_noise - image_with_noise)**2) / (n * m)

    return mse


def calculate_psnr(image_without_noise: np.ndarray, mse_value: float) -> float:
    max_value = np.max(image_without_noise)

    psnr = 10 * np.log10((max_value ** 2) / mse_value)

    return psnr

