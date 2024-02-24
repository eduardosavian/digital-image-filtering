import numpy as np

from convolution import convolution

def blur(image: np.ndarray) -> np.ndarray:
    print("\nBlur")
    mask = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]]) * (1/9)

    res = convolution(image, mask).clip(0.0,1.0)
    return res
