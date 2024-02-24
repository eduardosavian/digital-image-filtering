import numpy as np

def put_noise(image: np.ndarray, standard_deviation: float=0.1, mean: float=0.0) -> np.ndarray:
    print("\nNoise")

    image_noise = np.random.normal(mean, standard_deviation, image.shape)
    image_noise += image
    image_noise = np.clip(image_noise, 0.0, 1.0)

    return image_noise